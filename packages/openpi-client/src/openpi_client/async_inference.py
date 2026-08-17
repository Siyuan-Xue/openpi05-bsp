import dataclasses
import logging
import threading
import time
import weakref
from typing import Callable, Dict, Optional

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


@dataclasses.dataclass(frozen=True)
class InferenceJob:
    request_id: int
    generation: int
    payload: bytes


@dataclasses.dataclass(frozen=True)
class InferenceOutcome:
    job: InferenceJob
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False


@dataclasses.dataclass(frozen=True)
class _Completion:
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False
    observed: bool = False

    def for_job(self, job: InferenceJob) -> InferenceOutcome:
        return InferenceOutcome(
            job=job,
            result=self.result,
            error=self.error,
            stale=self.stale,
            cancelled=self.cancelled,
        )


class BusyError(RuntimeError):
    """Raised when a request is submitted while another request owns the slot."""


class AsyncInferenceWorker:
    """Runs synchronous policy inference on one non-daemon owner thread.

    Completed outcomes are non-consuming: ``poll`` and ``wait`` may retrieve
    them repeatedly while the caller retains the corresponding job handle.
    The first retrieval establishes the outcome across later generation resets;
    a reset that acquires the state lock first invalidates an unseen outcome.
    Internal completion storage is weakly tied to that handle, so discarding it
    releases the payload and outcome instead of accumulating evaluation history.
    """

    def __init__(
        self,
        policy_factory: Callable[[threading.Event], websocket_client_policy.WebsocketClientPolicy],
        *,
        shutdown_timeout_s: float = 1.0,
        recv_poll_interval_s: float = 0.05,
    ) -> None:
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if recv_poll_interval_s <= 0:
            raise ValueError("recv_poll_interval_s must be positive")

        self._policy_factory = policy_factory
        self._shutdown_timeout_s = shutdown_timeout_s
        self._recv_poll_interval_s = recv_poll_interval_s
        self._packer = msgpack_numpy.Packer()

        self._condition = threading.Condition()
        self._generation = 0
        self._next_request_id = 0
        self._queued_job = None  # type: Optional[InferenceJob]
        self._active_job = None  # type: Optional[InferenceJob]
        self._jobs = {}  # type: Dict[int, weakref.ReferenceType]
        self._completions = {}  # type: Dict[int, _Completion]
        self._cancel_event = threading.Event()
        self._cancel_pending = False
        self._closing = False
        self._closed = False

        self._thread = threading.Thread(
            target=self._run,
            name="openpi-async-inference",
            daemon=False,
        )
        self._thread.start()

    def submit(self, observation: Dict) -> InferenceJob:  # noqa: UP006
        """Snapshot and enqueue one observation without touching the transport."""
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("Async inference worker is closed")
            if self._queued_job is not None or self._active_job is not None or self._cancel_pending:
                raise BusyError("Async inference worker already has an outstanding request")

            payload = self._packer.pack(observation)
            job = InferenceJob(
                request_id=self._next_request_id,
                generation=self._generation,
                payload=payload,
            )
            self._next_request_id += 1
            self._queued_job = job
            self._remember_job_locked(job)
            self._condition.notify_all()
            return job

    def poll(self, job: InferenceJob) -> Optional[InferenceOutcome]:
        """Return a completed outcome for exactly ``job`` without blocking or consuming it."""
        with self._condition:
            self._validate_job_locked(job)
            completion = self._completions.get(job.request_id)
            if completion is None:
                return None
            return self._observe_completion_locked(job.request_id, completion).for_job(job)

    def wait(self, job: InferenceJob, timeout: Optional[float] = None) -> InferenceOutcome:
        """Wait for exactly ``job`` without consuming it or cancelling it on timeout."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._validate_job_locked(job)
            while job.request_id not in self._completions:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for inference job {job.request_id}")
                self._condition.wait(remaining)
            completion = self._completions[job.request_id]
            return self._observe_completion_locked(job.request_id, completion).for_job(job)

    def reset_generation(self) -> int:
        """Invalidate outstanding work and request owner-side socket retirement."""
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("Async inference worker is closed")

            self._generation += 1
            self._invalidate_older_jobs_locked()
            if self._queued_job is not None:
                self._publish_cancelled_locked(self._queued_job)
                self._queued_job = None
            if self._active_job is not None:
                self._publish_cancelled_locked(self._active_job)

            self._cancel_pending = True
            self._cancel_event.set()
            self._condition.notify_all()
            return self._generation

    def close(self) -> None:
        """Stop and join the owner thread within the configured time bound."""
        with self._condition:
            if not self._closed and not self._closing:
                self._closing = True
                self._generation += 1
                if self._queued_job is not None:
                    self._publish_cancelled_locked(self._queued_job)
                    self._queued_job = None
                if self._active_job is not None:
                    self._publish_cancelled_locked(self._active_job)
                self._cancel_pending = True
                self._cancel_event.set()
                self._condition.notify_all()

        self._thread.join(self._shutdown_timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(
                f"Async inference worker thread did not stop within {self._shutdown_timeout_s} seconds"
            )

    def _run(self) -> None:
        policy = None  # type: Optional[websocket_client_policy.WebsocketClientPolicy]
        try:
            while True:
                with self._condition:
                    while (
                        not self._closing
                        and not self._cancel_pending
                        and self._queued_job is None
                    ):
                        self._condition.wait()

                    if self._closing:
                        break

                    cancel_event = self._cancel_event
                    if self._cancel_pending:
                        action = "retire"
                        job = None
                    elif policy is None:
                        action = "connect"
                        job = self._queued_job
                    else:
                        action = "infer"
                        job = self._queued_job
                        self._queued_job = None
                        self._active_job = job

                if action == "retire":
                    policy = self._retire_policy(policy)
                    self._acknowledge_cancellation(cancel_event)
                    continue

                if action == "connect":
                    try:
                        candidate = self._policy_factory(cancel_event)
                    except BaseException as error:
                        with self._condition:
                            cancelled = (
                                self._closing
                                or self._cancel_pending
                                or cancel_event.is_set()
                                or job is None
                                or job.generation != self._generation
                            )
                            if not cancelled and self._queued_job is job:
                                self._queued_job = None
                                self._publish_error_locked(job, error)
                        if cancelled:
                            self._acknowledge_cancellation(cancel_event)
                        job = None
                        continue

                    with self._condition:
                        candidate_is_stale = (
                            self._closing
                            or self._cancel_pending
                            or cancel_event is not self._cancel_event
                            or cancel_event.is_set()
                            or job is None
                            or job.generation != self._generation
                        )
                        if not candidate_is_stale:
                            policy = candidate

                    if candidate_is_stale:
                        self._retire_policy(candidate)
                        self._acknowledge_cancellation(cancel_event)
                    candidate = None
                    job = None
                    continue

                result = None
                error = None
                try:
                    result = policy.infer_packed(
                        job.payload,
                        cancel_event=cancel_event,
                        recv_poll_interval_s=self._recv_poll_interval_s,
                    )
                except BaseException as caught_error:
                    error = caught_error

                with self._condition:
                    cancelled = (
                        self._closing
                        or self._cancel_pending
                        or cancel_event.is_set()
                        or job.generation != self._generation
                    )
                    if job.request_id not in self._completions:
                        if cancelled:
                            self._publish_cancelled_locked(job)
                        elif error is not None:
                            self._publish_error_locked(job, error)
                        else:
                            self._publish_result_locked(job, result)
                    self._active_job = None
                    self._condition.notify_all()

                retire = cancelled or error is not None
                if retire:
                    policy = self._retire_policy(policy)
                if cancelled:
                    self._acknowledge_cancellation(cancel_event)
                job = None
                result = None
                error = None
        finally:
            self._retire_policy(policy)
            with self._condition:
                self._queued_job = None
                self._active_job = None
                self._cancel_pending = False
                self._closed = True
                self._condition.notify_all()

    def _validate_job_locked(self, job: InferenceJob) -> None:
        job_reference = self._jobs.get(job.request_id)
        if job_reference is None or job_reference() is not job:
            raise ValueError("Job was not submitted to this worker")

    def _remember_job_locked(self, job: InferenceJob) -> None:
        worker_reference = weakref.ref(self)
        request_id = job.request_id

        def discard_job(job_reference: weakref.ReferenceType) -> None:
            worker = worker_reference()
            if worker is None:
                return
            with worker._condition:
                if worker._jobs.get(request_id) is job_reference:
                    worker._jobs.pop(request_id, None)
                    worker._completions.pop(request_id, None)

        self._jobs[request_id] = weakref.ref(job, discard_job)

    def _publish_result_locked(self, job: InferenceJob, result: Dict) -> None:  # noqa: UP006
        self._completions[job.request_id] = _Completion(result=result)
        self._condition.notify_all()

    def _publish_error_locked(self, job: InferenceJob, error: BaseException) -> None:
        self._completions[job.request_id] = _Completion(error=error)
        self._condition.notify_all()

    def _publish_cancelled_locked(self, job: InferenceJob) -> None:
        if job.request_id not in self._completions:
            self._completions[job.request_id] = _Completion(stale=True, cancelled=True)
            self._condition.notify_all()

    def _invalidate_older_jobs_locked(self) -> None:
        invalidated = False
        for request_id, job_reference in list(self._jobs.items()):
            job = job_reference()
            completion = self._completions.get(request_id)
            if (
                job is not None
                and job.generation < self._generation
                and (completion is None or not completion.observed)
            ):
                self._completions[request_id] = _Completion(stale=True, cancelled=True)
                invalidated = True
        if invalidated:
            self._condition.notify_all()

    def _observe_completion_locked(self, request_id: int, completion: _Completion) -> _Completion:
        if completion.observed:
            return completion
        observed_completion = dataclasses.replace(completion, observed=True)
        self._completions[request_id] = observed_completion
        return observed_completion

    def _acknowledge_cancellation(self, cancel_event: threading.Event) -> None:
        with self._condition:
            if (
                not self._closing
                and self._cancel_pending
                and cancel_event is self._cancel_event
            ):
                self._cancel_event = threading.Event()
                self._cancel_pending = False
                self._condition.notify_all()

    @staticmethod
    def _retire_policy(
        policy: Optional[websocket_client_policy.WebsocketClientPolicy],
    ) -> Optional[websocket_client_policy.WebsocketClientPolicy]:
        """Attempt owner-side close, then discard the policy even if close reports failure.

        A close exception means transport cleanup isn't proven, so it is logged.
        The failed policy is never reused; later work creates a distinct policy.
        """
        if policy is not None:
            try:
                policy.close()
            except BaseException:
                logging.exception("Failed to close asynchronous inference policy")
        return None
