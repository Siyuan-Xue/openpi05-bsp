import dataclasses
import threading
import time
import weakref
from typing import Callable, Dict, Optional

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


@dataclasses.dataclass(frozen=True)
class ConnectionSnapshot:
    connection_id: int
    metadata_payload: bytes


@dataclasses.dataclass(frozen=True)
class InferenceJob:
    request_id: int
    generation: int
    payload: bytes
    submitted_monotonic_ns: int = 0


@dataclasses.dataclass(frozen=True)
class InferenceOutcome:
    job: InferenceJob
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False
    completed_monotonic_ns: Optional[int] = None
    connection: Optional[ConnectionSnapshot] = None


@dataclasses.dataclass(frozen=True)
class _Completion:
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False
    completed_monotonic_ns: Optional[int] = None
    connection: Optional[ConnectionSnapshot] = None
    observed: bool = False

    def for_job(self, job: InferenceJob) -> InferenceOutcome:
        return InferenceOutcome(
            job=job,
            result=self.result,
            error=self.error,
            stale=self.stale,
            cancelled=self.cancelled,
            completed_monotonic_ns=self.completed_monotonic_ns,
            connection=self.connection,
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
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if recv_poll_interval_s <= 0:
            raise ValueError("recv_poll_interval_s must be positive")

        self._policy_factory = policy_factory
        self._shutdown_timeout_s = shutdown_timeout_s
        self._recv_poll_interval_s = recv_poll_interval_s
        self._monotonic_ns = monotonic_ns
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
        self._retirement_pending = False
        self._connect_requested = False
        self._connection_error = None  # type: Optional[BaseException]
        self._connection = None  # type: Optional[ConnectionSnapshot]
        self._last_published_connection = None  # type: Optional[ConnectionSnapshot]
        self._connection_publication_count = 0
        self._next_connection_id = 0
        self._fatal_error = None  # type: Optional[BaseException]
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
            self._raise_fatal_locked()
            if self._closing or self._closed:
                raise RuntimeError("Async inference worker is closed")
            if (
                self._queued_job is not None
                or self._active_job is not None
                or self._cancel_pending
                or self._retirement_pending
            ):
                raise BusyError("Async inference worker already has an outstanding request")

            payload = self._packer.pack(observation)
            submitted_monotonic_ns = self._monotonic_ns()
            job = InferenceJob(
                request_id=self._next_request_id,
                generation=self._generation,
                payload=payload,
                submitted_monotonic_ns=submitted_monotonic_ns,
            )
            self._next_request_id += 1
            self._connection_error = None
            self._queued_job = job
            self._remember_job_locked(job)
            self._condition.notify_all()
            return job

    def connect(self, timeout: Optional[float] = None) -> ConnectionSnapshot:
        """Ask the owner thread to connect and return immutable server metadata."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._raise_fatal_locked()
            if self._closing or self._closed:
                raise RuntimeError("Async inference worker is closed")
            if self._connection is not None and not self._cancel_pending:
                return self._connection

            requested_generation = self._generation
            publication_count = self._connection_publication_count
            self._connection_error = None
            self._connect_requested = True
            self._condition.notify_all()
            while self._connection is None:
                self._raise_fatal_locked()
                if self._closing or self._closed:
                    raise RuntimeError("Async inference worker is closed")
                if requested_generation != self._generation:
                    raise ValueError("Connection request was invalidated by a generation reset")
                if self._connection_publication_count != publication_count:
                    if self._last_published_connection is None:
                        raise RuntimeError("Connection publication is missing its snapshot")
                    return self._last_published_connection
                if self._connection_error is not None and not self._connect_requested:
                    raise self._connection_error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for async inference connection")
                self._condition.wait(remaining)
            return self._connection

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
                self._raise_fatal_locked()
                if self._closed:
                    raise RuntimeError("Async inference worker closed before completing the job")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for inference job {job.request_id}")
                self._condition.wait(remaining)
            completion = self._completions[job.request_id]
            return self._observe_completion_locked(job.request_id, completion).for_job(job)

    def wait_until_ready(self, generation: int, timeout: Optional[float] = None) -> None:
        """Wait for owner acknowledgement of retirement for exactly ``generation``."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_fatal_locked()
                if generation != self._generation:
                    raise ValueError(
                        f"Generation {generation} is not current (current generation is {self._generation})"
                    )
                if self._closing or self._closed:
                    raise RuntimeError("Async inference worker is closed")
                if not self._cancel_pending and not self._retirement_pending:
                    return
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for generation {generation} readiness")
                self._condition.wait(remaining)

    def reset_generation(self) -> int:
        """Invalidate outstanding work and request owner-side socket retirement."""
        with self._condition:
            self._raise_fatal_locked()
            if self._closing or self._closed:
                raise RuntimeError("Async inference worker is closed")

            self._generation += 1
            self._invalidate_older_jobs_locked()
            if self._queued_job is not None:
                self._publish_cancelled_locked(self._queued_job)
                self._queued_job = None
            if self._active_job is not None:
                self._publish_cancelled_locked(self._active_job)

            self._connect_requested = False
            self._connection_error = None
            self._connection = None
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
                self._connect_requested = False
                self._connection = None
                self._cancel_pending = True
                self._cancel_event.set()
                self._condition.notify_all()

        self._thread.join(self._shutdown_timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(
                f"Async inference worker thread did not stop within {self._shutdown_timeout_s} seconds"
            )
        with self._condition:
            self._raise_fatal_locked()

    def _run(self) -> None:
        policy = None  # type: Optional[websocket_client_policy.WebsocketClientPolicy]
        try:
            while True:
                with self._condition:
                    while (
                        not self._closing
                        and self._fatal_error is None
                        and not self._cancel_pending
                        and self._queued_job is None
                        and not self._connect_requested
                    ):
                        self._condition.wait()

                    if self._closing or self._fatal_error is not None:
                        break

                    cancel_event = self._cancel_event
                    if self._cancel_pending:
                        action = "retire"
                        job = None
                    elif policy is None:
                        action = "connect"
                        job = self._queued_job
                    elif self._queued_job is not None:
                        action = "infer"
                        job = self._queued_job
                        self._queued_job = None
                        self._active_job = job
                        connection = self._connection
                        if connection is None:
                            raise RuntimeError("Live policy is missing its connection snapshot")
                    else:
                        self._connect_requested = False
                        self._condition.notify_all()
                        continue

                if action == "retire":
                    policy, retirement_error = self._retire_policy(policy)
                    if retirement_error is not None:
                        self._record_fatal_error(retirement_error)
                        return
                    self._acknowledge_retirement()
                    self._acknowledge_cancellation(cancel_event)
                    continue

                if action == "connect":
                    candidate = None
                    try:
                        candidate = self._policy_factory(cancel_event)
                        metadata_payload = msgpack_numpy.packb(candidate.get_server_metadata())
                    except BaseException as error:
                        candidate, retirement_error = self._retire_policy(candidate)
                        if retirement_error is not None:
                            self._record_fatal_error(retirement_error)
                            return
                        with self._condition:
                            cancelled = (
                                self._closing
                                or self._cancel_pending
                                or cancel_event.is_set()
                                or cancel_event is not self._cancel_event
                            )
                            failed_job = self._queued_job
                            if failed_job is not None and failed_job.generation != self._generation:
                                cancelled = True
                            self._connect_requested = False
                            if not cancelled:
                                self._connection_error = error
                                if failed_job is not None:
                                    self._queued_job = None
                                    self._publish_error_locked(
                                        failed_job,
                                        error,
                                        completed_monotonic_ns=None,
                                        connection=None,
                                    )
                            self._condition.notify_all()
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
                            or (job is not None and job.generation != self._generation)
                        )
                        if not candidate_is_stale:
                            connection = ConnectionSnapshot(
                                connection_id=self._next_connection_id,
                                metadata_payload=metadata_payload,
                            )
                            self._next_connection_id += 1
                            policy = candidate
                            self._connection = connection
                            self._last_published_connection = connection
                            self._connection_publication_count += 1
                            self._connection_error = None
                            self._connect_requested = False
                            self._condition.notify_all()

                    if candidate_is_stale:
                        candidate, retirement_error = self._retire_policy(candidate)
                        if retirement_error is not None:
                            self._record_fatal_error(retirement_error)
                            return
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
                    completed_monotonic_ns = self._monotonic_ns()
                else:
                    completed_monotonic_ns = self._monotonic_ns()

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
                            self._publish_error_locked(
                                job,
                                error,
                                completed_monotonic_ns=completed_monotonic_ns,
                                connection=connection,
                            )
                        else:
                            self._publish_result_locked(
                                job,
                                result,
                                completed_monotonic_ns=completed_monotonic_ns,
                                connection=connection,
                            )
                    self._active_job = None
                    retire = cancelled or error is not None
                    if retire:
                        self._connection = None
                        self._retirement_pending = True
                    self._condition.notify_all()

                if retire:
                    policy, retirement_error = self._retire_policy(policy)
                    if retirement_error is not None:
                        self._record_fatal_error(retirement_error)
                        return
                    self._acknowledge_retirement()
                if cancelled:
                    self._acknowledge_cancellation(cancel_event)
                job = None
                result = None
                error = None
        except BaseException as error:
            self._record_fatal_error(error)
        finally:
            policy, retirement_error = self._retire_policy(policy)
            if retirement_error is not None:
                self._record_fatal_error(retirement_error)
            with self._condition:
                self._queued_job = None
                self._active_job = None
                self._connect_requested = False
                self._connection = None
                self._cancel_pending = False
                self._retirement_pending = False
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

    def _publish_result_locked(
        self,
        job: InferenceJob,
        result: Dict,  # noqa: UP006
        *,
        completed_monotonic_ns: int,
        connection: ConnectionSnapshot,
    ) -> None:
        self._completions[job.request_id] = _Completion(
            result=result,
            completed_monotonic_ns=completed_monotonic_ns,
            connection=connection,
        )
        self._condition.notify_all()

    def _publish_error_locked(
        self,
        job: InferenceJob,
        error: BaseException,
        *,
        completed_monotonic_ns: Optional[int],
        connection: Optional[ConnectionSnapshot],
    ) -> None:
        self._completions[job.request_id] = _Completion(
            error=error,
            completed_monotonic_ns=completed_monotonic_ns,
            connection=connection,
        )
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

    def _raise_fatal_locked(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    def _record_fatal_error(self, error: BaseException) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
            self._connect_requested = False
            self._connection = None
            self._condition.notify_all()

    def _acknowledge_retirement(self) -> None:
        with self._condition:
            if self._retirement_pending:
                self._retirement_pending = False
                self._condition.notify_all()

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
    ):
        """Owner-side close that never reuses a policy and preserves close failure."""
        retirement_error = None
        if policy is not None:
            try:
                policy.close()
            except BaseException as error:
                retirement_error = error
        return None, retirement_error
