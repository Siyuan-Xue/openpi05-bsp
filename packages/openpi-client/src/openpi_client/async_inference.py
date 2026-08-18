import dataclasses
import threading
import time
import weakref
from typing import Callable, Dict, Optional

from openpi_client import latency_sampling
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


@dataclasses.dataclass(frozen=True)
class ConnectionSnapshot:
    connection_id: int
    metadata_payload: bytes


@dataclasses.dataclass(frozen=True, eq=False)
class _ConnectionAttempt:
    generation: int


@dataclasses.dataclass(frozen=True)
class _ConnectionAttemptResult:
    snapshot: Optional[ConnectionSnapshot] = None
    error: Optional[BaseException] = None


@dataclasses.dataclass(frozen=True)
class InferenceJob:
    request_id: int
    generation: int
    payload: bytes
    submitted_monotonic_ns: int = 0
    latency_sample_key: Optional[latency_sampling.LatencySampleKeyV1] = None
    sampled_target_latency_ns: int = 0


@dataclasses.dataclass(frozen=True)
class InferenceOutcome:
    job: InferenceJob
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False
    completed_monotonic_ns: Optional[int] = None
    connection: Optional[ConnectionSnapshot] = None
    sampled_target_latency_ns: int = 0
    raw_inference_latency_ns: Optional[int] = None
    requested_synthetic_delay_ns: Optional[int] = None
    observed_synthetic_delay_ns: Optional[int] = None
    observed_effective_latency_ns: Optional[int] = None
    latency_overshoot_ns: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class _Completion:
    result: Optional[Dict] = None  # noqa: UP006
    error: Optional[BaseException] = None
    stale: bool = False
    cancelled: bool = False
    completed_monotonic_ns: Optional[int] = None
    connection: Optional[ConnectionSnapshot] = None
    raw_inference_latency_ns: Optional[int] = None
    requested_synthetic_delay_ns: Optional[int] = None
    observed_synthetic_delay_ns: Optional[int] = None
    observed_effective_latency_ns: Optional[int] = None
    latency_overshoot_ns: Optional[int] = None
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
            sampled_target_latency_ns=job.sampled_target_latency_ns,
            raw_inference_latency_ns=self.raw_inference_latency_ns,
            requested_synthetic_delay_ns=self.requested_synthetic_delay_ns,
            observed_synthetic_delay_ns=self.observed_synthetic_delay_ns,
            observed_effective_latency_ns=self.observed_effective_latency_ns,
            latency_overshoot_ns=self.latency_overshoot_ns,
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
        wait_until_ns: Optional[Callable[[int], None]] = None,
        synthetic_latency_target_ms: int = 0,
        latency_sampler: Optional[latency_sampling.NormalLatencySamplerV1] = None,
    ) -> None:
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if recv_poll_interval_s <= 0:
            raise ValueError("recv_poll_interval_s must be positive")
        if (
            isinstance(synthetic_latency_target_ms, bool)
            or not isinstance(synthetic_latency_target_ms, int)
            or synthetic_latency_target_ms < 0
        ):
            raise ValueError("synthetic_latency_target_ms must be a nonnegative integer")
        if latency_sampler is not None and synthetic_latency_target_ms != 0:
            raise ValueError("latency_sampler and synthetic_latency_target_ms are mutually exclusive")
        if latency_sampler is not None and not isinstance(latency_sampler, latency_sampling.NormalLatencySamplerV1):
            raise ValueError("latency_sampler must be a NormalLatencySamplerV1")

        self._policy_factory = policy_factory
        self._shutdown_timeout_s = shutdown_timeout_s
        self._recv_poll_interval_s = recv_poll_interval_s
        self._monotonic_ns = monotonic_ns
        self._wait_until_ns = wait_until_ns or self._default_wait_until_ns
        self._synthetic_latency_target_ns = synthetic_latency_target_ms * 1_000_000
        self._latency_sampler = latency_sampler
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
        self._connection_attempt = None  # type: Optional[_ConnectionAttempt]
        self._connection_attempt_results = weakref.WeakKeyDictionary()
        self._connection = None  # type: Optional[ConnectionSnapshot]
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

    def submit(
        self,
        observation: Dict,  # noqa: UP006
        *,
        latency_sample_key: Optional[latency_sampling.LatencySampleKeyV1] = None,
    ) -> InferenceJob:
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

            if self._latency_sampler is not None:
                if latency_sample_key is None:
                    raise ValueError("latency_sample_key is required when latency_sampler is configured")
                sampled_target_latency_ns = self._latency_sampler.sample_target_ns(latency_sample_key)
            else:
                if latency_sample_key is not None:
                    raise ValueError("latency_sample_key requires a configured latency_sampler")
                sampled_target_latency_ns = self._synthetic_latency_target_ns

            payload = self._packer.pack(observation)
            submitted_monotonic_ns = self._monotonic_ns()
            job = InferenceJob(
                request_id=self._next_request_id,
                generation=self._generation,
                payload=payload,
                submitted_monotonic_ns=submitted_monotonic_ns,
                latency_sample_key=latency_sample_key,
                sampled_target_latency_ns=sampled_target_latency_ns,
            )
            self._next_request_id += 1
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
            attempt = self._connection_attempt
            if attempt is None:
                attempt = _ConnectionAttempt(generation=requested_generation)
                self._connection_attempt = attempt
            self._condition.notify_all()
            while True:
                attempt_result = self._connection_attempt_results.get(attempt)
                if attempt_result is not None:
                    if attempt_result.error is not None:
                        raise attempt_result.error
                    if attempt_result.snapshot is None:
                        raise RuntimeError("Connection attempt result is missing its snapshot")
                    return attempt_result.snapshot
                if requested_generation != self._generation:
                    raise ValueError("Connection request was invalidated by a generation reset")
                self._raise_fatal_locked()
                if self._closing or self._closed:
                    raise RuntimeError("Async inference worker is closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for async inference connection")
                self._condition.wait(remaining)

    def poll(self, job: InferenceJob) -> Optional[InferenceOutcome]:
        """Return a completed outcome for exactly ``job`` without blocking or consuming it."""
        with self._condition:
            self._validate_job_locked(job)
            completion = self._completions.get(job.request_id)
            if completion is None:
                self._raise_fatal_locked()
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
                if generation != self._generation:
                    raise ValueError(
                        f"Generation {generation} is not current (current generation is {self._generation})"
                    )
                self._raise_fatal_locked()
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

            self._connection_attempt = None
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
                self._connection_attempt = None
                self._connection = None
                self._cancel_pending = True
                self._cancel_event.set()
                self._condition.notify_all()

        self._thread.join(self._shutdown_timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(f"Async inference worker thread did not stop within {self._shutdown_timeout_s} seconds")
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
                        and self._connection_attempt is None
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
                        connection_attempt = self._connection_attempt
                        if connection_attempt is None:
                            connection_attempt = _ConnectionAttempt(generation=self._generation)
                            self._connection_attempt = connection_attempt
                    elif self._queued_job is not None:
                        action = "infer"
                        job = self._queued_job
                        self._queued_job = None
                        self._active_job = job
                        connection = self._connection
                        if connection is None:
                            raise RuntimeError("Live policy is missing its connection snapshot")
                    else:
                        connection_attempt = self._connection_attempt
                        if connection_attempt is not None:
                            if self._connection is None:
                                raise RuntimeError("Live policy is missing its connection snapshot")
                            self._publish_connection_attempt_locked(
                                connection_attempt,
                                snapshot=self._connection,
                            )
                            connection_attempt = None
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
                        with self._condition:
                            cancelled = (
                                self._closing
                                or self._cancel_pending
                                or cancel_event.is_set()
                                or cancel_event is not self._cancel_event
                                or self._fatal_error is not None
                                or connection_attempt.generation != self._generation
                                or self._connection_attempt is not connection_attempt
                            )
                            failed_job = self._queued_job
                            if failed_job is not None and failed_job.generation != self._generation:
                                cancelled = True
                            if not cancelled:
                                self._publish_connection_attempt_locked(
                                    connection_attempt,
                                    error=error,
                                )
                                if failed_job is not None:
                                    self._queued_job = None
                                    self._publish_error_locked(
                                        failed_job,
                                        error,
                                        completed_monotonic_ns=None,
                                        connection=None,
                                    )
                            if candidate is not None:
                                self._retirement_pending = True
                            self._condition.notify_all()
                        candidate, retirement_error = self._retire_policy(candidate)
                        if retirement_error is not None:
                            self._record_fatal_error(retirement_error)
                            return
                        self._acknowledge_retirement()
                        if cancelled:
                            self._acknowledge_cancellation(cancel_event)
                        job = None
                        connection_attempt = None
                        continue

                    with self._condition:
                        candidate_is_stale = (
                            self._closing
                            or self._cancel_pending
                            or cancel_event is not self._cancel_event
                            or cancel_event.is_set()
                            or self._fatal_error is not None
                            or (job is not None and job.generation != self._generation)
                            or connection_attempt.generation != self._generation
                            or self._connection_attempt is not connection_attempt
                        )
                        if not candidate_is_stale:
                            connection = ConnectionSnapshot(
                                connection_id=self._next_connection_id,
                                metadata_payload=metadata_payload,
                            )
                            self._next_connection_id += 1
                            policy = candidate
                            self._connection = connection
                            self._publish_connection_attempt_locked(
                                connection_attempt,
                                snapshot=connection,
                            )

                    if candidate_is_stale:
                        candidate, retirement_error = self._retire_policy(candidate)
                        if retirement_error is not None:
                            self._record_fatal_error(retirement_error)
                            return
                        self._acknowledge_cancellation(cancel_event)
                    candidate = None
                    job = None
                    connection_attempt = None
                    continue

                result = None
                error = None
                raw_inference_latency_ns = None
                requested_synthetic_delay_ns = None
                observed_synthetic_delay_ns = None
                observed_effective_latency_ns = None
                latency_overshoot_ns = None
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
                    raw_completed_monotonic_ns = self._monotonic_ns()
                    raw_inference_latency_ns = raw_completed_monotonic_ns - job.submitted_monotonic_ns
                    target_completed_monotonic_ns = job.submitted_monotonic_ns + job.sampled_target_latency_ns
                    if raw_completed_monotonic_ns < target_completed_monotonic_ns:
                        self._wait_until_ns(target_completed_monotonic_ns)
                        completed_monotonic_ns = self._monotonic_ns()
                    else:
                        completed_monotonic_ns = raw_completed_monotonic_ns
                    scheduled_effective_latency_ns = max(
                        raw_inference_latency_ns,
                        job.sampled_target_latency_ns,
                    )
                    requested_synthetic_delay_ns = scheduled_effective_latency_ns - raw_inference_latency_ns
                    observed_synthetic_delay_ns = completed_monotonic_ns - raw_completed_monotonic_ns
                    observed_effective_latency_ns = completed_monotonic_ns - job.submitted_monotonic_ns
                    latency_overshoot_ns = observed_effective_latency_ns - scheduled_effective_latency_ns

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
                                raw_inference_latency_ns=raw_inference_latency_ns,
                                requested_synthetic_delay_ns=requested_synthetic_delay_ns,
                                observed_synthetic_delay_ns=observed_synthetic_delay_ns,
                                observed_effective_latency_ns=observed_effective_latency_ns,
                                latency_overshoot_ns=latency_overshoot_ns,
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
                self._connection_attempt = None
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
        raw_inference_latency_ns: int,
        requested_synthetic_delay_ns: int,
        observed_synthetic_delay_ns: int,
        observed_effective_latency_ns: int,
        latency_overshoot_ns: int,
    ) -> None:
        self._completions[job.request_id] = _Completion(
            result=result,
            completed_monotonic_ns=completed_monotonic_ns,
            connection=connection,
            raw_inference_latency_ns=raw_inference_latency_ns,
            requested_synthetic_delay_ns=requested_synthetic_delay_ns,
            observed_synthetic_delay_ns=observed_synthetic_delay_ns,
            observed_effective_latency_ns=observed_effective_latency_ns,
            latency_overshoot_ns=latency_overshoot_ns,
        )
        self._condition.notify_all()

    @staticmethod
    def _default_wait_until_ns(deadline_ns: int) -> None:
        while True:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            time.sleep(remaining_ns / 1_000_000_000)

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

    def _publish_connection_attempt_locked(
        self,
        attempt: _ConnectionAttempt,
        *,
        snapshot: Optional[ConnectionSnapshot] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if (snapshot is None) == (error is None):
            raise RuntimeError("Connection attempt must publish exactly one snapshot or error")
        if attempt in self._connection_attempt_results:
            raise RuntimeError("Connection attempt already has a terminal result")
        self._connection_attempt_results[attempt] = _ConnectionAttemptResult(
            snapshot=snapshot,
            error=error,
        )
        if self._connection_attempt is attempt:
            self._connection_attempt = None
        self._condition.notify_all()

    def _raise_fatal_locked(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    def _record_fatal_error(self, error: BaseException) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
            self._connection_attempt = None
            self._connection = None
            self._condition.notify_all()

    def _acknowledge_retirement(self) -> None:
        with self._condition:
            if self._retirement_pending:
                self._retirement_pending = False
                self._condition.notify_all()

    def _acknowledge_cancellation(self, cancel_event: threading.Event) -> None:
        with self._condition:
            if not self._closing and self._cancel_pending and cancel_event is self._cancel_event:
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
