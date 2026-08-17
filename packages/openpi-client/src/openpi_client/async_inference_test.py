import gc
import threading
import time
import weakref

import numpy as np
import pytest

from openpi_client import async_inference
from openpi_client import msgpack_numpy


class _GatePolicy:
    def __init__(
        self,
        *,
        result=None,
        error=None,
        metadata=None,
        metadata_error=None,
        return_after_cancel=False,
        ignore_cancel=False,
        close_gate=None,
        close_error=None,
    ):
        self.result = {"actions": [1, 2, 3]} if result is None else result
        self.error = error
        self.metadata = {"server": "test"} if metadata is None else metadata
        self.metadata_error = metadata_error
        self.return_after_cancel = return_after_cancel
        self.ignore_cancel = ignore_cancel
        self.close_gate = close_gate
        self.close_error = close_error
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_observed = threading.Event()
        self.close_started = threading.Event()
        self.closed = threading.Event()
        self.payloads = []
        self.metadata_thread_ids = []
        self.infer_thread_ids = []
        self.close_thread_ids = []
        self.active_calls = 0
        self.max_active_calls = 0

    def get_server_metadata(self):
        self.metadata_thread_ids.append(threading.get_ident())
        if self.metadata_error is not None:
            raise self.metadata_error
        return self.metadata

    def infer_packed(self, payload, *, cancel_event=None, recv_poll_interval_s=None):
        self.infer_thread_ids.append(threading.get_ident())
        self.payloads.append(payload)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.started.set()
        try:
            while not self.release.wait(0.005):
                if cancel_event is not None and cancel_event.is_set() and not self.ignore_cancel:
                    self.cancel_observed.set()
                    if self.return_after_cancel:
                        return {"late": True}
                    raise RuntimeError("fake policy observed cancellation")
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.active_calls -= 1

    def close(self):
        self.close_thread_ids.append(threading.get_ident())
        self.close_started.set()
        if self.close_gate is not None:
            self.close_gate.wait()
        self.closed.set()
        if self.close_error is not None:
            raise self.close_error


class _Factory:
    def __init__(self, policies):
        self.policies = list(policies)
        self.cancel_events = []
        self.thread_ids = []

    def __call__(self, cancel_event):
        self.cancel_events.append(cancel_event)
        self.thread_ids.append(threading.get_ident())
        return self.policies[len(self.cancel_events) - 1]


class _BlockingFactory:
    def __init__(self, policy, *, ignore_cancel=False):
        self.policy = policy
        self.ignore_cancel = ignore_cancel
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_observed = threading.Event()
        self.created_policies = []

    def __call__(self, cancel_event):
        self.started.set()
        while not self.release.wait(0.005):
            if cancel_event.is_set() and not self.ignore_cancel:
                self.cancel_observed.set()
                raise RuntimeError("fake connect retry cancelled")
        self.created_policies.append(self.policy)
        return self.policy


class _GatedSequenceFactory:
    def __init__(self, policies):
        self.policies = list(policies)
        self.started = [threading.Event() for _ in self.policies]
        self.release = [threading.Event() for _ in self.policies]
        self.cancel_events = []

    def __call__(self, cancel_event):
        index = len(self.cancel_events)
        self.cancel_events.append(cancel_event)
        self.started[index].set()
        self.release[index].wait()
        return self.policies[index]


class _FailingFactory:
    def __init__(self, error):
        self.error = error
        self.started = threading.Event()

    def __call__(self, cancel_event):
        self.started.set()
        raise self.error


class _ManualMonotonicNs:
    def __init__(self, initial_ns=0):
        self._condition = threading.Condition()
        self._now_ns = initial_ns
        self._call_count = 0

    def __call__(self):
        with self._condition:
            self._call_count += 1
            self._condition.notify_all()
            return self._now_ns

    def set(self, now_ns):
        with self._condition:
            self._now_ns = now_ns
            self._condition.notify_all()

    def wait_for_call_count(self, count, timeout=1.0):
        with self._condition:
            return self._condition.wait_for(lambda: self._call_count >= count, timeout=timeout)

    @property
    def call_count(self):
        with self._condition:
            return self._call_count


def _wait_for_stored_completion(worker, job):
    with worker._condition:
        assert worker._condition.wait_for(
            lambda: job.request_id in worker._completions,
            timeout=1.0,
        )


def test_connect_reuses_live_snapshot_and_reconnects_after_retirement():
    first_policy = _GatePolicy(metadata={"server": "one"})
    second_policy = _GatePolicy(metadata={"server": "one"})
    factory = _Factory([first_policy, second_policy])
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        first = worker.connect(timeout=1.0)
        repeated = worker.connect(timeout=1.0)

        assert repeated is first
        assert msgpack_numpy.unpackb(first.metadata_payload) == {"server": "one"}
        assert len(factory.cancel_events) == 1

        generation = worker.reset_generation()
        worker.wait_until_ready(generation, timeout=1.0)
        second = worker.connect(timeout=1.0)

        assert second.connection_id == first.connection_id + 1
        assert second is not first
        assert msgpack_numpy.unpackb(second.metadata_payload) == {"server": "one"}
        assert len(factory.cancel_events) == 2
    finally:
        first_policy.release.set()
        second_policy.release.set()
        worker.close()


def test_connect_publishes_defensive_metadata_snapshot_read_by_owner():
    metadata = {
        "nested": {"version": 1},
        "array": np.array([1, 2], dtype=np.int32),
    }
    policy = _GatePolicy(metadata=metadata)
    factory = _Factory([policy])
    calling_thread_id = threading.get_ident()
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        connection = worker.connect(timeout=1.0)
        metadata["nested"]["version"] = 99
        metadata["array"][:] = 0

        assert isinstance(connection.metadata_payload, bytes)
        unpacked = msgpack_numpy.unpackb(connection.metadata_payload)
        assert unpacked["nested"] == {"version": 1}
        np.testing.assert_array_equal(unpacked["array"], np.array([1, 2], dtype=np.int32))
        assert factory.thread_ids == policy.metadata_thread_ids
        assert factory.thread_ids[0] != calling_thread_id
    finally:
        policy.release.set()
        worker.close()


def test_queued_job_and_connect_converge_on_same_owner_connection():
    policy = _GatePolicy(metadata={"server": "queued"}, result={"answer": 7})
    policy.release.set()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory)
    connect_result = []
    connect_finished = threading.Event()

    def connect():
        try:
            connect_result.append(worker.connect(timeout=1.0))
        finally:
            connect_finished.set()

    try:
        job = worker.submit({"request": "queued"})
        assert factory.started.wait(1.0)
        connect_thread = threading.Thread(target=connect)
        connect_thread.start()

        factory.release.set()
        assert connect_finished.wait(1.0)
        connect_thread.join()
        outcome = worker.wait(job, timeout=1.0)

        assert len(connect_result) == 1
        assert outcome.connection is connect_result[0]
        assert outcome.result == {"answer": 7}
        assert len(factory.created_policies) == 1
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_queued_job_error_and_connect_share_snapshot_even_after_retirement():
    error = OSError("inference failed")
    policy = _GatePolicy(metadata={"server": "queued-error"}, error=error)
    policy.release.set()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory)
    connect_result = []
    connect_finished = threading.Event()

    def connect():
        try:
            connect_result.append(worker.connect(timeout=1.0))
        finally:
            connect_finished.set()

    try:
        job = worker.submit({"request": "queued-error"})
        assert factory.started.wait(1.0)
        connect_thread = threading.Thread(target=connect)
        connect_thread.start()

        factory.release.set()
        assert connect_finished.wait(1.0)
        connect_thread.join()
        outcome = worker.wait(job, timeout=1.0)

        assert outcome.error is error
        assert len(connect_result) == 1
        assert outcome.connection is connect_result[0]
        assert policy.closed.wait(1.0)
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_delayed_connect_waiter_keeps_exact_registered_attempt_across_reconnect(monkeypatch):
    first_policy = _GatePolicy(metadata={"server": "first"}, result={"request": "first"})
    first_policy.release.set()
    second_policy = _GatePolicy(metadata={"server": "second"})
    factory = _GatedSequenceFactory([first_policy, second_policy])
    worker = async_inference.AsyncInferenceWorker(factory)
    waiter_registered = threading.Event()
    waiter_woke = threading.Event()
    release_waiter = threading.Event()
    first_connect_result = []
    real_condition_wait = worker._condition.wait
    delayed_once = []
    connect_thread = None

    def delayed_condition_wait(timeout=None):
        if threading.current_thread() is connect_thread:
            waiter_registered.set()
        result = real_condition_wait(timeout)
        if threading.current_thread() is connect_thread and not delayed_once:
            delayed_once.append(True)
            waiter_woke.set()
            worker._condition.release()
            try:
                release_waiter.wait()
            finally:
                worker._condition.acquire()
        return result

    monkeypatch.setattr(worker._condition, "wait", delayed_condition_wait)

    def connect_first():
        first_connect_result.append(worker.connect(timeout=1.0))

    try:
        job = worker.submit({"request": "first"})
        assert factory.started[0].wait(1.0)
        connect_thread = threading.Thread(target=connect_first)
        connect_thread.start()
        assert waiter_registered.wait(1.0)

        factory.release[0].set()
        assert waiter_woke.wait(1.0)
        first_outcome = worker.wait(job, timeout=1.0)
        first_connection = first_outcome.connection

        generation = worker.reset_generation()
        worker.wait_until_ready(generation, timeout=1.0)
        factory.release[1].set()
        second_connection = worker.connect(timeout=1.0)

        release_waiter.set()
        connect_thread.join()
        assert first_connect_result == [first_connection]
        assert second_connection.connection_id == first_connection.connection_id + 1
        assert msgpack_numpy.unpackb(first_connect_result[0].metadata_payload) == {"server": "first"}
        assert msgpack_numpy.unpackb(second_connection.metadata_payload) == {"server": "second"}
    finally:
        release_waiter.set()
        for release in factory.release:
            release.set()
        first_policy.release.set()
        second_policy.release.set()
        if connect_thread is not None:
            connect_thread.join()
        worker.close()


def test_submit_and_completion_timestamps_use_injected_clock_at_exact_seams():
    clock = _ManualMonotonicNs(11)
    policy = _GatePolicy(metadata={"server": "timed"}, result={"answer": 7})
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]), monotonic_ns=clock)
    try:
        job = worker.submit({"request": "timed"})
        assert job.submitted_monotonic_ns == 11
        assert policy.started.wait(1.0)

        clock.set(47)
        with worker._condition:
            policy.release.set()
            assert clock.wait_for_call_count(2)
            assert job.request_id not in worker._completions
        outcome = worker.wait(job, timeout=1.0)

        assert outcome.completed_monotonic_ns == 47
        assert outcome.connection is not None
        assert msgpack_numpy.unpackb(outcome.connection.metadata_payload) == {"server": "timed"}
        assert clock.call_count == 2
    finally:
        policy.release.set()
        worker.close()


def test_current_generation_inference_error_carries_timestamp_and_connection():
    clock = _ManualMonotonicNs(3)
    error = OSError("transport failed")
    policy = _GatePolicy(metadata={"server": "error"}, error=error)
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]), monotonic_ns=clock)
    try:
        job = worker.submit({"request": "error"})
        assert policy.started.wait(1.0)
        clock.set(29)
        policy.release.set()

        outcome = worker.wait(job, timeout=1.0)
        assert outcome.error is error
        assert outcome.completed_monotonic_ns == 29
        assert outcome.connection is not None
        assert msgpack_numpy.unpackb(outcome.connection.metadata_payload) == {"server": "error"}
    finally:
        policy.release.set()
        worker.close()


def test_wait_until_ready_acknowledges_retirement_without_starting_connection():
    policy = _GatePolicy()
    factory = _Factory([policy])
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        generation = worker.reset_generation()
        worker.wait_until_ready(generation, timeout=1.0)

        assert factory.cancel_events == []
        job = worker.submit({"generation": generation})
        policy.release.set()
        assert worker.wait(job, timeout=1.0).result == policy.result
    finally:
        policy.release.set()
        worker.close()


def test_wait_until_ready_rejects_generation_invalidated_by_later_reset():
    worker = async_inference.AsyncInferenceWorker(_Factory([]))
    try:
        old_generation = worker.reset_generation()
        current_generation = worker.reset_generation()

        with pytest.raises(ValueError, match="generation"):
            worker.wait_until_ready(old_generation, timeout=1.0)
        worker.wait_until_ready(current_generation, timeout=1.0)
    finally:
        worker.close()


def test_poll_and_wait_do_not_complete_before_policy_gate_opens():
    policy = _GatePolicy(result={"answer": 7})
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"observation": 1})
        assert policy.started.wait(1.0)
        assert worker.poll(job) is None

        with pytest.raises(TimeoutError, match="inference job"):
            worker.wait(job, timeout=0.01)

        assert not policy.cancel_observed.is_set()
        policy.release.set()
        outcome = worker.wait(job, timeout=1.0)
        assert outcome.job == job
        assert outcome.result == {"answer": 7}
        assert outcome.error is None
        assert not outcome.stale
        assert not outcome.cancelled
    finally:
        worker.close()


def test_submit_rejects_second_job_while_first_is_queued():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        first = worker.submit({"request": 1})
        assert factory.started.wait(1.0)

        with pytest.raises(async_inference.BusyError):
            worker.submit({"request": 2})

        factory.release.set()
        policy.release.set()
        assert worker.wait(first, timeout=1.0).result == policy.result
        assert len(factory.created_policies) == 1
        assert len(policy.payloads) == 1
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_submit_rejects_second_job_while_first_is_active():
    policy = _GatePolicy()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        first = worker.submit({"request": 1})
        assert policy.started.wait(1.0)

        with pytest.raises(async_inference.BusyError):
            worker.submit({"request": 2})

        policy.release.set()
        assert worker.wait(first, timeout=1.0).result == policy.result
        assert policy.max_active_calls == 1
        assert len(policy.payloads) == 1
    finally:
        policy.release.set()
        worker.close()


def test_submit_snapshots_nested_values_and_numpy_arrays_immediately():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory)
    observation = {
        "nested": {"value": 3},
        "image": np.array([[1, 2], [3, 4]], dtype=np.uint8),
    }
    try:
        job = worker.submit(observation)
        assert factory.started.wait(1.0)
        observation["nested"]["value"] = 99
        observation["image"][:] = 0

        snapshot = msgpack_numpy.unpackb(job.payload)
        assert snapshot["nested"] == {"value": 3}
        np.testing.assert_array_equal(snapshot["image"], np.array([[1, 2], [3, 4]], dtype=np.uint8))

        factory.release.set()
        policy.release.set()
        worker.wait(job, timeout=1.0)
        delivered = msgpack_numpy.unpackb(policy.payloads[0])
        assert delivered["nested"] == {"value": 3}
        np.testing.assert_array_equal(delivered["image"], np.array([[1, 2], [3, 4]], dtype=np.uint8))
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_current_generation_result_is_delivered_to_exact_job():
    policy = _GatePolicy(result={"request": "first"})
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"request": "first"})
        outcome = worker.wait(job, timeout=1.0)
        assert outcome.job is job
        assert outcome.result == {"request": "first"}
        repeated = worker.poll(job)
        assert repeated == outcome
        assert repeated.job is job
    finally:
        worker.close()


def test_completed_job_storage_is_released_when_caller_drops_handle():
    policy = _GatePolicy(result={"request": "complete"})
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"image": np.ones((64, 64, 3), dtype=np.uint8)})
        worker.wait(job, timeout=1.0)
        job_reference = weakref.ref(job)

        del job
        deadline = time.monotonic() + 1.0
        while job_reference() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.005)

        assert job_reference() is None
        assert worker._jobs == {}
        assert worker._completions == {}
    finally:
        worker.close()


def test_current_generation_error_is_delivered_unchanged():
    error = OSError("transport failed")
    policy = _GatePolicy(error=error)
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"request": "error"})
        outcome = worker.wait(job, timeout=1.0)
        assert outcome.error is error
        assert outcome.result is None
        assert isinstance(outcome.completed_monotonic_ns, int)
        assert outcome.connection is not None
        assert not outcome.stale
        assert not outcome.cancelled
    finally:
        worker.close()


def test_current_generation_factory_error_is_delivered_unchanged():
    error = ConnectionError("factory connect failed")
    factory = _FailingFactory(error)
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        job = worker.submit({"request": "connect error"})
        assert factory.started.wait(1.0)
        outcome = worker.wait(job, timeout=1.0)
        assert outcome.error is error
        assert outcome.result is None
        assert outcome.completed_monotonic_ns is None
        assert outcome.connection is None
        assert not outcome.stale
        assert not outcome.cancelled
    finally:
        worker.close()


def test_reset_invalidates_completed_unpolled_result():
    policy = _GatePolicy(result={"old": "result"})
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"generation": 0})
        _wait_for_stored_completion(worker, job)

        assert worker.reset_generation() == 1
        outcome = worker.poll(job)
        assert outcome.result is None
        assert outcome.error is None
        assert outcome.stale
        assert outcome.cancelled
    finally:
        worker.close()


def test_reset_invalidates_completed_unpolled_error():
    error = OSError("old generation transport failure")
    policy = _GatePolicy(error=error)
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"generation": 0})
        _wait_for_stored_completion(worker, job)

        assert worker.reset_generation() == 1
        outcome = worker.poll(job)
        assert outcome.result is None
        assert outcome.error is None
        assert outcome.stale
        assert outcome.cancelled
    finally:
        worker.close()


def test_reset_preserves_completed_result_observed_by_wait():
    policy = _GatePolicy(result={"observed": "result"})
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"generation": 0})
        first_outcome = worker.wait(job, timeout=1.0)

        assert worker.reset_generation() == 1
        repeated_outcome = worker.poll(job)
        assert repeated_outcome == first_outcome
        assert repeated_outcome.result == {"observed": "result"}
        assert repeated_outcome.error is None
        assert not repeated_outcome.stale
        assert not repeated_outcome.cancelled
    finally:
        worker.close()


def test_reset_preserves_completed_error_observed_by_poll():
    error = OSError("observed transport failure")
    policy = _GatePolicy(error=error)
    policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"generation": 0})
        _wait_for_stored_completion(worker, job)
        first_outcome = worker.poll(job)

        assert worker.reset_generation() == 1
        repeated_outcome = worker.poll(job)
        assert repeated_outcome == first_outcome
        assert repeated_outcome.result is None
        assert repeated_outcome.error is error
        assert not repeated_outcome.stale
        assert not repeated_outcome.cancelled
    finally:
        worker.close()


def test_reset_of_queued_job_is_stale_without_transport_use():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        job = worker.submit({"request": "queued"})
        assert factory.started.wait(1.0)

        assert worker.reset_generation() == 1
        outcome = worker.wait(job, timeout=1.0)
        assert outcome.job == job
        assert outcome.stale
        assert outcome.cancelled
        assert outcome.result is None
        assert outcome.error is None
        assert factory.cancel_observed.wait(1.0)
        assert factory.created_policies == []
        assert policy.payloads == []
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_active_reset_retires_socket_before_fresh_generation_is_accepted():
    close_gate = threading.Event()
    old_policy = _GatePolicy(close_gate=close_gate)
    new_policy = _GatePolicy(result={"generation": 1})
    new_policy.release.set()
    factory = _Factory([old_policy, new_policy])
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        old_job = worker.submit({"generation": 0})
        assert old_policy.started.wait(1.0)
        old_connection = worker.connect(timeout=1.0)

        generation = worker.reset_generation()
        assert generation == 1
        stale = worker.wait(old_job, timeout=1.0)
        assert stale.stale and stale.cancelled
        assert old_policy.close_started.wait(1.0)
        with pytest.raises(async_inference.BusyError):
            worker.submit({"too": "early"})
        with pytest.raises(TimeoutError, match="generation"):
            worker.wait_until_ready(generation, timeout=0)

        ready = threading.Event()

        def wait_until_ready():
            worker.wait_until_ready(generation, timeout=1.0)
            ready.set()

        ready_thread = threading.Thread(target=wait_until_ready)
        ready_thread.start()
        assert not ready.is_set()

        close_gate.set()
        assert old_policy.closed.wait(1.0)
        assert ready.wait(1.0)
        ready_thread.join()
        new_job = worker.submit({"generation": 1})
        fresh = worker.wait(new_job, timeout=1.0)
        assert new_job.generation == 1
        assert fresh.result == {"generation": 1}
        assert fresh.connection.connection_id == old_connection.connection_id + 1
        assert factory.cancel_events[0] is not factory.cancel_events[1]
        assert len(factory.cancel_events) == 2
    finally:
        close_gate.set()
        old_policy.release.set()
        new_policy.release.set()
        worker.close()


def test_repeated_reset_keeps_submit_blocked_until_owner_retires_old_socket():
    close_gate = threading.Event()
    old_policy = _GatePolicy(close_gate=close_gate)
    new_policy = _GatePolicy(result={"generation": 2})
    new_policy.release.set()
    worker = async_inference.AsyncInferenceWorker(_Factory([old_policy, new_policy]))
    try:
        old_job = worker.submit({"generation": 0})
        assert old_policy.started.wait(1.0)
        assert worker.reset_generation() == 1
        assert old_policy.close_started.wait(1.0)

        assert worker.reset_generation() == 2
        with pytest.raises(async_inference.BusyError):
            worker.submit({"too": "early"})

        close_gate.set()
        assert old_policy.closed.wait(1.0)
        worker.wait_until_ready(2, timeout=1.0)
        new_job = worker.submit({"generation": 2})
        outcome = worker.wait(new_job, timeout=1.0)
        assert new_job.generation == 2
        assert outcome.result == {"generation": 2}
        stale = worker.poll(old_job)
        assert stale.stale and stale.cancelled
        assert stale.result is None and stale.error is None
    finally:
        close_gate.set()
        old_policy.release.set()
        new_policy.release.set()
        worker.close()


def test_late_factory_candidate_after_reset_is_closed_without_inference():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy, ignore_cancel=True)
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        job = worker.submit({"request": "queued"})
        assert factory.started.wait(1.0)
        worker.reset_generation()

        factory.release.set()
        assert policy.closed.wait(1.0)
        outcome = worker.poll(job)
        assert outcome.stale and outcome.cancelled
        assert outcome.result is None and outcome.error is None
        assert policy.payloads == []
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_policy_close_failure_is_fatal_and_observable_on_readiness_and_close():
    close_error = OSError("close handshake failed")
    old_policy = _GatePolicy(close_error=close_error)
    worker = async_inference.AsyncInferenceWorker(_Factory([old_policy]))
    try:
        old_job = worker.submit({"generation": 0})
        assert old_policy.started.wait(1.0)
        generation = worker.reset_generation()
        assert old_policy.closed.wait(1.0)

        assert worker.poll(old_job).stale
        with pytest.raises(ValueError, match="Generation 0 is not current"):
            worker.wait_until_ready(0, timeout=1.0)
        with pytest.raises(OSError, match="close handshake failed"):
            worker.wait_until_ready(generation, timeout=1.0)
        with pytest.raises(OSError, match="close handshake failed"):
            worker.connect(timeout=1.0)
        with pytest.raises(OSError, match="close handshake failed"):
            worker.close()
        with pytest.raises(OSError, match="close handshake failed"):
            worker.close()
        assert len(old_policy.payloads) == 1
    finally:
        old_policy.release.set()
        try:
            worker.close()
        except OSError:
            pass


def test_metadata_and_close_failure_publish_job_error_before_fatal_retirement():
    metadata_error = ValueError("metadata malformed")
    close_error = OSError("metadata candidate close failed")
    policy = _GatePolicy(metadata_error=metadata_error, close_error=close_error)
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        job = worker.submit({"request": "metadata-error"})
        assert policy.closed.wait(1.0)

        outcome = worker.poll(job)
        assert outcome.error is metadata_error
        assert outcome.completed_monotonic_ns is None
        assert outcome.connection is None
        with pytest.raises(OSError, match="metadata candidate close failed"):
            worker.close()
    finally:
        try:
            worker.close()
        except OSError:
            pass


def test_metadata_and_close_failure_preserve_explicit_connect_error():
    metadata_error = ValueError("metadata unavailable")
    close_error = OSError("failed to retire metadata candidate")
    policy = _GatePolicy(metadata_error=metadata_error, close_error=close_error)
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        with pytest.raises(ValueError, match="metadata unavailable"):
            worker.connect(timeout=1.0)
        assert policy.closed.wait(1.0)
        with pytest.raises(OSError, match="failed to retire metadata candidate"):
            worker.close()
    finally:
        try:
            worker.close()
        except OSError:
            pass


def test_poll_raises_terminal_fatal_when_job_has_no_completion():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy, ignore_cancel=True)
    fatal_error = RuntimeError("owner terminated")
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        job = worker.submit({"request": "never-completed"})
        assert factory.started.wait(1.0)
        worker._record_fatal_error(fatal_error)

        with pytest.raises(RuntimeError, match="owner terminated"):
            worker.poll(job)
    finally:
        factory.release.set()
        policy.release.set()
        try:
            worker.close()
        except RuntimeError:
            pass


def test_close_surfaces_owner_policy_close_failure_idempotently():
    close_error = OSError("close failed during shutdown")
    policy = _GatePolicy(close_error=close_error)
    calling_thread_id = threading.get_ident()
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]))
    try:
        worker.connect(timeout=1.0)

        with pytest.raises(OSError, match="close failed during shutdown"):
            worker.close()
        with pytest.raises(OSError, match="close failed during shutdown"):
            worker.close()

        assert len(policy.close_thread_ids) == 1
        assert policy.close_thread_ids[0] != calling_thread_id
    finally:
        try:
            worker.close()
        except OSError:
            pass


def test_late_result_after_active_reset_cannot_replace_stale_outcome():
    old_policy = _GatePolicy(return_after_cancel=True)
    worker = async_inference.AsyncInferenceWorker(_Factory([old_policy]))
    try:
        job = worker.submit({"request": "old"})
        assert old_policy.started.wait(1.0)
        worker.reset_generation()
        first_outcome = worker.wait(job, timeout=1.0)
        assert old_policy.closed.wait(1.0)

        repeated_outcome = worker.poll(job)
        assert repeated_outcome == first_outcome
        assert repeated_outcome.job is job
        assert first_outcome.result is None
        assert first_outcome.stale
        assert first_outcome.cancelled
    finally:
        old_policy.release.set()
        worker.close()


def test_close_interrupts_policy_factory_retry_and_exits_worker_thread():
    policy = _GatePolicy()
    factory = _BlockingFactory(policy)
    worker = async_inference.AsyncInferenceWorker(factory, shutdown_timeout_s=0.5)
    try:
        worker.submit({"request": "connect"})
        assert factory.started.wait(1.0)

        worker.close()
        worker.close()

        assert factory.cancel_observed.is_set()
        assert not worker._thread.daemon
        assert not worker._thread.is_alive()
        assert factory.created_policies == []
    finally:
        factory.release.set()
        policy.release.set()
        worker.close()


def test_close_interrupts_active_receive_and_owner_closes_socket():
    policy = _GatePolicy()
    factory = _Factory([policy])
    calling_thread_id = threading.get_ident()
    worker = async_inference.AsyncInferenceWorker(factory, shutdown_timeout_s=0.5)
    try:
        worker.submit({"request": "active"})
        assert policy.started.wait(1.0)

        worker.close()

        assert policy.cancel_observed.is_set()
        assert policy.closed.is_set()
        assert not worker._thread.is_alive()
        assert (
            factory.thread_ids
            == policy.metadata_thread_ids
            == policy.infer_thread_ids
            == policy.close_thread_ids
        )
        assert factory.thread_ids[0] != calling_thread_id
    finally:
        policy.release.set()
        worker.close()


def test_close_raises_timeout_if_non_daemon_owner_does_not_stop_in_bound():
    policy = _GatePolicy(ignore_cancel=True)
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]), shutdown_timeout_s=0.01)
    try:
        worker.submit({"request": "stuck"})
        assert policy.started.wait(1.0)

        with pytest.raises(TimeoutError, match="worker thread"):
            worker.close()

        policy.release.set()
        worker._shutdown_timeout_s = 1.0
        worker.close()
        assert not worker._thread.is_alive()
    finally:
        policy.release.set()
        worker._shutdown_timeout_s = 1.0
        worker.close()


def test_repeated_close_joins_even_after_owner_publishes_closed(monkeypatch):
    worker = async_inference.AsyncInferenceWorker(_Factory([]), shutdown_timeout_s=0.5)
    try:
        worker.close()
        join_calls = []
        real_join = worker._thread.join

        def recording_join(timeout):
            join_calls.append(timeout)
            return real_join(timeout)

        monkeypatch.setattr(worker._thread, "join", recording_join)
        worker.close()

        assert join_calls == [0.5]
        assert not worker._thread.is_alive()
    finally:
        worker.close()
