import gc
import threading
import time
import weakref

import numpy as np
import pytest

from openpi_client import async_inference
from openpi_client import msgpack_numpy


class _GatePolicy:
    def __init__(self, *, result=None, error=None, return_after_cancel=False, ignore_cancel=False):
        self.result = {"actions": [1, 2, 3]} if result is None else result
        self.error = error
        self.return_after_cancel = return_after_cancel
        self.ignore_cancel = ignore_cancel
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_observed = threading.Event()
        self.closed = threading.Event()
        self.payloads = []
        self.infer_thread_ids = []
        self.close_thread_ids = []
        self.active_calls = 0
        self.max_active_calls = 0

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
        self.closed.set()


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
    def __init__(self, policy):
        self.policy = policy
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_observed = threading.Event()
        self.created_policies = []

    def __call__(self, cancel_event):
        self.started.set()
        while not self.release.wait(0.005):
            if cancel_event.is_set():
                self.cancel_observed.set()
                raise RuntimeError("fake connect retry cancelled")
        self.created_policies.append(self.policy)
        return self.policy


def _submit_when_ready(worker, observation):
    deadline = time.monotonic() + 1.0
    while True:
        try:
            return worker.submit(observation)
        except async_inference.BusyError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.005)


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
        assert not outcome.stale
        assert not outcome.cancelled
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
    old_policy = _GatePolicy()
    new_policy = _GatePolicy(result={"generation": 1})
    new_policy.release.set()
    factory = _Factory([old_policy, new_policy])
    worker = async_inference.AsyncInferenceWorker(factory)
    try:
        old_job = worker.submit({"generation": 0})
        assert old_policy.started.wait(1.0)

        assert worker.reset_generation() == 1
        stale = worker.wait(old_job, timeout=1.0)
        assert stale.stale and stale.cancelled
        with pytest.raises(async_inference.BusyError):
            worker.submit({"too": "early"})

        assert old_policy.cancel_observed.wait(1.0)
        assert old_policy.closed.wait(1.0)
        new_job = _submit_when_ready(worker, {"generation": 1})
        fresh = worker.wait(new_job, timeout=1.0)
        assert new_job.generation == 1
        assert fresh.result == {"generation": 1}
        assert factory.cancel_events[0] is not factory.cancel_events[1]
        assert len(factory.cancel_events) == 2
    finally:
        old_policy.release.set()
        new_policy.release.set()
        worker.close()


def test_late_result_after_active_reset_cannot_replace_stale_outcome():
    old_policy = _GatePolicy(return_after_cancel=True)
    worker = async_inference.AsyncInferenceWorker(_Factory([old_policy]))
    try:
        job = worker.submit({"request": "old"})
        assert old_policy.started.wait(1.0)
        worker.reset_generation()
        first_outcome = worker.wait(job, timeout=1.0)
        assert old_policy.closed.wait(1.0)

        assert worker.poll(job) is first_outcome
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
    worker.submit({"request": "connect"})
    assert factory.started.wait(1.0)

    worker.close()
    worker.close()

    assert factory.cancel_observed.is_set()
    assert not worker._thread.daemon
    assert not worker._thread.is_alive()
    assert factory.created_policies == []


def test_close_interrupts_active_receive_and_owner_closes_socket():
    policy = _GatePolicy()
    factory = _Factory([policy])
    calling_thread_id = threading.get_ident()
    worker = async_inference.AsyncInferenceWorker(factory, shutdown_timeout_s=0.5)
    worker.submit({"request": "active"})
    assert policy.started.wait(1.0)

    worker.close()

    assert policy.cancel_observed.is_set()
    assert policy.closed.is_set()
    assert not worker._thread.is_alive()
    assert factory.thread_ids == policy.infer_thread_ids == policy.close_thread_ids
    assert factory.thread_ids[0] != calling_thread_id


def test_close_raises_timeout_if_non_daemon_owner_does_not_stop_in_bound():
    policy = _GatePolicy(ignore_cancel=True)
    worker = async_inference.AsyncInferenceWorker(_Factory([policy]), shutdown_timeout_s=0.01)
    worker.submit({"request": "stuck"})
    assert policy.started.wait(1.0)

    with pytest.raises(TimeoutError, match="worker thread"):
        worker.close()

    policy.release.set()
    worker.close()
    assert not worker._thread.is_alive()
