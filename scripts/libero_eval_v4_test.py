# ruff: noqa: SLF001 -- focused integration tests exercise private controller seams.

import dataclasses
import threading
from types import SimpleNamespace

import numpy as np
from openpi_client import async_inference
from openpi_client import inference
from openpi_client import libero_control_v4 as control
from openpi_client import libero_eval_v4 as evaluation
from openpi_client import libero_video_timing_v4 as timing
from openpi_client import msgpack_numpy
import pytest

from examples.libero import main_v4


NS_PER_MS = 1_000_000


class ManualClock:
    def __init__(self, now_ns=0):
        self.now_ns = now_ns
        self.waits = []
        self._lock = threading.Lock()

    def monotonic_ns(self):
        with self._lock:
            return self.now_ns

    def wait_until_ns(self, deadline_ns):
        with self._lock:
            self.waits.append(deadline_ns)
            self.now_ns = max(self.now_ns, deadline_ns)

    def advance_to(self, target_ns):
        with self._lock:
            self.now_ns = max(self.now_ns, target_ns)


def _metadata_payload(*, revision="same"):
    return msgpack_numpy.packb(
        {
            "server_revision": revision,
            inference.INFERENCE_CAPABILITIES_KEY: {
                "schema_version": 1,
                "action_representation": "native",
                "model_action_horizon": 16,
                "model_action_dim": 32,
                "supported_protocols": [
                    "baseline_h16_n5_v1",
                    "baseline_rtc_h16_v1",
                ],
            },
        }
    )


def _rtc_response(offset=0.0):
    actions = np.arange(16 * 7, dtype=np.float32).reshape(16, 7) + offset
    model_actions = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) + offset
    return {
        "actions": actions,
        "rtc": {"schema_version": 1, "model_actions": model_actions},
    }


@dataclasses.dataclass
class _ScriptedCall:
    latency_ns: int
    result: object = None
    error: BaseException = None
    metadata_payload: bytes = dataclasses.field(default_factory=_metadata_payload)


class FakeWorker:
    def __init__(self, clock, calls, *, connect_payload=None, reset_advance_ns=0):
        self.clock = clock
        self.calls = list(calls)
        self.connect_payload = connect_payload or _metadata_payload()
        self.reset_advance_ns = reset_advance_ns
        self.requests = []
        self.jobs = []
        self._pending = None
        self.generation = 0
        self.reset_calls = 0
        self.ready_calls = []
        self.connect_calls = 0
        self.close_calls = 0

    def connect(self, timeout=None):
        del timeout
        self.connect_calls += 1
        return async_inference.ConnectionSnapshot(
            connection_id=self.connect_calls,
            metadata_payload=self.connect_payload,
        )

    def submit(self, request):
        if self._pending is not None:
            raise async_inference.BusyError("one request already pending")
        call = self.calls.pop(0)
        job = SimpleNamespace(
            request_id=len(self.jobs),
            generation=self.generation,
            submitted_monotonic_ns=self.clock.monotonic_ns(),
        )
        self.requests.append(request)
        self.jobs.append(job)
        self._pending = (job, call, job.submitted_monotonic_ns + call.latency_ns)
        return job

    def poll(self, job):
        if self._pending is None or self._pending[0] is not job:
            return None
        if self.clock.monotonic_ns() < self._pending[2]:
            return None
        return self._finish()

    def wait(self, job, timeout=None):
        del timeout
        if self._pending is None or self._pending[0] is not job:
            raise ValueError("unknown job")
        self.clock.advance_to(self._pending[2])
        return self._finish()

    def _finish(self):
        job, call, completed_ns = self._pending
        self._pending = None
        return async_inference.InferenceOutcome(
            job=job,
            result=call.result,
            error=call.error,
            completed_monotonic_ns=completed_ns,
            connection=async_inference.ConnectionSnapshot(
                connection_id=len(self.jobs),
                metadata_payload=call.metadata_payload,
            ),
        )

    def reset_generation(self):
        self.reset_calls += 1
        self.generation += 1
        self._pending = None
        if self.reset_calls > 1:
            self.clock.advance_to(self.clock.monotonic_ns() + self.reset_advance_ns)
        return self.generation

    def wait_until_ready(self, generation, timeout=None):
        del timeout
        self.ready_calls.append(generation)

    def close(self):
        self.close_calls += 1


class FakeEnvironment:
    def __init__(self, *, done_after_real_steps, dummy_steps=0, step_advance_ns=0):
        self.done_after_real_steps = done_after_real_steps
        self.dummy_steps = dummy_steps
        self.step_advance_ns = step_advance_ns
        self.clock = None
        self.actions = []
        self.reset_calls = 0

    def reset_to(self, initial_state):
        self.reset_calls += 1
        self.initial_state = initial_state
        return {"index": 0}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        if self.clock is not None:
            self.clock.advance_to(self.clock.monotonic_ns() + self.step_advance_ns)
        real_steps = max(0, len(self.actions) - self.dummy_steps)
        return {"index": len(self.actions)}, 0.0, real_steps >= self.done_after_real_steps, {}

    def invalidate(self):
        pass


def _identity():
    return evaluation.EpisodeIdentity(
        suite="libero_spatial",
        task_id=0,
        task_name="pick up the block",
        init_state_index=0,
        init_state_fingerprint="a" * 64,
    )


def _args(**overrides):
    values = {
        "num_steps_wait": 0,
        "resize_size": 224,
        "eval_seed": 42,
        "inference_timeout_s": 10.0,
        "connection_timeout_s": 10.0,
        "execution_mode": "baseline_sync_n5",
    }
    values.update(overrides)
    return dataclasses.replace(main_v4.ArgsV4(), **values)


def _prepare(obs, task_description, resize_size):
    del task_description, resize_size
    index = obs["index"]
    return {"observation/index": index}, np.full((2, 2, 3), index, dtype=np.uint8)


def _run(clock, worker, environment, *, args=None, scheduler=None, max_steps=20):
    args = args or _args()
    environment.clock = clock
    mode = control.EXECUTION_MODES[args.execution_mode]
    fingerprint = control.validate_server_metadata(
        mode,
        msgpack_numpy.unpackb(worker.connect_payload),
    )
    return main_v4._run_attempt_v4(
        environment=environment,
        worker=worker,
        scheduler=scheduler or control.make_scheduler_v4(mode, None),
        initial_state=np.array([1.0], dtype=np.float32),
        identity=_identity(),
        task_description="pick up the block",
        args=args,
        max_steps=max_steps,
        expected_server_metadata_fingerprint=fingerprint,
        clock=clock,
        prepare_observation=_prepare,
    )


def test_dummy_phase_is_paced_and_excluded_from_episode_timeline():
    clock = ManualClock()
    worker = FakeWorker(clock, [_ScriptedCall(125 * NS_PER_MS, _rtc_response())])
    environment = FakeEnvironment(done_after_real_steps=1, dummy_steps=10)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(num_steps_wait=10),
    )

    assert clock.waits[:9] == [index * 50 * NS_PER_MS for index in range(1, 10)]
    assert len(environment.actions) == 11
    assert all(np.array_equal(action, main_v4.LIBERO_DUMMY_ACTION) for action in environment.actions[:10])
    assert result.steps == 1
    assert result.episode_duration_ns == 125 * NS_PER_MS
    assert result.inference_requests[0].submitted_offset_ns == 0
    assert result.inference_latencies[0].completed_offset_ns == 125 * NS_PER_MS
    assert result.plan_activations[0].activated_offset_ns == 125 * NS_PER_MS
    assert result.control_stalls[0].duration_ns == 125 * NS_PER_MS


@pytest.mark.parametrize(
    ("second_latency_ms", "expected_stalls"),
    (
        (30, [(0, 0), (8, 0)]),
        (80, [(0, 0), (8, 30)]),
    ),
)
def test_native_replan_uses_idle_time_and_only_records_deadline_suffix(
    second_latency_ms, expected_stalls
):
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _rtc_response()),
            _ScriptedCall(second_latency_ms * NS_PER_MS, _rtc_response(1000.0)),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=9)

    result = _run(clock, worker, environment)

    measured = [(stall.control_step, stall.duration_ns // NS_PER_MS) for stall in result.control_stalls]
    measured_with_absent_suffix = measured if len(measured) == 2 else measured + [(8, 0)]
    assert measured_with_absent_suffix == expected_stalls
    assert result.inference_requests[1].submitted_offset_ns == 350 * NS_PER_MS
    assert result.inference_latencies[1].completed_offset_ns == (350 + second_latency_ms) * NS_PER_MS
    if second_latency_ms == 30:
        assert result.episode_duration_ns == 400 * NS_PER_MS
    else:
        assert result.episode_duration_ns == 430 * NS_PER_MS


def test_every_baseline_sync_request_has_schema_one_rtc_envelope_and_fresh_seed():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, _rtc_response()), _ScriptedCall(0, _rtc_response(100.0))],
    )
    environment = FakeEnvironment(done_after_real_steps=9)

    result = _run(clock, worker, environment)

    assert len(worker.requests) == 2
    assert [request[inference.RTC_REQUEST_KEY] for request in worker.requests] == [
        {"schema_version": 1},
        {"schema_version": 1},
    ]
    assert [request[inference.INFERENCE_SEED_KEY] for request in worker.requests] == [
        evaluation.stable_replan_seed(42, _identity(), 0),
        evaluation.stable_replan_seed(42, _identity(), 1),
    ]
    assert all("previous_model_actions" not in request[inference.RTC_REQUEST_KEY] for request in worker.requests)
    assert np.array_equal(environment.actions[0], _rtc_response()["actions"][0])
    assert np.array_equal(environment.actions[7], _rtc_response()["actions"][7])
    assert np.array_equal(environment.actions[8], _rtc_response(100.0)["actions"][0])
    assert result.success


def test_malformed_initial_result_is_a_complete_audited_policy_failure():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(17 * NS_PER_MS, {"actions": np.zeros((16, 7), dtype=np.float32)})],
    )
    environment = FakeEnvironment(done_after_real_steps=1)

    result = _run(clock, worker, environment)

    assert not result.success
    assert result.failure_kind == "policy"
    assert result.steps == 0
    assert result.inference_requests[0].disposition == "failed"
    assert result.inference_latencies[0].outcome == "policy_failure"
    assert result.control_stalls == (
        timing.ControlStallV4(
            request_id=0,
            control_step=0,
            started_offset_ns=0,
            duration_ns=17 * NS_PER_MS,
            reason="synchronous_inference",
        ),
    )
    assert len(result.stall_source_frames) == 1
    assert not environment.actions


class _BackgroundScheduler(control.ModeSchedulerV4):
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0
        self.action_index = 0
        self._pending_intent = None

    def maybe_request(self, now_ns, *, at_due, request_in_flight):
        del now_ns, at_due
        if request_in_flight or self._pending_intent is not None:
            return None
        if self.phase == 0:
            self._pending_intent = control.RequestIntentV4(
                dispatch="blocking_initial",
                trigger="initial_plan",
                scheduler_context={},
                request_overlay={inference.RTC_REQUEST_KEY: {"schema_version": 1}},
            )
        elif self.phase == 1 and self.action_index == 1:
            self._pending_intent = control.RequestIntentV4(
                dispatch="background",
                trigger="rtc_launch",
                scheduler_context={"s": 8, "d": 0},
                request_overlay={
                    inference.RTC_REQUEST_KEY: {
                        "schema_version": 1,
                        "previous_model_actions": np.zeros((16, 32), dtype=np.float32),
                        "s": 8,
                        "d": 0,
                    }
                },
            )
        return self._pending_intent

    def install_response(self, intent, response, *, now_ns, control_step):
        del response, now_ns, control_step
        assert intent is self._pending_intent
        self._pending_intent = None
        if self.phase == 0:
            self.phase = 1
            return control.ActivationDecisionV4("initial", {"action_cursor": 0})
        self.phase = 2
        return control.ActivationDecisionV4("immediate_swap", {"action_cursor": 0})

    def take_action(self, now_ns):
        del now_ns
        self.action_index += 1
        return control.ActionDecisionV4(
            action=np.full(7, self.action_index, dtype=np.float32),
            underflow=False,
        )


def test_current_background_policy_error_stops_before_old_plan_action():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, {"ok": True}),
            _ScriptedCall(10 * NS_PER_MS, error=ValueError("bad guided response")),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=20)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="baseline_rtc"),
        scheduler=_BackgroundScheduler(),
    )

    assert not result.success
    assert result.steps == 1
    assert len(environment.actions) == 1
    assert result.inference_requests[-1].disposition == "failed"
    assert result.inference_latencies[-1].outcome == "policy_failure"
    assert not result.action_underflows


class _UnderflowScheduler(_BackgroundScheduler):
    def take_action(self, now_ns):
        if self.phase == 1 and self.action_index == 1:
            return control.ActionDecisionV4(action=None, underflow=True)
        return super().take_action(now_ns)


def test_async_underflow_waits_once_and_reanchors_next_deadline():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, {"ok": True}),
            _ScriptedCall(85 * NS_PER_MS, {"ok": True}),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=2)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="baseline_rtc"),
        scheduler=_UnderflowScheduler(),
    )

    assert result.action_underflows == (
        timing.ActionUnderflowV4(
            request_id=1,
            control_step=1,
            started_offset_ns=50 * NS_PER_MS,
            duration_ns=35 * NS_PER_MS,
        ),
    )
    assert result.control_stalls[-1].reason == "async_action_underflow"
    assert result.control_stalls[-1].duration_ns == 35 * NS_PER_MS
    assert result.episode_duration_ns == 85 * NS_PER_MS


def test_done_abandons_background_request_without_latency_or_activation():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, {"ok": True}),
            _ScriptedCall(500 * NS_PER_MS, {"ok": True}),
        ],
        reset_advance_ns=700 * NS_PER_MS,
    )
    environment = FakeEnvironment(done_after_real_steps=2)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="baseline_rtc"),
        scheduler=_BackgroundScheduler(),
    )

    assert result.success
    assert result.inference_requests[-1].disposition == "abandoned"
    assert len(result.inference_latencies) == 1
    assert len(result.plan_activations) == 1
    assert result.episode_duration_ns == 50 * NS_PER_MS
    assert worker.reset_calls == 2
    assert worker.ready_calls[-1] == worker.generation


def test_metadata_change_on_result_is_run_fatal_before_another_action():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, _rtc_response(), metadata_payload=_metadata_payload(revision="changed"))],
    )
    environment = FakeEnvironment(done_after_real_steps=1)

    with pytest.raises(control.CalibrationIdentityError, match="fingerprint changed"):
        _run(clock, worker, environment)

    assert not environment.actions


def test_v4_video_frames_hold_control_twice_and_insert_reason_driven_stalls():
    stalls = (
        timing.ControlStallV4(0, 0, 0, 25 * NS_PER_MS, "synchronous_inference"),
        timing.ControlStallV4(1, 1, 50 * NS_PER_MS, 50 * NS_PER_MS, "async_action_underflow"),
    )
    rendered = []

    def renderer(frame, lines):
        rendered.append(lines)
        return "{}:{}".format(frame, lines[0])

    frames = main_v4._build_video_frames_v4(
        ("frame-0", "frame-1"),
        stalls,
        stall_source_frames=((0, "wait-0"), (1, "wait-1")),
        include_stalls=True,
        overlay_renderer=renderer,
    )

    assert rendered == [
        ("Synchronous inference", "Control stalled: 0.03 s"),
        ("Waiting for policy actions", "Control stalled: 0.05 s"),
    ]
    assert frames == (
        "wait-0:Synchronous inference",
        "frame-0",
        "frame-0",
        "wait-1:Waiting for policy actions",
        "wait-1:Waiting for policy actions",
        "frame-1",
        "frame-1",
    )


def test_worker_shutdown_failure_preserves_primary_exception():
    primary = KeyboardInterrupt("stop")

    class Worker:
        def reset_generation(self):
            return 3

        def wait_until_ready(self, generation, timeout=None):
            del generation, timeout

        def close(self):
            raise TimeoutError("join timeout")

    with pytest.raises(main_v4.RunCleanupError) as caught:
        main_v4._close_worker_v4(Worker(), primary_error=primary, timeout_s=1.0)

    assert caught.value.primary_error is primary
    assert isinstance(caught.value.cleanup_error, TimeoutError)
