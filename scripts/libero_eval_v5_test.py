# ruff: noqa: SLF001 -- focused integration tests exercise private controller seams.

import dataclasses
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from openpi_client import async_inference
from openpi_client import inference
from openpi_client import latency_sampling
from openpi_client import libero_control_v5 as control
from openpi_client import libero_eval_v5 as evaluation
from openpi_client import libero_video_timing_v5 as timing
from openpi_client import msgpack_numpy
import pytest

from examples.libero import main_v5

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


def _metadata_payload(*, revision="same", policy_variant="baseline"):
    if policy_variant == "baseline":
        representation = "native"
        protocols = [
            "baseline_h16_n5_v1",
            "baseline_sync_n5_h16_full_v2",
            "baseline_async_h16_v1",
            "baseline_async_h16_blocking_recovery_v2",
            "baseline_rtc_h16_v1",
        ]
    else:
        representation = "bsp"
        protocols = [
            "bsp_spline_sync_speedup2_phase0_v2",
            "bsp_spline_async_phase_skip_speedup2_v2",
            "bsp_spline_async_phase_skip_speedup1_v1",
            "bsp_spline_async_phase_skip_speedup4_delta_accum_v2",
            "bsp_spline_async_phase_skip_speedup8_delta_accum_v2",
        ]
    return msgpack_numpy.packb(
        {
            "server_revision": revision,
            inference.INFERENCE_CAPABILITIES_KEY: {
                "schema_version": 1,
                "action_representation": representation,
                "model_action_horizon": 16,
                "model_action_dim": 32,
                "supported_protocols": protocols,
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


def _bsp_response(offset=0.0, *, duration_ticks=None, speedup=2):
    parameters = np.zeros((16, 8), dtype=np.float32)
    parameters[:, :7] = offset
    if duration_ticks is None:
        parameters[:, 7] = np.arange(16, dtype=np.float32)
    else:
        parameters[:, 7] = np.concatenate(
            (
                np.zeros(4, dtype=np.float32),
                np.linspace(0, duration_ticks, 10, dtype=np.float32)[1:9],
                np.full(4, duration_ticks, dtype=np.float32),
            )
        )
    return {
        "actions": np.full((8, 7), offset, dtype=np.float32),
        "bsp": {
            "schema_version": 1,
            "parameters": parameters,
            "origin_hz": 10,
            "degree": 3,
            "speedup": speedup,
            "alignment": "disabled_delta_eff",
        },
    }


def _calibration(mode_name, *, latency_ns=0):
    mode = control.EXECUTION_MODES[mode_name]
    canonical_request = {"observation/index": 0}
    identity = control.CalibrationObservationIdentityV1(
        suite="libero_spatial",
        task_id=0,
        init_state_index=0,
        init_state_fingerprint="a" * 64,
        request_fingerprint=control.canonical_fingerprint(canonical_request),
    )
    sampled_ns = 0 if control.is_native_latency_mode_v5(mode.name) else 300 * NS_PER_MS
    effective_ns = max(latency_ns, sampled_ns)
    synthetic_ns = effective_ns - latency_ns
    return control.LatencyCalibrationV2.create(
        execution_mode=mode.name,
        checkpoint_identity_fingerprint="b" * 64,
        server_metadata_fingerprint="c" * 64,
        canonical_observation_identity=identity,
        seed_namespace="openpi-libero-calibration-v2/{}/{}".format(mode.name, "b" * 64),
        bootstrap_request_fingerprint=("d" * 64 if mode.calibration_kind == "rtc" else None),
        warmup_request_fingerprints=["e" * 64] * 5,
        measurement_request_fingerprints=["f" * 64] * 20,
        warmup_raw_inference_latency_ns=[latency_ns] * 5,
        warmup_sampled_target_latency_ns=[sampled_ns] * 5,
        warmup_requested_synthetic_delay_ns=[synthetic_ns] * 5,
        warmup_observed_synthetic_delay_ns=[synthetic_ns] * 5,
        warmup_observed_effective_latency_ns=[effective_ns] * 5,
        warmup_latency_overshoot_ns=[0] * 5,
        measurement_raw_inference_latency_ns=[latency_ns] * 20,
        measurement_sampled_target_latency_ns=[sampled_ns] * 20,
        measurement_requested_synthetic_delay_ns=[synthetic_ns] * 20,
        measurement_observed_synthetic_delay_ns=[synthetic_ns] * 20,
        measurement_observed_effective_latency_ns=[effective_ns] * 20,
        measurement_latency_overshoot_ns=[0] * 20,
    )


@dataclasses.dataclass
class _ScriptedCall:
    latency_ns: int
    result: object = None
    error: BaseException = None
    metadata_payload: bytes = dataclasses.field(default_factory=_metadata_payload)


class FakeWorker:
    def __init__(
        self,
        clock,
        calls,
        *,
        connect_payload=None,
        reset_advance_ns=0,
        formal_sampling=False,
    ):
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
        self.formal_sampling = formal_sampling

    def connect(self, timeout=None):
        del timeout
        self.connect_calls += 1
        return async_inference.ConnectionSnapshot(
            connection_id=self.connect_calls,
            metadata_payload=self.connect_payload,
        )

    def submit(self, request, *, latency_sample_key=None):
        if self._pending is not None:
            raise async_inference.BusyError("one request already pending")
        call = self.calls.pop(0)
        sampled_target = (
            latency_sampling.NormalLatencySamplerV1().sample_target_ns(latency_sample_key)
            if self.formal_sampling
            else 0
        )
        effective_latency = max(call.latency_ns, sampled_target)
        job = SimpleNamespace(
            request_id=len(self.jobs),
            generation=self.generation,
            submitted_monotonic_ns=self.clock.monotonic_ns(),
            latency_sample_key=latency_sample_key,
            sampled_target_latency_ns=sampled_target,
        )
        self.requests.append(request)
        self.jobs.append(job)
        self._pending = (
            job,
            call,
            job.submitted_monotonic_ns + effective_latency,
        )
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
            sampled_target_latency_ns=job.sampled_target_latency_ns,
            raw_inference_latency_ns=call.latency_ns,
            requested_synthetic_delay_ns=(completed_ns - job.submitted_monotonic_ns - call.latency_ns),
            observed_synthetic_delay_ns=(completed_ns - job.submitted_monotonic_ns - call.latency_ns),
            observed_effective_latency_ns=completed_ns - job.submitted_monotonic_ns,
            latency_overshoot_ns=0,
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


class _StaleReplanSubmissionOverheadWorker(FakeWorker):
    """Make the stale-response-to-replan handoff consume measurable wall time."""

    def submit(self, request, *, latency_sample_key=None):
        if len(self.jobs) == 2:
            self.clock.advance_to(self.clock.monotonic_ns() + NS_PER_MS)
        return super().submit(request, latency_sample_key=latency_sample_key)


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
        "execution_mode": "baseline_async",
    }
    values.update(overrides)
    return dataclasses.replace(main_v5.ArgsV5(), **values)


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
    return main_v5._run_attempt_v5(
        environment=environment,
        worker=worker,
        scheduler=scheduler or control.make_scheduler_v5(mode, _calibration(mode.name) if mode.asynchronous else None),
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
    assert all(np.array_equal(action, main_v5.LIBERO_DUMMY_ACTION) for action in environment.actions[:10])
    assert result.steps == 1
    assert result.episode_duration_ns == 125 * NS_PER_MS
    assert result.inference_requests[0].submitted_offset_ns == 0
    assert result.inference_latencies[0].completed_offset_ns == 125 * NS_PER_MS
    assert result.plan_activations[0].activated_offset_ns == 125 * NS_PER_MS
    assert result.control_stalls[0].duration_ns == 125 * NS_PER_MS


def test_video_wait_display_switch_cannot_change_rollout_actions_or_result():
    outcomes = []
    action_traces = []
    for show_waits in (False, True):
        clock = ManualClock()
        worker = FakeWorker(clock, [_ScriptedCall(80 * NS_PER_MS, _rtc_response())])
        environment = FakeEnvironment(done_after_real_steps=3)
        outcomes.append(
            _run(
                clock,
                worker,
                environment,
                args=_args(video_show_inference_waits=show_waits),
            )
        )
        action_traces.append(tuple(action.copy() for action in environment.actions))

    assert len(action_traces[0]) == len(action_traces[1])
    assert all(
        np.array_equal(before, after)
        for before, after in zip(  # noqa: B905 -- LIBERO client runs on Python 3.8.
            action_traces[0], action_traces[1]
        )
    )
    assert outcomes[0].success == outcomes[1].success
    assert outcomes[0].failure_kind == outcomes[1].failure_kind
    assert outcomes[0].steps == outcomes[1].steps
    assert outcomes[0].episode_duration_ns == outcomes[1].episode_duration_ns


@pytest.mark.parametrize(
    "second_latency_ms",
    [pytest.param(30), pytest.param(80)],
)
def test_baseline_async_hides_completed_background_latency_and_records_a_seam(
    second_latency_ms,
):
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _rtc_response()),
            _ScriptedCall(second_latency_ms * NS_PER_MS, _rtc_response(1000.0)),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=12)

    result = _run(clock, worker, environment)

    assert not result.action_underflows
    assert all(stall.request_id == 0 for stall in result.control_stalls)
    assert result.inference_requests[1].submitted_offset_ns == 350 * NS_PER_MS
    assert result.inference_latencies[1].completed_offset_ns == (350 + second_latency_ms) * NS_PER_MS
    assert result.action_seams[0].request_id == 1
    assert result.action_seams[0].control_step in (8, 9)


def test_baseline_async_capacity_miss_blocks_for_fresh_actions_instead_of_failing():
    """The old INFEASIBLE branch fails this real runner sequence at control step 33."""
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _rtc_response()),
            _ScriptedCall(250 * NS_PER_MS, _rtc_response(1_000.0)),
            _ScriptedCall(250 * NS_PER_MS, _rtc_response(2_000.0)),
            _ScriptedCall(500 * NS_PER_MS, _rtc_response(3_000.0)),
            _ScriptedCall(50 * NS_PER_MS, _rtc_response(4_000.0)),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=34)
    mode = control.EXECUTION_MODES["baseline_async_recovery"]

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode=mode.name),
        scheduler=control.make_scheduler_v5(mode, _calibration(mode.name)),
        max_steps=40,
    )

    assert result.success, result.error
    assert [request.trigger for request in result.inference_requests] == [
        "initial_plan",
        "baseline_async_launch",
        "baseline_async_launch",
        "baseline_async_launch",
        "baseline_async_capacity_replan",
    ]
    assert result.inference_requests[-1].observation_control_step == 33
    assert result.plan_activations[-1].activation == "blocking_replace"
    assert result.plan_activations[-1].activation_context == {"action_cursor": 0}
    assert len(result.action_underflows) == 1
    assert result.action_underflows[0].request_id == 3
    recovery_stall = result.control_stalls[-1]
    assert recovery_stall.request_id == 4
    assert recovery_stall.control_step == 33
    assert recovery_stall.reason == "synchronous_inference"
    assert np.array_equal(environment.actions[33], _rtc_response(4_000.0)["actions"][0])


def test_every_baseline_sync_request_has_schema_one_rtc_envelope_and_fresh_seed():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(300 * NS_PER_MS, _rtc_response()),
            _ScriptedCall(300 * NS_PER_MS, _rtc_response(100.0)),
        ],
        formal_sampling=True,
    )
    environment = FakeEnvironment(done_after_real_steps=17)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="baseline_sync"),
    )

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
    assert np.array_equal(environment.actions[15], _rtc_response()["actions"][15])
    assert np.array_equal(environment.actions[16], _rtc_response(100.0)["actions"][0])
    assert [stall.request_id for stall in result.control_stalls] == [0, 1]
    assert all(stall.reason == "synchronous_inference" for stall in result.control_stalls)
    assert result.control_stalls[0].duration_ns == result.inference_latencies[0].duration_ns
    assert result.control_stalls[1].duration_ns == result.inference_latencies[1].duration_ns - 50 * NS_PER_MS
    assert result.success, result.error


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
        timing.ControlStallV5(
            request_id=0,
            control_step=0,
            started_offset_ns=0,
            duration_ns=17 * NS_PER_MS,
            reason="synchronous_inference",
        ),
    )
    assert len(result.stall_source_frames) == 1
    assert not environment.actions


class _BackgroundScheduler(control.ModeSchedulerV5):
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0
        self.action_index = 0
        self._pending_intent = None

    def maybe_request(self, now_ns, *, at_due, request_in_flight, control_step=0):
        del now_ns, at_due, control_step
        if request_in_flight or self._pending_intent is not None:
            return None
        if self.phase == 0:
            self._pending_intent = control.RequestIntentV5(
                dispatch="blocking_initial",
                trigger="initial_plan",
                scheduler_context={},
                request_overlay={inference.RTC_REQUEST_KEY: {"schema_version": 1}},
            )
        elif self.phase == 1 and self.action_index == 1:
            self._pending_intent = control.RequestIntentV5(
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
            return control.ActivationDecisionV5("initial", {"action_cursor": 0})
        self.phase = 2
        return control.ActivationDecisionV5("immediate_swap", {"action_cursor": 0})

    def take_action(self, now_ns, *, control_step=0):
        del now_ns, control_step
        self.action_index += 1
        return control.ActionDecisionV5(
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


def test_completed_policy_error_is_polled_before_next_observation_preparation():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, {"ok": True}),
            _ScriptedCall(50 * NS_PER_MS, error=ValueError("current policy failure")),
        ],
    )
    environment = FakeEnvironment(done_after_real_steps=20, step_advance_ns=20 * NS_PER_MS)
    prepared_indices = []

    def prepare(obs, task_description, resize_size):
        del task_description, resize_size
        prepared_indices.append(obs["index"])
        if obs["index"] > 1:
            raise RuntimeError("preparation must not mask a completed policy failure")
        return {"observation/index": obs["index"]}, np.zeros((2, 2, 3), dtype=np.uint8)

    result = main_v5._run_attempt_v5(
        environment=environment,
        worker=worker,
        scheduler=_BackgroundScheduler(),
        initial_state=np.array([1.0], dtype=np.float32),
        identity=_identity(),
        task_description="pick up the block",
        args=_args(execution_mode="baseline_rtc"),
        max_steps=20,
        expected_server_metadata_fingerprint=control.validate_server_metadata(
            control.EXECUTION_MODES["baseline_rtc"],
            msgpack_numpy.unpackb(worker.connect_payload),
        ),
        clock=clock,
        prepare_observation=prepare,
    )

    assert result.failure_kind == "policy"
    assert "current policy failure" in result.error
    assert prepared_indices == [0, 1]


class _WaitFailureWorker(FakeWorker):
    def wait(self, job, timeout=None):
        del job, timeout
        raise ConnectionError("blocked request transport failed")


class _InvalidSubmitTimestampWorker(FakeWorker):
    def submit(self, request, *, latency_sample_key=None):
        job = super().submit(request, latency_sample_key=latency_sample_key)
        job.submitted_monotonic_ns = None
        return job


def test_post_submit_validation_failure_cancels_the_owned_job():
    clock = ManualClock()
    worker = _InvalidSubmitTimestampWorker(clock, [_ScriptedCall(0, _rtc_response())])

    result = _run(clock, worker, FakeEnvironment(done_after_real_steps=1))

    assert result.failure_kind == "policy"
    assert "submit timestamp" in result.error
    assert worker.reset_calls == 2
    assert worker.ready_calls == [1, 2]
    assert worker._pending is None


def test_blocking_wait_failure_keeps_pending_owned_until_reset_acknowledgement():
    clock = ManualClock()
    worker = _WaitFailureWorker(clock, [_ScriptedCall(100 * NS_PER_MS, _rtc_response())])
    environment = FakeEnvironment(done_after_real_steps=1)

    with pytest.raises(evaluation.InfrastructureFailure, match="transport failed"):
        _run(clock, worker, environment)

    assert worker.reset_calls == 2
    assert worker.ready_calls == [1, 2]


class _TakeActionFailureScheduler(_BackgroundScheduler):
    def take_action(self, now_ns, *, control_step=0):
        if self.phase == 1 and self.action_index == 1:
            raise ValueError("active background policy failure")
        return super().take_action(now_ns, control_step=control_step)


def test_policy_failure_return_gate_abandons_and_acknowledges_active_background_job():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, {"ok": True}),
            _ScriptedCall(500 * NS_PER_MS, {"ok": True}),
        ],
    )

    result = _run(
        clock,
        worker,
        FakeEnvironment(done_after_real_steps=20),
        args=_args(execution_mode="baseline_rtc"),
        scheduler=_TakeActionFailureScheduler(),
    )

    assert result.failure_kind == "policy"
    assert "active background policy failure" in result.error
    assert [event.disposition for event in result.inference_requests] == [
        "activated",
        "abandoned",
    ]
    assert worker.reset_calls == 2
    assert worker.ready_calls == [1, 2]
    assert worker._pending is None


def test_infrastructure_exhaustion_resets_and_acknowledges_every_attempt():
    clock = ManualClock()
    worker = _WaitFailureWorker(
        clock,
        [_ScriptedCall(100 * NS_PER_MS, _rtc_response()) for _ in range(3)],
    )
    environment = FakeEnvironment(done_after_real_steps=1)

    record = evaluation.run_episode_with_retries_v5(
        _identity(),
        lambda _attempt: _run(clock, worker, environment),
        eval_seed=42,
        execution_mode="baseline_async",
    )

    assert record.status == "infrastructure_incomplete"
    assert record.attempts == 3
    assert worker.reset_calls == 6
    assert worker.ready_calls == [1, 2, 3, 4, 5, 6]


def test_real_rtc_scheduler_bootstraps_then_installs_guided_result():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, _rtc_response()), _ScriptedCall(0, _rtc_response(1000.0))],
    )
    environment = FakeEnvironment(done_after_real_steps=9)
    calibration = _calibration("baseline_rtc")

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="baseline_rtc"),
        scheduler=control.make_scheduler_v5(control.EXECUTION_MODES["baseline_rtc"], calibration),
    )

    assert result.success
    assert [event.trigger for event in result.inference_requests] == ["initial_plan", "rtc_launch"]
    assert result.inference_requests[1].scheduler_context == {"s": 8, "d": 8}
    assert [event.activation for event in result.plan_activations] == ["initial", "immediate_swap"]
    assert np.array_equal(environment.actions[8], _rtc_response(1000.0)["actions"][0])


def test_real_bsp_async_arrival_skips_elapsed_prefix_then_immediately_prefetches_short_tail():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _bsp_response(), metadata_payload=payload),
            _ScriptedCall(500 * NS_PER_MS, _bsp_response(3.0), metadata_payload=payload),
            _ScriptedCall(500 * NS_PER_MS, _bsp_response(5.0), metadata_payload=payload),
        ],
        connect_payload=payload,
    )
    environment = FakeEnvironment(done_after_real_steps=14)
    calibration = _calibration("bsp_spline_async", latency_ns=50 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="bsp_spline_async"),
        scheduler=control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_async"], calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert result.inference_requests[1].trigger == "bsp_prefetch"
    assert result.inference_requests[1].scheduler_context == {
        "remaining_plan_ns": 400 * NS_PER_MS,
        "budget_ns": 400 * NS_PER_MS,
        "request_control_step": 4,
    }
    assert result.action_underflows == ()
    assert result.plan_activations[1].activation_context == {
        "request_control_step": 4,
        "activation_control_step": 13,
        "executed_prefix_steps": 9,
        "phase_offset_microindices": 9_000_000,
        "first_sample_microindices": 9_000_000,
        "remaining_curve_microindices": 3_000_000,
        "remaining_curve_ns": 150 * NS_PER_MS,
        "immediate_prefetch": 1,
    }
    assert result.inference_requests[2].trigger == "bsp_prefetch"
    assert result.inference_requests[2].disposition == "abandoned"
    assert np.allclose(environment.actions[-1], 3.0)


def test_real_bsp_async_speedup1_advances_half_index_and_prefetches_at_four_remaining_indices():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _bsp_response(duration_ticks=9, speedup=1), metadata_payload=payload),
            _ScriptedCall(
                200 * NS_PER_MS,
                _bsp_response(3.0, duration_ticks=9, speedup=1),
                metadata_payload=payload,
            ),
        ],
        connect_payload=payload,
    )
    environment = FakeEnvironment(done_after_real_steps=15)
    mode = control.EXECUTION_MODES["bsp_spline_async_speedup1"]
    calibration = _calibration(mode.name, latency_ns=50 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode=mode.name),
        scheduler=control.make_scheduler_v5(mode, calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert result.inference_requests[1].scheduler_context == {
        "remaining_plan_ns": 400 * NS_PER_MS,
        "budget_ns": 400 * NS_PER_MS,
        "request_control_step": 10,
    }
    assert worker.requests[0][inference.BSP_EXECUTION_KEY] == {"schema_version": 1, "speedup": 1}
    assert worker.requests[1][inference.BSP_EXECUTION_KEY] == {"schema_version": 1, "speedup": 1}
    assert result.plan_activations[1].activation_context == {
        "request_control_step": 10,
        "activation_control_step": 13,
        "executed_prefix_steps": 3,
        "phase_offset_microindices": 1_500_000,
        "first_sample_microindices": 1_500_000,
        "remaining_curve_microindices": 7_500_000,
        "remaining_curve_ns": 750 * NS_PER_MS,
        "immediate_prefetch": 0,
    }


def test_native_bsp_async_records_the_online_budget_that_triggered_each_activation():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(250 * NS_PER_MS, _bsp_response(duration_ticks=9), metadata_payload=payload),
            _ScriptedCall(250 * NS_PER_MS, _bsp_response(3.0, duration_ticks=9), metadata_payload=payload),
        ],
        connect_payload=payload,
        formal_sampling=False,
    )
    environment = FakeEnvironment(done_after_real_steps=11)
    mode = control.EXECUTION_MODES["bsp_spline_async_native"]
    calibration = _calibration(mode.name, latency_ns=80 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode=mode.name),
        scheduler=control.make_scheduler_v5(mode, calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert result.inference_requests[1].scheduler_context["budget_ns"] == 100 * NS_PER_MS
    assert result.plan_activations[1].activation_context["prefetch_budget_ns"] == 250 * NS_PER_MS
    assert all(request.sampled_target_latency_ns == 0 for request in result.inference_requests)
    assert all(latency.observed_synthetic_delay_ns == 0 for latency in result.inference_latencies)


def test_real_bsp_async_discards_stale_short_curve_and_blocks_on_latest_observation():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = _StaleReplanSubmissionOverheadWorker(
        clock,
        [
            _ScriptedCall(0, _bsp_response(), metadata_payload=payload),
            _ScriptedCall(
                550 * NS_PER_MS,
                _bsp_response(3.0, duration_ticks=4),
                metadata_payload=payload,
            ),
            _ScriptedCall(50 * NS_PER_MS, _bsp_response(5.0), metadata_payload=payload),
        ],
        connect_payload=payload,
    )
    environment = FakeEnvironment(done_after_real_steps=14)
    calibration = _calibration("bsp_spline_async", latency_ns=50 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="bsp_spline_async"),
        scheduler=control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_async"], calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert [request.trigger for request in result.inference_requests] == [
        "initial_plan",
        "bsp_prefetch",
        "bsp_stale_replan",
    ]
    assert [request.disposition for request in result.inference_requests] == [
        "activated",
        "discarded_stale_phase",
        "activated",
    ]
    stale_context = result.inference_requests[2].scheduler_context
    assert stale_context["discarded_request_control_step"] == 4
    assert stale_context["discarded_activation_control_step"] == 13
    assert stale_context["phase_offset_microindices"] == 9_000_000
    assert stale_context["curve_t_max_microindices"] == 4_000_000
    assert [activation.activation for activation in result.plan_activations] == [
        "initial",
        "blocking_replace",
    ]
    assert result.plan_activations[1].control_step == 13
    assert result.plan_activations[1].activation_context["executed_prefix_steps"] == 0
    assert len(result.action_underflows) == 1
    assert result.action_underflows[0].request_id == 1
    assert result.action_underflows[0].control_step == 13
    assert result.action_underflows[0].duration_ns == 101 * NS_PER_MS
    assert result.control_stalls[-1].request_id == 1
    assert result.control_stalls[-1].duration_ns == 101 * NS_PER_MS
    assert all(stall.request_id != 2 for stall in result.control_stalls)
    assert np.allclose(environment.actions[-1], 5.0)


def test_real_bsp_async_stale_response_with_old_tail_records_only_blocking_replan_stall():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _bsp_response(), metadata_payload=payload),
            _ScriptedCall(
                400 * NS_PER_MS,
                _bsp_response(3.0, duration_ticks=4),
                metadata_payload=payload,
            ),
            _ScriptedCall(50 * NS_PER_MS, _bsp_response(5.0), metadata_payload=payload),
        ],
        connect_payload=payload,
    )
    environment = FakeEnvironment(done_after_real_steps=14)
    calibration = _calibration("bsp_spline_async", latency_ns=50 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="bsp_spline_async"),
        scheduler=control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_async"], calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert [request.trigger for request in result.inference_requests[:3]] == [
        "initial_plan",
        "bsp_prefetch",
        "bsp_stale_replan",
    ]
    assert [request.disposition for request in result.inference_requests[:3]] == [
        "activated",
        "discarded_stale_phase",
        "activated",
    ]
    assert result.action_underflows == ()
    stale_replan_stall = next(stall for stall in result.control_stalls if stall.request_id == 2)
    assert stale_replan_stall.reason == "synchronous_inference"
    assert stale_replan_stall.control_step == 11
    assert stale_replan_stall.duration_ns == 50 * NS_PER_MS
    assert np.allclose(environment.actions[-1], 5.0)


def test_real_bsp_async_prefetches_immediately_when_underflow_returns_only_an_endpoint():
    clock = ManualClock()
    payload = _metadata_payload(policy_variant="bsp")
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(0, _bsp_response(), metadata_payload=payload),
            _ScriptedCall(
                550 * NS_PER_MS,
                _bsp_response(3.0, duration_ticks=9),
                metadata_payload=payload,
            ),
            _ScriptedCall(50 * NS_PER_MS, _bsp_response(5.0), metadata_payload=payload),
        ],
        connect_payload=payload,
    )
    environment = FakeEnvironment(done_after_real_steps=14)
    calibration = _calibration("bsp_spline_async", latency_ns=50 * NS_PER_MS)

    result = _run(
        clock,
        worker,
        environment,
        args=_args(execution_mode="bsp_spline_async"),
        scheduler=control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_async"], calibration),
        max_steps=30,
    )

    assert result.success, result.error
    assert [request.trigger for request in result.inference_requests] == [
        "initial_plan",
        "bsp_prefetch",
        "bsp_prefetch",
    ]
    assert result.plan_activations[1].activation_context["first_sample_microindices"] == 9_000_000
    assert result.plan_activations[1].activation_context["remaining_curve_ns"] == 0
    assert result.plan_activations[1].activation_context["immediate_prefetch"] == 1
    assert result.inference_requests[2].observation_control_step == 13
    assert result.inference_requests[2].disposition == "abandoned"
    assert np.allclose(environment.actions[-1], 3.0)


class _UnderflowScheduler(_BackgroundScheduler):
    def take_action(self, now_ns, *, control_step=0):
        if self.phase == 1 and self.action_index == 1:
            return control.ActionDecisionV5(action=None, underflow=True)
        return super().take_action(now_ns, control_step=control_step)


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
        timing.ActionUnderflowV5(
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


def test_v5_video_frames_hold_control_twice_and_show_persistent_cumulative_wait():
    stalls = (
        timing.ControlStallV5(0, 0, 0, 25 * NS_PER_MS, "synchronous_inference"),
        timing.ControlStallV5(1, 1, 50 * NS_PER_MS, 50 * NS_PER_MS, "async_action_underflow"),
    )
    rendered = []

    def renderer(frame, lines):
        rendered.append(lines)
        return f"{frame}:{lines[0]}"

    frames = main_v5._build_video_frames_v5(
        ("frame-0", "frame-1"),
        stalls,
        stall_source_frames=((0, "wait-0"), (1, "wait-1")),
        include_stalls=True,
        overlay_renderer=renderer,
    )

    assert rendered == [
        ("Cumulative inference wait: 0.00 s",),
        ("Cumulative inference wait: 0.03 s",),
        ("Cumulative inference wait: 0.03 s",),
        ("Cumulative inference wait: 0.03 s",),
        ("Cumulative inference wait: 0.05 s",),
        ("Cumulative inference wait: 0.08 s",),
        ("Cumulative inference wait: 0.08 s",),
    ]
    assert frames == (
        "wait-0:Cumulative inference wait: 0.00 s",
        "frame-0:Cumulative inference wait: 0.03 s",
        "frame-0:Cumulative inference wait: 0.03 s",
        "wait-1:Cumulative inference wait: 0.03 s",
        "wait-1:Cumulative inference wait: 0.05 s",
        "frame-1:Cumulative inference wait: 0.08 s",
        "frame-1:Cumulative inference wait: 0.08 s",
    )


def test_cumulative_wait_overlay_uses_text_stroke_without_a_black_rectangle():
    frame = np.full((80, 240, 3), 127, dtype=np.uint8)

    rendered = main_v5._draw_cumulative_wait_overlay_v5(
        frame,
        ("Cumulative inference wait: 0.00 s",),
    )

    assert np.array_equal(rendered[0, 220], frame[0, 220])
    assert np.array_equal(rendered[30, 220], frame[30, 220])
    assert np.array_equal(frame, np.full((80, 240, 3), 127, dtype=np.uint8))
    assert np.any(rendered != frame)


def test_worker_shutdown_failure_preserves_primary_exception():
    primary = KeyboardInterrupt("stop")

    class Worker:
        def reset_generation(self):
            return 3

        def wait_until_ready(self, generation, timeout=None):
            del generation, timeout

        def close(self):
            raise TimeoutError("join timeout")

    with pytest.raises(main_v5.RunCleanupError) as caught:
        main_v5._close_worker_v5(Worker(), primary_error=primary, timeout_s=1.0)

    assert caught.value.primary_error is primary
    assert isinstance(caught.value.cleanup_error, TimeoutError)


def test_async_calibration_failure_occurs_before_manifest_writer_creation(monkeypatch, tmp_path):
    args = _args(
        execution_mode="baseline_rtc",
        output_dir=str(tmp_path / "run"),
        config_name="pi05_libero",
        checkpoint_step=0,
        checkpoint="checkpoint/0",
        norm_hash="1" * 64,
        container_digest="sha256:" + "2" * 64,
    )
    mode = control.EXECUTION_MODES["baseline_rtc"]
    worker = FakeWorker(ManualClock(), [], connect_payload=_metadata_payload())
    canonical_request = {"observation/index": 0}
    canonical_identity = control.CalibrationObservationIdentityV1(
        suite="libero_spatial",
        task_id=0,
        init_state_index=0,
        init_state_fingerprint="a" * 64,
        request_fingerprint=control.canonical_fingerprint(canonical_request),
    )
    writer_creations = []

    monkeypatch.setattr(main_v5, "_resolve_code_sha_v5", lambda: "3" * 40)
    monkeypatch.setattr(
        main_v5,
        "_calibration_request_v5",
        lambda **_kwargs: (canonical_request, canonical_identity),
    )

    def fail_calibration(*_args, **_kwargs):
        raise control.CalibrationPolicyError("calibration response malformed")

    class WriterMustNotExist:
        def __init__(self, output_dir):
            writer_creations.append(Path(output_dir))

    monkeypatch.setattr(main_v5._control, "calibrate_async_mode", fail_calibration)
    monkeypatch.setattr(main_v5._eval, "ArtifactWriterV5", WriterMustNotExist)

    with pytest.raises(control.CalibrationPolicyError, match="malformed"):
        main_v5._evaluate_run_v5(
            args=args,
            suites=("libero_spatial",),
            task_ids=(0,),
            mode=mode,
            worker=worker,
            clock=worker.clock,
        )

    assert writer_creations == []
    assert not (tmp_path / "run").exists()


def test_selected_zero_frame_video_persists_episode_before_padding_and_audit(monkeypatch, tmp_path):
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, {"actions": np.zeros((16, 7), dtype=np.float32)})],
        formal_sampling=True,
    )
    attempt = _run(clock, worker, FakeEnvironment(done_after_real_steps=1))
    record = evaluation.EpisodeRecordV5.from_attempt(
        _identity(),
        42,
        1,
        execution_mode="baseline_async",
        result=attempt,
    )
    order = []

    class Writer:
        def append_episode(self, persisted):
            order.append(("episode", persisted))

        def append_video_audit(self, audit):
            order.append(("audit", audit))

        def append_artifact_error(self, error):
            order.append(("error", error))

    class Selector:
        def claim(self, persisted):
            assert order == [("episode", persisted)]
            return tmp_path / "selected.mp4"

    class StreamingWriter:
        def __init__(self, path, *, fps):
            order.append(("open", (path, fps)))

        def append_data(self, frame):
            order.append(("append", np.asarray(frame).copy()))

        def close(self):
            order.append(("close", None))

    monkeypatch.setattr(main_v5, "_read_encoded_video", lambda _path: (40.0, 1, 0.025))

    persisted, artifact_error = main_v5._persist_episode_artifacts_v5(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=False,
        video_writer_factory=StreamingWriter,
    )

    assert artifact_error is None
    assert persisted.replay_frames == ()
    assert [entry[0] for entry in order] == ["episode", "open", "append", "close", "audit"]
    audit = order[-1][1]
    assert audit.artifact_padding_frame_count == 1
    assert audit.encoded_frame_count == 1
    assert audit.planned.video_frame_count == 0


def test_production_video_is_closed_locally_before_copy_and_final_readback(monkeypatch, tmp_path):
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, {"actions": np.zeros((16, 7), dtype=np.float32)})],
        formal_sampling=True,
    )
    attempt = _run(clock, worker, FakeEnvironment(done_after_real_steps=1))
    record = evaluation.EpisodeRecordV5.from_attempt(
        _identity(),
        42,
        1,
        execution_mode="baseline_async",
        result=attempt,
    )
    final_path = tmp_path / "selected.mp4"
    audits = []

    class Writer:
        def append_episode(self, _persisted):
            pass

        def append_video_audit(self, audit):
            audits.append(audit)

        def append_artifact_error(self, error):
            raise AssertionError(error)

    class Selector:
        def claim(self, _persisted):
            return final_path

    class StreamingWriter:
        def __init__(self, path, *, fps):
            assert fps == 40
            self.path = Path(path)
            assert self.path != final_path
            assert self.path.suffix == ".mp4"

        def append_data(self, _frame):
            pass

        def close(self):
            self.path.write_bytes(b"closed-mp4-with-moov")

    def read_encoded(path):
        assert Path(path) == final_path
        assert final_path.read_bytes() == b"closed-mp4-with-moov"
        return 40.0, 1, 0.025

    monkeypatch.setattr(main_v5.imageio, "get_writer", StreamingWriter)
    monkeypatch.setattr(main_v5, "_read_encoded_video", read_encoded)

    persisted, artifact_error = main_v5._persist_episode_artifacts_v5(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=False,
    )

    assert persisted.replay_frames == ()
    assert artifact_error is None
    assert len(audits) == 1
    assert audits[0].path == str(final_path)


def test_success_video_stop_requires_successful_video_artifact():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [_ScriptedCall(0, _rtc_response())],
        formal_sampling=True,
    )
    attempt = _run(clock, worker, FakeEnvironment(done_after_real_steps=1))
    success = evaluation.EpisodeRecordV5.from_attempt(
        _identity(),
        42,
        1,
        execution_mode="baseline_async",
        result=attempt,
    )
    artifact_error = evaluation.ArtifactErrorV5(
        episode_id=success.episode_id,
        artifact_type="video",
        path="failed.mp4",
        error="ffmpeg failed",
    )

    assert main_v5._has_success_video_v5(success, None)
    assert not main_v5._has_success_video_v5(success, artifact_error)


def test_native_bsp_attempt_uses_zero_synthetic_latency_and_dynamic_activation_budget():
    clock = ManualClock()
    worker = FakeWorker(
        clock,
        [
            _ScriptedCall(
                80 * NS_PER_MS,
                _bsp_response(speedup=2),
                metadata_payload=_metadata_payload(policy_variant="bsp"),
            )
        ],
        connect_payload=_metadata_payload(policy_variant="bsp"),
        formal_sampling=False,
    )
    mode = control.EXECUTION_MODES["bsp_spline_async_native"]
    attempt = _run(
        clock,
        worker,
        FakeEnvironment(done_after_real_steps=1),
        args=_args(execution_mode=mode.name),
        scheduler=control.make_scheduler_v5(mode, _calibration(mode.name, latency_ns=80 * NS_PER_MS)),
    )
    record = evaluation.EpisodeRecordV5.from_attempt(
        _identity(),
        42,
        1,
        execution_mode=mode.name,
        result=attempt,
    )

    assert record.status == "success"
    assert record.inference_requests[0].sampled_target_latency_ns == 0
    assert record.inference_latencies[0].observed_synthetic_delay_ns == 0
    assert record.plan_activations[0].activation_context["prefetch_budget_ns"] == 100 * NS_PER_MS


@pytest.mark.parametrize(
    "failure",
    [pytest.param(None), pytest.param(KeyboardInterrupt("stop"))],
)
def test_eval_entrypoint_closes_single_worker_on_normal_and_exception_exit(monkeypatch, failure):
    workers = []

    class Worker:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.generation = 0
            self.reset_calls = 0
            self.ready_calls = []
            self.close_calls = 0
            workers.append(self)

        def reset_generation(self):
            self.generation += 1
            self.reset_calls += 1
            return self.generation

        def wait_until_ready(self, generation, timeout=None):
            self.ready_calls.append((generation, timeout))

        def close(self):
            self.close_calls += 1

    def evaluate(**_kwargs):
        if failure is not None:
            raise failure
        return {"acceptance_complete": True}

    monkeypatch.setattr(main_v5._async, "AsyncInferenceWorker", Worker)
    monkeypatch.setattr(main_v5, "_evaluate_run_v5", evaluate)
    args = _args()

    if failure is None:
        assert main_v5.eval_libero_v5(args) == {"acceptance_complete": True}
    else:
        with pytest.raises(KeyboardInterrupt, match="stop"):
            main_v5.eval_libero_v5(args)

    assert len(workers) == 1
    assert workers[0].reset_calls == 1
    assert workers[0].ready_calls == [(1, args.connection_timeout_s)]
    assert workers[0].close_calls == 1


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param("baseline_async"),
        pytest.param("baseline_rtc"),
        pytest.param("bsp_spline_async"),
        pytest.param("bsp_spline_async_native"),
        pytest.param("bsp_spline_async_native_speedup4"),
        pytest.param("bsp_spline_async_native_speedup8"),
    ],
)
def test_all_execution_modes_share_the_worker_latency_injection_point(monkeypatch, execution_mode):
    captured = []

    class Worker:
        def __init__(self, *args, **kwargs):
            del args
            captured.append(kwargs)

        def reset_generation(self):
            return 1

        def wait_until_ready(self, generation, timeout=None):
            del generation, timeout

        def close(self):
            pass

    monkeypatch.setattr(main_v5._async, "AsyncInferenceWorker", Worker)
    monkeypatch.setattr(
        main_v5,
        "_evaluate_run_v5",
        lambda **_kwargs: {"acceptance_complete": True},
    )
    args = _args(execution_mode=execution_mode)

    assert main_v5.eval_libero_v5(args) == {"acceptance_complete": True}
    sampler = captured[0]["latency_sampler"]
    assert isinstance(sampler, latency_sampling.NormalLatencySamplerV1)
    assert sampler.mean_ns == 300_000_000
    assert sampler.stddev_ns == 60_000_000
    assert captured[0]["inject_sampled_latency"] is (not control.is_native_latency_mode_v5(execution_mode))
    assert captured[0]["wait_until_ns"].__self__.__class__ is main_v5._SystemClock
