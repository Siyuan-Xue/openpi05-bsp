import copy
import dataclasses

import numpy as np
import pytest

from openpi_client import async_inference
from openpi_client import inference
from openpi_client import libero_control_v4 as control
from openpi_client import msgpack_numpy


_SHA = "a" * 64
_OTHER_SHA = "b" * 64


class _ManualClock:
    def __init__(self, now_ns=0):
        self.now_ns = now_ns
        self.waits = []

    def monotonic_ns(self):
        return self.now_ns

    def wait_until_ns(self, deadline_ns):
        self.waits.append(deadline_ns)
        self.now_ns = max(self.now_ns, deadline_ns)


def _native_metadata(extra=None):
    metadata = {
        "model": "pi0",
        inference.INFERENCE_CAPABILITIES_KEY: {
            "schema_version": 1,
            "action_representation": "native",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": ["baseline_h16_n5_v1", "baseline_rtc_h16_v1"],
        },
    }
    if extra is not None:
        metadata["extra"] = extra
    return metadata


def _bsp_metadata(extra=None):
    metadata = {
        "model": "pi0-bsp",
        inference.INFERENCE_CAPABILITIES_KEY: {
            "schema_version": 1,
            "action_representation": "bsp",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": ["bsp_spline_h8_v1"],
        },
    }
    if extra is not None:
        metadata["extra"] = extra
    return metadata


def _rtc_response(offset=0.0):
    return {
        "actions": np.arange(16 * 7, dtype=np.float32).reshape(16, 7) + offset,
        "rtc": {
            "schema_version": 1,
            "model_actions": np.arange(16 * 32, dtype=np.float32).reshape(16, 32) + offset,
        },
    }


def _bsp_sidecar(duration_ticks=9, value=1.0):
    parameters = np.zeros((16, 8), dtype=np.float32)
    parameters[:12, :7] = value
    parameters[:, 7] = (
        [0, 0, 0, 0]
        + np.linspace(0, duration_ticks, 10, dtype=np.float32)[1:9].tolist()
        + [duration_ticks, duration_ticks, duration_ticks, duration_ticks]
    )
    return {
        "schema_version": 1,
        "parameters": parameters,
        "origin_hz": 10,
        "degree": 3,
        "speedup": 1,
        "alignment": "disabled_delta_eff",
    }


def _bsp_response(duration_ticks=9, value=1.0):
    return {
        "actions": np.full((8, 7), value, dtype=np.float32),
        "bsp": _bsp_sidecar(duration_ticks=duration_ticks, value=value),
    }


def _checkpoint_identity(*, bsp=False):
    return control.CheckpointIdentityV1(
        code_sha="c" * 40,
        config_name="pi0_libero",
        checkpoint_step=1000,
        checkpoint="/checkpoints/run/1000/",
        container_digest="sha256:" + "d" * 64,
        norm_hash="e" * 64,
        bsp_cache_hash="f" * 64 if bsp else None,
        bsp_cache_manifest_fingerprint="1" * 64 if bsp else None,
    )


def _observation_identity(request):
    return control.CalibrationObservationIdentityV1(
        suite="libero_spatial",
        task_id=0,
        init_state_index=0,
        init_state_fingerprint="2" * 64,
        request_fingerprint=control.canonical_fingerprint(request),
    )


def _calibration(mode_name, p95_ns, *, request=None):
    if request is None:
        request = {"state": np.zeros(8, dtype=np.float32)}
    mode = control.EXECUTION_MODES[mode_name]
    measurements = [p95_ns] * 20
    return control.LatencyCalibrationV1.create(
        execution_mode=mode_name,
        checkpoint_identity_fingerprint=_checkpoint_identity(
            bsp=mode_name == "bsp_spline_async"
        ).fingerprint,
        server_metadata_fingerprint=_OTHER_SHA,
        canonical_observation_identity=_observation_identity(request),
        seed_namespace="openpi-libero-calibration-v1/{}/{}".format(
            mode_name,
            _checkpoint_identity(bsp=mode_name == "bsp_spline_async").fingerprint,
        ),
        bootstrap_request_fingerprint=_SHA if mode_name == "baseline_rtc" else None,
        warmup_request_fingerprints=[_SHA] * 5,
        measurement_request_fingerprints=[_OTHER_SHA] * 20,
        warmup_latency_ns=[3] * 5,
        measurement_latency_ns=measurements,
    )


class _CalibrationWorker:
    def __init__(self, mode_name, durations, *, fail_once_at=None, metadata=None):
        self.mode_name = mode_name
        self.durations = list(durations)
        self.fail_once_at = fail_once_at
        if metadata is None:
            metadata = _native_metadata() if mode_name == "baseline_rtc" else _bsp_metadata()
        self.connection = async_inference.ConnectionSnapshot(
            connection_id=0,
            metadata_payload=msgpack_numpy.packb(metadata),
        )
        self.requests = []
        self.jobs = []
        self.reset_calls = 0
        self.ready_generations = []
        self._generation = 0
        self._failed = False

    def connect(self, timeout=None):
        return self.connection

    def submit(self, observation):
        request = copy.deepcopy(observation)
        index = len(self.requests)
        self.requests.append(request)
        job = async_inference.InferenceJob(
            request_id=index,
            generation=self._generation,
            payload=b"payload",
            submitted_monotonic_ns=index * 1_000_000_000,
        )
        self.jobs.append(job)
        return job

    def wait(self, job, timeout=None):
        index = job.request_id
        if self.fail_once_at == index and not self._failed:
            self._failed = True
            return async_inference.InferenceOutcome(
                job=job,
                error=ConnectionError("transient disconnect"),
                completed_monotonic_ns=job.submitted_monotonic_ns + 1,
                connection=self.connection,
            )
        request = self.requests[index]
        if self.mode_name == "baseline_rtc":
            result = _rtc_response(float(index))
        else:
            result = _bsp_response(value=float(index))
        duration = self.durations[index % len(self.durations)]
        return async_inference.InferenceOutcome(
            job=job,
            result=result,
            completed_monotonic_ns=job.submitted_monotonic_ns + duration,
            connection=self.connection,
        )

    def reset_generation(self):
        self.reset_calls += 1
        self._generation += 1
        self.connection = dataclasses.replace(
            self.connection,
            connection_id=self.connection.connection_id + 1,
        )
        return self._generation

    def wait_until_ready(self, generation, timeout=None):
        self.ready_generations.append(generation)


def test_no_catchup_pacer_reanchors_every_deadline_to_the_actual_action_start():
    clock = _ManualClock()
    pacer = control.NoCatchupPacer(clock)

    assert pacer.wait_until_due() == 0
    assert pacer.mark_action_started(0) == 50_000_000
    assert pacer.wait_until_due() == 50_000_000
    assert pacer.mark_action_started(50_000_000) == 100_000_000

    clock.now_ns = 120_000_000
    assert pacer.wait_until_due() == 120_000_000
    assert pacer.mark_action_started(120_000_000) == 170_000_000
    assert clock.waits == [50_000_000]


def test_first_action_mark_uses_the_due_time_returned_by_wait_without_rereading_clock():
    class AdvancingReadClock:
        def __init__(self):
            self.now_ns = 0
            self.reads = 0

        def monotonic_ns(self):
            result = self.now_ns
            self.now_ns += 1
            self.reads += 1
            return result

        def wait_until_ns(self, deadline_ns):
            self.now_ns = max(self.now_ns, deadline_ns)

    clock = AdvancingReadClock()
    pacer = control.NoCatchupPacer(clock)

    first_due = pacer.wait_until_due()
    assert first_due == 0
    assert pacer.mark_action_started(first_due) == 50_000_000
    assert clock.reads == 1


def test_pacer_waits_through_early_wakeups_and_rejects_starts_before_due_time():
    class EarlyClock(_ManualClock):
        def wait_until_ns(self, deadline_ns):
            self.waits.append(deadline_ns)
            self.now_ns = min(deadline_ns, self.now_ns + 20)

    clock = EarlyClock()
    pacer = control.NoCatchupPacer(clock, period_ns=50)
    pacer.mark_action_started(0)

    assert pacer.wait_until_due() == 50
    assert clock.waits == [50, 50, 50]
    with pytest.raises(ValueError, match="due"):
        pacer.mark_action_started(49)


def test_frozen_execution_modes_emit_only_the_exact_protocol_parameters():
    assert tuple(control.EXECUTION_MODES) == (
        "baseline_sync_n5",
        "baseline_rtc",
        "bsp_spline_sync",
        "bsp_spline_async",
    )
    assert control.EXECUTION_MODES["baseline_sync_n5"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "synchronous",
        "model_action_horizon": 16,
        "execution_horizon": 8,
        "num_inference_steps": 5,
    }
    assert control.EXECUTION_MODES["baseline_rtc"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "num_inference_steps": 5,
        "guidance_beta": 5,
        "delay_history_size": 10,
        "activation_policy": "immediate",
    }
    assert control.EXECUTION_MODES["bsp_spline_sync"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "synchronous",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 1,
        "alignment": "disabled_delta_eff",
        "activation_policy": "blocking_replace",
    }
    assert control.EXECUTION_MODES["bsp_spline_async"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 1,
        "alignment": "disabled_delta_eff",
        "activation_policy": "immediate",
        "prefetch_comparison": "remaining_lte_budget",
    }
    with pytest.raises(TypeError):
        control.EXECUTION_MODES["extra"] = control.EXECUTION_MODES["baseline_rtc"]


@pytest.mark.parametrize(
    ("mode_name", "metadata"),
    [
        ("baseline_sync_n5", _native_metadata()),
        ("baseline_rtc", _native_metadata()),
        ("bsp_spline_sync", _bsp_metadata()),
        ("bsp_spline_async", _bsp_metadata()),
    ],
)
def test_server_metadata_validation_binds_the_full_outer_object(mode_name, metadata):
    mode = control.EXECUTION_MODES[mode_name]

    fingerprint = control.validate_server_metadata(mode, metadata)

    assert fingerprint == control.canonical_fingerprint(metadata)
    assert fingerprint != control.canonical_fingerprint({**metadata, "unrelated": "changed"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("model_action_dim"),
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(model_action_horizon=True),
        lambda value: value.update(supported_protocols=list(reversed(value["supported_protocols"]))),
        lambda value: value.update(supported_protocols=value["supported_protocols"] * 2),
        lambda value: value.update(action_representation="bsp"),
    ],
)
def test_server_metadata_rejects_nonexact_or_unrelated_capabilities(mutation):
    metadata = _native_metadata()
    mutation(metadata[inference.INFERENCE_CAPABILITIES_KEY])

    with pytest.raises(ValueError, match="capabilit"):
        control.validate_server_metadata(control.EXECUTION_MODES["baseline_rtc"], metadata)


def test_baseline_sync_validates_every_rtc_sidecar_and_executes_only_rows_zero_through_seven():
    scheduler = control.make_scheduler_v4(control.EXECUTION_MODES["baseline_sync_n5"], None)
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)

    assert initial.dispatch == "blocking_initial"
    assert initial.trigger == "initial_plan"
    assert dict(initial.scheduler_context) == {}
    assert initial.request_overlay == {inference.RTC_REQUEST_KEY: {"schema_version": 1}}
    activation = scheduler.install_response(initial, _rtc_response(), now_ns=10, control_step=0)
    assert activation.activation == "initial"
    assert dict(activation.activation_context) == {"action_cursor": 0}

    actions = [scheduler.take_action(10 + index) for index in range(8)]
    for index, decision in enumerate(actions):
        assert not decision.underflow
        np.testing.assert_array_equal(
            decision.action,
            np.arange(16 * 7, dtype=np.float32).reshape(16, 7)[index],
        )
    replan = scheduler.maybe_request(18, at_due=False, request_in_flight=False)
    assert replan.dispatch == "blocking_replan"
    assert replan.trigger == "baseline_chunk_exhausted"
    assert replan.request_overlay == {inference.RTC_REQUEST_KEY: {"schema_version": 1}}
    with pytest.raises(ValueError, match="rtc"):
        scheduler.install_response(
            replan,
            {"actions": np.zeros((16, 7), dtype=np.float32)},
            now_ns=19,
            control_step=8,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: control.RequestIntentV4(
            dispatch="background",
            trigger="initial_plan",
            scheduler_context={},
            request_overlay={},
        ),
        lambda: control.RequestIntentV4(
            dispatch="blocking_replan",
            trigger="rtc_launch",
            scheduler_context={"s": 8, "d": 8},
            request_overlay={},
        ),
        lambda: control.RequestIntentV4(
            dispatch="background",
            trigger="rtc_launch",
            scheduler_context={"s": 8, "d": 8, "extra": 0},
            request_overlay={},
        ),
        lambda: control.RequestIntentV4(
            dispatch="background",
            trigger="rtc_launch",
            scheduler_context={"s": 9, "d": 8},
            request_overlay={},
        ),
        lambda: control.RequestIntentV4(
            dispatch="background",
            trigger="bsp_prefetch",
            scheduler_context={"remaining_plan_ns": 51, "budget_ns": 50},
            request_overlay={},
        ),
    ],
)
def test_request_intents_reject_dispatch_trigger_and_exact_context_mismatches(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"action_cursor": 16},
        {"curve_elapsed_ns": 1},
        {"action_cursor": 0, "curve_elapsed_ns": 0},
        {"unexpected": 0},
    ],
)
def test_activation_decisions_require_one_exact_native_or_bsp_context(context):
    with pytest.raises(ValueError):
        control.ActivationDecisionV4(activation="initial", activation_context=context)


def test_scheduler_install_requires_the_exact_pending_transition_object():
    scheduler = control.make_scheduler_v4(control.EXECUTION_MODES["baseline_sync_n5"], None)
    pending = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    fabricated = dataclasses.replace(pending)

    with pytest.raises(ValueError, match="pending"):
        scheduler.install_response(fabricated, _rtc_response(), now_ns=1, control_step=0)
    scheduler.install_response(pending, _rtc_response(), now_ns=1, control_step=0)
    with pytest.raises(ValueError, match="pending"):
        scheduler.install_response(pending, _rtc_response(), now_ns=2, control_step=0)


def test_rtc_adapter_uses_plan_launch_gates_q_install_and_nonadvancing_underflow():
    scheduler = control.make_scheduler_v4(
        control.EXECUTION_MODES["baseline_rtc"],
        _calibration("baseline_rtc", 400_000_000),
    )
    bootstrap = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    assert bootstrap.dispatch == "blocking_initial"
    scheduler.install_response(bootstrap, _rtc_response(), now_ns=1, control_step=0)

    for index in range(8):
        assert not scheduler.take_action(index + 2).underflow
    guided = scheduler.maybe_request(10, at_due=False, request_in_flight=False)
    assert guided.dispatch == "background"
    assert guided.trigger == "rtc_launch"
    assert dict(guided.scheduler_context) == {"s": 8, "d": 8}

    for index in range(8):
        assert not scheduler.take_action(index + 11).underflow
    assert scheduler.take_action(20).underflow
    assert scheduler.take_action(21).underflow

    activation = scheduler.install_response(
        guided,
        _rtc_response(1000),
        now_ns=22,
        control_step=18,
    )
    assert activation.activation == "immediate_swap"
    assert dict(activation.activation_context) == {"action_cursor": 8}
    np.testing.assert_array_equal(
        scheduler.take_action(23).action,
        _rtc_response(1000)["actions"][8],
    )


def test_bsp_async_launches_on_budget_equality_and_immediately_restarts_curve_clock():
    scheduler = control.make_scheduler_v4(
        control.EXECUTION_MODES["bsp_spline_async"],
        _calibration("bsp_spline_async", 100_000_000),
    )
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(), now_ns=0, control_step=0)

    assert scheduler.maybe_request(
        799_999_999, at_due=False, request_in_flight=False
    ) is None
    prefetch = scheduler.maybe_request(800_000_000, at_due=True, request_in_flight=False)
    assert prefetch.dispatch == "background"
    assert prefetch.trigger == "bsp_prefetch"
    assert dict(prefetch.scheduler_context) == {
        "remaining_plan_ns": 100_000_000,
        "budget_ns": 100_000_000,
    }

    activation = scheduler.install_response(
        prefetch,
        _bsp_response(value=7),
        now_ns=825_000_000,
        control_step=17,
    )
    assert activation.activation == "immediate_swap"
    assert dict(activation.activation_context) == {"curve_elapsed_ns": 0}
    np.testing.assert_array_equal(
        scheduler.take_action(825_000_000).action,
        np.full(7, 7, dtype=np.float32),
    )


def test_bsp_async_rejects_budget_equal_to_a_new_curves_usable_duration():
    scheduler = control.make_scheduler_v4(
        control.EXECUTION_MODES["bsp_spline_async"],
        _calibration("bsp_spline_async", 400_000_000),
    )
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)

    with pytest.raises(control.BspBudgetError, match="usable"):
        scheduler.install_response(
            initial,
            _bsp_response(duration_ticks=4),
            now_ns=0,
            control_step=0,
        )

    later = control.make_scheduler_v4(
        control.EXECUTION_MODES["bsp_spline_async"],
        _calibration("bsp_spline_async", 400_000_000),
    )
    first = later.maybe_request(0, at_due=True, request_in_flight=False)
    later.install_response(first, _bsp_response(value=3), now_ns=0, control_step=0)
    prefetch = later.maybe_request(500_000_000, at_due=True, request_in_flight=False)
    with pytest.raises(control.BspBudgetError, match="usable"):
        later.install_response(
            prefetch,
            _bsp_response(duration_ticks=4, value=9),
            now_ns=525_000_000,
            control_step=11,
        )
    np.testing.assert_array_equal(
        later.take_action(525_000_000).action,
        np.full(7, 3, dtype=np.float32),
    )


def test_bsp_sync_blocks_only_after_the_closed_right_endpoint_is_no_longer_executable():
    scheduler = control.make_scheduler_v4(control.EXECUTION_MODES["bsp_spline_sync"], None)
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(), now_ns=0, control_step=0)

    endpoint = scheduler.take_action(900_000_000)
    assert not endpoint.underflow
    assert scheduler.maybe_request(900_000_000, at_due=True, request_in_flight=False) is None
    assert scheduler.take_action(900_000_001).underflow
    assert scheduler.maybe_request(
        900_000_001,
        at_due=False,
        request_in_flight=False,
    ) is None
    replan = scheduler.maybe_request(900_000_001, at_due=True, request_in_flight=False)
    assert replan.dispatch == "blocking_replan"
    assert replan.trigger == "bsp_curve_exhausted"


@pytest.mark.parametrize(
    "response",
    [
        {"bsp": _bsp_sidecar()},
        {"actions": np.zeros((8, 7), dtype=np.float32)},
        {"actions": np.zeros((8, 8), dtype=np.float32), "bsp": _bsp_sidecar()},
        {"actions": np.full((8, 7), np.nan, dtype=np.float32), "bsp": _bsp_sidecar()},
        {"actions": np.full((8, 7), np.finfo(np.float64).max), "bsp": _bsp_sidecar()},
        {"actions": np.full((8, 7), 1 + 2j, dtype=np.complex64), "bsp": _bsp_sidecar()},
        {"actions": [[object()] * 7 for _ in range(8)], "bsp": _bsp_sidecar()},
    ],
)
def test_bsp_scheduler_requires_a_complete_finite_float32_representable_dual_response(response):
    scheduler = control.make_scheduler_v4(control.EXECUTION_MODES["bsp_spline_sync"], None)
    intent = scheduler.maybe_request(0, at_due=True, request_in_flight=False)

    with pytest.raises(ValueError, match="BSP|actions"):
        scheduler.install_response(intent, response, now_ns=0, control_step=0)


def test_bsp_calibration_uses_the_same_complete_dual_response_validator():
    request = {"state": np.arange(4, dtype=np.float32)}
    worker = _CalibrationWorker("bsp_spline_async", [1] * 25)
    original_wait = worker.wait

    def missing_legacy_actions(job, timeout=None):
        outcome = original_wait(job, timeout=timeout)
        return dataclasses.replace(outcome, result={"bsp": outcome.result["bsp"]})

    worker.wait = missing_legacy_actions

    with pytest.raises(control.CalibrationPolicyError, match="BSP"):
        control.calibrate_async_mode(
            control.EXECUTION_MODES["bsp_spline_async"],
            request,
            _observation_identity(request),
            worker,
            _checkpoint_identity(bsp=True),
            control.canonical_fingerprint(_bsp_metadata()),
        )


def test_nearest_rank_p95_uses_only_exactly_twenty_nonnegative_integer_measurements():
    rank, value = control.nearest_rank_p95_ns(list(range(1, 21)))
    assert (rank, value) == (19, 19)

    for invalid in ([1] * 19, [1] * 21, [1] * 19 + [-1], [1] * 19 + [True]):
        with pytest.raises(ValueError):
            control.nearest_rank_p95_ns(invalid)


@pytest.mark.parametrize(
    ("mode_name", "p95_ns", "delay_ticks", "budget_ns"),
    [
        ("baseline_rtc", 0, 0, None),
        ("baseline_rtc", 50_000_000, 1, None),
        ("baseline_rtc", 50_000_001, 2, None),
        ("baseline_rtc", 400_000_000, 8, None),
        ("bsp_spline_async", 0, None, 0),
        ("bsp_spline_async", 50_000_000, None, 50_000_000),
        ("bsp_spline_async", 50_000_001, None, 100_000_000),
        ("bsp_spline_async", 400_000_000, None, 400_000_000),
        ("bsp_spline_async", 400_000_001, None, 450_000_000),
    ],
)
def test_calibration_derivation_uses_exact_ceiling_boundaries(
    mode_name, p95_ns, delay_ticks, budget_ns
):
    calibration = _calibration(mode_name, p95_ns)

    assert calibration.derived_delay_ticks == delay_ticks
    assert calibration.derived_prefetch_budget_ns == budget_ns


def test_rtc_calibration_rejects_a_p95_above_eight_control_ticks():
    with pytest.raises(ValueError, match="eight"):
        _calibration("baseline_rtc", 400_000_001)


def test_rtc_calibration_uses_one_untimed_bootstrap_then_five_plus_twenty_chained_guided_calls():
    request = {"state": np.arange(8, dtype=np.float32), "prompt": "pick"}
    durations = [999] + [1000 + index for index in range(25)]
    worker = _CalibrationWorker("baseline_rtc", durations)
    metadata_fingerprint = control.canonical_fingerprint(_native_metadata())

    calibration = control.calibrate_async_mode(
        control.EXECUTION_MODES["baseline_rtc"],
        request,
        _observation_identity(request),
        worker,
        _checkpoint_identity(),
        metadata_fingerprint,
    )

    assert len(worker.requests) == 26
    assert worker.requests[0][inference.RTC_REQUEST_KEY] == {"schema_version": 1}
    for index, guided_request in enumerate(worker.requests[1:], start=1):
        envelope = guided_request[inference.RTC_REQUEST_KEY]
        assert envelope["s"] == 8
        assert envelope["d"] == 8
        np.testing.assert_array_equal(
            envelope["previous_model_actions"],
            _rtc_response(float(index - 1))["rtc"]["model_actions"],
        )
    assert calibration.bootstrap_request_fingerprint == control.canonical_fingerprint(
        worker.requests[0]
    )
    assert list(calibration.warmup_latency_ns) == durations[1:6]
    assert list(calibration.measurement_latency_ns) == durations[6:26]
    assert calibration.p95_latency_ns == sorted(durations[6:26])[18]


def test_bsp_calibration_discards_partial_samples_and_restarts_whole_sequence_after_infrastructure():
    request = {"state": np.arange(4, dtype=np.float32)}
    worker = _CalibrationWorker(
        "bsp_spline_async",
        [1] * 25,
        fail_once_at=3,
    )

    calibration = control.calibrate_async_mode(
        control.EXECUTION_MODES["bsp_spline_async"],
        request,
        _observation_identity(request),
        worker,
        _checkpoint_identity(bsp=True),
        control.canonical_fingerprint(_bsp_metadata()),
    )

    assert worker.reset_calls == 1
    assert worker.ready_generations == [1]
    assert len(worker.requests) == 4 + 25
    assert len(calibration.warmup_latency_ns) == 5
    assert len(calibration.measurement_latency_ns) == 20


def test_changed_outcome_metadata_is_fatal_before_a_transport_error_can_trigger_retry():
    request = {"state": np.arange(8, dtype=np.float32)}
    worker = _CalibrationWorker("baseline_rtc", [1] * 26)
    original_wait = worker.wait
    changed_connection = async_inference.ConnectionSnapshot(
        connection_id=1,
        metadata_payload=msgpack_numpy.packb(_native_metadata(extra="changed")),
    )

    def changed_metadata_transport_error(job, timeout=None):
        outcome = original_wait(job, timeout=timeout)
        return dataclasses.replace(
            outcome,
            result=None,
            error=ConnectionError("disconnect after metadata changed"),
            connection=changed_connection,
        )

    worker.wait = changed_metadata_transport_error

    with pytest.raises(control.CalibrationIdentityError, match="metadata"):
        control.calibrate_async_mode(
            control.EXECUTION_MODES["baseline_rtc"],
            request,
            _observation_identity(request),
            worker,
            _checkpoint_identity(),
            control.canonical_fingerprint(_native_metadata()),
        )
    assert worker.reset_calls == 0


def test_calibration_fingerprint_binds_canonical_inputs_raw_samples_and_derived_values():
    request = {
        "state": np.arange(4, dtype=np.float32),
        "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "gain": 0.5,
        "payload": b"\x00\xff",
    }
    first = _calibration("baseline_rtc", 50_000_000, request=request)
    second = control.LatencyCalibrationV1.from_dict(first.to_dict())
    assert first.fingerprint == second.fingerprint

    mutated = first.to_dict()
    mutated["measurement_latency_ns"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        control.LatencyCalibrationV1.from_dict(mutated)

    changed_request = dict(request)
    changed_request["image"] = request["image"].astype(np.int16)
    assert control.canonical_fingerprint(request) != control.canonical_fingerprint(changed_request)
    changed_request = dict(request)
    changed_request["image"] = request["image"].reshape(1, 4, 3)
    assert control.canonical_fingerprint(request) != control.canonical_fingerprint(changed_request)
    changed_request = dict(request)
    changed_request["image"] = request["image"].copy()
    changed_request["image"].flat[0] += 1
    assert control.canonical_fingerprint(request) != control.canonical_fingerprint(changed_request)


def test_canonical_encoding_has_literal_float_bytes_and_c_contiguous_ndarray_golden():
    value = {
        "float": 1.5,
        "bytes": b"\x00\xff",
        "array": np.asarray([[1, 2], [3, 4]], dtype="<i2"),
    }
    expected = (
        b'{"__mapping__":{"array":{"__ndarray__":{"dtype":"<i2",'
        b'"sha256":"ea99f710d9d0b8ba192295c969a63ed7ce8fc5743da20d2057fa2b6d2c404bfb",'
        b'"shape":[2,2]}},"bytes":{"__bytes__":"00ff"},'
        b'"float":{"__float_hex__":"0x1.8000000000000p+0"}}}'
    )

    assert control.canonical_json_bytes(value) == expected
    assert control.canonical_fingerprint(value) == (
        "4afa4d2be877c5232bb1a06e501768fc36689e69ff5d66b37ed3d3b054415245"
    )


def test_canonical_type_tags_cannot_collide_with_literal_user_mappings():
    array = np.asarray([[1, 2], [3, 4]], dtype="<i2")
    ndarray_literal = {
        "__ndarray__": {
            "dtype": "<i2",
            "shape": [2, 2],
            "sha256": "ea99f710d9d0b8ba192295c969a63ed7ce8fc5743da20d2057fa2b6d2c404bfb",
        }
    }

    assert control.canonical_json_bytes(1.5) != control.canonical_json_bytes(
        {"__float_hex__": 1.5.hex()}
    )
    assert control.canonical_json_bytes(b"\x00\xff") != control.canonical_json_bytes(
        {"__bytes__": "00ff"}
    )
    assert control.canonical_json_bytes(array) != control.canonical_json_bytes(
        ndarray_literal
    )


def test_seed_namespace_phase_index_and_checkpoint_identity_are_fingerprint_inputs():
    checkpoint = _checkpoint_identity()

    assert control.calibration_seed("namespace", "warmup", 0) == int("d099f188", 16)
    assert control.calibration_seed("namespace", "warmup", 0) != control.calibration_seed(
        "namespace", "measurement", 0
    )
    assert control.calibration_seed("namespace", "warmup", 0) != control.calibration_seed(
        "namespace", "warmup", 1
    )
    assert checkpoint.fingerprint != dataclasses.replace(
        checkpoint,
        checkpoint="/different/1000",
    ).fingerprint
    assert checkpoint.fingerprint != dataclasses.replace(
        checkpoint,
        norm_hash="3" * 64,
    ).fingerprint


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("clock"),
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(p95_rank=19.0),
        lambda value: value.update(warmup_latency_ns=tuple(value["warmup_latency_ns"])),
        lambda value: value.update(derived_delay_ticks=True),
    ],
)
def test_latency_calibration_from_dict_rejects_nonexact_json_schema_mutations(mutation):
    payload = _calibration("baseline_rtc", 50_000_000).to_dict()
    mutation(payload)

    with pytest.raises(ValueError):
        control.LatencyCalibrationV1.from_dict(payload)


@pytest.mark.parametrize(
    "invalid",
    [
        {"bad": object()},
        {1: "non-string key"},
        {"array": np.asarray([object()], dtype=object)},
        {"float": float("nan")},
        {"array": np.asarray([float("inf")], dtype=np.float32)},
    ],
)
def test_canonical_encoder_rejects_ambiguous_unsupported_or_nonfinite_inputs(invalid):
    with pytest.raises((TypeError, ValueError)):
        control.canonical_fingerprint(invalid)


def test_checkpoint_identity_normalizes_the_checkpoint_before_fingerprinting_and_serialization():
    identity = _checkpoint_identity()
    without_slash = dataclasses.replace(identity, checkpoint="/checkpoints/run/1000")

    assert identity.checkpoint == "/checkpoints/run/1000"
    assert identity.to_dict()["checkpoint"] == "/checkpoints/run/1000"
    assert identity.fingerprint == without_slash.fingerprint


def test_calibration_rejects_metadata_change_in_an_unrelated_outer_field():
    request = {"state": np.zeros(4, dtype=np.float32)}
    worker = _CalibrationWorker(
        "baseline_rtc",
        [1] * 26,
        metadata=_native_metadata(extra="changed"),
    )

    with pytest.raises(control.CalibrationIdentityError, match="metadata"):
        control.calibrate_async_mode(
            control.EXECUTION_MODES["baseline_rtc"],
            request,
            _observation_identity(request),
            worker,
            _checkpoint_identity(),
            control.canonical_fingerprint(_native_metadata()),
        )
