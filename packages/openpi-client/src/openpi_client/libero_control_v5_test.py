import copy
import dataclasses
import hashlib

import numpy as np
import pytest

from openpi_client import async_inference
from openpi_client import inference
from openpi_client import latency_sampling
from openpi_client import libero_control_v5 as control
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
            "supported_protocols": [
                "baseline_h16_n5_v1",
                "baseline_sync_n5_h16_full_v2",
                "baseline_async_h16_v1",
                "baseline_async_h16_blocking_recovery_v2",
                "baseline_rtc_h16_v1",
            ],
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
            "supported_protocols": [
                "bsp_spline_sync_speedup2_phase0_v2",
                "bsp_spline_async_phase_skip_speedup2_v2",
                "bsp_spline_async_phase_skip_speedup1_v1",
                "bsp_spline_async_phase_skip_speedup4_v1",
                "bsp_spline_async_phase_skip_speedup8_v1",
            ],
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


def _bsp_sidecar(duration_ticks=9, value=1.0, *, speedup=2):
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
        "speedup": speedup,
        "alignment": "disabled_delta_eff",
    }


def _bsp_response(duration_ticks=9, value=1.0, *, speedup=2):
    return {
        "actions": np.full((8, 7), value, dtype=np.float32),
        "bsp": _bsp_sidecar(duration_ticks=duration_ticks, value=value, speedup=speedup),
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
    measurements = [p95_ns] * 20
    return control.LatencyCalibrationV1.create(
        execution_mode=mode_name,
        checkpoint_identity_fingerprint=_checkpoint_identity(bsp=mode_name == "bsp_spline_async").fingerprint,
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
            metadata = _native_metadata() if mode_name.startswith("baseline") else _bsp_metadata()
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
        self.latency_sample_keys = []
        self._sampler = latency_sampling.NormalLatencySamplerV1()

    def connect(self, timeout=None):
        return self.connection

    def submit(self, observation, *, latency_sample_key):
        request = copy.deepcopy(observation)
        index = len(self.requests)
        self.requests.append(request)
        self.latency_sample_keys.append(latency_sample_key)
        sampled_target_latency_ns = self._sampler.sample_target_ns(latency_sample_key)
        job = async_inference.InferenceJob(
            request_id=index,
            generation=self._generation,
            payload=b"payload",
            submitted_monotonic_ns=index * 1_000_000_000,
            latency_sample_key=latency_sample_key,
            sampled_target_latency_ns=sampled_target_latency_ns,
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
        if self.mode_name in ("baseline_async", "baseline_async_recovery", "baseline_rtc"):
            result = _rtc_response(float(index))
        else:
            result = _bsp_response(
                value=float(index),
                speedup=1 if self.mode_name == "bsp_spline_async_speedup1" else 2,
            )
        duration = self.durations[index % len(self.durations)]
        effective = max(duration, job.sampled_target_latency_ns)
        return async_inference.InferenceOutcome(
            job=job,
            result=result,
            completed_monotonic_ns=job.submitted_monotonic_ns + effective,
            connection=self.connection,
            sampled_target_latency_ns=job.sampled_target_latency_ns,
            raw_inference_latency_ns=duration,
            requested_synthetic_delay_ns=effective - duration,
            observed_synthetic_delay_ns=effective - duration,
            observed_effective_latency_ns=effective,
            latency_overshoot_ns=0,
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


def test_frozen_execution_modes_preserve_async_parameters_and_add_exact_sync_parameters():
    assert tuple(control.EXECUTION_MODES) == (
        "baseline_async",
        "baseline_async_recovery",
        "baseline_rtc",
        "bsp_spline_async",
        "bsp_spline_async_speedup1",
        "bsp_spline_async_native",
        "bsp_spline_async_native_speedup4",
        "bsp_spline_async_native_speedup8",
        "baseline_sync",
        "bsp_spline_sync",
    )
    assert control.EXECUTION_MODES["baseline_async"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "forecast_delay_ticks": 8,
        "num_inference_steps": 5,
        "continuity_guidance": False,
        "activation_policy": "immediate_skip_elapsed_prefix",
    }
    assert control.EXECUTION_MODES["baseline_async_recovery"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial_with_blocking_capacity_recovery",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "forecast_delay_ticks": 8,
        "num_inference_steps": 5,
        "continuity_guidance": False,
        "activation_policy": "immediate_skip_elapsed_prefix",
        "capacity_recovery": "blocking_replan_from_latest_observation",
    }
    assert control.EXECUTION_MODES["baseline_rtc"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "num_inference_steps": 5,
        "guidance_beta": 5,
        "forecast_delay_ticks": 8,
        "continuity_guidance": True,
        "activation_policy": "immediate",
    }
    assert control.EXECUTION_MODES["bsp_spline_async"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
    }
    assert control.EXECUTION_MODES["bsp_spline_async_speedup1"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 1,
        "effective_curve_rate_hz": 10,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
    }
    assert control.EXECUTION_MODES["bsp_spline_async_native"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
        "latency_injection": "disabled_native",
        "prefetch_budget_policy": "rolling_raw_p95_20_ceil_control_period",
        "prefetch_window_size": 20,
        "prefetch_budget_rounding_ns": 50_000_000,
    }
    assert control.EXECUTION_MODES["baseline_sync"].to_parameters_dict() == {
        "action_representation": "native",
        "dispatch": "synchronous",
        "model_action_horizon": 16,
        "execution_horizon": 16,
        "num_inference_steps": 5,
        "activation_policy": "blocking_replace_after_full_chunk",
    }
    assert control.EXECUTION_MODES["bsp_spline_sync"].to_parameters_dict() == {
        "action_representation": "bsp",
        "dispatch": "synchronous",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "blocking_replace_from_curve_start",
        "replan_policy": "after_closed_curve_endpoint",
    }
    with pytest.raises(TypeError):
        control.EXECUTION_MODES["extra"] = control.EXECUTION_MODES["baseline_rtc"]


@pytest.mark.parametrize(
    ("mode_name", "metadata"),
    [
        ("baseline_async", _native_metadata()),
        ("baseline_rtc", _native_metadata()),
        ("bsp_spline_async", _bsp_metadata()),
    ],
)
def test_server_metadata_validation_binds_the_full_outer_object(mode_name, metadata):
    mode = control.EXECUTION_MODES[mode_name]

    fingerprint = control.validate_server_metadata(mode, metadata)

    assert fingerprint == control.canonical_fingerprint(metadata)
    assert fingerprint != control.canonical_fingerprint({**metadata, "unrelated": "changed"})


def test_recovery_mode_requires_the_new_baseline_policy_capability():
    metadata = _native_metadata()
    assert control.validate_server_metadata(control.EXECUTION_MODES["baseline_async_recovery"], metadata)

    metadata[inference.INFERENCE_CAPABILITIES_KEY]["supported_protocols"].remove(
        "baseline_async_h16_blocking_recovery_v2"
    )
    with pytest.raises(ValueError, match="capabilit"):
        control.validate_server_metadata(control.EXECUTION_MODES["baseline_async_recovery"], metadata)


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


@pytest.mark.parametrize(
    "factory",
    [
        lambda: control.RequestIntentV5(
            dispatch="background",
            trigger="initial_plan",
            scheduler_context={},
            request_overlay={},
        ),
        lambda: control.RequestIntentV5(
            dispatch="blocking_replan",
            trigger="rtc_launch",
            scheduler_context={"s": 8, "d": 8},
            request_overlay={},
        ),
        lambda: control.RequestIntentV5(
            dispatch="background",
            trigger="rtc_launch",
            scheduler_context={"s": 8, "d": 8, "extra": 0},
            request_overlay={},
        ),
        lambda: control.RequestIntentV5(
            dispatch="background",
            trigger="rtc_launch",
            scheduler_context={"s": 9, "d": 8},
            request_overlay={},
        ),
        lambda: control.RequestIntentV5(
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
        {"curve_elapsed_ns": 0},
        {"action_cursor": 0, "curve_elapsed_ns": 0},
        {"unexpected": 0},
    ],
)
def test_activation_decisions_require_one_exact_native_or_bsp_context(context):
    with pytest.raises(ValueError):
        control.ActivationDecisionV5(activation="initial", activation_context=context)


def test_scheduler_install_requires_the_exact_pending_transition_object():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_async"],
        _schema5_calibration("baseline_async"),
    )
    pending = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    fabricated = dataclasses.replace(pending)

    with pytest.raises(ValueError, match="pending"):
        scheduler.install_response(fabricated, _rtc_response(), now_ns=1, control_step=0)
    scheduler.install_response(pending, _rtc_response(), now_ns=1, control_step=0)
    with pytest.raises(ValueError, match="pending"):
        scheduler.install_response(pending, _rtc_response(), now_ns=2, control_step=0)


def test_rtc_adapter_uses_plan_launch_gates_q_install_and_nonadvancing_underflow():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_rtc"],
        _schema5_calibration("baseline_rtc"),
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


def test_bsp_async_launches_on_budget_equality_and_skips_the_executed_prefix():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(), now_ns=0, control_step=0)

    assert (
        scheduler.maybe_request(
            49_999_999,
            control_step=0,
            at_due=False,
            request_in_flight=False,
        )
        is None
    )
    prefetch = scheduler.maybe_request(
        50_000_000,
        control_step=1,
        at_due=True,
        request_in_flight=False,
    )
    assert prefetch.dispatch == "background"
    assert prefetch.trigger == "bsp_prefetch"
    assert dict(prefetch.scheduler_context) == {
        "remaining_plan_ns": 400_000_000,
        "budget_ns": 400_000_000,
        "request_control_step": 1,
    }

    activation = scheduler.install_response(
        prefetch,
        _bsp_response(value=7),
        now_ns=75_000_000,
        control_step=7,
    )
    assert activation.activation == "immediate_swap"
    assert activation.activation_context["executed_prefix_steps"] == 6
    assert activation.activation_context["first_sample_microindices"] == 6_000_000
    np.testing.assert_array_equal(
        scheduler.take_action(75_000_000, control_step=7).action,
        np.full(7, 7, dtype=np.float32),
    )


def test_bsp_async_accepts_a_curve_shorter_than_the_budget_and_immediately_prefetches():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    activation = scheduler.install_response(
        initial,
        _bsp_response(duration_ticks=4),
        now_ns=0,
        control_step=0,
    )
    assert activation.activation_context["remaining_curve_ns"] == 200_000_000
    assert activation.activation_context["immediate_prefetch"] == 1
    assert (
        scheduler.maybe_request(
            1,
            control_step=0,
            at_due=False,
            request_in_flight=False,
        ).trigger
        == "bsp_prefetch"
    )

    later = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    first = later.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    later.install_response(first, _bsp_response(value=3), now_ns=0, control_step=0)
    prefetch = later.maybe_request(
        50_000_000,
        control_step=1,
        at_due=True,
        request_in_flight=False,
    )
    replacement = later.install_response(
        prefetch,
        _bsp_response(duration_ticks=4, value=9),
        now_ns=75_000_000,
        control_step=3,
    )
    assert replacement.activation_context["remaining_curve_ns"] == 100_000_000
    assert replacement.activation_context["immediate_prefetch"] == 1
    np.testing.assert_array_equal(
        later.take_action(75_000_000, control_step=3).action,
        np.full(7, 9, dtype=np.float32),
    )


def test_bsp_async_canonicalizes_fractional_knot_remaining_time_for_strict_audit():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)

    activation = scheduler.install_response(
        initial,
        _bsp_response(duration_ticks=9.1234564),
        now_ns=0,
        control_step=0,
    )

    remaining_microindices = activation.activation_context["remaining_curve_microindices"]
    assert (
        activation.activation_context["remaining_curve_ns"]
        == (remaining_microindices * 1_000_000_000 + 20_000_000 - 1) // 20_000_000
    )


def test_bsp_async_uses_control_steps_to_skip_elapsed_prefix_and_immediately_prefetches_short_tail():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    initial = scheduler.maybe_request(
        0,
        control_step=0,
        at_due=True,
        request_in_flight=False,
    )
    first_activation = scheduler.install_response(initial, _bsp_response(), now_ns=1, control_step=0)
    assert first_activation.activation == "initial"
    assert not scheduler.take_action(2, control_step=0).underflow

    prefetch = scheduler.maybe_request(
        10_000_000_000,
        control_step=1,
        at_due=True,
        request_in_flight=False,
    )
    assert prefetch.trigger == "bsp_prefetch"
    assert dict(prefetch.scheduler_context) == {
        "remaining_plan_ns": 400_000_000,
        "budget_ns": 400_000_000,
        "request_control_step": 1,
    }

    activation = scheduler.install_response(
        prefetch,
        _bsp_response(value=7),
        now_ns=20_000_000_000,
        control_step=7,
    )
    assert activation.activation == "immediate_swap"
    assert dict(activation.activation_context) == {
        "request_control_step": 1,
        "activation_control_step": 7,
        "executed_prefix_steps": 6,
        "phase_offset_microindices": 6_000_000,
        "first_sample_microindices": 6_000_000,
        "remaining_curve_microindices": 3_000_000,
        "remaining_curve_ns": 150_000_000,
        "immediate_prefetch": 1,
    }
    np.testing.assert_array_equal(
        scheduler.take_action(30_000_000_000, control_step=7).action,
        np.full(7, 7, dtype=np.float32),
    )
    immediate = scheduler.maybe_request(
        40_000_000_000,
        control_step=7,
        at_due=False,
        request_in_flight=False,
    )
    assert immediate.trigger == "bsp_prefetch"
    assert immediate.scheduler_context["remaining_plan_ns"] == 150_000_000


def test_bsp_async_discards_expired_response_and_requires_latest_observation_blocking_replan():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(value=3), now_ns=0, control_step=0)
    scheduler.take_action(0, control_step=0)
    prefetch = scheduler.maybe_request(1, control_step=1, at_due=True, request_in_flight=False)

    discarded = scheduler.install_response(
        prefetch,
        _bsp_response(value=9),
        now_ns=2,
        control_step=11,
    )

    assert discarded.activation == "discarded_stale_phase"
    assert discarded.activation_context["executed_prefix_steps"] == 10
    assert discarded.activation_context["phase_offset_microindices"] == 10_000_000
    replan = scheduler.maybe_request(3, control_step=11, at_due=False, request_in_flight=False)
    assert replan.dispatch == "blocking_replan"
    assert replan.trigger == "bsp_stale_replan"
    assert replan.scheduler_context["discarded_request_control_step"] == 1
    assert replan.scheduler_context["discarded_activation_control_step"] == 11

    replacement = scheduler.install_response(
        replan,
        _bsp_response(value=5),
        now_ns=4,
        control_step=11,
    )
    assert replacement.activation == "blocking_replace"
    assert replacement.activation_context["executed_prefix_steps"] == 0
    np.testing.assert_array_equal(
        scheduler.take_action(100_000_000_000, control_step=11).action,
        np.full(7, 5, dtype=np.float32),
    )


def test_formal_bsp_scheduler_samples_continuous_curve_and_ignores_legacy_eight_action_values():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
    intent = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    response = _bsp_response(value=6)
    response["actions"][:] = -12345

    scheduler.install_response(intent, response, now_ns=0, control_step=0)
    action = scheduler.take_action(1, control_step=0).action

    np.testing.assert_array_equal(action, np.full(7, 6, dtype=np.float32))


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
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["bsp_spline_async"],
        _schema5_calibration("bsp_spline_async"),
    )
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
    assert calibration.bootstrap_request_fingerprint == control.canonical_fingerprint(worker.requests[0])
    assert len(calibration.warmup_observed_effective_latency_ns) == 5
    assert len(calibration.measurement_observed_effective_latency_ns) == 20
    assert calibration.derived_delay_ticks == 8
    assert calibration.scheduling_latency_budget_ns == 400_000_000
    assert all(key.namespace.startswith("calibration/") for key in worker.latency_sample_keys)


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
    assert len(calibration.warmup_observed_effective_latency_ns) == 5
    assert len(calibration.measurement_observed_effective_latency_ns) == 20


def test_bsp_speedup1_calibration_uses_the_execution_envelope_for_all_twenty_five_probes():
    request = {"state": np.arange(4, dtype=np.float32)}
    mode = control.EXECUTION_MODES["bsp_spline_async_speedup1"]
    worker = _CalibrationWorker(mode.name, [80_000_000] * 25)

    calibration = control.calibrate_async_mode(
        mode,
        request,
        _observation_identity(request),
        worker,
        _checkpoint_identity(bsp=True),
        control.canonical_fingerprint(_bsp_metadata()),
    )

    assert len(worker.requests) == 25
    assert all(item[inference.BSP_EXECUTION_KEY] == {"schema_version": 1, "speedup": 1} for item in worker.requests)
    assert calibration.execution_mode == mode.name
    assert calibration.derived_prefetch_budget_ns == 400_000_000


def test_calibration_records_requested_and_observed_latency_breakdown():
    request = {"state": np.arange(4, dtype=np.float32)}
    worker = _CalibrationWorker("bsp_spline_async", [80_000_000] * 25)

    calibration = control.calibrate_async_mode(
        control.EXECUTION_MODES["bsp_spline_async"],
        request,
        _observation_identity(request),
        worker,
        _checkpoint_identity(bsp=True),
        control.canonical_fingerprint(_bsp_metadata()),
    )

    assert set(calibration.measurement_raw_inference_latency_ns) == {80_000_000}
    assert all(value > 0 for value in calibration.measurement_sampled_target_latency_ns)
    assert all(
        requested == max(raw, sampled) - raw
        and observed_synthetic == requested
        and observed_effective == max(raw, sampled)
        and overshoot == 0
        for raw, sampled, requested, observed_synthetic, observed_effective, overshoot in zip(
            calibration.measurement_raw_inference_latency_ns,
            calibration.measurement_sampled_target_latency_ns,
            calibration.measurement_requested_synthetic_delay_ns,
            calibration.measurement_observed_synthetic_delay_ns,
            calibration.measurement_observed_effective_latency_ns,
            calibration.measurement_latency_overshoot_ns,
        )
    )


def test_calibration_accepts_real_wait_overshoot_and_records_it_separately_from_requested_delay():
    class OvershootingCalibrationWorker(_CalibrationWorker):
        def wait(self, job, timeout=None):
            outcome = super().wait(job, timeout=timeout)
            overshoot_ns = 500_000
            requested_delay_ns = outcome.requested_synthetic_delay_ns
            observed_delay_ns = requested_delay_ns + overshoot_ns
            observed_effective_ns = outcome.observed_effective_latency_ns + overshoot_ns
            return dataclasses.replace(
                outcome,
                completed_monotonic_ns=outcome.completed_monotonic_ns + overshoot_ns,
                requested_synthetic_delay_ns=requested_delay_ns,
                observed_synthetic_delay_ns=observed_delay_ns,
                observed_effective_latency_ns=observed_effective_ns,
                latency_overshoot_ns=overshoot_ns,
            )

    request = {"state": np.arange(4, dtype=np.float32)}
    worker = OvershootingCalibrationWorker("bsp_spline_async", [80_000_000] * 25)

    calibration = control.calibrate_async_mode(
        control.EXECUTION_MODES["bsp_spline_async"],
        request,
        _observation_identity(request),
        worker,
        _checkpoint_identity(bsp=True),
        control.canonical_fingerprint(_bsp_metadata()),
    )

    assert set(calibration.measurement_requested_synthetic_delay_ns) == {
        sampled - 80_000_000 for sampled in calibration.measurement_sampled_target_latency_ns
    }
    assert set(calibration.measurement_latency_overshoot_ns) == {500_000}
    assert all(
        raw + observed_delay == observed_effective and observed_effective == max(raw, sampled) + overshoot
        for raw, sampled, observed_delay, observed_effective, overshoot in zip(
            calibration.measurement_raw_inference_latency_ns,
            calibration.measurement_sampled_target_latency_ns,
            calibration.measurement_observed_synthetic_delay_ns,
            calibration.measurement_observed_effective_latency_ns,
            calibration.measurement_latency_overshoot_ns,
        )
    )


def test_calibration_rejects_an_inconsistent_effective_latency_breakdown():
    class InconsistentLatencyWorker(_CalibrationWorker):
        def wait(self, job, timeout=None):
            outcome = super().wait(job, timeout=timeout)
            return dataclasses.replace(
                outcome,
                raw_inference_latency_ns=80_000_000,
                observed_synthetic_delay_ns=220_000_000,
                observed_effective_latency_ns=299_000_000,
            )

    request = {"state": np.arange(4, dtype=np.float32)}
    worker = InconsistentLatencyWorker("bsp_spline_async", [300_000_000] * 25)

    with pytest.raises(control.CalibrationPolicyError, match="effective latency"):
        control.calibrate_async_mode(
            control.EXECUTION_MODES["bsp_spline_async"],
            request,
            _observation_identity(request),
            worker,
            _checkpoint_identity(bsp=True),
            control.canonical_fingerprint(_bsp_metadata()),
        )


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
        b'{"array":{"__ndarray__":{"dtype":"<i2",'
        b'"sha256":"ea99f710d9d0b8ba192295c969a63ed7ce8fc5743da20d2057fa2b6d2c404bfb",'
        b'"shape":[2,2]}},"bytes":{"__bytes__":"00ff"},'
        b'"float":{"__float_hex__":"0x1.8000000000000p+0"}}'
    )

    assert control.canonical_json_bytes(value) == expected
    assert control.canonical_fingerprint(value) == ("0f8914286b06f6b96f42bf65fc4165f96fec7cca4a6b250e19d0279c5c4145f9")


@pytest.mark.parametrize(
    "reserved_mapping",
    [
        {"__float_hex__": "0x1.8000000000000p+0"},
        {"__float_hex__": "not-a-float-tag"},
        {"__bytes__": "00ff"},
        {"__bytes__": 2},
        {
            "__ndarray__": {
                "dtype": "<i2",
                "shape": [2, 2],
                "sha256": ("ea99f710d9d0b8ba192295c969a63ed7ce8fc5743da20d2057fa2b6d2c404bfb"),
            }
        },
        {"__ndarray__": "not-an-array-tag"},
    ],
)
def test_canonical_encoder_fail_closes_every_single_reserved_tag_envelope(
    reserved_mapping,
):
    with pytest.raises(ValueError, match="reserved canonical type tag"):
        control.canonical_json_bytes(reserved_mapping)


@pytest.mark.parametrize(
    ("ordinary_mapping", "expected"),
    [
        (
            {"__float_hex__": "literal", "kind": "metadata"},
            b'{"__float_hex__":"literal","kind":"metadata"}',
        ),
        (
            {"__bytes__": "00ff", "length": 2},
            b'{"__bytes__":"00ff","length":2}',
        ),
        (
            {"__ndarray__": "literal", "source": "metadata"},
            b'{"__ndarray__":"literal","source":"metadata"}',
        ),
    ],
)
def test_canonical_encoder_keeps_non_tag_shaped_reserved_key_mappings_ordinary(
    ordinary_mapping,
    expected,
):
    assert control.canonical_json_bytes(ordinary_mapping) == expected


def test_seed_namespace_phase_index_and_checkpoint_identity_are_fingerprint_inputs():
    checkpoint = _checkpoint_identity()

    assert control.calibration_seed("namespace", "warmup", 0) == int("d099f188", 16)
    assert control.calibration_seed("namespace", "warmup", 0) != control.calibration_seed("namespace", "measurement", 0)
    assert control.calibration_seed("namespace", "warmup", 0) != control.calibration_seed("namespace", "warmup", 1)
    assert (
        checkpoint.fingerprint
        != dataclasses.replace(
            checkpoint,
            checkpoint="/different/1000",
        ).fingerprint
    )
    assert (
        checkpoint.fingerprint
        != dataclasses.replace(
            checkpoint,
            norm_hash="3" * 64,
        ).fingerprint
    )


def test_calibration_seed_uses_the_shared_canonical_json_encoder(monkeypatch):
    encoded = b"canonical-seed-vector"
    calls = []

    def canonical_json_bytes(value):
        calls.append(value)
        return encoded

    monkeypatch.setattr(control, "canonical_json_bytes", canonical_json_bytes)

    assert control.calibration_seed("namespace", "measurement", 7) == int.from_bytes(
        hashlib.sha256(encoded).digest()[:4],
        "big",
        signed=False,
    )
    assert calls == [{"namespace": "namespace", "phase": "measurement", "index": 7}]


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


def _schema5_calibration(mode_name, *, empirical_observed_effective_ns=300_000_000):
    checkpoint = _checkpoint_identity(bsp=mode_name.startswith("bsp_spline_async"))
    return control.LatencyCalibrationV2.create(
        execution_mode=mode_name,
        checkpoint_identity_fingerprint=checkpoint.fingerprint,
        server_metadata_fingerprint=_OTHER_SHA,
        canonical_observation_identity=_observation_identity({"state": np.zeros(8, dtype=np.float32)}),
        seed_namespace="openpi-libero-calibration-v2/{}/{}".format(
            mode_name,
            checkpoint.fingerprint,
        ),
        bootstrap_request_fingerprint=_SHA if mode_name == "baseline_rtc" else None,
        warmup_request_fingerprints=[_SHA] * 5,
        measurement_request_fingerprints=[_OTHER_SHA] * 20,
        warmup_raw_inference_latency_ns=[80_000_000] * 5,
        warmup_sampled_target_latency_ns=[300_000_000] * 5,
        warmup_requested_synthetic_delay_ns=[220_000_000] * 5,
        warmup_observed_synthetic_delay_ns=[220_000_000] * 5,
        warmup_observed_effective_latency_ns=[300_000_000] * 5,
        warmup_latency_overshoot_ns=[0] * 5,
        measurement_raw_inference_latency_ns=[80_000_000] * 20,
        measurement_sampled_target_latency_ns=[empirical_observed_effective_ns] * 20,
        measurement_requested_synthetic_delay_ns=[empirical_observed_effective_ns - 80_000_000] * 20,
        measurement_observed_synthetic_delay_ns=[empirical_observed_effective_ns - 80_000_000] * 20,
        measurement_observed_effective_latency_ns=[empirical_observed_effective_ns] * 20,
        measurement_latency_overshoot_ns=[0] * 20,
    )


def _native_bsp_calibration(raw_latency_ns=80_000_000, *, mode_name="bsp_spline_async_native"):
    checkpoint = _checkpoint_identity(bsp=True)
    return control.LatencyCalibrationV2.create(
        execution_mode=mode_name,
        checkpoint_identity_fingerprint=checkpoint.fingerprint,
        server_metadata_fingerprint=_OTHER_SHA,
        canonical_observation_identity=_observation_identity({"state": np.zeros(8, dtype=np.float32)}),
        seed_namespace="openpi-libero-calibration-v2/{}/{}".format(mode_name, checkpoint.fingerprint),
        bootstrap_request_fingerprint=None,
        warmup_request_fingerprints=[_SHA] * 5,
        measurement_request_fingerprints=[_OTHER_SHA] * 20,
        warmup_raw_inference_latency_ns=[raw_latency_ns] * 5,
        warmup_sampled_target_latency_ns=[0] * 5,
        warmup_requested_synthetic_delay_ns=[0] * 5,
        warmup_observed_synthetic_delay_ns=[0] * 5,
        warmup_observed_effective_latency_ns=[raw_latency_ns] * 5,
        warmup_latency_overshoot_ns=[0] * 5,
        measurement_raw_inference_latency_ns=[raw_latency_ns] * 20,
        measurement_sampled_target_latency_ns=[0] * 20,
        measurement_requested_synthetic_delay_ns=[0] * 20,
        measurement_observed_synthetic_delay_ns=[0] * 20,
        measurement_observed_effective_latency_ns=[raw_latency_ns] * 20,
        measurement_latency_overshoot_ns=[0] * 20,
    )


def test_schema5_adds_baseline_async_blocking_recovery_without_replacing_v1():
    assert "baseline_async" in control.EXECUTION_MODES
    assert "baseline_async_recovery" in control.EXECUTION_MODES
    assert control.EXECUTION_MODES["baseline_async"].policy_protocol == "baseline_async_h16_v1"
    assert (
        control.EXECUTION_MODES["baseline_async_recovery"].policy_protocol == "baseline_async_h16_blocking_recovery_v2"
    )


def test_schema5_adds_bsp_speedup1_without_replacing_existing_modes():
    assert set(control.EXECUTION_MODES) == {
        "baseline_async",
        "baseline_async_recovery",
        "baseline_rtc",
        "bsp_spline_async",
        "bsp_spline_async_speedup1",
        "bsp_spline_async_native",
        "bsp_spline_async_native_speedup4",
        "bsp_spline_async_native_speedup8",
        "baseline_sync",
        "bsp_spline_sync",
    }
    assert (
        control.EXECUTION_MODES["bsp_spline_async_speedup1"].policy_protocol
        == "bsp_spline_async_phase_skip_speedup1_v1"
    )
    assert control.EXECUTION_MODES["baseline_sync"].policy_protocol == "baseline_sync_n5_h16_full_v2"
    assert control.EXECUTION_MODES["bsp_spline_sync"].policy_protocol == "bsp_spline_sync_speedup2_phase0_v2"


def test_native_bsp_mode_preserves_speedup2_and_declares_dynamic_raw_latency_prefetch():
    mode = control.EXECUTION_MODES["bsp_spline_async_native"]
    assert mode.policy_protocol == "bsp_spline_async_phase_skip_speedup2_v2"
    assert mode.to_parameters_dict()["speedup"] == 2
    assert mode.to_parameters_dict()["latency_injection"] == "disabled_native"
    assert mode.to_parameters_dict()["prefetch_budget_policy"] == "rolling_raw_p95_20_ceil_control_period"


@pytest.mark.parametrize(
    ("mode_name", "speedup", "protocol", "effective_rate_hz", "phase_offset_microindices"),
    [
        (
            "bsp_spline_async_native_speedup4",
            4,
            "bsp_spline_async_phase_skip_speedup4_v1",
            40,
            2_000_000,
        ),
        (
            "bsp_spline_async_native_speedup8",
            8,
            "bsp_spline_async_phase_skip_speedup8_v1",
            80,
            4_000_000,
        ),
    ],
)
def test_native_high_speedup_modes_bind_protocol_request_overlay_and_phase_progression(
    mode_name, speedup, protocol, effective_rate_hz, phase_offset_microindices
):
    mode = control.EXECUTION_MODES[mode_name]
    parameters = mode.to_parameters_dict()
    assert mode.policy_protocol == protocol
    assert parameters["speedup"] == speedup
    assert parameters["effective_curve_rate_hz"] == effective_rate_hz
    assert parameters["latency_injection"] == "disabled_native"

    scheduler = control.make_scheduler_v5(
        mode,
        _native_bsp_calibration(mode_name=mode_name),
    )
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    assert initial.request_overlay == {inference.BSP_EXECUTION_KEY: {"schema_version": 1, "speedup": speedup}}
    activation = scheduler.install_response(
        initial,
        _bsp_response(speedup=speedup),
        now_ns=80_000_000,
        control_step=1,
    )
    assert activation.activation_context["phase_offset_microindices"] == phase_offset_microindices


def test_native_bsp_scheduler_updates_prefetch_budget_from_rolling_raw_p95():
    """Replacing the rolling p95 with a fixed 400 ms budget must fail this test."""
    mode = control.EXECUTION_MODES["bsp_spline_async_native"]
    scheduler = control.make_scheduler_v5(mode, _native_bsp_calibration())
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(speedup=2), now_ns=80_000_000, control_step=0)

    assert scheduler.maybe_request(0, control_step=6, at_due=True, request_in_flight=False) is None
    first_prefetch = scheduler.maybe_request(0, control_step=7, at_due=True, request_in_flight=False)
    assert first_prefetch.scheduler_context["remaining_plan_ns"] == 100_000_000
    assert first_prefetch.scheduler_context["budget_ns"] == 100_000_000

    scheduler = control.make_scheduler_v5(mode, _native_bsp_calibration())
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _bsp_response(speedup=2), now_ns=80_000_000, control_step=0)
    scheduler.observe_raw_inference_latency(250_000_000)
    scheduler.observe_raw_inference_latency(250_000_000)
    adaptive_prefetch = scheduler.maybe_request(0, control_step=4, at_due=True, request_in_flight=False)
    assert adaptive_prefetch.scheduler_context["remaining_plan_ns"] == 250_000_000
    assert adaptive_prefetch.scheduler_context["budget_ns"] == 250_000_000


def test_bsp_speedup1_scheduler_requests_server_identity_and_uses_real_remaining_wall_clock():
    mode = control.EXECUTION_MODES["bsp_spline_async_speedup1"]
    scheduler = control.make_scheduler_v5(mode, _schema5_calibration(mode.name))

    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    assert initial.request_overlay == {inference.BSP_EXECUTION_KEY: {"schema_version": 1, "speedup": 1}}
    activation = scheduler.install_response(
        initial,
        _bsp_response(speedup=1),
        now_ns=300_000_000,
        control_step=0,
    )
    assert activation.activation_context["phase_offset_microindices"] == 0
    assert activation.activation_context["remaining_curve_ns"] == 900_000_000

    assert scheduler.maybe_request(0, control_step=9, at_due=True, request_in_flight=False) is None
    prefetch = scheduler.maybe_request(0, control_step=10, at_due=True, request_in_flight=False)
    assert prefetch.trigger == "bsp_prefetch"
    assert prefetch.scheduler_context["remaining_plan_ns"] == 400_000_000
    assert prefetch.request_overlay == initial.request_overlay


def test_baseline_sync_executes_all_sixteen_actions_before_blocking_replan():
    scheduler = control.make_scheduler_v5(control.EXECUTION_MODES["baseline_sync"], None)
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    assert initial.dispatch == "blocking_initial"
    assert initial.trigger == "initial_plan"
    assert initial.request_overlay == {inference.RTC_REQUEST_KEY: {"schema_version": inference.RTC_SCHEMA_VERSION}}
    response = _rtc_response(offset=100)
    scheduler.install_response(initial, response, now_ns=1, control_step=0)

    for step in range(16):
        assert (
            scheduler.maybe_request(
                step + 2,
                control_step=step,
                at_due=True,
                request_in_flight=False,
            )
            is None
        )
        decision = scheduler.take_action(step + 2, control_step=step)
        assert not decision.underflow
        np.testing.assert_array_equal(decision.action, response["actions"][step])

    assert scheduler.take_action(20, control_step=16).underflow
    replan = scheduler.maybe_request(21, control_step=16, at_due=True, request_in_flight=False)
    assert replan.dispatch == "blocking_replan"
    assert replan.trigger == "baseline_chunk_exhausted"
    assert replan.request_overlay == initial.request_overlay


def test_bsp_sync_executes_the_closed_continuous_curve_then_replans_from_phase_zero():
    scheduler = control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_sync"], None)
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)
    first = _bsp_response(duration_ticks=9, value=3)
    activation = scheduler.install_response(initial, first, now_ns=300_000_000, control_step=0)
    assert activation.activation == "initial"
    assert activation.activation_context["executed_prefix_steps"] == 0
    assert activation.activation_context["phase_offset_microindices"] == 0
    assert activation.activation_context["first_sample_microindices"] == 0

    for step in range(10):
        decision = scheduler.take_action(300_000_001 + step, control_step=step)
        assert not decision.underflow
        np.testing.assert_array_equal(decision.action, np.full(7, 3, dtype=np.float32))
    assert scheduler.take_action(400_000_000, control_step=10).underflow

    replan = scheduler.maybe_request(400_000_001, control_step=10, at_due=True, request_in_flight=False)
    assert replan.dispatch == "blocking_replan"
    assert replan.trigger == "bsp_curve_exhausted"
    second = _bsp_response(duration_ticks=9, value=7)
    replacement = scheduler.install_response(replan, second, now_ns=900_000_000, control_step=10)
    assert replacement.activation == "blocking_replace"
    assert replacement.activation_context["request_control_step"] == 10
    assert replacement.activation_context["activation_control_step"] == 10
    assert replacement.activation_context["executed_prefix_steps"] == 0
    assert replacement.activation_context["phase_offset_microindices"] == 0
    assert replacement.activation_context["first_sample_microindices"] == 0
    np.testing.assert_array_equal(
        scheduler.take_action(900_000_001, control_step=10).action,
        np.full(7, 7, dtype=np.float32),
    )


def test_bsp_sync_accepts_a_short_curve_without_async_immediate_prefetch():
    scheduler = control.make_scheduler_v5(control.EXECUTION_MODES["bsp_spline_sync"], None)
    initial = scheduler.maybe_request(0, control_step=0, at_due=True, request_in_flight=False)

    activation = scheduler.install_response(
        initial,
        _bsp_response(duration_ticks=4, value=5),
        now_ns=300_000_000,
        control_step=0,
    )

    assert activation.activation_context["remaining_curve_ns"] == 200_000_000
    assert activation.activation_context["immediate_prefetch"] == 0
    assert (
        scheduler.maybe_request(
            300_000_001,
            control_step=0,
            at_due=False,
            request_in_flight=False,
        )
        is None
    )


def test_theoretical_budget_is_fixed_at_eight_ticks_when_empirical_p95_is_longer():
    calibration = _schema5_calibration(
        "baseline_rtc",
        empirical_observed_effective_ns=450_000_000,
    )

    assert calibration.theoretical_p95_latency_ns == 398_691_218
    assert calibration.scheduling_latency_budget_ns == 400_000_000
    assert calibration.derived_delay_ticks == 8
    assert calibration.empirical_observed_effective_p95_ns == 450_000_000
    assert calibration.empirical_p95_exceeds_budget


def test_baseline_async_and_rtc_launch_at_same_cursor_but_only_rtc_has_guidance():
    raw = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_async"],
        _schema5_calibration("baseline_async"),
    )
    guided = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_rtc"],
        _schema5_calibration("baseline_rtc"),
    )
    raw_initial = raw.maybe_request(0, at_due=True, request_in_flight=False)
    guided_initial = guided.maybe_request(0, at_due=True, request_in_flight=False)
    raw.install_response(raw_initial, _rtc_response(), now_ns=0, control_step=0)
    guided.install_response(guided_initial, _rtc_response(), now_ns=0, control_step=0)

    for step in range(8):
        assert not raw.take_action(step).underflow
        assert not guided.take_action(step).underflow

    raw_request = raw.maybe_request(8, at_due=True, request_in_flight=False)
    guided_request = guided.maybe_request(8, at_due=True, request_in_flight=False)
    assert raw_request.trigger == "baseline_async_launch"
    assert guided_request.trigger == "rtc_launch"
    assert (
        dict(raw_request.scheduler_context)
        == dict(guided_request.scheduler_context)
        == {
            "s": 8,
            "d": 8,
        }
    )
    assert raw_request.request_overlay == {inference.RTC_REQUEST_KEY: {"schema_version": inference.RTC_SCHEMA_VERSION}}
    assert set(guided_request.request_overlay[inference.RTC_REQUEST_KEY]) == {
        "schema_version",
        "previous_model_actions",
        "s",
        "d",
    }


def test_baseline_async_install_skips_elapsed_prefix_and_underflow_does_not_advance():
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_async"],
        _schema5_calibration("baseline_async"),
    )
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _rtc_response(), now_ns=0, control_step=0)
    for step in range(8):
        scheduler.take_action(step)
    background = scheduler.maybe_request(8, at_due=True, request_in_flight=False)
    for step in range(8, 16):
        scheduler.take_action(step)
    assert scheduler.take_action(16).underflow
    assert scheduler.take_action(17).underflow

    response = _rtc_response(offset=1_000.0)
    activation = scheduler.install_response(background, response, now_ns=18, control_step=16)
    assert dict(activation.activation_context) == {"action_cursor": 8}
    decision = scheduler.take_action(19)
    assert not decision.underflow
    np.testing.assert_array_equal(decision.action, response["actions"][8])


def test_baseline_async_recovery_blocks_at_capacity_miss_and_restarts_from_zero():
    """Replacing the recovery with the old INFEASIBLE raise must fail this test."""
    scheduler = control.make_scheduler_v5(
        control.EXECUTION_MODES["baseline_async_recovery"],
        _schema5_calibration("baseline_async_recovery"),
    )
    initial = scheduler.maybe_request(0, at_due=True, request_in_flight=False)
    scheduler.install_response(initial, _rtc_response(), now_ns=0, control_step=0)
    for step in range(9):
        scheduler.take_action(step)

    recovery = scheduler.maybe_request(9, at_due=True, request_in_flight=False, control_step=9)
    assert recovery.dispatch == "blocking_replan"
    assert recovery.trigger == "baseline_async_capacity_replan"
    assert dict(recovery.scheduler_context) == {"action_cursor": 9, "forecast_delay_ticks": 8}
    assert recovery.request_overlay == {inference.RTC_REQUEST_KEY: {"schema_version": inference.RTC_SCHEMA_VERSION}}

    response = _rtc_response(offset=3_000.0)
    activation = scheduler.install_response(recovery, response, now_ns=10, control_step=9)
    assert activation.activation == "blocking_replace"
    assert dict(activation.activation_context) == {"action_cursor": 0}
    decision = scheduler.take_action(11, control_step=9)
    assert not decision.underflow
    np.testing.assert_array_equal(decision.action, response["actions"][0])
