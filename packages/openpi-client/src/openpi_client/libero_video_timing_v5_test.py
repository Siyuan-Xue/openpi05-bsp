"""Strict schema-v5 timing and video accounting contracts.

These tests are intentionally dependency-free.  Per the integration plan they
are authored before production code, but are not executed on this checkout.
"""

import dataclasses
from copy import deepcopy
import operator

import pytest

from openpi_client import libero_eval
from openpi_client import libero_video_timing_v5 as timing
from openpi_client import latency_sampling


_EVAL_SEED = 42
_IDENTITY = libero_eval.EpisodeIdentity(
    suite="libero_spatial",
    task_id=0,
    task_name="timing contract",
    init_state_index=0,
    init_state_fingerprint="a" * 64,
)


def _flow_seed(request_id):
    return libero_eval.stable_replan_seed(_EVAL_SEED, _IDENTITY, request_id)


def _sample_key(request_id):
    return latency_sampling.LatencySampleKeyV1(
        namespace="formal",
        seed=42,
        suite=_IDENTITY.suite,
        task_id=_IDENTITY.task_id,
        trial_index=_IDENTITY.init_state_index,
        request_ordinal=request_id,
    )


def _assert_raises(exception_type, callback):
    try:
        callback()
    except exception_type:
        pass
    else:
        raise AssertionError("expected {}".format(exception_type.__name__))


def _request(
    request_id,
    submitted_offset_ns,
    *,
    observation_control_step=0,
    flow_seed=None,
    dispatch="background",
    trigger="rtc_launch",
    scheduler_context=None,
    disposition="activated",
):
    if flow_seed is None:
        flow_seed = _flow_seed(request_id)
    if scheduler_context is None:
        scheduler_context = {"s": 8, "d": 8}
    return timing.RequestEventV5(
        request_id=request_id,
        observation_control_step=observation_control_step,
        submitted_offset_ns=submitted_offset_ns,
        flow_seed=flow_seed,
        dispatch=dispatch,
        trigger=trigger,
        scheduler_context=scheduler_context,
        disposition=disposition,
        latency_sample_key=_sample_key(request_id),
        sampled_target_latency_ns=0,
    )


def _initial_request(*, disposition="activated"):
    return _request(
        0,
        0,
        flow_seed=_flow_seed(0),
        dispatch="blocking_initial",
        trigger="initial_plan",
        scheduler_context={},
        disposition=disposition,
    )


def _latency(request_id, completed_offset_ns, duration_ns, *, outcome="success"):
    return timing.LatencyEventV5(
        request_id=request_id,
        completed_offset_ns=completed_offset_ns,
        duration_ns=duration_ns,
        outcome=outcome,
        sampled_target_latency_ns=0,
    )


def test_request_and_latency_records_expose_paired_sample_identity_and_target():
    request = dataclasses.replace(_initial_request(), sampled_target_latency_ns=300_000_000)
    latency = timing.LatencyEventV5(
        request_id=0,
        completed_offset_ns=300_000_000,
        duration_ns=300_000_000,
        outcome="success",
        raw_inference_latency_ns=80_000_000,
        requested_synthetic_delay_ns=220_000_000,
        observed_synthetic_delay_ns=220_000_000,
        observed_effective_latency_ns=300_000_000,
        latency_overshoot_ns=0,
        sampled_target_latency_ns=300_000_000,
    )

    assert request.to_dict()["latency_sample_key"] == _sample_key(0).to_dict()
    assert request.to_dict()["sampled_target_latency_ns"] == 300_000_000
    assert latency.to_dict()["sampled_target_latency_ns"] == 300_000_000


def test_action_seam_records_exact_arm_and_gripper_jumps():
    seam = timing.ActionSeamV5.from_actions(
        plan_id=1,
        request_id=1,
        control_step=8,
        previous_action=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0),
        activated_action=(1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0),
    )

    assert seam.arm_l2_jump == pytest.approx(1.0)
    assert seam.arm_max_abs_jump == pytest.approx(1.0)
    assert seam.gripper_abs_jump == pytest.approx(2.0)
    assert timing.ActionSeamV5.from_dict(seam.to_dict()) == seam


def test_latency_event_records_requested_and_observed_durations_separately():
    event = timing.LatencyEventV5(
        request_id=3,
        completed_offset_ns=400_000_000,
        duration_ns=300_000_000,
        outcome="success",
        raw_inference_latency_ns=80_000_000,
        requested_synthetic_delay_ns=220_000_000,
        observed_synthetic_delay_ns=220_000_000,
        observed_effective_latency_ns=300_000_000,
        latency_overshoot_ns=0,
    )

    payload = event.to_dict()
    assert payload["raw_inference_latency_ns"] == 80_000_000
    assert payload["requested_synthetic_delay_ns"] == 220_000_000
    assert payload["observed_synthetic_delay_ns"] == 220_000_000
    assert payload["observed_effective_latency_ns"] == 300_000_000
    assert payload["latency_overshoot_ns"] == 0
    assert timing.LatencyEventV5.from_dict(payload) == event


def test_latency_event_records_requested_and_observed_wait_when_real_clock_overshoots():
    event = timing.LatencyEventV5(
        request_id=3,
        completed_offset_ns=400_500_000,
        duration_ns=300_500_000,
        outcome="success",
        raw_inference_latency_ns=80_000_000,
        requested_synthetic_delay_ns=220_000_000,
        observed_synthetic_delay_ns=220_500_000,
        observed_effective_latency_ns=300_500_000,
        latency_overshoot_ns=500_000,
        sampled_target_latency_ns=300_000_000,
    )

    payload = event.to_dict()
    assert payload["requested_synthetic_delay_ns"] == 220_000_000
    assert payload["observed_synthetic_delay_ns"] == 220_500_000
    assert payload["observed_effective_latency_ns"] == 300_500_000
    assert payload["latency_overshoot_ns"] == 500_000
    assert timing.LatencyEventV5.from_dict(payload) == event


def test_latency_event_rejects_breakdowns_that_do_not_sum_to_observed_effective_latency():
    with pytest.raises(ValueError, match="raw.*synthetic.*effective"):
        timing.LatencyEventV5(
            request_id=3,
            completed_offset_ns=400_000_000,
            duration_ns=300_000_000,
            outcome="success",
            raw_inference_latency_ns=80_000_000,
            requested_synthetic_delay_ns=220_000_000,
            observed_synthetic_delay_ns=10_000_000,
            observed_effective_latency_ns=300_000_000,
            latency_overshoot_ns=0,
        )


def _native_activation(plan_id, request_id, control_step, activated_offset_ns, *, activation):
    return timing.PlanActivationV5(
        plan_id=plan_id,
        request_id=request_id,
        control_step=control_step,
        activated_offset_ns=activated_offset_ns,
        activation=activation,
        activation_context={"action_cursor": 0},
    )


def _stall(request_id, control_step, started_offset_ns, duration_ns, *, reason):
    return timing.ControlStallV5(
        request_id=request_id,
        control_step=control_step,
        started_offset_ns=started_offset_ns,
        duration_ns=duration_ns,
        reason=reason,
    )


def _mixed_async_timeline():
    requests = (
        _initial_request(),
        _request(1, 350_000_000, observation_control_step=8),
        _request(2, 800_000_000, observation_control_step=16),
    )
    latencies = (
        _latency(0, 125_000_000, 125_000_000),
        _latency(1, 380_000_000, 30_000_000),
        _latency(2, 935_000_000, 135_000_000),
    )
    activations = (
        _native_activation(0, 0, 0, 125_000_000, activation="initial"),
        _native_activation(1, 1, 8, 380_000_000, activation="immediate_swap"),
        _native_activation(2, 2, 16, 935_000_000, activation="immediate_swap"),
    )
    underflows = (
        timing.ActionUnderflowV5(
            request_id=2,
            control_step=16,
            started_offset_ns=900_000_000,
            duration_ns=35_000_000,
        ),
    )
    stalls = (
        _stall(0, 0, 0, 125_000_000, reason="synchronous_inference"),
        _stall(2, 16, 900_000_000, 35_000_000, reason="async_action_underflow"),
    )
    return requests, latencies, activations, underflows, stalls


def test_event_records_round_trip_exact_fields_and_defensively_copy_contexts():
    scheduler_context = {"s": 8, "d": 8}
    request = _request(3, 40, scheduler_context=scheduler_context)
    scheduler_context["s"] = 9
    assert request.to_dict() == {
        "clock": "episode_monotonic_ns",
        "request_id": 3,
        "observation_control_step": 0,
        "submitted_offset_ns": 40,
        "flow_seed": _flow_seed(3),
        "dispatch": "background",
        "trigger": "rtc_launch",
        "scheduler_context": {"s": 8, "d": 8},
        "disposition": "activated",
        "latency_sample_key": _sample_key(3).to_dict(),
        "sampled_target_latency_ns": 0,
    }
    _assert_raises(TypeError, lambda: operator.setitem(request.scheduler_context, "s", 9))
    _assert_raises(dataclasses.FrozenInstanceError, lambda: setattr(request, "request_id", 4))

    records = (
        request,
        _latency(3, 90, 50),
        _native_activation(1, 3, 4, 91, activation="immediate_swap"),
        timing.ActionUnderflowV5(3, 4, 75, 16),
        _stall(3, 4, 75, 16, reason="async_action_underflow"),
    )
    for record in records:
        assert type(record).from_dict(record.to_dict()) == record


def test_every_event_rejects_missing_extra_bool_nonfinite_and_wrong_json_containers():
    records = (
        _initial_request(),
        _latency(0, 125, 125),
        _native_activation(0, 0, 0, 125, activation="initial"),
        timing.ActionUnderflowV5(0, 0, 100, 25),
        _stall(0, 0, 100, 25, reason="async_action_underflow"),
    )
    integer_fields = (
        "request_id",
        "request_id",
        "plan_id",
        "request_id",
        "request_id",
    )
    for record, integer_field in zip(records, integer_fields):
        payload = record.to_dict()
        missing = deepcopy(payload)
        missing.pop(next(iter(missing)))
        extra = dict(payload, unexpected=1)
        bool_integer = deepcopy(payload)
        bool_integer[integer_field] = True
        nonfinite = deepcopy(payload)
        nonfinite[integer_field] = float("inf")
        for malformed in (missing, extra, bool_integer, nonfinite, []):
            _assert_raises(ValueError, lambda malformed=malformed, record=record: type(record).from_dict(malformed))

    malformed_context = _initial_request().to_dict()
    malformed_context["scheduler_context"] = []
    _assert_raises(ValueError, lambda: timing.RequestEventV5.from_dict(malformed_context))
    malformed_activation_context = records[2].to_dict()
    malformed_activation_context["activation_context"] = []
    _assert_raises(
        ValueError,
        lambda: timing.PlanActivationV5.from_dict(malformed_activation_context),
    )
    wrong_clock = records[1].to_dict()
    wrong_clock["clock"] = "wall_clock"
    _assert_raises(ValueError, lambda: timing.LatencyEventV5.from_dict(wrong_clock))


def test_request_context_and_enum_validation_is_exact_and_trigger_aware():
    invalid_payloads = []
    for field, value in (
        ("dispatch", "queued"),
        ("trigger", "periodic"),
        ("disposition", "ignored"),
    ):
        payload = _initial_request().to_dict()
        payload[field] = value
        invalid_payloads.append(payload)

    wrong_initial_context = _initial_request().to_dict()
    wrong_initial_context["scheduler_context"] = {"s": 8, "d": 8}
    invalid_payloads.append(wrong_initial_context)
    wrong_rtc_context = _request(1, 1).to_dict()
    wrong_rtc_context["scheduler_context"] = {"s": 8, "d": 8, "extra": 0}
    invalid_payloads.append(wrong_rtc_context)
    wrong_bsp_context = _request(
        1,
        1,
        trigger="bsp_prefetch",
        scheduler_context={"remaining_plan_ns": 50, "budget_ns": 50},
    ).to_dict()
    wrong_bsp_context["scheduler_context"]["budget_ns"] = True
    invalid_payloads.append(wrong_bsp_context)

    for payload in invalid_payloads:
        _assert_raises(ValueError, lambda payload=payload: timing.RequestEventV5.from_dict(payload))

    latency = _latency(0, 1, 1).to_dict()
    latency["outcome"] = "transport_error"
    activation = _native_activation(0, 0, 0, 1, activation="initial").to_dict()
    activation["activation"] = "deferred"
    stall = _stall(0, 0, 0, 1, reason="synchronous_inference").to_dict()
    stall["reason"] = "network"
    for record_type, payload in (
        (timing.LatencyEventV5, latency),
        (timing.PlanActivationV5, activation),
        (timing.ControlStallV5, stall),
    ):
        _assert_raises(ValueError, lambda record_type=record_type, payload=payload: record_type.from_dict(payload))


def test_cross_event_validation_accepts_initial_sync_background_overlap_and_later_underflow():
    events = _mixed_async_timeline()
    normalized = timing.validate_timing_events_v5(
        requests=events[0],
        latencies=events[1],
        activations=events[2],
        underflows=events[3],
        stalls=events[4],
        steps=20,
        episode_duration_ns=1_000_000_000,
        execution_mode="baseline_rtc",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
    )

    assert normalized == events
    assert len(events[1]) == 3
    assert len(events[4]) == 2
    assert all(stall.request_id != 1 for stall in events[4])


def test_cross_event_validation_rejects_id_orphan_order_interval_and_seed_mutations():
    requests, latencies, activations, underflows, stalls = _mixed_async_timeline()

    def validate(**changes):
        values = {
            "requests": requests,
            "latencies": latencies,
            "activations": activations,
            "underflows": underflows,
            "stalls": stalls,
            "steps": 20,
            "episode_duration_ns": 1_000_000_000,
            "execution_mode": "baseline_rtc",
            "eval_seed": _EVAL_SEED,
            "identity": _IDENTITY,
        }
        values.update(changes)
        timing.validate_timing_events_v5(**values)

    gapped_requests = (
        requests[0],
        dataclasses.replace(
            requests[1],
            request_id=2,
            latency_sample_key=_sample_key(2),
        ),
        requests[2],
    )
    orphan_latency = latencies + (_latency(9, 950_000_000, 1),)
    orphan_activation = activations + (_native_activation(3, 9, 17, 950_000_000, activation="immediate_swap"),)
    early_activation = (
        activations[0],
        dataclasses.replace(activations[1], activated_offset_ns=379_999_999),
        activations[2],
    )
    wrong_disposition = (
        requests[0],
        dataclasses.replace(requests[1], disposition="failed"),
        requests[2],
    )
    overlapping_requests = (
        requests[0],
        dataclasses.replace(requests[1], submitted_offset_ns=124_999_999),
        requests[2],
    )
    mismatched_stall = (
        stalls[0],
        dataclasses.replace(stalls[1], duration_ns=34_999_999),
    )
    past_end_latency = (
        latencies[0],
        latencies[1],
        dataclasses.replace(
            latencies[2],
            completed_offset_ns=1_000_000_001,
            duration_ns=200_000_001,
            raw_inference_latency_ns=200_000_001,
            requested_synthetic_delay_ns=0,
            observed_synthetic_delay_ns=0,
            observed_effective_latency_ns=200_000_001,
            latency_overshoot_ns=0,
        ),
    )
    wrong_seed_requests = (
        requests[0],
        dataclasses.replace(requests[1], flow_seed=(requests[1].flow_seed + 1) % (2**32)),
        requests[2],
    )

    mutations = (
        {"requests": gapped_requests},
        {"latencies": orphan_latency},
        {"activations": orphan_activation},
        {"activations": early_activation},
        {"requests": wrong_disposition},
        {"requests": overlapping_requests},
        {"stalls": mismatched_stall},
        {"latencies": past_end_latency},
        {"requests": wrong_seed_requests},
    )
    for mutation in mutations:
        _assert_raises(ValueError, lambda mutation=mutation: validate(**mutation))


def test_cross_event_validation_has_no_caller_supplied_seed_bypass():
    requests, latencies, activations, underflows, stalls = _mixed_async_timeline()
    forged_requests = (
        requests[0],
        dataclasses.replace(requests[1], flow_seed=(requests[1].flow_seed + 1) % (2**32)),
        requests[2],
    )
    forged_expected = tuple(request.flow_seed for request in forged_requests)

    _assert_raises(
        TypeError,
        lambda: timing.validate_timing_events_v5(
            requests=forged_requests,
            latencies=latencies,
            activations=activations,
            underflows=underflows,
            stalls=stalls,
            steps=20,
            episode_duration_ns=1_000_000_000,
            execution_mode="baseline_rtc",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
            expected_flow_seeds=forged_expected,
        ),
    )


def test_mode_aware_validation_rejects_invalid_rtc_and_bsp_contexts():
    requests, latencies, activations, underflows, stalls = _mixed_async_timeline()
    rtc_payload = requests[1].to_dict()
    rtc_payload["scheduler_context"] = {"s": 8, "d": 9}
    _assert_raises(ValueError, lambda: timing.RequestEventV5.from_dict(rtc_payload))

    bsp_requests = (
        _initial_request(),
        _request(
            1,
            150,
            observation_control_step=1,
            dispatch="background",
            trigger="bsp_prefetch",
            scheduler_context={"remaining_plan_ns": 50, "budget_ns": 50},
        ),
    )
    bsp_latencies = (_latency(0, 100, 100), _latency(1, 175, 25))
    bsp_activations = (
        timing.PlanActivationV5(0, 0, 0, 100, "initial", {"curve_elapsed_ns": 0}),
        timing.PlanActivationV5(1, 1, 1, 175, "immediate_swap", {"curve_elapsed_ns": 0}),
    )
    bsp_stalls = (_stall(0, 0, 0, 100, reason="synchronous_inference"),)
    assert (
        timing.validate_timing_events_v5(
            requests=bsp_requests,
            latencies=bsp_latencies,
            activations=bsp_activations,
            underflows=(),
            stalls=bsp_stalls,
            steps=2,
            episode_duration_ns=200,
            execution_mode="bsp_spline_async",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
            expected_bsp_prefetch_budget_ns=50,
        )[0]
        == bsp_requests
    )

    wrong_budget = dataclasses.replace(
        bsp_requests[1],
        scheduler_context={"remaining_plan_ns": 50, "budget_ns": 75},
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=(bsp_requests[0], wrong_budget),
            latencies=bsp_latencies,
            activations=bsp_activations,
            underflows=(),
            stalls=bsp_stalls,
            steps=2,
            episode_duration_ns=200,
            execution_mode="bsp_spline_async",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
            expected_bsp_prefetch_budget_ns=50,
        ),
    )

    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=bsp_requests,
            latencies=bsp_latencies,
            activations=bsp_activations,
            underflows=(),
            stalls=bsp_stalls,
            steps=2,
            episode_duration_ns=200,
            execution_mode="bsp_spline_async",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=(_initial_request(),),
            latencies=(_latency(0, 100, 100),),
            activations=(
                timing.PlanActivationV5(
                    0,
                    0,
                    0,
                    100,
                    "initial",
                    {"curve_elapsed_ns": 0},
                ),
            ),
            underflows=(),
            stalls=bsp_stalls,
            steps=0,
            episode_duration_ns=100,
            execution_mode="bsp_spline_async",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )
    timing.validate_timing_events_v5(
        requests=(_initial_request(),),
        latencies=(_latency(0, 100, 100),),
        activations=(
            timing.PlanActivationV5(
                0,
                0,
                0,
                100,
                "initial",
                {"curve_elapsed_ns": 0},
            ),
        ),
        underflows=(),
        stalls=bsp_stalls,
        steps=0,
        episode_duration_ns=100,
        execution_mode="bsp_spline_async",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
        expected_bsp_prefetch_budget_ns=0,
    )
    for invalid_budget in (True, -1):
        _assert_raises(
            ValueError,
            lambda invalid_budget=invalid_budget: timing.validate_timing_events_v5(
                requests=(_initial_request(),),
                latencies=(_latency(0, 100, 100),),
                activations=(
                    timing.PlanActivationV5(
                        0,
                        0,
                        0,
                        100,
                        "initial",
                        {"curve_elapsed_ns": 0},
                    ),
                ),
                underflows=(),
                stalls=bsp_stalls,
                steps=0,
                episode_duration_ns=100,
                execution_mode="bsp_spline_async",
                eval_seed=_EVAL_SEED,
                identity=_IDENTITY,
                expected_bsp_prefetch_budget_ns=invalid_budget,
            ),
        )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=(_initial_request(),),
            latencies=(_latency(0, 100, 100),),
            activations=(_native_activation(0, 0, 0, 100, activation="initial"),),
            underflows=(),
            stalls=bsp_stalls,
            steps=0,
            episode_duration_ns=100,
            execution_mode="baseline_sync_n5",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
            expected_bsp_prefetch_budget_ns=0,
        ),
    )


def test_cross_event_chronology_binds_submissions_activation_steps_and_blocking_stalls():
    requests, latencies, activations, underflows, stalls = _mixed_async_timeline()
    delayed_prior_activation = (
        dataclasses.replace(activations[0], activated_offset_ns=360_000_000),
        activations[1],
        activations[2],
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=requests,
            latencies=latencies,
            activations=delayed_prior_activation,
            underflows=underflows,
            stalls=stalls,
            steps=20,
            episode_duration_ns=1_000_000_000,
            execution_mode="baseline_rtc",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )

    bsp_requests = (
        _initial_request(),
        _request(
            1,
            150,
            observation_control_step=2,
            dispatch="background",
            trigger="bsp_prefetch",
            scheduler_context={"remaining_plan_ns": 50, "budget_ns": 50},
        ),
    )
    bsp_activations = (
        timing.PlanActivationV5(0, 0, 0, 100, "initial", {"curve_elapsed_ns": 0}),
        timing.PlanActivationV5(1, 1, 1, 175, "immediate_swap", {"curve_elapsed_ns": 0}),
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=bsp_requests,
            latencies=(_latency(0, 100, 100), _latency(1, 175, 25)),
            activations=bsp_activations,
            underflows=(),
            stalls=(_stall(0, 0, 0, 100, reason="synchronous_inference"),),
            steps=2,
            episode_duration_ns=200,
            execution_mode="bsp_spline_async",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
            expected_bsp_prefetch_budget_ns=50,
        ),
    )


def test_background_policy_failure_underflow_cannot_precede_observation_step():
    failed_request = dataclasses.replace(
        _request(1, 150, observation_control_step=2),
        disposition="failed",
    )
    underflow = timing.ActionUnderflowV5(1, 1, 175, 25)
    stalls = (
        _stall(0, 0, 0, 100, reason="synchronous_inference"),
        _stall(1, 1, 175, 25, reason="async_action_underflow"),
    )

    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v5(
            requests=(_initial_request(), failed_request),
            latencies=(
                _latency(0, 100, 100),
                _latency(1, 200, 50, outcome="policy_failure"),
            ),
            activations=(_native_activation(0, 0, 0, 100, activation="initial"),),
            underflows=(underflow,),
            stalls=stalls,
            steps=2,
            episode_duration_ns=200,
            execution_mode="baseline_rtc",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )


def test_zero_duration_events_touch_at_endpoints_and_accept_control_step_equal_to_steps():
    requests = (
        _initial_request(),
        _request(1, 0, observation_control_step=1),
    )
    latencies = (_latency(0, 0, 0), _latency(1, 0, 0))
    activations = (
        _native_activation(0, 0, 0, 0, activation="initial"),
        _native_activation(1, 1, 1, 0, activation="immediate_swap"),
    )
    underflows = (timing.ActionUnderflowV5(1, 1, 0, 0),)
    stalls = (
        _stall(0, 0, 0, 0, reason="synchronous_inference"),
        _stall(1, 1, 0, 0, reason="async_action_underflow"),
    )

    assert timing.validate_timing_events_v5(
        requests=requests,
        latencies=latencies,
        activations=activations,
        underflows=underflows,
        stalls=stalls,
        steps=1,
        episode_duration_ns=0,
        execution_mode="baseline_rtc",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
    ) == (requests, latencies, activations, underflows, stalls)


def test_failed_and_abandoned_requests_have_exactly_the_allowed_relations():
    failed_request = dataclasses.replace(_initial_request(), disposition="failed")
    failed_latency = _latency(0, 25, 25, outcome="policy_failure")
    failed_stall = _stall(0, 0, 0, 25, reason="synchronous_inference")
    timing.validate_timing_events_v5(
        requests=(failed_request,),
        latencies=(failed_latency,),
        activations=(),
        underflows=(),
        stalls=(failed_stall,),
        steps=0,
        episode_duration_ns=25,
        execution_mode="baseline_async",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
    )

    abandoned_request = dataclasses.replace(
        _request(1, 150, observation_control_step=1),
        disposition="abandoned",
    )
    timing.validate_timing_events_v5(
        requests=(_initial_request(), abandoned_request),
        latencies=(_latency(0, 100, 100),),
        activations=(_native_activation(0, 0, 0, 100, activation="initial"),),
        underflows=(),
        stalls=(_stall(0, 0, 0, 100, reason="synchronous_inference"),),
        steps=1,
        episode_duration_ns=200,
        execution_mode="baseline_rtc",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
    )


def test_wait_overlay_is_one_persistent_cumulative_line_and_stalls_quantize_cumulatively():
    stalls = (
        _stall(0, 0, 0, 12_500_000, reason="synchronous_inference"),
        _stall(1, 1, 20_000_000, 12_500_000, reason="async_action_underflow"),
        _stall(2, 2, 40_000_000, 12_500_000, reason="synchronous_inference"),
    )

    assert timing.cumulative_wait_overlay_line_v5(0) == ("Cumulative inference wait: 0.00 s",)
    assert timing.cumulative_wait_overlay_line_v5(12_500_000) == ("Cumulative inference wait: 0.01 s",)
    assert timing.cumulative_wait_overlay_line_v5(37_500_000) == ("Cumulative inference wait: 0.04 s",)
    assert timing.quantize_stall_frames_v5(stalls) == (0, 1, 0)

    audit = timing.build_video_timing_audit_v5(
        control_frame_count=0,
        requests=(),
        latencies=(),
        activations=(),
        underflows=(timing.ActionUnderflowV5(1, 1, 20_000_000, 12_500_000),),
        stalls=stalls,
        include_stalls=True,
    )
    assert audit.included_stall_frame_counts == (0, 1, 0)
    assert audit.stall_frame_count == 1


def test_control_frames_are_held_twice_and_fifty_ms_underflow_adds_two_frames():
    assert timing.expand_control_frames_v5(("a", "b")) == ("a", "a", "b", "b")
    request = _request(1, 0)
    latency = _latency(1, 75_000_000, 75_000_000)
    underflow = timing.ActionUnderflowV5(1, 2, 25_000_000, 50_000_000)
    stall = _stall(1, 2, 25_000_000, 50_000_000, reason="async_action_underflow")

    audit = timing.build_video_timing_audit_v5(
        control_frame_count=2,
        requests=(request,),
        latencies=(latency,),
        activations=(),
        underflows=(underflow,),
        stalls=(stall,),
        include_stalls=True,
    )

    assert audit.held_frame_count == 4
    assert audit.stall_frame_count == 2
    assert audit.video_frame_count == 6
    assert audit.total_request_latency_ns == 75_000_000
    assert audit.total_underflow_ns == 50_000_000


def test_audit_distinguishes_measured_stalls_from_included_overlay_stalls():
    request = _request(1, 0)
    latency = _latency(1, 50_000_000, 50_000_000)
    underflow = timing.ActionUnderflowV5(1, 1, 0, 50_000_000)
    stall = _stall(1, 1, 0, 50_000_000, reason="async_action_underflow")

    audit = timing.build_video_timing_audit_v5(
        control_frame_count=3,
        requests=(request,),
        latencies=(latency,),
        activations=(),
        underflows=(underflow,),
        stalls=(stall,),
        include_stalls=False,
    )

    assert audit.to_dict() == {
        "control_hz": 20,
        "video_fps": 40,
        "control_frame_count": 3,
        "held_frame_count": 6,
        "request_count": 1,
        "latency_count": 1,
        "activation_count": 0,
        "underflow_count": 1,
        "total_request_latency_ns": 50_000_000,
        "total_underflow_ns": 50_000_000,
        "measured_stall_count": 1,
        "measured_control_stall_ns": 50_000_000,
        "included_stall_count": 0,
        "included_control_stall_ns": 0,
        "included_stall_reasons": [],
        "included_stall_frame_counts": [],
        "stall_frame_count": 0,
        "video_frame_count": 6,
        "control_duration_ns": 150_000_000,
        "video_duration_ns": 150_000_000,
        "expected_duration_ns": 150_000_000,
        "duration_deviation_ns": 0,
    }
    assert timing.VideoTimingAuditV5.from_dict(audit.to_dict()) == audit


def test_background_request_latency_without_stall_adds_no_video_frames():
    audit = timing.build_video_audit_v5(
        control_frame_count=1,
        requests=(_request(1, 0),),
        latencies=(_latency(1, 500_000_000, 500_000_000),),
        activations=(_native_activation(1, 1, 1, 500_000_000, activation="immediate_swap"),),
        underflows=(),
        stalls=(),
        include_stalls=True,
    )
    assert audit.total_request_latency_ns == 500_000_000
    assert audit.stall_frame_count == 0
    assert audit.video_frame_count == 2


def test_video_audit_from_dict_rejects_exact_field_and_derived_value_mutations():
    audit = timing.build_video_timing_audit_v5(
        control_frame_count=1,
        requests=(),
        latencies=(),
        activations=(),
        underflows=(),
        stalls=(),
        include_stalls=False,
    )
    payload = audit.to_dict()
    missing = deepcopy(payload)
    missing.pop("video_fps")
    extra = dict(payload, extra=1)
    boolean = dict(payload, control_frame_count=True)
    nonfinite = dict(payload, control_duration_ns=float("nan"))
    wrong_list = dict(payload, included_stall_reasons=())
    wrong_derived = dict(payload, video_frame_count=3)
    impossible_counts = dict(payload, latency_count=1)
    impossible_total = dict(payload, total_underflow_ns=1)
    for malformed in (
        missing,
        extra,
        boolean,
        nonfinite,
        wrong_list,
        wrong_derived,
        impossible_counts,
        impossible_total,
    ):
        _assert_raises(ValueError, lambda malformed=malformed: timing.VideoTimingAuditV5.from_dict(malformed))


def test_video_audit_from_dict_defensively_copies_included_lists():
    audit = timing.build_video_timing_audit_v5(
        control_frame_count=1,
        requests=(_request(1, 0),),
        latencies=(_latency(1, 50_000_000, 50_000_000),),
        activations=(),
        underflows=(timing.ActionUnderflowV5(1, 1, 0, 50_000_000),),
        stalls=(_stall(1, 1, 0, 50_000_000, reason="async_action_underflow"),),
        include_stalls=True,
    )
    payload = audit.to_dict()
    impossible_underflow_total = dict(payload, total_underflow_ns=50_000_001)
    wrong_underflow_reason_count = deepcopy(payload)
    wrong_underflow_reason_count["included_stall_reasons"] = ["synchronous_inference"]
    coherent_quantization_tamper = deepcopy(payload)
    coherent_quantization_tamper.update(
        {
            "included_stall_frame_counts": [3],
            "stall_frame_count": 3,
            "video_frame_count": 5,
            "video_duration_ns": 125_000_000,
            "duration_deviation_ns": 25_000_000,
        }
    )
    for malformed in (
        impossible_underflow_total,
        wrong_underflow_reason_count,
        coherent_quantization_tamper,
    ):
        _assert_raises(ValueError, lambda malformed=malformed: timing.VideoTimingAuditV5.from_dict(malformed))

    restored = timing.VideoTimingAuditV5.from_dict(payload)
    payload["included_stall_reasons"][0] = "synchronous_inference"
    payload["included_stall_frame_counts"][0] = 99
    assert restored.included_stall_reasons == ("async_action_underflow",)
    assert restored.included_stall_frame_counts == (2,)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
