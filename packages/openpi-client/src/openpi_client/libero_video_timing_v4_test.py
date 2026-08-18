"""Strict schema-v4 timing and video accounting contracts.

These tests are intentionally dependency-free.  Per the integration plan they
are authored before production code, but are not executed on this checkout.
"""

import dataclasses
from copy import deepcopy
import operator

from openpi_client import libero_eval
from openpi_client import libero_video_timing_v4 as timing


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
    return timing.RequestEventV4(
        request_id=request_id,
        observation_control_step=observation_control_step,
        submitted_offset_ns=submitted_offset_ns,
        flow_seed=flow_seed,
        dispatch=dispatch,
        trigger=trigger,
        scheduler_context=scheduler_context,
        disposition=disposition,
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
    return timing.LatencyEventV4(
        request_id=request_id,
        completed_offset_ns=completed_offset_ns,
        duration_ns=duration_ns,
        outcome=outcome,
    )


def _native_activation(plan_id, request_id, control_step, activated_offset_ns, *, activation):
    return timing.PlanActivationV4(
        plan_id=plan_id,
        request_id=request_id,
        control_step=control_step,
        activated_offset_ns=activated_offset_ns,
        activation=activation,
        activation_context={"action_cursor": 0},
    )


def _stall(request_id, control_step, started_offset_ns, duration_ns, *, reason):
    return timing.ControlStallV4(
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
        timing.ActionUnderflowV4(
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
    }
    _assert_raises(TypeError, lambda: operator.setitem(request.scheduler_context, "s", 9))
    _assert_raises(dataclasses.FrozenInstanceError, lambda: setattr(request, "request_id", 4))

    records = (
        request,
        _latency(3, 90, 50),
        _native_activation(1, 3, 4, 91, activation="immediate_swap"),
        timing.ActionUnderflowV4(3, 4, 75, 16),
        _stall(3, 4, 75, 16, reason="async_action_underflow"),
    )
    for record in records:
        assert type(record).from_dict(record.to_dict()) == record


def test_every_event_rejects_missing_extra_bool_nonfinite_and_wrong_json_containers():
    records = (
        _initial_request(),
        _latency(0, 125, 125),
        _native_activation(0, 0, 0, 125, activation="initial"),
        timing.ActionUnderflowV4(0, 0, 100, 25),
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
    _assert_raises(ValueError, lambda: timing.RequestEventV4.from_dict(malformed_context))
    malformed_activation_context = records[2].to_dict()
    malformed_activation_context["activation_context"] = []
    _assert_raises(
        ValueError,
        lambda: timing.PlanActivationV4.from_dict(malformed_activation_context),
    )
    wrong_clock = records[1].to_dict()
    wrong_clock["clock"] = "wall_clock"
    _assert_raises(ValueError, lambda: timing.LatencyEventV4.from_dict(wrong_clock))


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
        _assert_raises(ValueError, lambda payload=payload: timing.RequestEventV4.from_dict(payload))

    latency = _latency(0, 1, 1).to_dict()
    latency["outcome"] = "transport_error"
    activation = _native_activation(0, 0, 0, 1, activation="initial").to_dict()
    activation["activation"] = "deferred"
    stall = _stall(0, 0, 0, 1, reason="synchronous_inference").to_dict()
    stall["reason"] = "network"
    for record_type, payload in (
        (timing.LatencyEventV4, latency),
        (timing.PlanActivationV4, activation),
        (timing.ControlStallV4, stall),
    ):
        _assert_raises(ValueError, lambda record_type=record_type, payload=payload: record_type.from_dict(payload))


def test_cross_event_validation_accepts_initial_sync_background_overlap_and_later_underflow():
    events = _mixed_async_timeline()
    normalized = timing.validate_timing_events_v4(
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
        timing.validate_timing_events_v4(**values)

    gapped_requests = (requests[0], dataclasses.replace(requests[1], request_id=2), requests[2])
    orphan_latency = latencies + (_latency(9, 950_000_000, 1),)
    orphan_activation = activations + (
        _native_activation(3, 9, 17, 950_000_000, activation="immediate_swap"),
    )
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
        dataclasses.replace(latencies[2], completed_offset_ns=1_000_000_001, duration_ns=200_000_001),
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
        lambda: timing.validate_timing_events_v4(
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
    _assert_raises(ValueError, lambda: timing.RequestEventV4.from_dict(rtc_payload))

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
        timing.PlanActivationV4(0, 0, 0, 100, "initial", {"curve_elapsed_ns": 0}),
        timing.PlanActivationV4(1, 1, 1, 175, "immediate_swap", {"curve_elapsed_ns": 0}),
    )
    bsp_stalls = (_stall(0, 0, 0, 100, reason="synchronous_inference"),)
    assert timing.validate_timing_events_v4(
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
    )[0] == bsp_requests

    wrong_budget = dataclasses.replace(
        bsp_requests[1],
        scheduler_context={"remaining_plan_ns": 50, "budget_ns": 75},
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
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
        lambda: timing.validate_timing_events_v4(
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
        lambda: timing.validate_timing_events_v4(
            requests=(_initial_request(),),
            latencies=(_latency(0, 100, 100),),
            activations=(
                timing.PlanActivationV4(
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
    timing.validate_timing_events_v4(
        requests=(_initial_request(),),
        latencies=(_latency(0, 100, 100),),
        activations=(
            timing.PlanActivationV4(
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
            lambda invalid_budget=invalid_budget: timing.validate_timing_events_v4(
                requests=(_initial_request(),),
                latencies=(_latency(0, 100, 100),),
                activations=(
                    timing.PlanActivationV4(
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
        lambda: timing.validate_timing_events_v4(
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
        lambda: timing.validate_timing_events_v4(
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
        timing.PlanActivationV4(0, 0, 0, 100, "initial", {"curve_elapsed_ns": 0}),
        timing.PlanActivationV4(1, 1, 1, 175, "immediate_swap", {"curve_elapsed_ns": 0}),
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
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

    sync_requests = (
        _initial_request(),
        _request(
            1,
            150,
            observation_control_step=1,
            dispatch="blocking_replan",
            trigger="baseline_chunk_exhausted",
            scheduler_context={},
        ),
    )
    sync_latencies = (_latency(0, 100, 100), _latency(1, 200, 50))
    sync_activations = (
        _native_activation(0, 0, 0, 100, activation="initial"),
        _native_activation(1, 1, 2, 200, activation="blocking_replace"),
    )
    sync_stalls = (
        _stall(0, 0, 0, 100, reason="synchronous_inference"),
        _stall(1, 1, 175, 25, reason="synchronous_inference"),
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
            requests=sync_requests,
            latencies=sync_latencies,
            activations=sync_activations,
            underflows=(),
            stalls=sync_stalls,
            steps=2,
            episode_duration_ns=200,
            execution_mode="baseline_sync_n5",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
            requests=sync_requests,
            latencies=sync_latencies,
            activations=sync_activations,
            underflows=(),
            stalls=(sync_stalls[0], dataclasses.replace(sync_stalls[1], control_step=2)),
            steps=2,
            episode_duration_ns=200,
            execution_mode="baseline_sync_n5",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )

    failed_request = dataclasses.replace(sync_requests[1], disposition="failed")
    failed_latency = _latency(1, 200, 50, outcome="policy_failure")
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
            requests=(sync_requests[0], failed_request),
            latencies=(sync_latencies[0], failed_latency),
            activations=(sync_activations[0],),
            underflows=(),
            stalls=(sync_stalls[0], dataclasses.replace(sync_stalls[1], control_step=2)),
            steps=2,
            episode_duration_ns=200,
            execution_mode="baseline_sync_n5",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )

    wrong_initial_step = dataclasses.replace(sync_stalls[0], control_step=1)
    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
            requests=(sync_requests[0],),
            latencies=(sync_latencies[0],),
            activations=(sync_activations[0],),
            underflows=(),
            stalls=(wrong_initial_step,),
            steps=1,
            episode_duration_ns=100,
            execution_mode="baseline_sync_n5",
            eval_seed=_EVAL_SEED,
            identity=_IDENTITY,
        ),
    )


def test_background_policy_failure_underflow_cannot_precede_observation_step():
    failed_request = dataclasses.replace(
        _request(1, 150, observation_control_step=2),
        disposition="failed",
    )
    underflow = timing.ActionUnderflowV4(1, 1, 175, 25)
    stalls = (
        _stall(0, 0, 0, 100, reason="synchronous_inference"),
        _stall(1, 1, 175, 25, reason="async_action_underflow"),
    )

    _assert_raises(
        ValueError,
        lambda: timing.validate_timing_events_v4(
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
    underflows = (timing.ActionUnderflowV4(1, 1, 0, 0),)
    stalls = (
        _stall(0, 0, 0, 0, reason="synchronous_inference"),
        _stall(1, 1, 0, 0, reason="async_action_underflow"),
    )

    assert timing.validate_timing_events_v4(
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
    timing.validate_timing_events_v4(
        requests=(failed_request,),
        latencies=(failed_latency,),
        activations=(),
        underflows=(),
        stalls=(failed_stall,),
        steps=0,
        episode_duration_ns=25,
        execution_mode="baseline_sync_n5",
        eval_seed=_EVAL_SEED,
        identity=_IDENTITY,
    )

    abandoned_request = dataclasses.replace(
        _request(1, 150, observation_control_step=1),
        disposition="abandoned",
    )
    timing.validate_timing_events_v4(
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


def test_reason_alone_selects_overlay_and_mixed_stalls_quantize_cumulatively():
    stalls = (
        _stall(0, 0, 0, 12_500_000, reason="synchronous_inference"),
        _stall(1, 1, 20_000_000, 12_500_000, reason="async_action_underflow"),
        _stall(2, 2, 40_000_000, 12_500_000, reason="synchronous_inference"),
    )

    assert timing.stall_overlay_lines_v4(stalls[0]) == (
        "Synchronous inference",
        "Control stalled: 0.01 s",
    )
    assert timing.stall_overlay_lines_v4(stalls[1]) == (
        "Waiting for policy actions",
        "Control stalled: 0.01 s",
    )
    assert timing.quantize_stall_frames_v4(stalls) == (0, 1, 0)

    audit = timing.build_video_timing_audit_v4(
        control_frame_count=0,
        requests=(),
        latencies=(),
        activations=(),
        underflows=(timing.ActionUnderflowV4(1, 1, 20_000_000, 12_500_000),),
        stalls=stalls,
        include_stalls=True,
    )
    assert audit.included_stall_frame_counts == (0, 1, 0)
    assert audit.stall_frame_count == 1


def test_control_frames_are_held_twice_and_fifty_ms_underflow_adds_two_frames():
    assert timing.expand_control_frames_v4(("a", "b")) == ("a", "a", "b", "b")
    request = _request(1, 0)
    latency = _latency(1, 75_000_000, 75_000_000)
    underflow = timing.ActionUnderflowV4(1, 2, 25_000_000, 50_000_000)
    stall = _stall(1, 2, 25_000_000, 50_000_000, reason="async_action_underflow")

    audit = timing.build_video_timing_audit_v4(
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
    underflow = timing.ActionUnderflowV4(1, 1, 0, 50_000_000)
    stall = _stall(1, 1, 0, 50_000_000, reason="async_action_underflow")

    audit = timing.build_video_timing_audit_v4(
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
    assert timing.VideoTimingAuditV4.from_dict(audit.to_dict()) == audit


def test_background_request_latency_without_stall_adds_no_video_frames():
    audit = timing.build_video_audit_v4(
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
    audit = timing.build_video_timing_audit_v4(
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
        _assert_raises(ValueError, lambda malformed=malformed: timing.VideoTimingAuditV4.from_dict(malformed))


def test_video_audit_from_dict_defensively_copies_included_lists():
    audit = timing.build_video_timing_audit_v4(
        control_frame_count=1,
        requests=(_request(1, 0),),
        latencies=(_latency(1, 50_000_000, 50_000_000),),
        activations=(),
        underflows=(timing.ActionUnderflowV4(1, 1, 0, 50_000_000),),
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
        _assert_raises(ValueError, lambda malformed=malformed: timing.VideoTimingAuditV4.from_dict(malformed))

    restored = timing.VideoTimingAuditV4.from_dict(payload)
    payload["included_stall_reasons"][0] = "synchronous_inference"
    payload["included_stall_frame_counts"][0] = 99
    assert restored.included_stall_reasons == ("async_action_underflow",)
    assert restored.included_stall_frame_counts == (2,)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
