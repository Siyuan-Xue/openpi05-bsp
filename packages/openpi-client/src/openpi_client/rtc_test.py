import numpy as np
import pytest

from openpi_client import inference
from openpi_client import rtc


def _response(action_offset=0.0, model_offset=0.0):
    return {
        "actions": np.arange(16 * 7, dtype=np.float64).reshape(16, 7) + action_offset,
        "rtc": {
            "schema_version": 1,
            "model_actions": np.arange(16 * 32, dtype=np.float64).reshape(16, 32) + model_offset,
        },
        "policy_timing": {"infer_ms": 10.0},
    }


def _bootstrapped_plan(d_init=3):
    plan = rtc.RtcPlan(d_init=d_init)
    request = plan.begin_bootstrap()
    assert request == {inference.RTC_REQUEST_KEY: {"schema_version": 1}}
    plan.install_result(_response())
    return plan


def test_response_keeps_native_and_normalized_chunks_as_separate_read_only_float32_copies():
    response = _response()

    chunk = rtc.RtcActionChunk.from_response(response)
    response["actions"][0, 0] = -1
    response["rtc"]["model_actions"][0, 0] = -2

    assert chunk.actions.shape == (16, 7)
    assert chunk.actions.dtype == np.float32
    assert chunk.model_actions.shape == (16, 32)
    assert chunk.model_actions.dtype == np.float32
    assert chunk.actions[0, 0] == 0
    assert chunk.model_actions[0, 0] == 0
    assert not chunk.actions.flags.writeable
    assert not chunk.model_actions.flags.writeable
    assert not np.shares_memory(chunk.actions, chunk.model_actions)


@pytest.mark.parametrize(
    "response",
    [
        {"actions": np.zeros((16, 7), dtype=np.float32)},
        {"actions": np.zeros((16, 7), dtype=np.float32), "rtc": {"schema_version": 1}},
        {
            "actions": np.zeros((16, 7), dtype=np.float32),
            "rtc": {
                "schema_version": 1,
                "model_actions": np.zeros((16, 32), dtype=np.float32),
                "extra": True,
            },
        },
        {
            "actions": np.zeros((16, 7), dtype=np.float32),
            "rtc": {"schema_version": True, "model_actions": np.zeros((16, 32), dtype=np.float32)},
        },
        {
            "actions": np.zeros((16, 8), dtype=np.float32),
            "rtc": {"schema_version": 1, "model_actions": np.zeros((16, 32), dtype=np.float32)},
        },
        {
            "actions": np.zeros((16, 7), dtype=np.float32),
            "rtc": {"schema_version": 1, "model_actions": np.zeros((16, 7), dtype=np.float32)},
        },
        {
            "actions": np.full((16, 7), np.nan, dtype=np.float32),
            "rtc": {"schema_version": 1, "model_actions": np.zeros((16, 32), dtype=np.float32)},
        },
        {
            "actions": np.zeros((16, 7), dtype=np.float32),
            "rtc": {"schema_version": 1, "model_actions": np.full((16, 32), np.inf, dtype=np.float32)},
        },
    ],
)
def test_response_rejects_malformed_or_native_model_mixed_chunks(response):
    with pytest.raises(ValueError):
        rtc.RtcActionChunk.from_response(response)


def test_delay_history_starts_with_one_calibration_sample_and_is_bounded_to_ten():
    plan = _bootstrapped_plan(d_init=2)

    assert plan.delay_history == (2,)
    assert plan.forecast_delay == 2

    for actual_delay in range(1, 11):
        while plan.cursor < 8:
            plan.consume_action()
        plan.begin_guided()
        for _ in range(min(actual_delay, 8)):
            plan.consume_action()
        plan.install_result(_response(model_offset=actual_delay))

    assert len(plan.delay_history) == 10
    assert plan.delay_history[-1] == 8
    assert plan.forecast_delay == 8


@pytest.mark.parametrize("d_init", [True, -1, 9, 1.5])
def test_initial_delay_is_a_non_boolean_integer_from_zero_through_eight(d_init):
    with pytest.raises(ValueError):
        rtc.RtcPlan(d_init=d_init)


def test_cursor_consumes_native_actions_and_launch_captures_only_the_full_model_sidecar():
    plan = _bootstrapped_plan(d_init=3)
    expected_response = _response()

    for index in range(8):
        np.testing.assert_array_equal(plan.consume_action(), expected_response["actions"][index].astype(np.float32))

    request = plan.begin_guided()
    context = request[inference.RTC_REQUEST_KEY]

    assert context["schema_version"] == 1
    assert context["s"] == 8
    assert context["d"] == 3
    assert context["previous_model_actions"].shape == (16, 32)
    np.testing.assert_array_equal(
        context["previous_model_actions"], expected_response["rtc"]["model_actions"].astype(np.float32)
    )
    assert not np.shares_memory(context["previous_model_actions"], plan.model_actions)


def test_launch_waits_for_dynamic_threshold_and_surfaces_infeasible_horizon():
    plan = _bootstrapped_plan(d_init=5)

    with pytest.raises(rtc.RtcLaunchNotReadyError):
        plan.begin_guided()
    for _ in range(8):
        plan.consume_action()
    assert plan.state is rtc.RtcPlanState.READY_TO_LAUNCH
    plan.begin_guided()

    infeasible = _bootstrapped_plan(d_init=5)
    for _ in range(12):
        infeasible.consume_action()
    assert infeasible.state is rtc.RtcPlanState.INFEASIBLE
    with pytest.raises(rtc.RtcLaunchInfeasibleError):
        infeasible.begin_guided()


def test_result_installs_immediately_at_elapsed_cursor_and_enqueues_actual_delay():
    plan = _bootstrapped_plan(d_init=3)
    for _ in range(8):
        plan.consume_action()
    plan.begin_guided()
    for _ in range(3):
        plan.consume_action()

    new_response = _response(action_offset=1000, model_offset=2000)
    plan.install_result(new_response)

    assert plan.cursor == 3
    assert plan.delay_history == (3, 3)
    np.testing.assert_array_equal(plan.consume_action(), new_response["actions"][3].astype(np.float32))
    np.testing.assert_array_equal(plan.model_actions, new_response["rtc"]["model_actions"].astype(np.float32))


def test_second_request_and_result_without_request_are_rejected():
    plan = _bootstrapped_plan(d_init=1)
    for _ in range(8):
        plan.consume_action()
    plan.begin_guided()

    with pytest.raises(rtc.RtcRequestInFlightError):
        plan.begin_guided()
    plan.install_result(_response())
    with pytest.raises(rtc.RtcNoRequestInFlightError):
        plan.install_result(_response())


def test_malformed_result_does_not_discard_the_in_flight_request():
    plan = _bootstrapped_plan(d_init=1)
    for _ in range(8):
        plan.consume_action()
    plan.begin_guided()

    with pytest.raises(ValueError):
        plan.install_result({"actions": np.zeros((16, 7), dtype=np.float32)})

    assert plan.request_in_flight
    plan.install_result(_response())


def test_invalid_elapsed_delay_is_rejected_instead_of_clipped(monkeypatch):
    negative = _bootstrapped_plan(d_init=1)
    for _ in range(8):
        negative.consume_action()
    negative.begin_guided()
    monkeypatch.setattr(negative, "_cursor", 7)
    negative_model_actions = negative.model_actions
    with pytest.raises(rtc.RtcInvalidDelayError):
        negative.install_result(_response())
    assert negative.cursor == 7
    assert negative.request_in_flight
    assert negative.model_actions is negative_model_actions

    too_late = _bootstrapped_plan(d_init=1)
    for _ in range(8):
        too_late.consume_action()
    too_late.begin_guided()
    monkeypatch.setattr(too_late, "_cursor", 24)
    too_late_model_actions = too_late.model_actions
    with pytest.raises(rtc.RtcInvalidDelayError):
        too_late.install_result(_response())
    assert too_late.cursor == 24
    assert too_late.request_in_flight
    assert too_late.model_actions is too_late_model_actions


def test_exhausted_state_and_reset_return_to_a_clean_bootstrap_seam():
    plan = _bootstrapped_plan(d_init=2)
    for _ in range(16):
        plan.consume_action()

    assert plan.state is rtc.RtcPlanState.EXHAUSTED
    with pytest.raises(rtc.RtcPlanExhaustedError):
        plan.consume_action()

    plan.reset(d_init=4)

    assert plan.state is rtc.RtcPlanState.BOOTSTRAP_REQUIRED
    assert plan.cursor == 0
    assert plan.delay_history == (4,)
    assert not plan.request_in_flight


def test_raw_async_and_rtc_share_the_same_eight_tick_launch_gate():
    raw = rtc.RawAsyncPlan(d_init=8)
    guided = rtc.RtcPlan(d_init=8, fixed_delay=True)
    raw.begin_bootstrap()
    guided.begin_bootstrap()
    response = _response()
    raw.install_result(response)
    guided.install_result(response)

    for _ in range(7):
        raw.consume_action()
        guided.consume_action()
    assert raw.state is rtc.RtcPlanState.EXECUTING
    assert guided.state is rtc.RtcPlanState.EXECUTING

    raw.consume_action()
    guided.consume_action()
    assert raw.state is rtc.RtcPlanState.READY_TO_LAUNCH
    assert guided.state is rtc.RtcPlanState.READY_TO_LAUNCH
    assert raw.forecast_delay == guided.forecast_delay == 8


def test_raw_async_request_has_no_continuity_guidance_and_skips_elapsed_prefix():
    plan = rtc.RawAsyncPlan(d_init=8)
    plan.begin_bootstrap()
    plan.install_result(_response(action_offset=0.0))
    for _ in range(8):
        plan.consume_action()

    overlay = plan.begin_background()
    assert overlay == {inference.RTC_REQUEST_KEY: {"schema_version": inference.RTC_SCHEMA_VERSION}}

    for _ in range(3):
        plan.consume_action()
    response = _response(action_offset=1_000.0)
    plan.install_result(response)

    assert plan.cursor == 3
    np.testing.assert_array_equal(plan.consume_action(), response["actions"][3].astype(np.float32))


def test_fixed_rtc_delay_does_not_adapt_below_theoretical_budget():
    plan = rtc.RtcPlan(d_init=8, fixed_delay=True)
    plan.begin_bootstrap()
    plan.install_result(_response())

    for cycle in range(12):
        while plan.cursor < 8:
            plan.consume_action()
        overlay = plan.begin_guided()
        assert overlay[inference.RTC_REQUEST_KEY]["s"] == 8
        assert overlay[inference.RTC_REQUEST_KEY]["d"] == 8
        plan.install_result(_response(action_offset=float(cycle + 1)))
        assert plan.forecast_delay == 8
