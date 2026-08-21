import numpy as np
import pytest

from openpi_client import bsp_spline


def _parameters(knots=None):
    parameters = np.zeros((16, 8), dtype=np.float32)
    if knots is None:
        knots = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    parameters[:, 7] = knots
    for channel in range(7):
        parameters[:12, channel] = np.arange(12, dtype=np.float32) + 10 * channel
    return parameters


def _response(parameters=None):
    return {
        "schema_version": 1,
        "parameters": _parameters() if parameters is None else parameters,
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "alignment": "disabled_delta_eff",
    }


def _speedup_one_response(parameters=None):
    response = _response(parameters)
    response["speedup"] = 1
    return response


def test_speedup_one_must_be_explicitly_expected_and_advances_half_an_index_per_control_step():
    with pytest.raises(ValueError, match="speedup"):
        bsp_spline.BspSpline.from_response(_speedup_one_response())

    spline = bsp_spline.BspSpline.from_response(_speedup_one_response(), expected_speedup=1)
    assert spline.speedup == 1

    plan = bsp_spline.BspControlActionPlan(expected_speedup=1)
    installed = plan.install(
        _speedup_one_response(),
        request_control_step=4,
        activation_control_step=10,
        control_freq_hz=20,
    )
    assert installed.executed_prefix_steps == 6
    assert installed.phase_offset_indices == pytest.approx(3.0)
    assert installed.remaining_time_ns == 600_000_000
    assert plan.sample(11).spline_time == pytest.approx(3.5)


def test_speedup_one_prefetches_at_four_remaining_indices_for_four_hundred_ms():
    plan = bsp_spline.BspControlActionPlan(expected_speedup=1)
    plan.install(
        _speedup_one_response(),
        request_control_step=0,
        activation_control_step=0,
        control_freq_hz=20,
    )

    before = plan.prefetch_decision(9, lead_time_ns=400_000_000)
    due = plan.prefetch_decision(10, lead_time_ns=400_000_000)

    assert before.remaining_time_ns == 450_000_000
    assert not before.should_prefetch
    assert due.remaining_time_ns == 400_000_000
    assert due.should_prefetch


def _constant_response(value, knots=None):
    parameters = _parameters(knots)
    parameters[:12, :7] = value
    return _response(parameters)


def test_hand_derived_cubic_values_cover_both_closed_endpoints_and_an_interior_time():
    spline = bsp_spline.BspSpline.from_response(_response())

    left = spline.evaluate(0.0)
    interior = spline.evaluate(0.5)
    right = spline.evaluate(9.0)

    assert left.shape == (7,)
    assert left.dtype == np.float32
    np.testing.assert_array_equal(left, np.arange(7, dtype=np.float32) * 10)
    np.testing.assert_allclose(
        interior,
        np.arange(7, dtype=np.float32) * 10 + np.float32(113.0 / 96.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(right, np.arange(7, dtype=np.float32) * 10 + 11)


def test_vector_evaluation_preserves_arbitrary_query_shape_and_float32_output():
    spline = bsp_spline.BspSpline.from_response(_response())

    values = spline.evaluate(np.asarray([[0.0, 0.5], [4.5, 9.0]], dtype=np.float64))

    assert values.shape == (2, 2, 7)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values[0, 0], np.arange(7, dtype=np.float32) * 10)
    np.testing.assert_array_equal(values[1, 1], np.arange(7, dtype=np.float32) * 10 + 11)


def test_descending_knot_repair_is_sequential_and_preserves_raw_parameters():
    parameters = _parameters([0, 0, 0, 0, 2, 1, 1, 4, 5, 6, 7, 8, 9, 9, 9, 9])
    spline = bsp_spline.BspSpline.from_response(_response(parameters))

    np.testing.assert_array_equal(spline.parameters[:, 7], parameters[:, 7])
    np.testing.assert_allclose(spline.knots[4:7], [2.0, 2.000001, 2.000002], rtol=0.0, atol=1e-12)


def test_valid_equal_knots_are_not_perturbed():
    spline = bsp_spline.BspSpline.from_response(_response())

    np.testing.assert_array_equal(spline.knots[:4], np.zeros(4, dtype=np.float64))
    np.testing.assert_array_equal(spline.knots[-4:], np.full(4, 9.0, dtype=np.float64))
    np.testing.assert_array_equal(spline.evaluate(spline.t_min), np.arange(7, dtype=np.float32) * 10)
    np.testing.assert_array_equal(spline.evaluate(spline.t_max), np.arange(7, dtype=np.float32) * 10 + 11)


def test_raw_projected_and_active_control_arrays_are_defensive_read_only_copies():
    parameters = _parameters()
    expected_parameters = parameters.copy()
    spline = bsp_spline.BspSpline.from_response(_response(parameters))
    expected_knots = spline.knots.copy()
    expected_controls = spline.controls.copy()

    parameters[:] = -1

    np.testing.assert_array_equal(spline.parameters, expected_parameters)
    np.testing.assert_array_equal(spline.knots, expected_knots)
    np.testing.assert_array_equal(spline.controls, expected_controls)
    for array in (spline.parameters, spline.knots, spline.controls):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 123


def test_inactive_control_rows_do_not_affect_continuous_or_eight_point_decode():
    first = _parameters()
    second = first.copy()
    first[12:, :7] = -1000
    second[12:, :7] = 1000
    first_spline = bsp_spline.BspSpline.from_response(_response(first))
    second_spline = bsp_spline.BspSpline.from_response(_response(second))
    times = np.linspace(first_spline.t_min, first_spline.t_max, 31, dtype=np.float64)

    np.testing.assert_array_equal(first_spline.evaluate(times), second_spline.evaluate(times))
    np.testing.assert_array_equal(first_spline.decode_eight(), second_spline.decode_eight())


def test_decode_eight_includes_both_endpoints():
    spline = bsp_spline.BspSpline.from_response(_response())

    decoded = spline.decode_eight()

    assert decoded.shape == (8, 7)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded[0], np.arange(7, dtype=np.float32) * 10)
    np.testing.assert_array_equal(decoded[-1], np.arange(7, dtype=np.float32) * 10 + 11)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", 2),
        ("origin_hz", True),
        ("origin_hz", 10.0),
        ("origin_hz", 20),
        ("degree", True),
        ("degree", 2),
        ("speedup", True),
        ("speedup", 1.0),
        ("speedup", 1),
        ("alignment", "aligned"),
    ],
)
def test_response_rejects_inexact_metadata_and_boolean_integer_impostors(field, value):
    response = _response()
    response[field] = value

    with pytest.raises(ValueError, match=field):
        bsp_spline.BspSpline.from_response(response)


def test_response_requires_the_exact_schema_one_field_set():
    missing = _response()
    missing.pop("alignment")
    extra = _response()
    extra["padding"] = np.zeros((16, 24), dtype=np.float32)

    with pytest.raises(ValueError, match="fields"):
        bsp_spline.BspSpline.from_response(missing)
    with pytest.raises(ValueError, match="fields"):
        bsp_spline.BspSpline.from_response(extra)
    with pytest.raises(ValueError, match="mapping"):
        bsp_spline.BspSpline.from_response(None)


def test_response_accepts_convertible_parameters_as_a_float32_defensive_copy():
    source = _parameters().astype(np.float64).tolist()

    spline = bsp_spline.BspSpline.from_response(_response(source))

    assert spline.parameters.dtype == np.float32
    assert spline.parameters.shape == (16, 8)
    np.testing.assert_array_equal(spline.parameters, _parameters())


@pytest.mark.parametrize(
    "parameters",
    [
        np.zeros((16, 7), dtype=np.float32),
        np.full((16, 8), np.nan, dtype=np.float32),
        [[object()] * 8 for _ in range(16)],
        np.zeros((16, 8), dtype=np.float32),
    ],
)
def test_response_rejects_malformed_nonfinite_unconvertible_or_closed_parameters(parameters):
    with pytest.raises(ValueError, match=r"parameters|range"):
        bsp_spline.BspSpline.from_response(_response(parameters))


@pytest.mark.parametrize(
    "times",
    [
        -1e-9,
        9.0 + 1e-9,
        np.nan,
        np.inf,
        np.asarray([0.0, np.nan]),
    ],
)
def test_evaluate_rejects_extrapolation_and_nonfinite_queries(times):
    spline = bsp_spline.BspSpline.from_response(_response())

    with pytest.raises(ValueError, match="times"):
        spline.evaluate(times)


def test_plan_uses_activation_clock_and_immediately_swaps_without_alignment_or_latency_offset():
    activation_time_ns = 1_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_response(), activation_time_ns=activation_time_ns)

    old_sample = plan.sample(activation_time_ns + 50_000_000)

    assert old_sample.spline_time == pytest.approx(1.0)
    np.testing.assert_allclose(
        old_sample.action,
        bsp_spline.BspSpline.from_response(_response()).evaluate(1.0),
        rtol=1e-6,
        atol=1e-6,
    )

    swap_time_ns = activation_time_ns + 50_000_000
    plan.install(_constant_response(42.0), activation_time_ns=swap_time_ns)
    new_sample = plan.sample(swap_time_ns)

    assert plan.activation_time_ns == swap_time_ns
    assert new_sample.spline_time == 0.0
    assert not new_sample.underflow
    np.testing.assert_array_equal(new_sample.action, np.full(7, 42.0, dtype=np.float32))


def test_plan_clamps_only_below_t_min_and_keeps_the_right_endpoint_actionable():
    shifted_knots = [2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11]
    activation_time_ns = 2_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(7.0, shifted_knots), activation_time_ns=activation_time_ns)

    below = plan.sample(activation_time_ns)
    endpoint = plan.sample(activation_time_ns + 550_000_000)
    exhausted = plan.sample(activation_time_ns + 550_000_001)

    assert below.spline_time == 2.0
    assert not below.underflow
    np.testing.assert_array_equal(below.action, np.full(7, 7.0, dtype=np.float32))
    assert endpoint.spline_time == 11.0
    assert not endpoint.underflow
    np.testing.assert_array_equal(endpoint.action, np.full(7, 7.0, dtype=np.float32))
    assert exhausted.spline_time > 11.0
    assert exhausted.underflow
    assert exhausted.action is None


def test_plan_exposes_deterministic_remaining_time_and_calibrated_prefetch_decision():
    activation_time_ns = 3_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(1.0), activation_time_ns=activation_time_ns)

    early = plan.prefetch_decision(activation_time_ns + 400_000_000, lead_time_ns=49_999_999)
    due = plan.prefetch_decision(activation_time_ns + 400_000_000, lead_time_ns=50_000_000)
    endpoint = plan.prefetch_decision(activation_time_ns + 450_000_000, lead_time_ns=0)
    exhausted = plan.prefetch_decision(activation_time_ns + 450_000_001, lead_time_ns=0)

    assert early.remaining_time_ns == 50_000_000
    assert not early.should_prefetch
    assert not early.underflow
    assert due.remaining_time_ns == 50_000_000
    assert due.should_prefetch
    assert not due.underflow
    assert endpoint.remaining_time_ns == 0
    assert endpoint.should_prefetch
    assert not endpoint.underflow
    assert exhausted.remaining_time_ns == 0
    assert exhausted.should_prefetch
    assert exhausted.underflow


def test_plan_rejects_non_integer_boolean_and_negative_nanosecond_values():
    plan = bsp_spline.BspActionPlan()
    for invalid in (True, 1.0, -1):
        with pytest.raises(ValueError, match="activation_time_ns"):
            plan.install(_response(), activation_time_ns=invalid)

    plan.install(_response(), activation_time_ns=0)
    for invalid in (True, 1.0, -1):
        with pytest.raises(ValueError, match="now_ns"):
            plan.sample(invalid)
        with pytest.raises(ValueError, match="lead_time_ns"):
            plan.prefetch_decision(0, lead_time_ns=invalid)


def test_plan_rejects_monotonic_clock_regression_before_activation():
    plan = bsp_spline.BspActionPlan()
    plan.install(_response(), activation_time_ns=100)

    with pytest.raises(ValueError, match="must not precede activation_time_ns"):
        plan.sample(99)
    with pytest.raises(ValueError, match="must not precede activation_time_ns"):
        plan.remaining_time_ns(99)
    with pytest.raises(ValueError, match="must not precede activation_time_ns"):
        plan.prefetch_decision(99, lead_time_ns=0)


def test_plan_rejects_sample_rewind_after_a_later_sample_and_allows_equality():
    activation_time_ns = 1_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_response(), activation_time_ns=activation_time_ns)

    first = plan.sample(activation_time_ns + 200_000_000)
    equal = plan.sample(activation_time_ns + 200_000_000)

    np.testing.assert_array_equal(equal.action, first.action)
    with pytest.raises(ValueError, match="high-water"):
        plan.sample(activation_time_ns + 100_000_000)


def test_plan_rejects_cross_method_rewind_without_reverting_prefetch_state():
    activation_time_ns = 2_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(1.0), activation_time_ns=activation_time_ns)

    assert plan.remaining_time_ns(activation_time_ns + 150_000_000) == 300_000_000
    with pytest.raises(ValueError, match="high-water"):
        plan.prefetch_decision(activation_time_ns + 100_000_000, lead_time_ns=300_000_000)

    equal = plan.prefetch_decision(activation_time_ns + 150_000_000, lead_time_ns=300_000_000)
    later = plan.prefetch_decision(activation_time_ns + 200_000_000, lead_time_ns=250_000_000)
    assert equal.should_prefetch
    assert later.should_prefetch
    with pytest.raises(ValueError, match="high-water"):
        plan.remaining_time_ns(activation_time_ns + 199_999_999)
    assert not plan.sample(activation_time_ns + 200_000_000).underflow


def test_plan_rejects_install_before_high_water_and_allows_immediate_swap_at_high_water():
    activation_time_ns = 3_000_000_000
    current_time_ns = activation_time_ns + 200_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(5.0), activation_time_ns=activation_time_ns)
    plan.sample(current_time_ns)

    with pytest.raises(ValueError, match="high-water"):
        plan.install(_constant_response(7.0), activation_time_ns=current_time_ns - 1)

    assert plan.activation_time_ns == activation_time_ns
    np.testing.assert_array_equal(plan.sample(current_time_ns).action, np.full(7, 5.0, dtype=np.float32))

    plan.install(_constant_response(7.0), activation_time_ns=current_time_ns)
    swapped = plan.sample(current_time_ns)
    assert plan.activation_time_ns == current_time_ns
    assert swapped.spline_time == 0.0
    np.testing.assert_array_equal(swapped.action, np.full(7, 7.0, dtype=np.float32))


def test_invalid_install_preserves_plan_and_high_water_transactionally():
    activation_time_ns = 4_000_000_000
    current_time_ns = activation_time_ns + 300_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(9.0), activation_time_ns=activation_time_ns)
    plan.sample(current_time_ns)
    malformed = _constant_response(11.0)
    malformed["degree"] = 2

    with pytest.raises(ValueError, match="degree"):
        plan.install(malformed, activation_time_ns=current_time_ns + 100_000_000)

    assert plan.activation_time_ns == activation_time_ns
    np.testing.assert_array_equal(plan.sample(current_time_ns).action, np.full(7, 9.0, dtype=np.float32))
    with pytest.raises(ValueError, match="high-water"):
        plan.sample(current_time_ns - 1)


def test_invalid_prefetch_does_not_advance_high_water():
    activation_time_ns = 5_000_000_000
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(3.0), activation_time_ns=activation_time_ns)
    plan.sample(activation_time_ns + 100_000_000)

    with pytest.raises(ValueError, match="lead_time_ns"):
        plan.prefetch_decision(activation_time_ns + 300_000_000, lead_time_ns=True)

    assert not plan.sample(activation_time_ns + 200_000_000).underflow


def test_invalid_install_leaves_the_active_curve_and_clock_unchanged():
    plan = bsp_spline.BspActionPlan()
    plan.install(_constant_response(5.0), activation_time_ns=100)
    malformed = _response()
    malformed["degree"] = 2

    with pytest.raises(ValueError, match="degree"):
        plan.install(malformed, activation_time_ns=200)

    assert plan.activation_time_ns == 100
    np.testing.assert_array_equal(plan.sample(100).action, np.full(7, 5.0, dtype=np.float32))


def test_plan_requires_an_installed_curve_before_sampling_or_prefetching():
    plan = bsp_spline.BspActionPlan()

    with pytest.raises(RuntimeError, match="installed"):
        plan.sample(0)
    with pytest.raises(RuntimeError, match="installed"):
        plan.prefetch_decision(0, lead_time_ns=0)


def test_control_plan_skips_six_executed_steps_and_advances_one_index_per_step():
    plan = bsp_spline.BspControlActionPlan()

    installed = plan.install(
        _response(),
        request_control_step=100,
        activation_control_step=106,
        control_freq_hz=20,
    )

    assert not installed.stale
    assert installed.executed_prefix_steps == 6
    assert installed.phase_offset_indices == pytest.approx(6.0)
    assert installed.first_sample_time == pytest.approx(6.0)
    assert installed.remaining_time_ns == 150_000_000
    first = plan.sample(106)
    second = plan.sample(107)
    assert first.spline_time == pytest.approx(6.0)
    assert second.spline_time == pytest.approx(7.0)
    np.testing.assert_array_equal(first.action, bsp_spline.BspSpline.from_response(_response()).evaluate(6.0))
    np.testing.assert_array_equal(second.action, bsp_spline.BspSpline.from_response(_response()).evaluate(7.0))


def test_control_plan_uses_completed_steps_not_wall_clock_or_underflow_wait():
    plan = bsp_spline.BspControlActionPlan()
    plan.install(
        _constant_response(4.0),
        request_control_step=20,
        activation_control_step=20,
        control_freq_hz=20,
    )

    first = plan.sample(20)
    repeated = plan.sample(20)

    assert first.spline_time == repeated.spline_time == 0.0
    assert plan.remaining_time_ns(20) == 450_000_000


def test_control_plan_prefetches_at_eight_remaining_indices_and_accepts_shorter_tail():
    plan = bsp_spline.BspControlActionPlan()
    plan.install(
        _constant_response(1.0),
        request_control_step=0,
        activation_control_step=0,
        control_freq_hz=20,
    )

    before = plan.prefetch_decision(0, lead_time_ns=400_000_000)
    due = plan.prefetch_decision(1, lead_time_ns=400_000_000)

    assert before.remaining_time_ns == 450_000_000
    assert not before.should_prefetch
    assert due.remaining_time_ns == 400_000_000
    assert due.should_prefetch

    installed = plan.install(
        _constant_response(2.0),
        request_control_step=1,
        activation_control_step=7,
        control_freq_hz=20,
    )
    assert not installed.stale
    assert installed.remaining_time_ns == 150_000_000
    assert plan.prefetch_decision(7, lead_time_ns=400_000_000).should_prefetch


def test_control_plan_rejects_expired_candidate_transactionally():
    plan = bsp_spline.BspControlActionPlan()
    plan.install(
        _constant_response(3.0),
        request_control_step=0,
        activation_control_step=0,
        control_freq_hz=20,
    )
    previous = plan.sample(1)

    discarded = plan.install(
        _constant_response(9.0),
        request_control_step=1,
        activation_control_step=11,
        control_freq_hz=20,
    )

    assert discarded.stale
    assert discarded.phase_offset_indices == pytest.approx(10.0)
    assert discarded.first_sample_time == pytest.approx(10.0)
    assert discarded.remaining_time_ns == 0
    np.testing.assert_array_equal(plan.sample(1).action, previous.action)


def test_control_plan_allows_closed_endpoint_once_then_reports_underflow():
    plan = bsp_spline.BspControlActionPlan()
    plan.install(
        _constant_response(8.0),
        request_control_step=0,
        activation_control_step=0,
        control_freq_hz=20,
    )

    endpoint = plan.sample(9)
    exhausted = plan.sample(10)

    assert endpoint.spline_time == 9.0
    assert not endpoint.underflow
    np.testing.assert_array_equal(endpoint.action, np.full(7, 8.0, dtype=np.float32))
    assert exhausted.spline_time == 10.0
    assert exhausted.underflow
    assert exhausted.action is None
