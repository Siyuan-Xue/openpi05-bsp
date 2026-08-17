"""Pure contracts for LIBERO video timing; executable without pytest."""

from copy import deepcopy

from openpi_client import libero_video_timing as timing


def test_frequency_validation_requires_20_hz_control_and_integral_video_holds():
    assert timing.validate_video_frequencies() == 2
    assert timing.validate_video_frequencies(control_hz=20, video_fps=60) == 3
    for control_hz, video_fps in ((10, 40), (20, 0), (20, 30), (True, 40)):
        try:
            timing.validate_video_frequencies(control_hz=control_hz, video_fps=video_fps)
        except ValueError:
            pass
        else:
            raise AssertionError((control_hz, video_fps))


def test_synchronous_and_async_request_stall_records_keep_latency_separate_from_stall():
    request = timing.InferenceRequest(replan_index=3, started_offset_ns=100, duration_ns=250)
    synchronous_stall = timing.ControlStall(control_step=24, replan_index=3, started_offset_ns=100, duration_ns=250)
    future_async_request = timing.InferenceRequest(replan_index=4, started_offset_ns=400, duration_ns=300)
    future_async_stall = timing.ControlStall(
        control_step=32,
        replan_index=4,
        started_offset_ns=900,
        duration_ns=0,
        reason="async_action_underflow",
    )

    assert request.duration_ns == 250
    assert synchronous_stall.duration_ns == 250
    assert future_async_stall.duration_ns == 0
    assert request.to_dict() == {
        "clock": "episode_monotonic_ns",
        "replan_index": 3,
        "started_offset_ns": 100,
        "duration_ns": 250,
    }
    assert future_async_request.duration_ns == 300
    assert future_async_stall.to_dict() == {
        "clock": "episode_monotonic_ns",
        "control_step": 32,
        "replan_index": 4,
        "started_offset_ns": 900,
        "duration_ns": 0,
        "reason": "async_action_underflow",
    }


def test_stall_reason_and_schedule_select_exact_sync_and_async_overlay_labels():
    synchronous = timing.ControlStall(0, 0, 0, 125_000_000)
    asynchronous = timing.ControlStall(
        1,
        1,
        200_000_000,
        75_000_000,
        reason="async_action_underflow",
    )

    assert timing.stall_overlay_lines(synchronous, inference_schedule="synchronous") == (
        "Synchronous inference",
        "Control stalled: 0.12 s",
    )
    assert timing.stall_overlay_lines(asynchronous, inference_schedule="asynchronous") == (
        "Waiting for policy actions",
        "Control stalled: 0.07 s",
    )
    for invalid in (
        lambda: timing.ControlStall(0, 0, 0, 1, reason="network"),
        lambda: timing.stall_overlay_lines(synchronous, inference_schedule="asynchronous"),
        lambda: timing.stall_overlay_lines(asynchronous, inference_schedule="synchronous"),
        lambda: timing.stall_overlay_lines(synchronous, inference_schedule="batched"),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)


def test_control_frames_hold_exactly_twice_at_40_fps_without_dataset_rate_input():
    frames = ("control-0", "control-1", "control-2")

    assert timing.expand_control_frames(frames, control_hz=20, video_fps=40) == (
        "control-0",
        "control-0",
        "control-1",
        "control-1",
        "control-2",
        "control-2",
    )


def test_stall_quantization_carries_fractional_frames_across_events():
    stalls = (
        timing.ControlStall(control_step=0, replan_index=0, started_offset_ns=0, duration_ns=12_500_000),
        timing.ControlStall(control_step=1, replan_index=1, started_offset_ns=20_000_000, duration_ns=12_500_000),
        timing.ControlStall(control_step=2, replan_index=2, started_offset_ns=40_000_000, duration_ns=12_500_000),
    )

    assert timing.quantize_stall_frames(stalls, video_fps=40) == (0, 1, 0)


def test_stall_quantization_requires_chronological_non_overlapping_control_steps():
    chronological = (
        timing.ControlStall(control_step=8, replan_index=1, started_offset_ns=0, duration_ns=18_750_000),
        timing.ControlStall(control_step=9, replan_index=2, started_offset_ns=18_750_000, duration_ns=6_250_000),
    )

    assert timing.quantize_stall_frames(chronological, video_fps=40) == (0, 1)
    for invalid in (
        tuple(reversed(chronological)),
        (
            chronological[0],
            timing.ControlStall(control_step=9, replan_index=2, started_offset_ns=18_000_000, duration_ns=6_250_000),
        ),
    ):
        try:
            timing.quantize_stall_frames(invalid, video_fps=40)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)


def test_overlay_renderer_receives_a_copy_and_cannot_mutate_the_source_frame():
    source = {"pixels": [[1, 2], [3, 4]], "labels": []}

    def renderer(frame, lines):
        frame["pixels"][0][0] = 99
        frame["labels"].extend(lines)
        return frame

    rendered = timing.render_overlay(source, ("latency: 25 ms",), renderer=renderer)

    assert source == {"pixels": [[1, 2], [3, 4]], "labels": []}
    assert rendered == {"pixels": [[99, 2], [3, 4]], "labels": ["latency: 25 ms"]}
    assert rendered is not source
    assert deepcopy(source) == source


def test_audit_uses_exact_integer_durations_and_cumulative_stall_frames():
    requests = (
        timing.InferenceRequest(replan_index=0, started_offset_ns=100, duration_ns=125_000_000),
        timing.InferenceRequest(replan_index=1, started_offset_ns=200, duration_ns=225_000_000),
    )
    stalls = (
        timing.ControlStall(control_step=0, replan_index=0, started_offset_ns=0, duration_ns=12_500_000),
        timing.ControlStall(control_step=1, replan_index=1, started_offset_ns=12_500_000, duration_ns=12_500_000),
    )

    audit = timing.build_video_audit(
        control_frame_count=3,
        requests=requests,
        stalls=stalls,
        control_hz=20,
        video_fps=40,
    )

    assert audit.to_dict() == {
        "control_hz": 20,
        "video_fps": 40,
        "control_frame_count": 3,
        "held_frame_count": 6,
        "stall_frame_count": 1,
        "video_frame_count": 7,
        "control_duration_ns": 150_000_000,
        "request_count": 2,
        "total_request_latency_ns": 350_000_000,
        "included_stall_count": 2,
        "included_control_stall_ns": 25_000_000,
        "video_duration_ns": 175_000_000,
        "expected_duration_ns": 175_000_000,
        "duration_deviation_ns": 0,
    }


def test_audit_uses_same_replan_zero_stall_instead_of_async_request_latency():
    request = timing.InferenceRequest(replan_index=4, started_offset_ns=100, duration_ns=300_000_000)
    zero_stall = timing.ControlStall(
        control_step=7,
        replan_index=4,
        started_offset_ns=400_000_000,
        duration_ns=0,
        reason="async_action_underflow",
    )

    audit = timing.build_video_audit(control_frame_count=3, requests=(request,), stalls=(zero_stall,))

    assert audit.total_request_latency_ns == 300_000_000
    assert audit.included_control_stall_ns == 0
    assert audit.stall_frame_count == 0
    assert audit.video_frame_count == 6
    assert audit.duration_deviation_ns == 0


def test_partial_async_underflow_freezes_only_the_measured_stall_duration():
    request = timing.InferenceRequest(4, started_offset_ns=100, duration_ns=300_000_000)
    partial_stall = timing.ControlStall(
        7,
        4,
        started_offset_ns=350_000_000,
        duration_ns=50_000_000,
        reason="async_action_underflow",
    )

    audit = timing.build_video_audit(
        control_frame_count=3,
        requests=(request,),
        stalls=(partial_stall,),
    )

    assert audit.total_request_latency_ns == 300_000_000
    assert audit.included_control_stall_ns == 50_000_000
    assert audit.stall_frame_count == 2
    assert audit.video_frame_count == 8
    assert audit.expected_duration_ns == 200_000_000


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
