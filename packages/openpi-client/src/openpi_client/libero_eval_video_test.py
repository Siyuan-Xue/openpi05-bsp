"""Pure pytest-style contracts for encoded LIBERO video audits."""

import json

from openpi_client import libero_eval
from openpi_client import libero_video_timing as timing


def _measured_events():
    request = timing.InferenceRequest(
        replan_index=0,
        started_offset_ns=10,
        duration_ns=25_000_000,
    )
    stall = timing.ControlStall(
        control_step=0,
        replan_index=0,
        started_offset_ns=10,
        duration_ns=25_000_000,
    )
    return request, stall


def _planned_audit(*, include_stall=True):
    request, stall = _measured_events()
    return timing.build_video_audit(
        control_frame_count=2,
        requests=(request,),
        stalls=(stall,) if include_stall else (),
    )


def _build_artifact_audit(*, show_waits, encoded_duration_s, encoded_frame_count):
    _, stall = _measured_events()
    included_stalls = (stall,) if show_waits else ()
    return libero_eval.build_video_artifact_audit(
        episode_id="episode-1",
        path="videos/episode-1.mp4",
        planned=_planned_audit(include_stall=show_waits),
        measured_stalls=(stall,),
        included_stalls=included_stalls,
        video_show_inference_waits=show_waits,
        inference_schedule="synchronous",
        encoded_fps=40.0,
        encoded_frame_count=encoded_frame_count,
        encoded_duration_s=encoded_duration_s,
    )


def test_video_artifact_audit_accepts_readback_within_one_output_frame():
    audit = _build_artifact_audit(
        show_waits=True,
        encoded_duration_s=0.14,
        encoded_frame_count=5,
    )

    assert audit.timing_gate_pass
    assert audit.warning is None
    assert audit.encoded_duration_ns == 140_000_000
    assert audit.encoded_duration_deviation_ns == 15_000_000
    assert audit.timing_tolerance_ns == 25_000_000


def test_episode_record_preserves_sparse_request_and_stall_events():
    request = timing.InferenceRequest(0, started_offset_ns=10, duration_ns=25_000_000)
    stall = timing.ControlStall(0, 0, started_offset_ns=10, duration_ns=25_000_000)
    identity = libero_eval.EpisodeIdentity(
        suite="libero_spatial",
        task_id=0,
        task_name="pick up the block",
        init_state_index=0,
        init_state_fingerprint="state-0",
    )
    result = libero_eval.AttemptResult(
        success=True,
        steps=1,
        replans=1,
        inference_ms=(25.0,),
        inference_requests=(request,),
        control_stalls=(stall,),
    )

    record = libero_eval.EpisodeRecord.from_attempt(
        identity,
        42,
        1,
        success=True,
        result=result,
    )

    assert record.inference_requests == (request,)
    assert record.control_stalls == (stall,)
    assert record.to_dict()["inference_requests"] == [request.to_dict()]
    assert record.to_dict()["control_stalls"] == [stall.to_dict()]


def test_video_artifact_audit_warns_but_does_not_error_on_duration_only_mismatch():
    audit = _build_artifact_audit(
        show_waits=True,
        encoded_duration_s=0.2,
        encoded_frame_count=5,
    )

    assert not audit.timing_gate_pass
    assert audit.encoded_duration_deviation_ns == 75_000_000
    assert audit.warning == (
        "encoded duration deviates from expected duration by 0.075000 s "
        "(tolerance 0.025000 s)"
    )


def test_video_artifact_audit_rejects_invalid_or_mismatched_readback_metadata():
    for overrides in (
        {"encoded_fps": 20.0},
        {"encoded_frame_count": 4},
        {"encoded_duration_s": float("nan")},
    ):
        values = {
            "encoded_fps": 40.0,
            "encoded_frame_count": 5,
            "encoded_duration_s": 0.125,
            **overrides,
        }
        try:
            libero_eval.build_video_artifact_audit(
                episode_id="episode-1",
                path="videos/episode-1.mp4",
                planned=_planned_audit(),
                measured_stalls=(_measured_events()[1],),
                included_stalls=(_measured_events()[1],),
                video_show_inference_waits=True,
                inference_schedule="synchronous",
                **values,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(overrides)


def test_artifact_writer_appends_video_audit_jsonl(tmp_path):
    audit = _build_artifact_audit(
        show_waits=True,
        encoded_duration_s=0.125,
        encoded_frame_count=5,
    )

    writer = libero_eval.ArtifactWriter(tmp_path)
    writer.append_video_audit(audit)

    payload = json.loads((tmp_path / "video_audit.jsonl").read_text(encoding="utf-8"))
    assert payload["episode_id"] == "episode-1"
    assert payload["control_hz"] == 20
    assert payload["video_fps"] == 40
    assert payload["encoded_frame_count"] == 5
    assert payload["timing_gate_pass"] is True
    assert payload["video_show_inference_waits"] is True
    assert payload["inference_schedule"] == "synchronous"
    assert payload["measured_stall_count"] == 1
    assert payload["measured_control_stall_ns"] == 25_000_000
    assert payload["included_stall_count"] == 1
    assert payload["included_control_stall_ns"] == 25_000_000
    assert "timing_gate" not in payload
    assert "total_control_stall_ns" not in payload


def test_disabled_video_audit_keeps_measured_stalls_but_excludes_them_from_timeline():
    audit = _build_artifact_audit(
        show_waits=False,
        encoded_duration_s=0.1,
        encoded_frame_count=4,
    )

    payload = audit.to_dict()
    assert payload["video_show_inference_waits"] is False
    assert payload["measured_stall_count"] == 1
    assert payload["measured_control_stall_ns"] == 25_000_000
    assert payload["included_stall_count"] == 0
    assert payload["included_control_stall_ns"] == 0
    assert payload["expected_duration_ns"] == 100_000_000
    assert payload["encoded_duration_ns"] == 100_000_000
    assert payload["timing_gate_pass"] is True


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for name, value in sorted(globals().items()):
        if not name.startswith("test_") or not callable(value):
            continue
        if name == "test_artifact_writer_appends_video_audit_jsonl":
            with tempfile.TemporaryDirectory() as directory:
                value(Path(directory))
        else:
            value()
        print("PASS", name)
