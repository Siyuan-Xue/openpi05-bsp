"""Pure pytest-style contracts for encoded LIBERO video audits."""

import json

from openpi_client import libero_eval
from openpi_client import libero_video_timing as timing


def _planned_audit():
    return timing.build_video_audit(
        control_frame_count=2,
        requests=(
            timing.InferenceRequest(
                replan_index=0,
                started_offset_ns=10,
                duration_ns=25_000_000,
            ),
        ),
        stalls=(
            timing.ControlStall(
                control_step=0,
                replan_index=0,
                started_offset_ns=10,
                duration_ns=25_000_000,
            ),
        ),
    )


def test_video_artifact_audit_accepts_readback_within_one_output_frame():
    audit = libero_eval.build_video_artifact_audit(
        episode_id="episode-1",
        path="videos/episode-1.mp4",
        planned=_planned_audit(),
        encoded_fps=40.0,
        encoded_frame_count=5,
        encoded_duration_s=0.14,
    )

    assert audit.timing_gate
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
    audit = libero_eval.build_video_artifact_audit(
        episode_id="episode-1",
        path="videos/episode-1.mp4",
        planned=_planned_audit(),
        encoded_fps=40.0,
        encoded_frame_count=5,
        encoded_duration_s=0.2,
    )

    assert not audit.timing_gate
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
                **values,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(overrides)


def test_artifact_writer_appends_video_audit_jsonl(tmp_path):
    audit = libero_eval.build_video_artifact_audit(
        episode_id="episode-1",
        path="videos/episode-1.mp4",
        planned=_planned_audit(),
        encoded_fps=40.0,
        encoded_frame_count=5,
        encoded_duration_s=0.125,
    )

    writer = libero_eval.ArtifactWriter(tmp_path)
    writer.append_video_audit(audit)

    payload = json.loads((tmp_path / "video_audit.jsonl").read_text(encoding="utf-8"))
    assert payload["episode_id"] == "episode-1"
    assert payload["control_hz"] == 20
    assert payload["video_fps"] == 40
    assert payload["encoded_frame_count"] == 5
    assert payload["timing_gate"] is True


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
