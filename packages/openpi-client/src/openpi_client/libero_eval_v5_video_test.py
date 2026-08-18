"""Schema-v5 encoded-video artifact audit contracts.

The tests deliberately avoid codecs and wall time.  They remain unexecuted on
this checkout under the Task 5 server-only verification boundary.
"""

import dataclasses

import pytest

from openpi_client import libero_eval_v5 as evaluation
from openpi_client import libero_video_timing_v5 as timing
from openpi_client import latency_sampling


def _record(*, steps=2):
    identity = evaluation.EpisodeIdentity(
        suite="libero_spatial",
        task_id=0,
        task_name="pick up the block",
        init_state_index=0,
        init_state_fingerprint="a" * 64,
    )
    seed = evaluation.stable_replan_seed(42, identity, 0)
    sample_key = latency_sampling.LatencySampleKeyV1(
        namespace="formal",
        seed=42,
        suite=identity.suite,
        task_id=identity.task_id,
        trial_index=identity.init_state_index,
        request_ordinal=0,
    )
    sampled_target = latency_sampling.NormalLatencySamplerV1().sample_target_ns(sample_key)
    request = timing.RequestEventV5(
        request_id=0,
        observation_control_step=0,
        submitted_offset_ns=0,
        flow_seed=seed,
        dispatch="blocking_initial",
        trigger="initial_plan",
        scheduler_context={},
        disposition="activated",
        latency_sample_key=sample_key,
        sampled_target_latency_ns=sampled_target,
    )
    latency = timing.LatencyEventV5(0, sampled_target, sampled_target, "success")
    activation = timing.PlanActivationV5(0, 0, 0, sampled_target, "initial", {"action_cursor": 0})
    stall = timing.ControlStallV5(0, 0, 0, sampled_target, "synchronous_inference")
    result = evaluation.AttemptResultV5(
        execution_mode="baseline_rtc",
        success=True,
        steps=steps,
        replans=1,
        episode_duration_ns=sampled_target + 100_000_000,
        inference_requests=(request,),
        inference_latencies=(latency,),
        plan_activations=(activation,),
        control_stalls=(stall,),
    )
    return evaluation.EpisodeRecordV5.from_attempt(
        identity,
        42,
        1,
        execution_mode="baseline_rtc",
        result=result,
    )


def _planned(record, *, include_stalls=True):
    return timing.build_video_timing_audit_v5(
        control_frame_count=record.steps,
        requests=record.inference_requests,
        latencies=record.inference_latencies,
        activations=record.plan_activations,
        underflows=record.action_underflows,
        stalls=record.control_stalls,
        include_stalls=include_stalls,
    )


def test_video_artifact_round_trip_crosschecks_the_exact_planned_timing_object():
    record = _record()
    planned = _planned(record)
    audit = evaluation.build_video_artifact_audit_v5(
        episode=record,
        path="videos/example.mp4",
        planned=planned,
        video_show_inference_waits=True,
        encoded_fps=40.0,
        encoded_frame_count=planned.video_frame_count,
        encoded_duration_s=planned.expected_duration_ns / 1_000_000_000,
    )
    assert audit.to_dict() == {
        "schema_version": 5,
        "episode_id": record.episode_id,
        "execution_mode": "baseline_rtc",
        "path": "videos/example.mp4",
        "video_show_inference_waits": True,
        "planned": planned.to_dict(),
        "encoded_fps": 40.0,
        "encoded_frame_count": planned.video_frame_count,
        "encoded_duration_ns": planned.expected_duration_ns,
        "artifact_padding_frame_count": 0,
        "timing_gate_pass": True,
        "warning": None,
    }
    assert evaluation.VideoArtifactAuditV5.from_dict(audit.to_dict()) == audit
    audit.validate_episode(record)


def test_duration_only_deviation_is_a_warning_but_fps_and_frame_mismatch_are_errors():
    record = _record()
    planned = _planned(record)
    warning = evaluation.build_video_artifact_audit_v5(
        episode=record,
        path="videos/example.mp4",
        planned=planned,
        video_show_inference_waits=True,
        encoded_fps=40,
        encoded_frame_count=planned.video_frame_count,
        encoded_duration_s=(planned.expected_duration_ns + 25_000_001) / 1_000_000_000,
    )
    assert not warning.timing_gate_pass
    assert warning.warning is not None

    for changes in (
        {"encoded_fps": 39.0},
        {"encoded_frame_count": planned.video_frame_count + 1},
    ):
        with pytest.raises(ValueError):
            evaluation.build_video_artifact_audit_v5(
                episode=record,
                path="videos/example.mp4",
                planned=planned,
                video_show_inference_waits=True,
                encoded_fps=changes.get("encoded_fps", 40),
                encoded_frame_count=changes.get("encoded_frame_count", planned.video_frame_count),
                encoded_duration_s=planned.expected_duration_ns / 1_000_000_000,
            )


def test_only_an_empty_timeline_accepts_the_documented_one_frame_padding():
    planned = timing.build_video_timing_audit_v5(
        control_frame_count=0,
        requests=(),
        latencies=(),
        activations=(),
        underflows=(),
        stalls=(),
        include_stalls=False,
    )
    audit = evaluation.VideoArtifactAuditV5(
        schema_version=5,
        episode_id="episode-zero",
        execution_mode="baseline_async",
        path="videos/zero.mp4",
        video_show_inference_waits=False,
        planned=planned,
        encoded_fps=40.0,
        encoded_frame_count=1,
        encoded_duration_ns=25_000_000,
        artifact_padding_frame_count=1,
        timing_gate_pass=True,
        warning=None,
    )
    assert audit.encoded_frame_count == 1

    with pytest.raises(ValueError, match="padding"):
        dataclasses.replace(audit, artifact_padding_frame_count=0)
    with pytest.raises(ValueError, match="padding"):
        dataclasses.replace(
            audit,
            planned=timing.build_video_timing_audit_v5(
                control_frame_count=1,
                requests=(),
                latencies=(),
                activations=(),
                underflows=(),
                stalls=(),
                include_stalls=False,
            ),
        )


def test_video_audit_rejects_exact_field_bool_nonfinite_wrong_object_and_identity_mutations():
    record = _record()
    planned = _planned(record)
    audit = evaluation.build_video_artifact_audit_v5(
        episode=record,
        path="videos/example.mp4",
        planned=planned,
        video_show_inference_waits=True,
        encoded_fps=40,
        encoded_frame_count=planned.video_frame_count,
        encoded_duration_s=planned.expected_duration_ns / 1_000_000_000,
    )
    payload = audit.to_dict()
    missing = dict(payload)
    missing.pop("planned")
    malformed = (
        missing,
        dict(payload, extra=True),
        dict(payload, schema_version=4.0),
        dict(payload, encoded_frame_count=True),
        dict(payload, encoded_fps=float("nan")),
        dict(payload, planned=[]),
        dict(payload, execution_mode="unknown"),
    )
    for value in malformed:
        with pytest.raises(ValueError):
            evaluation.VideoArtifactAuditV5.from_dict(value)

    other_audit = dataclasses.replace(audit, execution_mode="baseline_async")
    with pytest.raises(ValueError, match="execution_mode"):
        other_audit.validate_episode(record)

    with pytest.raises(ValueError, match="overlays"):
        dataclasses.replace(audit, video_show_inference_waits=False)


def test_video_writer_serializes_the_nested_timing_audit_without_flattening(tmp_path):
    record = _record()
    planned = _planned(record, include_stalls=False)
    audit = evaluation.build_video_artifact_audit_v5(
        episode=record,
        path="videos/example.mp4",
        planned=planned,
        video_show_inference_waits=False,
        encoded_fps=40,
        encoded_frame_count=planned.video_frame_count,
        encoded_duration_s=planned.expected_duration_ns / 1_000_000_000,
    )
    writer = evaluation.ArtifactWriterV5(tmp_path)
    writer.append_video_audit(audit)
    persisted = (tmp_path / "video_audit.jsonl").read_text()
    assert '"planned": {' in persisted
    assert '"schema_version": 5' in persisted
