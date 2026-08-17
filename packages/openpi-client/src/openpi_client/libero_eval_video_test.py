"""Pure pytest-style contracts for encoded LIBERO video audits."""

import json

from openpi_client import libero_eval
from openpi_client import libero_report
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


def test_video_artifact_audit_uses_ceiling_for_exact_60_fps_frame_tolerance():
    planned = timing.build_video_audit(
        control_frame_count=2,
        requests=(),
        stalls=(),
        video_fps=60,
    )
    one_frame_ns = 16_666_667

    exact_boundary = libero_eval.build_video_artifact_audit(
        episode_id="episode-60fps",
        path="videos/episode-60fps.mp4",
        planned=planned,
        measured_stalls=(),
        included_stalls=(),
        video_show_inference_waits=False,
        inference_schedule="synchronous",
        encoded_fps=60.0,
        encoded_frame_count=6,
        encoded_duration_s=(planned.expected_duration_ns + one_frame_ns) / 1_000_000_000,
    )
    beyond_boundary = libero_eval.build_video_artifact_audit(
        episode_id="episode-60fps",
        path="videos/episode-60fps.mp4",
        planned=planned,
        measured_stalls=(),
        included_stalls=(),
        video_show_inference_waits=False,
        inference_schedule="synchronous",
        encoded_fps=60.0,
        encoded_frame_count=6,
        encoded_duration_s=(planned.expected_duration_ns + one_frame_ns + 1) / 1_000_000_000,
    )

    assert exact_boundary.timing_tolerance_ns == one_frame_ns
    assert exact_boundary.timing_gate_pass
    assert exact_boundary.warning is None
    assert not beyond_boundary.timing_gate_pass
    assert beyond_boundary.warning is not None


def test_video_artifact_audit_records_real_one_frame_container_padding():
    planned = timing.build_video_audit(
        control_frame_count=0,
        requests=(timing.InferenceRequest(0, 0, 1),),
        stalls=(timing.ControlStall(0, 0, 0, 1),),
    )

    audit = libero_eval.build_video_artifact_audit(
        episode_id="zero-step-failure",
        path="videos/zero-step-failure.mp4",
        planned=planned,
        measured_stalls=(timing.ControlStall(0, 0, 0, 1),),
        included_stalls=(timing.ControlStall(0, 0, 0, 1),),
        video_show_inference_waits=True,
        inference_schedule="synchronous",
        artifact_padding_frame_count=1,
        encoded_fps=40.0,
        encoded_frame_count=1,
        encoded_duration_s=0.025,
    )

    assert audit.planned.control_frame_count == 0
    assert audit.planned.video_frame_count == 0
    assert audit.artifact_padding_frame_count == 1
    assert audit.expected_encoded_frame_count == 1
    assert audit.encoded_frame_count == 1
    assert audit.expected_duration_ns == 1
    assert audit.timing_gate_pass


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


def test_episode_record_keeps_stall_source_frames_transient():
    result = libero_eval.AttemptResult(
        success=False,
        steps=0,
        replans=0,
        failure_kind="policy",
        stall_source_frames=((0, "request-frame"),),
    )

    record = libero_eval.EpisodeRecord.from_attempt(
        libero_eval.EpisodeIdentity(
            suite="libero_spatial",
            task_id=0,
            task_name="pick up the block",
            init_state_index=0,
            init_state_fingerprint="state-0",
        ),
        42,
        1,
        success=False,
        failure_kind="policy",
        result=result,
    )

    assert record.steps == 0
    assert record.replans == 0
    assert record.stall_source_frames == ((0, "request-frame"),)
    assert "stall_source_frames" not in record.to_dict()


def _formal_manifest(**overrides):
    values = {
        "code_sha": "a" * 40,
        "dataset_revision": "v2.0",
        "config_name": "pi05_libero_baseline_h16",
        "checkpoint_step": 10000,
        "bsp_cache_hash": None,
        "bsp_cache_manifest_fingerprint": None,
        "norm_hash": "b" * 64,
        "checkpoint": "checkpoint/baseline/10000",
        "container_digest": "sha256:" + "d" * 64,
        "train_seed": 42,
        "eval_seed": 42,
        "policy_variant": "baseline",
        "bsp_parameters": dict(libero_eval.BSP_PARAMETERS),
        "policy_protocol": "baseline_h16",
        "expected_action_horizon": 16,
        "execution_horizon": 8,
        "suites": libero_eval.SUPPORTED_SUITES,
        "task_ids": tuple(range(10)),
        "trials_per_task": 50,
        "num_steps_wait": 10,
        "max_steps_by_suite": {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        },
        "connection_timeout_s": 30.0,
        "inference_timeout_s": 120.0,
        "infrastructure_retries": 2,
    }
    return libero_eval.EvaluationManifest(**dict(values, **overrides))


def test_manifest_producer_round_trips_through_formal_schema_v3_reader():
    payload = _formal_manifest().to_dict()

    assert libero_report._validate_manifest(payload) == ("baseline", 10000)


def test_manifest_producer_rejects_numeric_impostors_and_noncanonical_containers():
    invalid_overrides = (
        {"checkpoint_step": False},
        {"train_seed": 42.0},
        {"eval_seed": True},
        {"expected_action_horizon": 16.0},
        {"execution_horizon": True},
        {"trials_per_task": 50.0},
        {"num_steps_wait": False},
        {"infrastructure_retries": 2.0},
        {"task_ids": (0.0, *range(1, 10))},
        {"suites": "libero_spatial"},
        {
            "max_steps_by_suite": {
                "libero_spatial": 220.0,
                "libero_object": 280,
                "libero_goal": 300,
                "libero_10": 520,
            }
        },
        {"bsp_parameters": dict(libero_eval.BSP_PARAMETERS, degree=3.0)},
        {"config_name": 123},
        {"checkpoint": 10000},
        {"norm_hash": ["b"] * 64},
    )

    for overrides in invalid_overrides:
        try:
            _formal_manifest(**overrides)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(overrides)


def test_bsp_manifest_producer_rejects_nonstring_cache_hashes():
    common = {
        "config_name": "pi05_libero_bsp_h16",
        "checkpoint": "checkpoint/bsp/10000",
        "policy_variant": "bsp",
        "policy_protocol": "bsp_decoded_h8",
        "expected_action_horizon": 8,
    }
    for overrides in (
        {
            "bsp_cache_hash": ["c"] * 64,
            "bsp_cache_manifest_fingerprint": "d" * 64,
        },
        {
            "bsp_cache_hash": "c" * 64,
            "bsp_cache_manifest_fingerprint": tuple("d" * 64),
        },
    ):
        try:
            _formal_manifest(**dict(common, **overrides))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(overrides)


def test_manifest_to_dict_revalidates_caller_owned_mutable_containers():
    cases = []

    parameters = dict(libero_eval.BSP_PARAMETERS)
    cases.append((_formal_manifest(bsp_parameters=parameters), lambda: parameters.update(degree=3.0)))

    suites = list(libero_eval.SUPPORTED_SUITES)
    cases.append((_formal_manifest(suites=suites), suites.reverse))

    task_ids = list(range(10))
    cases.append((_formal_manifest(task_ids=task_ids), lambda: task_ids.__setitem__(0, True)))

    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    cases.append(
        (
            _formal_manifest(max_steps_by_suite=max_steps),
            lambda: max_steps.update(libero_spatial=220.0),
        )
    )

    for manifest, mutate in cases:
        mutate()
        try:
            manifest.to_dict()
        except ValueError:
            pass
        else:
            raise AssertionError("mutated caller-owned container bypassed schema-v3 validation")


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
