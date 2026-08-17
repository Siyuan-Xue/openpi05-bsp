"""Schema-v4 LIBERO artifact producer contracts.

These tests are authored before the implementation and intentionally remain
unexecuted on this checkout.  Task 5 verification is server-only.
"""

import dataclasses
import json

import pytest

from openpi_client import libero_control_v4 as control
from openpi_client import libero_eval_v4 as evaluation
from openpi_client import libero_video_timing_v4 as timing


_SHA = "a" * 64
_OTHER_SHA = "b" * 64


def _identity(init_index=0):
    return evaluation.EpisodeIdentity(
        suite="libero_spatial",
        task_id=0,
        task_name="pick up the block",
        init_state_index=init_index,
        init_state_fingerprint="c" * 64,
    )


def _checkpoint_identity(*, bsp=False):
    return control.CheckpointIdentityV1(
        code_sha="d" * 40,
        config_name="pi05_libero",
        checkpoint_step=1000,
        checkpoint="/checkpoints/run/1000",
        container_digest="sha256:" + "e" * 64,
        norm_hash="f" * 64,
        bsp_cache_hash="1" * 64 if bsp else None,
        bsp_cache_manifest_fingerprint="2" * 64 if bsp else None,
    )


def _calibration(mode_name):
    identity = _checkpoint_identity(bsp=mode_name == "bsp_spline_async")
    return control.LatencyCalibrationV1.create(
        execution_mode=mode_name,
        checkpoint_identity_fingerprint=identity.fingerprint,
        server_metadata_fingerprint=_SHA,
        canonical_observation_identity=control.CalibrationObservationIdentityV1(
            suite="libero_spatial",
            task_id=0,
            init_state_index=0,
            init_state_fingerprint="3" * 64,
            request_fingerprint="4" * 64,
        ),
        seed_namespace="openpi-libero-calibration-v1/{}/{}".format(
            mode_name, identity.fingerprint
        ),
        bootstrap_request_fingerprint="5" * 64 if mode_name == "baseline_rtc" else None,
        warmup_request_fingerprints=["6" * 64] * 5,
        measurement_request_fingerprints=["7" * 64] * 20,
        warmup_latency_ns=[1] * 5,
        measurement_latency_ns=[25_000_000] * 20,
    )


def _manifest(mode_name="baseline_sync_n5"):
    mode = control.EXECUTION_MODES[mode_name]
    bsp = mode.policy_variant == "bsp"
    checkpoint_identity = _checkpoint_identity(bsp=bsp)
    calibration = _calibration(mode_name) if mode.asynchronous else None
    return evaluation.EvaluationManifestV4(
        schema_version=4,
        dataset_fps=10,
        source_demo_control_hz=20,
        control_freq_hz=20,
        controller_period_ns=50_000_000,
        video_fps=40,
        video_show_inference_waits=True,
        execution_mode=mode_name,
        execution_parameters=mode.to_parameters_dict(),
        latency_calibration=calibration,
        server_metadata_fingerprint=_SHA,
        code_sha=checkpoint_identity.code_sha,
        dataset_revision="v2.0",
        config_name=checkpoint_identity.config_name,
        checkpoint_step=checkpoint_identity.checkpoint_step,
        bsp_cache_hash=checkpoint_identity.bsp_cache_hash,
        bsp_cache_manifest_fingerprint=checkpoint_identity.bsp_cache_manifest_fingerprint,
        norm_hash=checkpoint_identity.norm_hash,
        checkpoint=checkpoint_identity.checkpoint,
        container_digest=checkpoint_identity.container_digest,
        train_seed=42,
        eval_seed=42,
        policy_variant=mode.policy_variant,
        bsp_parameters=evaluation.BSP_PARAMETERS,
        policy_protocol=mode.policy_protocol,
        expected_action_horizon=mode.expected_action_horizon,
        suites=evaluation.SUPPORTED_SUITES,
        task_ids=tuple(range(10)),
        trials_per_task=50,
        num_steps_wait=10,
        max_steps_by_suite=evaluation.MAX_STEPS_BY_SUITE,
        connection_timeout_s=30.0,
        inference_timeout_s=120.0,
        infrastructure_retries=2,
    )


def _initial_events(identity=None, *, execution_mode="baseline_sync_n5", outcome="success"):
    if identity is None:
        identity = _identity()
    flow_seed = evaluation.stable_replan_seed(42, identity, 0)
    disposition = "activated" if outcome == "success" else "failed"
    requests = (
        timing.RequestEventV4(
            request_id=0,
            observation_control_step=0,
            submitted_offset_ns=0,
            flow_seed=flow_seed,
            dispatch="blocking_initial",
            trigger="initial_plan",
            scheduler_context={},
            disposition=disposition,
        ),
    )
    latencies = (
        timing.LatencyEventV4(
            request_id=0,
            completed_offset_ns=25_000_000,
            duration_ns=25_000_000,
            outcome=outcome,
        ),
    )
    if outcome == "success":
        activation_context = (
            {"action_cursor": 0}
            if execution_mode.startswith("baseline")
            else {"curve_elapsed_ns": 0}
        )
        activations = (
            timing.PlanActivationV4(
                plan_id=0,
                request_id=0,
                control_step=0,
                activated_offset_ns=25_000_000,
                activation="initial",
                activation_context=activation_context,
            ),
        )
    else:
        activations = ()
    stalls = (
        timing.ControlStallV4(
            request_id=0,
            control_step=0,
            started_offset_ns=0,
            duration_ns=25_000_000,
            reason="synchronous_inference",
        ),
    )
    return requests, latencies, activations, (), stalls


def _attempt(identity=None, *, execution_mode="baseline_sync_n5", success=True):
    events = _initial_events(
        identity,
        execution_mode=execution_mode,
        outcome="success" if success else "policy_failure",
    )
    return evaluation.AttemptResultV4(
        execution_mode=execution_mode,
        success=success,
        steps=1 if success else 0,
        replans=1 if success else 0,
        episode_duration_ns=75_000_000,
        failure_kind=None if success else "policy",
        error=None if success else "malformed response",
        inference_requests=events[0],
        inference_latencies=events[1],
        plan_activations=events[2],
        action_underflows=events[3],
        control_stalls=events[4],
        replay_frames=("frame",) if success else (),
        stall_source_frames=((0, "request-frame"),),
    )


def test_manifest_has_exactly_the_34_frozen_v4_fields_for_every_mode():
    expected_fields = {
        "schema_version",
        "dataset_fps",
        "source_demo_control_hz",
        "control_freq_hz",
        "controller_period_ns",
        "video_fps",
        "video_show_inference_waits",
        "execution_mode",
        "execution_parameters",
        "latency_calibration",
        "server_metadata_fingerprint",
        "code_sha",
        "dataset_revision",
        "config_name",
        "checkpoint_step",
        "bsp_cache_hash",
        "bsp_cache_manifest_fingerprint",
        "norm_hash",
        "checkpoint",
        "container_digest",
        "train_seed",
        "eval_seed",
        "policy_variant",
        "bsp_parameters",
        "policy_protocol",
        "expected_action_horizon",
        "suites",
        "task_ids",
        "trials_per_task",
        "num_steps_wait",
        "max_steps_by_suite",
        "connection_timeout_s",
        "inference_timeout_s",
        "infrastructure_retries",
    }
    assert len(expected_fields) == 34
    assert {field.name for field in dataclasses.fields(evaluation.EvaluationManifestV4)} == expected_fields

    for mode_name in control.EXECUTION_MODES:
        manifest = _manifest(mode_name)
        payload = manifest.to_dict()
        assert set(payload) == expected_fields
        assert payload["execution_parameters"] == control.EXECUTION_MODES[
            mode_name
        ].to_parameters_dict()
        assert evaluation.EvaluationManifestV4.from_dict(payload) == manifest
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.execution_mode = "other"


def test_manifest_defensively_owns_nested_containers_and_rejects_exact_field_mutations():
    execution_parameters = control.EXECUTION_MODES["bsp_spline_sync"].to_parameters_dict()
    bsp_parameters = dict(evaluation.BSP_PARAMETERS)
    suites = list(evaluation.SUPPORTED_SUITES)
    task_ids = list(range(10))
    max_steps = dict(evaluation.MAX_STEPS_BY_SUITE)
    manifest = dataclasses.replace(
        _manifest("bsp_spline_sync"),
        execution_parameters=execution_parameters,
        bsp_parameters=bsp_parameters,
        suites=suites,
        task_ids=task_ids,
        max_steps_by_suite=max_steps,
    )
    execution_parameters["parameter_shape"][0] = 99
    bsp_parameters["degree"] = 99
    suites.reverse()
    task_ids[0] = True
    max_steps["libero_spatial"] = 1
    assert manifest.to_dict() == _manifest("bsp_spline_sync").to_dict()

    payload = _manifest().to_dict()
    mutations = []
    missing = dict(payload)
    missing.pop("controller_period_ns")
    mutations.append(missing)
    mutations.append(dict(payload, extra=True))
    mutations.append(dict(payload, checkpoint_step=True))
    mutations.append(dict(payload, connection_timeout_s=float("inf")))
    mutations.append(dict(payload, suites=tuple(payload["suites"])))
    mutations.append(dict(payload, execution_mode="unknown"))
    mutations.append(dict(payload, server_metadata_fingerprint="A" * 64))
    bsp_payload = _manifest("bsp_spline_sync").to_dict()
    bsp_payload["execution_parameters"]["parameter_shape"] = (16, 8)
    mutations.append(bsp_payload)
    for malformed in mutations:
        with pytest.raises(ValueError):
            evaluation.EvaluationManifestV4.from_dict(malformed)


def test_manifest_binds_family_protocol_cache_and_async_calibration_identity():
    for mode_name in control.EXECUTION_MODES:
        manifest = _manifest(mode_name)
        mode = control.EXECUTION_MODES[mode_name]
        assert manifest.policy_variant == mode.policy_variant
        assert manifest.policy_protocol == mode.policy_protocol
        assert manifest.expected_action_horizon == mode.expected_action_horizon
        assert (manifest.latency_calibration is not None) == mode.asynchronous

    invalid = (
        lambda: dataclasses.replace(_manifest(), policy_protocol="baseline_rtc_h16_v1"),
        lambda: dataclasses.replace(
            _manifest(), latency_calibration=_calibration("baseline_rtc")
        ),
        lambda: dataclasses.replace(
            _manifest("baseline_rtc"), server_metadata_fingerprint=_OTHER_SHA
        ),
        lambda: dataclasses.replace(_manifest("bsp_spline_sync"), bsp_cache_hash=None),
    )
    for make_manifest in invalid:
        with pytest.raises(ValueError):
            make_manifest()


def test_episode_record_round_trip_has_exact_fields_and_revalidates_stable_seeds():
    identity = _identity()
    attempt = _attempt(identity)
    record = evaluation.EpisodeRecordV4.from_attempt(
        identity,
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=attempt,
    )
    expected_fields = {
        "schema_version",
        "episode_id",
        "paired_key",
        "suite",
        "task_id",
        "task_name",
        "init_state_index",
        "init_state_fingerprint",
        "eval_seed",
        "execution_mode",
        "status",
        "success",
        "include_in_success_rate",
        "attempts",
        "failure_kind",
        "infrastructure_kind",
        "error",
        "steps",
        "replans",
        "episode_duration_ns",
        "inference_requests",
        "inference_latencies",
        "plan_activations",
        "action_underflows",
        "control_stalls",
        "infrastructure_history",
    }
    assert set(record.to_dict()) == expected_fields
    assert len(expected_fields) == 26
    assert record.replans == len(record.plan_activations) == 1
    assert evaluation.EpisodeRecordV4.from_dict(record.to_dict()) == dataclasses.replace(
        record, replay_frames=(), stall_source_frames=()
    )

    wrong_seed = record.to_dict()
    wrong_seed["inference_requests"][0]["flow_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        evaluation.EpisodeRecordV4.from_dict(wrong_seed)


def test_episode_record_rejects_missing_extra_bool_nonfinite_wrong_list_and_bad_status():
    record = evaluation.EpisodeRecordV4.from_attempt(
        _identity(),
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=_attempt(),
    )
    payload = record.to_dict()
    missing = dict(payload)
    missing.pop("steps")
    malformed = (
        missing,
        dict(payload, extra=1),
        dict(payload, attempts=True),
        dict(payload, episode_duration_ns=float("nan")),
        dict(payload, inference_requests=tuple(payload["inference_requests"])),
        dict(payload, status="complete"),
    )
    for value in malformed:
        with pytest.raises(ValueError):
            evaluation.EpisodeRecordV4.from_dict(value)


def test_retry_discards_failed_attempt_metrics_and_restarts_final_attempt_ids_and_seeds():
    identity = _identity()
    discarded = _attempt(identity)
    calls = []

    def attempt(attempt_number):
        calls.append(attempt_number)
        if attempt_number == 1:
            assert discarded.inference_requests[0].request_id == 0
            raise evaluation.InfrastructureFailure("network", "disconnect")
        return _attempt(identity)

    record = evaluation.run_episode_with_retries_v4(
        identity,
        attempt,
        eval_seed=42,
        execution_mode="baseline_sync_n5",
    )
    assert calls == [1, 2]
    assert record.attempts == 2
    assert record.inference_requests[0].request_id == 0
    assert record.inference_requests[0].flow_seed == evaluation.stable_replan_seed(42, identity, 0)
    assert record.infrastructure_history == (
        {"attempt": 1, "kind": "network", "error": "disconnect"},
    )


def test_infrastructure_exhaustion_is_denominator_ineligible_with_empty_metrics():
    def attempt(_attempt_number):
        raise evaluation.InfrastructureFailure("simulator", "reset failed")

    record = evaluation.run_episode_with_retries_v4(
        _identity(),
        attempt,
        eval_seed=42,
        execution_mode="baseline_rtc",
    )
    assert record.status == "infrastructure_incomplete"
    assert record.success is None
    assert not record.include_in_success_rate
    assert len(record.infrastructure_history) == 3
    assert record.steps == record.replans == record.episode_duration_ns == 0
    assert record.inference_requests == ()
    assert record.inference_latencies == ()
    assert record.plan_activations == ()
    assert record.action_underflows == ()
    assert record.control_stalls == ()


def test_bsp_async_episode_validation_binds_prefetch_events_to_calibrated_budget():
    identity = _identity()
    requests = (
        timing.RequestEventV4(
            0,
            0,
            0,
            evaluation.stable_replan_seed(42, identity, 0),
            "blocking_initial",
            "initial_plan",
            {},
            "activated",
        ),
        timing.RequestEventV4(
            1,
            1,
            50_000_000,
            evaluation.stable_replan_seed(42, identity, 1),
            "background",
            "bsp_prefetch",
            {"remaining_plan_ns": 50_000_000, "budget_ns": 50_000_000},
            "activated",
        ),
    )
    result = evaluation.AttemptResultV4(
        execution_mode="bsp_spline_async",
        success=True,
        steps=2,
        replans=2,
        episode_duration_ns=100_000_000,
        inference_requests=requests,
        inference_latencies=(
            timing.LatencyEventV4(0, 25_000_000, 25_000_000, "success"),
            timing.LatencyEventV4(1, 75_000_000, 25_000_000, "success"),
        ),
        plan_activations=(
            timing.PlanActivationV4(
                0, 0, 0, 25_000_000, "initial", {"curve_elapsed_ns": 0}
            ),
            timing.PlanActivationV4(
                1, 1, 1, 75_000_000, "immediate_swap", {"curve_elapsed_ns": 0}
            ),
        ),
        control_stalls=(
            timing.ControlStallV4(
                0, 0, 0, 25_000_000, "synchronous_inference"
            ),
        ),
    )
    record = evaluation.EpisodeRecordV4.from_attempt(
        identity,
        42,
        1,
        execution_mode="bsp_spline_async",
        result=result,
        expected_bsp_prefetch_budget_ns=50_000_000,
    )
    assert evaluation.EpisodeRecordV4.from_dict(
        record.to_dict(), expected_bsp_prefetch_budget_ns=50_000_000
    ).to_dict() == record.to_dict()
    with pytest.raises(ValueError, match="budget"):
        evaluation.EpisodeRecordV4.from_dict(
            record.to_dict(), expected_bsp_prefetch_budget_ns=100_000_000
        )


def test_aggregate_excludes_infrastructure_and_writer_emits_only_v4_artifacts(tmp_path):
    success = evaluation.EpisodeRecordV4.from_attempt(
        _identity(0),
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=_attempt(_identity(0)),
    )
    failed = evaluation.EpisodeRecordV4.from_attempt(
        _identity(1),
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=_attempt(_identity(1), success=False),
    )
    incomplete = evaluation.EpisodeRecordV4.infrastructure_incomplete(
        _identity(2),
        42,
        3,
        execution_mode="baseline_sync_n5",
        kind="network",
        error="offline",
        infrastructure_history=(
            {"attempt": 1, "kind": "network", "error": "offline"},
            {"attempt": 2, "kind": "network", "error": "offline"},
            {"attempt": 3, "kind": "network", "error": "offline"},
        ),
    )
    artifact_error = evaluation.ArtifactErrorV4(
        episode_id=success.episode_id,
        artifact_type="video",
        path="videos/example.mp4",
        error="encode failed",
    )
    records = (success, failed, incomplete)
    summary = evaluation.aggregate_records_v4(records, artifact_errors=(artifact_error,))
    assert summary["requested_episodes"] == 3
    assert summary["eligible_episodes"] == 2
    assert summary["successes"] == 1
    assert summary["incomplete_infrastructure_count"] == 1
    assert summary["artifact_error_count"] == 1
    assert not summary["acceptance_complete"]
    assert evaluation.ArtifactErrorV4.from_dict(artifact_error.to_dict()) == artifact_error

    writer = evaluation.ArtifactWriterV4(tmp_path)
    writer.write_manifest(_manifest())
    for record in records:
        writer.append_episode(record)
    writer.append_artifact_error(artifact_error)
    persisted_summary = writer.write_summary(records, artifact_errors=(artifact_error,))
    assert persisted_summary == summary
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 4
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "artifact_errors.jsonl").read_text())["artifact_type"] == "video"
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert (tmp_path / "video_audit.jsonl").read_text() == ""
    assert not (tmp_path / "tasks.csv").exists()
    assert not (tmp_path / "suites.csv").exists()


def test_video_selector_claims_only_the_first_success_and_counted_failure(tmp_path):
    success = evaluation.EpisodeRecordV4.from_attempt(
        _identity(0),
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=_attempt(_identity(0)),
    )
    failure = evaluation.EpisodeRecordV4.from_attempt(
        _identity(1),
        42,
        1,
        execution_mode="baseline_sync_n5",
        result=_attempt(_identity(1), success=False),
    )
    selector = evaluation.VideoSelectorV4(tmp_path)
    assert selector.claim(success).name.startswith("success-")
    assert selector.claim(success) is None
    assert selector.claim(failure).name.startswith("failure-")
    assert selector.claim(failure) is None
