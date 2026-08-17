from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence

import pytest

from openpi_client import libero_control_v4
from openpi_client import libero_eval_v4
from openpi_client import libero_report
from openpi_client import libero_report_v4


_MODES = (
    "baseline_sync_n5",
    "baseline_rtc",
    "bsp_spline_sync",
    "bsp_spline_async",
)
_STEPS = (0, 1000, 2000, 5000, 10000)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint_identity(
    mode: str,
    step: int,
) -> libero_control_v4.CheckpointIdentityV1:
    family = "baseline" if mode.startswith("baseline_") else "bsp"
    bsp_hash = _sha("bsp-cache") if family == "bsp" else None
    bsp_manifest = _sha("bsp-cache-manifest") if family == "bsp" else None
    return libero_control_v4.CheckpointIdentityV1(
        code_sha="a" * 40,
        config_name="{}_config".format(family),
        checkpoint_step=step,
        checkpoint="/checkpoints/{}/{}".format(family, step),
        container_digest="sha256:" + "b" * 64,
        norm_hash=_sha("{}-norm".format(family)),
        bsp_cache_hash=bsp_hash,
        bsp_cache_manifest_fingerprint=bsp_manifest,
    )


def _calibration(
    mode: str,
    checkpoint: libero_control_v4.CheckpointIdentityV1,
) -> libero_control_v4.LatencyCalibrationV1:
    return libero_control_v4.LatencyCalibrationV1.create(
        execution_mode=mode,
        checkpoint_identity_fingerprint=checkpoint.fingerprint,
        server_metadata_fingerprint=_sha("server-metadata"),
        canonical_observation_identity=(
            libero_control_v4.CalibrationObservationIdentityV1(
                suite="libero_spatial",
                task_id=0,
                init_state_index=0,
                init_state_fingerprint=_sha("calibration-state"),
                request_fingerprint=_sha("calibration-observation"),
            )
        ),
        seed_namespace="openpi-libero-calibration-v1/{}/{}".format(
            mode,
            checkpoint.fingerprint,
        ),
        bootstrap_request_fingerprint=(
            _sha("bootstrap") if mode == "baseline_rtc" else None
        ),
        warmup_request_fingerprints=[
            _sha("warmup-{}".format(index)) for index in range(5)
        ],
        measurement_request_fingerprints=[
            _sha("measurement-{}".format(index)) for index in range(20)
        ],
        warmup_latency_ns=[100 + index for index in range(5)],
        measurement_latency_ns=[1_000 + index for index in range(20)],
    )


def _manifest(mode: str, step: int = 1000) -> Dict[str, Any]:
    spec = libero_control_v4.EXECUTION_MODES[mode]
    checkpoint = _checkpoint_identity(mode, step)
    calibration = _calibration(mode, checkpoint) if spec.asynchronous else None
    return {
        "schema_version": 4,
        "dataset_fps": 10,
        "source_demo_control_hz": 20,
        "control_freq_hz": 20,
        "controller_period_ns": 50_000_000,
        "video_fps": 40,
        "video_show_inference_waits": True,
        "execution_mode": mode,
        "execution_parameters": spec.to_parameters_dict(),
        "latency_calibration": (
            calibration.to_dict() if calibration is not None else None
        ),
        "server_metadata_fingerprint": _sha("server-metadata"),
        "code_sha": checkpoint.code_sha,
        "dataset_revision": "v2.0",
        "config_name": checkpoint.config_name,
        "checkpoint_step": step,
        "bsp_cache_hash": checkpoint.bsp_cache_hash,
        "bsp_cache_manifest_fingerprint": checkpoint.bsp_cache_manifest_fingerprint,
        "norm_hash": checkpoint.norm_hash,
        "checkpoint": checkpoint.checkpoint,
        "container_digest": checkpoint.container_digest,
        "train_seed": 42,
        "eval_seed": 42,
        "policy_variant": spec.policy_variant,
        "bsp_parameters": dict(libero_eval_v4.BSP_PARAMETERS),
        "policy_protocol": spec.policy_protocol,
        "expected_action_horizon": spec.expected_action_horizon,
        "suites": list(libero_eval_v4.SUPPORTED_SUITES),
        "task_ids": list(range(10)),
        "trials_per_task": 50,
        "num_steps_wait": 10,
        "max_steps_by_suite": dict(libero_eval_v4.MAX_STEPS_BY_SUITE),
        "connection_timeout_s": 10.0,
        "inference_timeout_s": 30.0,
        "infrastructure_retries": 2,
    }


def _identity(suite: str, task_id: int, init_state_index: int) -> Dict[str, Any]:
    fingerprint = _sha("{}:{}:{}".format(suite, task_id, init_state_index))
    paired_key = "{}/task-{:03d}/init-{:03d}/{}".format(
        suite,
        task_id,
        init_state_index,
        fingerprint,
    )
    return {
        "episode_id": "{}-task-{:03d}-init-{:03d}-{}".format(
            suite,
            task_id,
            init_state_index,
            fingerprint,
        ),
        "paired_key": paired_key,
        "init_state_fingerprint": fingerprint,
    }


def _flow_seed(
    suite: str,
    task_id: int,
    init_state_index: int,
    fingerprint: str,
) -> int:
    payload = json.dumps(
        {
            "namespace": "openpi-libero-flow-noise-v1",
            "eval_seed": 42,
            "suite": suite,
            "task_id": task_id,
            "init_state_index": init_state_index,
            "init_state_fingerprint": fingerprint,
            "replan_index": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:4],
        "big",
        signed=False,
    )


def _episode(
    mode: str,
    suite: str,
    task_id: int,
    init_state_index: int,
) -> Dict[str, Any]:
    identity = _identity(suite, task_id, init_state_index)
    request = {
        "clock": "episode_monotonic_ns",
        "request_id": 0,
        "observation_control_step": 0,
        "submitted_offset_ns": 0,
        "flow_seed": _flow_seed(
            suite,
            task_id,
            init_state_index,
            identity["init_state_fingerprint"],
        ),
        "dispatch": "blocking_initial",
        "trigger": "initial_plan",
        "scheduler_context": {},
        "disposition": "failed",
    }
    return {
        "schema_version": 4,
        **identity,
        "suite": suite,
        "task_id": task_id,
        "task_name": "{} task {}".format(suite, task_id),
        "init_state_index": init_state_index,
        "eval_seed": 42,
        "execution_mode": mode,
        "status": "policy_failure",
        "success": False,
        "include_in_success_rate": True,
        "attempts": 1,
        "failure_kind": "policy",
        "infrastructure_kind": None,
        "error": "policy rejected request",
        "steps": 0,
        "replans": 0,
        "episode_duration_ns": 0,
        "inference_requests": [request],
        "inference_latencies": [
            {
                "clock": "episode_monotonic_ns",
                "request_id": 0,
                "completed_offset_ns": 0,
                "duration_ns": 0,
                "outcome": "policy_failure",
            }
        ],
        "plan_activations": [],
        "action_underflows": [],
        "control_stalls": [
            {
                "clock": "episode_monotonic_ns",
                "request_id": 0,
                "control_step": 0,
                "started_offset_ns": 0,
                "duration_ns": 0,
                "reason": "synchronous_inference",
            }
        ],
        "infrastructure_history": [],
    }


def _video_audit(episode: Mapping[str, Any]) -> Dict[str, Any]:
    planned = {
        "control_hz": 20,
        "video_fps": 40,
        "control_frame_count": 0,
        "held_frame_count": 0,
        "request_count": 1,
        "latency_count": 1,
        "activation_count": 0,
        "underflow_count": 0,
        "total_request_latency_ns": 0,
        "total_underflow_ns": 0,
        "measured_stall_count": 1,
        "measured_control_stall_ns": 0,
        "included_stall_count": 1,
        "included_control_stall_ns": 0,
        "included_stall_reasons": ["synchronous_inference"],
        "included_stall_frame_counts": [0],
        "stall_frame_count": 0,
        "video_frame_count": 0,
        "control_duration_ns": 0,
        "video_duration_ns": 0,
        "expected_duration_ns": 0,
        "duration_deviation_ns": 0,
    }
    return {
        "schema_version": 4,
        "episode_id": episode["episode_id"],
        "execution_mode": episode["execution_mode"],
        "path": "videos/{}.mp4".format(episode["episode_id"]),
        "video_show_inference_waits": True,
        "planned": planned,
        "encoded_fps": 40.0,
        "encoded_frame_count": 1,
        "encoded_duration_ns": 25_000_000,
        "artifact_padding_frame_count": 1,
        "timing_gate_pass": True,
        "warning": None,
    }


def _summary() -> Dict[str, Any]:
    tasks = []
    for suite in sorted(libero_eval_v4.SUPPORTED_SUITES):
        for task_id in range(10):
            tasks.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": "{} task {}".format(suite, task_id),
                    "requested_episodes": 50,
                    "eligible_episodes": 50,
                    "successes": 0,
                    "failures": 50,
                    "incomplete_infrastructure_count": 0,
                    "success_rate": 0.0,
                }
            )
    suites = [
        {
            "suite": suite,
            "tasks": 10,
            "requested_episodes": 500,
            "eligible_episodes": 500,
            "successes": 0,
            "failures": 500,
            "incomplete_infrastructure_count": 0,
            "success_rate": 0.0,
            "task_macro_success_rate": 0.0,
        }
        for suite in sorted(libero_eval_v4.SUPPORTED_SUITES)
    ]
    return {
        "tasks": tasks,
        "suites": suites,
        "suite_macro_success_rate": 0.0,
        "four_suite_macro_success_rate": 0.0,
        "evaluated_suite_count": 4,
        "all_four_suites_evaluated": True,
        "requested_episodes": 2000,
        "eligible_episodes": 2000,
        "successes": 0,
        "incomplete_infrastructure_count": 0,
        "artifact_error_count": 0,
        "acceptance_complete": True,
    }


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


def _write_formal_run(root: Path, mode: str, step: int = 1000) -> Path:
    root.mkdir(parents=True)
    episodes = [
        _episode(mode, suite, task_id, init_state_index)
        for suite in libero_eval_v4.SUPPORTED_SUITES
        for task_id in range(10)
        for init_state_index in range(50)
    ]
    (root / "manifest.json").write_text(
        _json_text(_manifest(mode, step)),
        encoding="utf-8",
    )
    (root / "episodes.jsonl").write_text(
        "".join(_json_text(episode) for episode in episodes),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(_json_text(_summary()), encoding="utf-8")
    (root / "video_audit.jsonl").write_text(
        "".join(_json_text(_video_audit(episode)) for episode in episodes),
        encoding="utf-8",
    )
    (root / "artifact_errors.jsonl").write_text("", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def formal_baseline_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_formal_run(
        tmp_path_factory.mktemp("v4-formal") / "baseline",
        "baseline_sync_n5",
    )


def _copy_run(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(source, target)
    return target


def _mutate_jsonl(path: Path, index: int, mutate: Any) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[index])
    mutate(payload)
    lines[index] = _json_text(payload).rstrip("\n")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_run_v4_validates_formal_grid_and_hashes_all_five_artifacts(
    formal_baseline_run: Path,
) -> None:
    run = libero_report_v4.load_run_v4(formal_baseline_run)

    assert len(run.records) == 2000
    assert len(run.video_audits) == 2000
    assert set(run.file_sha256) == {
        "manifest.json",
        "episodes.jsonl",
        "summary.json",
        "video_audit.jsonl",
        "artifact_errors.jsonl",
    }
    for name, digest in run.file_sha256.items():
        assert digest == hashlib.sha256(
            (formal_baseline_run / name).read_bytes()
        ).hexdigest()


@pytest.mark.parametrize(
    "name",
    [
        "manifest.json",
        "episodes.jsonl",
        "summary.json",
        "video_audit.jsonl",
        "artifact_errors.jsonl",
    ],
)
def test_load_run_v4_rejects_each_missing_artifact(
    formal_baseline_run: Path,
    tmp_path: Path,
    name: str,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    (run_dir / name).unlink()

    with pytest.raises(
        libero_report_v4.ComparisonErrorV4,
        match="missing required artifacts",
    ):
        libero_report_v4.load_run_v4(run_dir)


@pytest.mark.parametrize("schema_version", [2, 3])
def test_new_loader_rejects_legacy_schemas(
    formal_baseline_run: Path,
    tmp_path: Path,
    schema_version: int,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    (run_dir / "manifest.json").write_text(_json_text(manifest), encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="schema"):
        libero_report_v4.load_run_v4(run_dir)


def test_legacy_loader_rejects_schema_v4(formal_baseline_run: Path) -> None:
    with pytest.raises(libero_report.ComparisonError, match="schema"):
        libero_report.load_run(formal_baseline_run)


def test_load_run_v4_rejects_duplicate_json_keys(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "manifest.json"
    original = path.read_text(encoding="utf-8")
    path.write_text('{"schema_version":4,' + original[1:], encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="duplicate JSON key"):
        libero_report_v4.load_run_v4(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("dataset_fps"), "fields mismatch"),
        (lambda value: value.__setitem__("unexpected", 1), "fields mismatch"),
        (lambda value: value.__setitem__("checkpoint_step", True), "integer"),
        (lambda value: value.__setitem__("suites", "libero_spatial"), "JSON list"),
        (lambda value: value.__setitem__("norm_hash", "not-a-hash"), "SHA256"),
    ],
)
def test_load_run_v4_rejects_manifest_shape_and_scalar_mutations(
    formal_baseline_run: Path,
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    path.write_text(_json_text(manifest), encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match=message):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_nonfinite_json_numbers(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["connection_timeout_s"] = float("nan")
    path.write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")

    with pytest.raises(
        libero_report_v4.ComparisonErrorV4,
        match="non-standard numeric",
    ):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_duplicate_formal_grid_cell(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "episodes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="duplicate|0..49"):
        libero_report_v4.load_run_v4(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("episode_id", "unknown-episode"), "unknown"),
        (
            lambda value: value["planned"].__setitem__("request_count", 2),
            "planned timing",
        ),
        (lambda value: value.__setitem__("encoded_fps", 39.0), "encoded_fps"),
        (lambda value: value.__setitem__("encoded_frame_count", 2), "frame count"),
    ],
)
def test_load_run_v4_rejects_invalid_video_audit(
    formal_baseline_run: Path,
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    _mutate_jsonl(run_dir / "video_audit.jsonl", 0, mutation)

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match=message):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_duplicate_video_episode(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "video_audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="duplicate video"):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_episode_without_video_audit(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "video_audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="exactly one"):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_artifact_errors(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    (run_dir / "artifact_errors.jsonl").write_text(
        _json_text(
            {
                "episode_id": "episode",
                "artifact_type": "video",
                "path": "video.mp4",
                "error": "encode failed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="artifact errors"):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_infrastructure_incomplete_episode(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)

    def make_incomplete(value: Dict[str, Any]) -> None:
        value.update(
            {
                "status": "infrastructure_incomplete",
                "success": None,
                "include_in_success_rate": False,
                "attempts": 3,
                "failure_kind": None,
                "infrastructure_kind": "network",
                "error": "network unavailable",
                "inference_requests": [],
                "inference_latencies": [],
                "control_stalls": [],
                "infrastructure_history": [
                    {
                        "attempt": index,
                        "kind": "network",
                        "error": "network unavailable",
                    }
                    for index in (1, 2, 3)
                ],
            }
        )

    _mutate_jsonl(run_dir / "episodes.jsonl", 0, make_incomplete)

    with pytest.raises(
        libero_report_v4.ComparisonErrorV4,
        match="infrastructure-incomplete",
    ):
        libero_report_v4.load_run_v4(run_dir)


def test_load_run_v4_rejects_summary_mismatch(
    formal_baseline_run: Path,
    tmp_path: Path,
) -> None:
    run_dir = _copy_run(formal_baseline_run, tmp_path)
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["successes"] = 1
    path.write_text(_json_text(summary), encoding="utf-8")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="summary.json"):
        libero_report_v4.load_run_v4(run_dir)


def test_classify_checkpoint_accepts_two_distinct_family_paths() -> None:
    classified = libero_report_v4.classify_checkpoint_manifests_v4(
        [_manifest(mode) for mode in _MODES]
    )

    assert tuple(classified) == _MODES
    assert classified["baseline_sync_n5"]["checkpoint"] == "/checkpoints/baseline/1000"
    assert classified["bsp_spline_async"]["checkpoint"] == "/checkpoints/bsp/1000"


@pytest.mark.parametrize(
    ("manifests", "message"),
    [
        ([_manifest(mode) for mode in _MODES[:-1]], "exactly four"),
        (
            [_manifest(mode) for mode in _MODES] + [_manifest(_MODES[0])],
            "exactly four",
        ),
        (
            [
                _manifest(mode, 2000 if mode == "bsp_spline_async" else 1000)
                for mode in _MODES
            ],
            "checkpoint_step",
        ),
    ],
)
def test_classify_checkpoint_rejects_wrong_mode_or_step_sets(
    manifests: Sequence[Mapping[str, Any]],
    message: str,
) -> None:
    with pytest.raises(libero_report_v4.ComparisonErrorV4, match=message):
        libero_report_v4.classify_checkpoint_manifests_v4(manifests)


def test_classify_checkpoint_rejects_duplicate_mode() -> None:
    manifests = [_manifest(mode) for mode in _MODES]
    manifests[-1] = _manifest("bsp_spline_sync")

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="duplicate|mode"):
        libero_report_v4.classify_checkpoint_manifests_v4(manifests)


def test_classify_checkpoint_rejects_cross_pair_checkpoint_mismatch() -> None:
    manifests = [_manifest(mode) for mode in _MODES]
    manifests[1] = copy.deepcopy(manifests[1])
    manifests[1]["checkpoint"] = "/checkpoints/other/1000"
    checkpoint = libero_control_v4.CheckpointIdentityV1(
        code_sha=manifests[1]["code_sha"],
        config_name=manifests[1]["config_name"],
        checkpoint_step=manifests[1]["checkpoint_step"],
        checkpoint=manifests[1]["checkpoint"],
        container_digest=manifests[1]["container_digest"],
        norm_hash=manifests[1]["norm_hash"],
        bsp_cache_hash=None,
        bsp_cache_manifest_fingerprint=None,
    )
    manifests[1]["latency_calibration"] = _calibration(
        "baseline_rtc",
        checkpoint,
    ).to_dict()

    with pytest.raises(
        libero_report_v4.ComparisonErrorV4,
        match="baseline family identity",
    ):
        libero_report_v4.classify_checkpoint_manifests_v4(manifests)


def test_classify_checkpoint_rejects_rollout_seed_mismatch() -> None:
    manifests = [_manifest(mode) for mode in _MODES]
    manifests[-1] = copy.deepcopy(manifests[-1])
    manifests[-1]["eval_seed"] = 43

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="rollout identity"):
        libero_report_v4.classify_checkpoint_manifests_v4(manifests)


def test_classify_five_checkpoint_manifests_groups_exact_twenty() -> None:
    manifests = [_manifest(mode, step) for step in _STEPS for mode in _MODES]

    classified = libero_report_v4.classify_five_checkpoint_manifests_v4(manifests)

    assert tuple(classified) == _STEPS
    assert all(tuple(group) == _MODES for group in classified.values())


@pytest.mark.parametrize("drop", [True, False])
def test_classify_five_checkpoint_manifests_rejects_wrong_count(drop: bool) -> None:
    manifests = [_manifest(mode, step) for step in _STEPS for mode in _MODES]
    if drop:
        manifests.pop()
    else:
        manifests.append(_manifest("baseline_sync_n5", 10000))

    with pytest.raises(libero_report_v4.ComparisonErrorV4, match="exactly 20"):
        libero_report_v4.classify_five_checkpoint_manifests_v4(manifests)


def _run_data(
    mode: str,
    step: int = 1000,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> libero_report_v4.RunDataV4:
    if records is None:
        records = tuple(
            _episode(mode, suite, task_id, init_state_index)
            for suite in libero_eval_v4.SUPPORTED_SUITES
            for task_id in range(10)
            for init_state_index in range(50)
        )
    return libero_report_v4.RunDataV4(
        path=Path("/runs/{}/{}".format(mode, step)),
        manifest=_manifest(mode, step),
        records=records,
        summary=_summary(),
        video_audits=(),
        file_sha256={"manifest.json": _sha("{}:{}".format(mode, step))},
    )


@pytest.fixture(scope="module")
def runs_by_mode() -> Mapping[str, libero_report_v4.RunDataV4]:
    return {mode: _run_data(mode) for mode in _MODES}


def test_compare_checkpoint_emits_four_rates_and_only_two_primary_deltas(
    runs_by_mode: Mapping[str, libero_report_v4.RunDataV4],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(libero_report, "BOOTSTRAP_RESAMPLES", 5)

    comparison = libero_report_v4.compare_checkpoint_v4(runs_by_mode)

    assert comparison["success_rates"] == {mode: 0.0 for mode in _MODES}
    assert set(comparison["primary_paired_deltas"]) == {
        "baseline_rtc_minus_baseline_sync_n5",
        "bsp_spline_async_minus_bsp_spline_sync",
    }
    assert (
        comparison["diagnostics"]["baseline_sync_n5"][
            "calibration_p95_latency_ns"
        ]
        is None
    )
    assert (
        comparison["diagnostics"]["baseline_rtc"]["calibration_p95_latency_ns"]
        == 1018
    )
    assert "inference_ms_p95" not in comparison["diagnostics"]["baseline_rtc"]


@pytest.mark.parametrize("field", ["init_state_fingerprint", "eval_seed"])
def test_compare_checkpoint_rejects_cross_mode_rollout_identity_mismatch(
    runs_by_mode: Mapping[str, libero_report_v4.RunDataV4],
    field: str,
) -> None:
    changed = dict(runs_by_mode)
    records = list(changed["bsp_spline_async"].records)
    records[0] = dict(records[0])
    records[0][field] = (
        _sha("different-state") if field == "init_state_fingerprint" else 43
    )
    changed["bsp_spline_async"] = dataclasses.replace(
        changed["bsp_spline_async"],
        records=tuple(records),
    )

    with pytest.raises(
        libero_report_v4.ComparisonErrorV4,
        match="rollout identity",
    ):
        libero_report_v4.compare_checkpoint_v4(changed)


def test_write_five_checkpoint_report_uses_only_v4_suffixed_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(libero_report, "BOOTSTRAP_RESAMPLES", 1)
    mode_records = {mode: _run_data(mode).records for mode in _MODES}
    runs = [
        _run_data(mode, step, records=mode_records[mode])
        for step in _STEPS
        for mode in _MODES
    ]

    report = libero_report_v4.write_five_checkpoint_report_v4(runs, output_dir=tmp_path)

    assert [row["checkpoint_step"] for row in report["checkpoints"]] == list(_STEPS)
    filenames = {path.name for path in tmp_path.iterdir()}
    assert filenames == set(libero_report_v4.OUTPUT_FILENAMES_V4)
    assert filenames
    assert all(Path(name).stem.endswith("_v4") for name in filenames)
    assert not filenames.intersection(libero_report.OUTPUT_FILENAMES)


def test_write_report_accepts_one_four_run_checkpoint(
    runs_by_mode: Mapping[str, libero_report_v4.RunDataV4],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(libero_report, "BOOTSTRAP_RESAMPLES", 1)

    report = libero_report_v4.write_five_checkpoint_report_v4(
        list(runs_by_mode.values()),
        output_dir=tmp_path,
    )

    assert [row["checkpoint_step"] for row in report["checkpoints"]] == [1000]
    assert {path.name for path in tmp_path.iterdir()} == set(
        libero_report_v4.OUTPUT_FILENAMES_V4
    )
