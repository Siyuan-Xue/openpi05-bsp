import copy
import csv
import hashlib
import importlib
import json

import pytest

from openpi_client import libero_eval
from openpi_client import libero_report
from openpi_client import libero_video_timing


_STEPS = (0, 1000, 2000, 5000, 10000)
_BASELINE_NORM = "b" * 64
_BSP_NORM = "c" * 64
_CACHE_SHA = "d" * 64
_CACHE_FINGERPRINT = "e" * 64
_CODE_SHA = "a" * 40
_CONTAINER_DIGEST = "sha256:" + "f" * 64
_CACHE_CONTENTS_SHA = "1" * 64
_BASELINE_ACTION_SHA = "2" * 64
_BSP_ACTION_SHA = "3" * 64
_VERIFICATION_FLAGS = (
    "strict_reconstruction_tolerance",
    "ground_truth_knots_nondecreasing",
    "tail_padding_valid",
    "future_segment_mapping_valid",
    "target_index_bounds_valid",
    "no_cross_episode_mapping",
    "all_frames_covered",
    "targets_match_rebuild",
    "mapping_matches_rebuild",
    "cache_contents_deterministic",
)


def _manifest(variant, step, *, family="full"):
    baseline = variant == "baseline"
    return {
        "schema_version": 3,
        "dataset_fps": 10,
        "source_demo_control_hz": 20,
        "control_freq_hz": 20,
        "video_fps": 40,
        "video_show_inference_waits": False,
        "inference_schedule": "synchronous",
        "replan_steps": 8,
        "code_sha": _CODE_SHA,
        "dataset_revision": "v2.0",
        "config_name": "pi05_libero_{}{}_h16".format(variant, "_lora" if family == "lora" else ""),
        "checkpoint_step": step,
        "bsp_cache_hash": None if baseline else _CACHE_SHA,
        "bsp_cache_manifest_fingerprint": None if baseline else _CACHE_FINGERPRINT,
        "norm_hash": _BASELINE_NORM if baseline else _BSP_NORM,
        "checkpoint": "checkpoint/{}/{}".format(variant, step),
        "container_digest": _CONTAINER_DIGEST,
        "train_seed": 42,
        "eval_seed": 42,
        "policy_variant": variant,
        "bsp_parameters": libero_eval.BSP_PARAMETERS,
        "policy_protocol": "baseline_h16" if baseline else "bsp_decoded_h8",
        "expected_action_horizon": 16 if baseline else 8,
        "execution_horizon": 8,
        "suites": list(libero_eval.SUPPORTED_SUITES),
        "task_ids": list(range(10)),
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


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def _episode(suite, task_id, init_index, success):
    fingerprint = hashlib.sha256("state-{}-{}-{}".format(suite, task_id, init_index).encode("utf-8")).hexdigest()
    identity = libero_eval.EpisodeIdentity(
        suite=suite,
        task_id=task_id,
        task_name="{} task {}".format(suite, task_id),
        init_state_index=init_index,
        init_state_fingerprint=fingerprint,
    )
    result = libero_eval.AttemptResult(
        success=success,
        steps=task_id + init_index + 1,
        replans=1,
        failure_kind=None if success else "policy",
        error=None if success else "policy output invalid",
        inference_ms=(1.0,),
        inference_requests=(
            libero_video_timing.InferenceRequest(0, 0, 1_000_000),
        ),
        control_stalls=(
            libero_video_timing.ControlStall(0, 0, 0, 1_000_000),
        ),
    )
    return libero_eval.EpisodeRecord.from_attempt(
        identity,
        42,
        1,
        success=success,
        failure_kind=None if success else "policy",
        result=result,
    )


def _write_run(path, variant, step, *, family="full"):
    path.mkdir(parents=True)
    records = []
    for suite in libero_eval.SUPPORTED_SUITES:
        for task_id in range(10):
            for init_index in range(50):
                records.append(_episode(suite, task_id, init_index, variant == "bsp"))
    _write_json(path / "manifest.json", _manifest(variant, step, family=family))
    with (path / "episodes.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    _write_json(path / "summary.json", libero_eval.aggregate_records(records))


def _bsp_diagnostics():
    return {
        **{flag: True for flag in _VERIFICATION_FLAGS},
        "verification_passed": True,
        "strict_comparison": True,
        "scipy_version": "1.15.3",
        "required_scipy_version": "1.15.3",
        "scipy_version_matches_required": True,
        "episode_count": 1693,
        "frame_count": 273465,
        "strict_max_reconstruction_error": 0.001,
        "mean_reconstruction_error": 0.0002,
        "p95_reconstruction_error": 0.0008,
        "max_error_threshold": 0.002,
        "cache_sha256": _CACHE_SHA,
        "cache_manifest_fingerprint": _CACHE_FINGERPRINT,
        "cache_contents_sha256": _CACHE_CONTENTS_SHA,
        "rebuilt_contents_sha256": _CACHE_CONTENTS_SHA,
        "code_sha": _CODE_SHA,
    }


def _norm_diagnostics():
    return {
        "state_stats_equal": True,
        "asset_directories_isolated": True,
        "action_stats_isolated": True,
        "baseline_norm_stats_sha256": _BASELINE_NORM,
        "bsp_norm_stats_sha256": _BSP_NORM,
        "baseline_action_stats_sha256": _BASELINE_ACTION_SHA,
        "bsp_action_stats_sha256": _BSP_ACTION_SHA,
        "baseline_asset_dir": "/experiments/assets/libero_baseline_h16",
        "bsp_asset_dir": "/experiments/assets/libero_bsp_h16",
        "state_fields": {field: {"equal": True} for field in ("mean", "std", "q01", "q99")},
        "rtol": 1e-7,
        "atol": 1e-8,
    }


def _write_diagnostics(root):
    bsp = root / "bsp-verification.json"
    norm = root / "norm-comparison.json"
    _write_json(bsp, _bsp_diagnostics())
    _write_json(norm, _norm_diagnostics())
    return bsp, norm


class TestLiberoPhaseOneReport:
    def test_full_20000_rollout_comparison_emits_all_five_fixed_milestone_artifacts(self, tmp_path):
        run_dirs = []
        identities = list(reversed([(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]))
        for index, (variant, step) in enumerate(identities):
            path = tmp_path / "opaque-run-{}".format(index)
            _write_run(path, variant, step)
            run_dirs.append(path)
        bsp_diagnostics, norm_diagnostics = _write_diagnostics(tmp_path)
        output_dir = tmp_path / "comparison"

        comparison = libero_report.compare_phase_one(
            run_dirs,
            bsp_diagnostics_path=bsp_diagnostics,
            norm_comparison_path=norm_diagnostics,
            output_dir=output_dir,
        )

        assert {path.name for path in output_dir.iterdir()} == {
            "task_comparison.csv",
            "suite_comparison.csv",
            "learning_curve.csv",
            "comparison.json",
            "report.md",
            "learning_curve.svg",
        }
        assert [row["checkpoint_step"] for row in comparison["milestones"]] == list(_STEPS)
        assert comparison["protocol"]["milestones"] == list(_STEPS)
        assert comparison["protocol"]["total_episodes"] == 20000
        assert comparison["protocol"]["training_family"] == "full"
        for row in comparison["milestones"]:
            assert row["bsp_minus_baseline"] == 1.0
            assert row["bootstrap_95_ci"] == [1.0, 1.0]
            assert row["bootstrap_resamples"] == 10000
            assert row["bootstrap_seed"] == 42
        for filename, count in (("task_comparison.csv", 200), ("suite_comparison.csv", 20), ("learning_curve.csv", 5)):
            with (output_dir / filename).open(newline="", encoding="utf-8") as input_file:
                assert len(list(csv.DictReader(input_file))) == count
        assert "best" not in (output_dir / "report.md").read_text(encoding="utf-8").lower()

    def test_manifest_classifier_accepts_one_lora_family_and_rejects_mixed_families(self):
        lora = [_manifest(variant, step, family="lora") for variant in ("baseline", "bsp") for step in _STEPS]

        classified = libero_report.classify_phase_one_manifests(lora)

        assert set(classified) == {(variant, step) for variant in ("baseline", "bsp") for step in _STEPS}
        assert {key: manifest["config_name"] for key, manifest in classified.items()} == {
            (variant, step): f"pi05_libero_{variant}_lora_h16" for variant in ("baseline", "bsp") for step in _STEPS
        }
        mixed = copy.deepcopy(lora)
        mixed[0]["config_name"] = "pi05_libero_baseline_h16"
        with pytest.raises(libero_report.ComparisonError, match="training family"):
            libero_report.classify_phase_one_manifests(mixed)

    def test_manifest_classifier_rejects_missing_duplicate_extra_and_wrong_protocol(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        classified = libero_report.classify_phase_one_manifests(valid)
        assert {key: manifest["config_name"] for key, manifest in classified.items()} == {
            (variant, step): f"pi05_libero_{variant}_h16" for variant in ("baseline", "bsp") for step in _STEPS
        }
        cases = []
        cases.append(valid[:-1])
        duplicate = copy.deepcopy(valid)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        cases.append(duplicate)
        extra = copy.deepcopy(valid)
        extra[-1]["checkpoint_step"] = 20000
        cases.append(extra)
        wrong = copy.deepcopy(valid)
        wrong[0]["policy_protocol"] = "baseline_h10_calibration"
        wrong[0]["expected_action_horizon"] = 10
        cases.append(wrong)
        wrong_config = copy.deepcopy(valid)
        wrong_config[0]["config_name"] = "pi05_libero"
        cases.append(wrong_config)
        for manifests in cases:
            with pytest.raises(libero_report.ComparisonError):
                libero_report.classify_phase_one_manifests(manifests)

    def test_manifest_classifier_binds_unique_checkpoint_paths_to_their_steps(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        trailing_slash = copy.deepcopy(valid)
        trailing_slash[0]["checkpoint"] += "/"
        libero_report.classify_phase_one_manifests(trailing_slash)

        duplicate_path = copy.deepcopy(valid)
        duplicate_path[3]["checkpoint"] = duplicate_path[0]["checkpoint"] + "///"
        wrong_step = copy.deepcopy(valid)
        wrong_step[0]["checkpoint"] = "checkpoint/baseline/9999"
        for manifests in (duplicate_path, wrong_step):
            with pytest.raises(libero_report.ComparisonError):
                libero_report.classify_phase_one_manifests(manifests)

    def test_manifest_requires_real_revision_hashes_finite_timeouts_and_shared_deadlines(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        mutations = (
            ("dataset_revision", "v2.1"),
            ("code_sha", "code-sha"),
            ("container_digest", "sha256:container"),
            ("norm_hash", 123),
            ("connection_timeout_s", 0.0),
            ("inference_timeout_s", float("inf")),
        )
        for field, value in mutations:
            manifests = copy.deepcopy(valid)
            manifests[0][field] = value
            with pytest.raises(libero_report.ComparisonError):
                libero_report.classify_phase_one_manifests(manifests)

        mismatched = copy.deepcopy(valid)
        mismatched[-1]["inference_timeout_s"] = 60.0
        with pytest.raises(libero_report.ComparisonError):
            libero_report.classify_phase_one_manifests(mismatched)

    def test_schema_two_is_archive_only_and_mixed_versions_are_rejected(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        for manifests in (
            [dict(manifest, schema_version=2) for manifest in valid],
            [dict(valid[0], schema_version=2), *valid[1:]],
        ):
            with pytest.raises(libero_report.ComparisonError, match="archive-only"):
                libero_report.classify_phase_one_manifests(manifests)

    def test_artifact_only_video_settings_may_differ_but_metric_clocks_must_match(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        artifact_differences = copy.deepcopy(valid)
        artifact_differences[0]["video_fps"] = 60
        artifact_differences[1]["video_show_inference_waits"] = True
        libero_report.classify_phase_one_manifests(artifact_differences)

        for field, value in (
            ("dataset_fps", 11),
            ("source_demo_control_hz", 10),
            ("control_freq_hz", 10),
            ("inference_schedule", "asynchronous"),
        ):
            changed = copy.deepcopy(valid)
            changed[0][field] = value
            with pytest.raises(libero_report.ComparisonError):
                libero_report.classify_phase_one_manifests(changed)

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            ("video_fps", 0),
            ("video_fps", 30),
            ("video_fps", True),
            ("video_show_inference_waits", 1),
        ),
    )
    def test_invalid_per_run_video_artifact_settings_are_rejected(self, field, value):
        manifests = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        manifests[0][field] = value
        with pytest.raises(libero_report.ComparisonError):
            libero_report.classify_phase_one_manifests(manifests)

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            ("schema_version", 3.0),
            ("schema_version", True),
            ("checkpoint_step", 0.0),
            ("checkpoint_step", False),
            ("expected_action_horizon", 16.0),
            ("expected_action_horizon", True),
            ("execution_horizon", 8.0),
            ("trials_per_task", 50.0),
            ("num_steps_wait", 10.0),
            ("infrastructure_retries", 2.0),
            ("train_seed", 42.0),
            ("eval_seed", True),
            ("replan_steps", 8.0),
        ),
    )
    def test_manifest_rejects_forged_non_integer_protocol_fields(self, field, value):
        manifest = _manifest("baseline", 0)
        manifest[field] = value
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

    @pytest.mark.parametrize("value", [0.0, True])
    def test_manifest_rejects_forged_task_ids_and_suite_max_steps(self, value):
        manifest = _manifest("baseline", 0)
        manifest["task_ids"] = list(manifest["task_ids"])
        manifest["task_ids"][0] = value
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

        manifest = _manifest("baseline", 0)
        manifest["max_steps_by_suite"] = dict(manifest["max_steps_by_suite"])
        manifest["max_steps_by_suite"]["libero_spatial"] = 220.0 if value == 0.0 else True
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

    @pytest.mark.parametrize(
        "suites",
        (
            {suite: None for suite in libero_eval.SUPPORTED_SUITES},
            tuple(libero_eval.SUPPORTED_SUITES),
            "".join(libero_eval.SUPPORTED_SUITES),
        ),
    )
    def test_manifest_requires_suites_to_be_a_json_list(self, suites):
        manifest = _manifest("baseline", 0)
        manifest["suites"] = suites
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

    @pytest.mark.parametrize("value", [0, True, None, ""])
    def test_manifest_requires_nonempty_supported_string_suite_items(self, value):
        manifest = _manifest("baseline", 0)
        manifest["suites"] = list(manifest["suites"])
        manifest["suites"][0] = value
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

    def test_manifest_accepts_canonical_suite_json_list(self):
        manifest = _manifest("baseline", 0)
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        assert manifest["suites"] == list(libero_eval.SUPPORTED_SUITES)
        assert libero_report._validate_manifest(manifest) == ("baseline", 0)

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            ("task_ids", tuple(range(10))),
            ("max_steps_by_suite", list(_manifest("baseline", 0)["max_steps_by_suite"].items())),
            ("bsp_parameters", list(libero_eval.BSP_PARAMETERS.items())),
        ),
    )
    def test_manifest_rejects_non_json_sequence_and_mapping_containers(self, field, value):
        manifest = _manifest("baseline", 0)
        manifest[field] = value
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_manifest(manifest)

    def test_replan_protocol_is_not_mutable_through_public_bsp_parameters(self):
        original_parameters = dict(libero_eval.BSP_PARAMETERS)
        try:
            libero_eval.BSP_PARAMETERS["executed_actions"] = 9
            reloaded_report = importlib.reload(libero_report)
            canonical = _manifest("baseline", 0)
            reloaded_report._validate_manifest(canonical)
            with pytest.raises(libero_report.ComparisonError):
                reloaded_report._validate_manifest(dict(canonical, replan_steps=9, execution_horizon=9))
        finally:
            libero_eval.BSP_PARAMETERS.clear()
            libero_eval.BSP_PARAMETERS.update(original_parameters)
            importlib.reload(libero_report)

    def test_strict_json_rejects_nan_and_truncated_input(self, tmp_path):
        nan_path = tmp_path / "nan.json"
        nan_path.write_text('{"value": NaN}\n', encoding="utf-8")
        truncated_path = tmp_path / "truncated.jsonl"
        truncated_path.write_text('{"value": 1}\n{"value":', encoding="utf-8")
        with pytest.raises(libero_report.ComparisonError):
            libero_report.load_strict_json(nan_path)
        with pytest.raises(libero_report.ComparisonError):
            libero_report.load_strict_jsonl(truncated_path)

    def test_pair_validator_rejects_duplicate_missing_or_identity_mismatch(self):
        baseline = [
            _episode("libero_spatial", 0, 0, False).to_dict(),
            _episode("libero_spatial", 0, 1, False).to_dict(),
        ]
        bsp = [
            _episode("libero_spatial", 0, 0, True).to_dict(),
            _episode("libero_spatial", 0, 1, True).to_dict(),
        ]
        libero_report.validate_paired_records(baseline, bsp)
        for invalid in (
            [bsp[0], bsp[0]],
            bsp[:1],
            [bsp[0], dict(bsp[1], task_name="different task")],
            [bsp[0], dict(bsp[1], eval_seed=7)],
        ):
            with pytest.raises(libero_report.ComparisonError):
                libero_report.validate_paired_records(baseline, invalid)

    def test_hierarchical_macro_averages_tasks_then_suites(self):
        paired = {}
        for suite_index, suite in enumerate(libero_eval.SUPPORTED_SUITES):
            for task_id in range(10):
                deltas = [1.0] * 50 if (suite_index == 0 and task_id == 0) else [0.0] * 50
                paired[(suite, task_id)] = deltas

        observed = libero_report.hierarchical_delta(paired)

        assert observed == pytest.approx(1.0 / 40.0)

    def test_bootstrap_is_reproducible_and_constant_one_has_exact_unit_interval(self):
        paired = {(suite, task_id): [1.0] * 50 for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)}

        first = libero_report.stratified_paired_bootstrap(paired)
        second = libero_report.stratified_paired_bootstrap(paired)

        assert first == second
        assert first == (1.0, 1.0)

    def test_bootstrap_is_reproducible_for_nonconstant_paired_deltas(self):
        paired = {(suite, task_id): [0.0] * 50 for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)}
        paired[("libero_spatial", 0)] = [-1.0, 1.0] * 25

        first = libero_report.stratified_paired_bootstrap(paired)
        second = libero_report.stratified_paired_bootstrap(paired)

        assert first == second
        assert first[0] < 0.0
        assert first[1] > 0.0

    def test_run_loader_rejects_artifact_errors_infrastructure_and_summary_mismatch(self, tmp_path):
        artifact_run = tmp_path / "artifact"
        _write_run(artifact_run, "baseline", 10000)
        (artifact_run / "artifact_errors.jsonl").write_text('{"error": "ffmpeg failed"}\n', encoding="utf-8")
        with pytest.raises(libero_report.ComparisonError):
            libero_report.load_run(artifact_run)

        infrastructure_run = tmp_path / "infrastructure"
        _write_run(infrastructure_run, "baseline", 10000)
        lines = (infrastructure_run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        first_record = json.loads(lines[0])
        first_record.update(
            {
                "status": "infrastructure_incomplete",
                "success": None,
                "include_in_success_rate": False,
                "infrastructure_kind": "network",
            }
        )
        lines[0] = json.dumps(first_record, allow_nan=False, sort_keys=True)
        (infrastructure_run / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(libero_report.ComparisonError):
            libero_report.load_run(infrastructure_run)

        summary_run = tmp_path / "summary"
        _write_run(summary_run, "baseline", 10000)
        summary = json.loads((summary_run / "summary.json").read_text(encoding="utf-8"))
        summary["successes"] = 1
        _write_json(summary_run / "summary.json", summary)
        with pytest.raises(libero_report.ComparisonError):
            libero_report.load_run(summary_run)

    def test_episode_validation_rejects_noncanonical_or_internally_inconsistent_records(self):
        manifest = _manifest("baseline", 10000)
        valid = _episode("libero_spatial", 0, 0, True).to_dict()
        libero_report._validate_episode(valid, manifest)

        mutations = (
            {"init_state_fingerprint": "not-a-sha"},
            {"init_state_fingerprint": "A" * 64},
            {"paired_key": "forged-pair"},
            {"episode_id": "forged-episode"},
            {"steps": 221},
            {"attempts": 0},
            {"attempts": 4},
            {"replans": 29, "inference_ms": [1.0] * 29, "mean_inference_ms": 1.0},
            {"status": "policy_failure"},
            {"failure_kind": "policy"},
            {"error": "unexpected"},
            {"inference_ms": []},
            {"mean_inference_ms": 2.0},
        )
        for mutation in mutations:
            record = dict(valid)
            record.update(mutation)
            with pytest.raises(libero_report.ComparisonError):
                libero_report._validate_episode(record, manifest)

        for missing in ("error", "mean_inference_ms"):
            record = dict(valid)
            record.pop(missing)
            with pytest.raises(libero_report.ComparisonError):
                libero_report._validate_episode(record, manifest)

        failure = _episode("libero_spatial", 0, 0, False).to_dict()
        libero_report._validate_episode(failure, manifest)
        for mutation in (
            {"failure_kind": None},
            {"failure_kind": "timeout"},
            {"error": None},
            {"status": "success"},
        ):
            record = dict(failure)
            record.update(mutation)
            with pytest.raises(libero_report.ComparisonError):
                libero_report._validate_episode(record, manifest)

    def test_episode_validation_accepts_real_retry_history_and_rejects_forged_history(self):
        manifest = _manifest("baseline", 10000)
        identity = _episode("libero_spatial", 0, 0, True).identity

        def attempt(attempt_number):
            if attempt_number == 1:
                raise libero_eval.InfrastructureFailure("network", "socket timeout")
            if attempt_number == 2:
                raise libero_eval.InfrastructureFailure("simulator", "EGL reset")
            return libero_eval.AttemptResult(
                success=True,
                steps=1,
                replans=1,
                inference_ms=(1.0,),
                inference_requests=(libero_video_timing.InferenceRequest(0, 0, 1_000_000),),
                control_stalls=(libero_video_timing.ControlStall(0, 0, 0, 1_000_000),),
            )

        valid = libero_eval.run_episode_with_retries(
            identity,
            attempt,
            eval_seed=42,
            infrastructure_retries=2,
        )
        valid = valid.to_dict()
        libero_report._validate_episode(valid, manifest)

        histories = (
            valid["infrastructure_history"][:1],
            [
                {"attempt": 2, "kind": "network", "error": "socket timeout"},
                valid["infrastructure_history"][1],
            ],
            [
                {"attempt": 1, "kind": "policy", "error": "wrong layer"},
                valid["infrastructure_history"][1],
            ],
            [
                {"attempt": 1, "kind": "network", "error": ""},
                valid["infrastructure_history"][1],
            ],
        )
        for history in histories:
            record = dict(valid, infrastructure_history=history)
            with pytest.raises(libero_report.ComparisonError):
                libero_report._validate_episode(record, manifest)

    def test_schema_three_episode_timing_rejects_forged_events(self):
        manifest = _manifest("baseline", 10000)
        valid = _episode("libero_spatial", 0, 0, True).to_dict()
        cases = []
        cases.append(dict(valid, inference_requests=None))
        wrong_clock = copy.deepcopy(valid)
        wrong_clock["inference_requests"][0]["clock"] = "absolute_monotonic_ns"
        cases.append(wrong_clock)
        bool_duration = copy.deepcopy(valid)
        bool_duration["control_stalls"][0]["duration_ns"] = True
        cases.append(bool_duration)
        bad_reason = copy.deepcopy(valid)
        bad_reason["control_stalls"][0]["reason"] = "async_action_underflow"
        cases.append(bad_reason)
        mismatch = copy.deepcopy(valid)
        mismatch["control_stalls"][0]["duration_ns"] += 1
        cases.append(mismatch)
        forged_latency = copy.deepcopy(valid)
        forged_latency["inference_ms"] = [2.0]
        forged_latency["mean_inference_ms"] = 2.0
        cases.append(forged_latency)
        extra_field = copy.deepcopy(valid)
        extra_field["inference_requests"][0]["wall_clock"] = 123
        cases.append(extra_field)
        for record in cases:
            with pytest.raises(libero_report.ComparisonError):
                libero_report._validate_episode(record, manifest)

    def test_sync_timeout_has_exact_pairs_and_policy_failure_may_record_failed_trailing_request(self):
        manifest = _manifest("baseline", 10000)
        timeout = _episode("libero_spatial", 0, 0, False).to_dict()
        timeout.update(status="timeout_failure", failure_kind="timeout", error="max steps")
        libero_report._validate_episode(timeout, manifest)

        policy_failure = _episode("libero_spatial", 0, 0, False).to_dict()
        policy_failure["inference_requests"].append(
            libero_video_timing.InferenceRequest(1, 2_000_000, 500_000).to_dict()
        )
        policy_failure["control_stalls"].append(
            libero_video_timing.ControlStall(1, 1, 2_000_000, 500_000).to_dict()
        )
        libero_report._validate_episode(policy_failure, manifest)

        success_with_extra = copy.deepcopy(policy_failure)
        success_with_extra.update(status="success", success=True, failure_kind=None, error=None)
        with pytest.raises(libero_report.ComparisonError):
            libero_report._validate_episode(success_with_extra, manifest)

    def test_diagnostics_must_match_all_variant_artifact_identities(self):
        manifests = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        bsp = _bsp_diagnostics()
        norm = _norm_diagnostics()
        libero_report.validate_diagnostics(manifests, bsp, norm)
        for payload, key, replacement in (
            (bsp, "cache_sha256", "f" * 64),
            (bsp, "verification_passed", False),
            (bsp, "strict_max_reconstruction_error", 0.002),
            (bsp, "mean_reconstruction_error", -0.1),
            (bsp, "p95_reconstruction_error", 0.0015),
            (bsp, "scipy_version_matches_required", False),
            (bsp, "required_scipy_version", "1.14.1"),
            (bsp, "code_sha", "4" * 40),
            (bsp, "cache_contents_sha256", "not-a-sha"),
            (bsp, "rebuilt_contents_sha256", "4" * 64),
            (norm, "state_stats_equal", False),
            (norm, "baseline_norm_stats_sha256", "f" * 64),
            (norm, "state_fields", {"mean": {"equal": True}}),
            (norm, "baseline_action_stats_sha256", "not-a-sha"),
            (norm, "bsp_action_stats_sha256", _BASELINE_ACTION_SHA),
            (norm, "bsp_asset_dir", "/experiments/assets/libero_baseline_h16"),
            (norm, "rtol", 1e-6),
            (norm, "atol", 1e-7),
        ):
            changed_bsp = copy.deepcopy(bsp)
            changed_norm = copy.deepcopy(norm)
            target = changed_bsp if payload is bsp else changed_norm
            target[key] = replacement
            with pytest.raises(libero_report.ComparisonError):
                libero_report.validate_diagnostics(manifests, changed_bsp, changed_norm)

        for flag in _VERIFICATION_FLAGS:
            changed = _bsp_diagnostics()
            changed[flag] = False
            with pytest.raises(libero_report.ComparisonError):
                libero_report.validate_diagnostics(manifests, changed, norm)

        for field in ("mean", "std", "q01", "q99"):
            changed = _norm_diagnostics()
            changed["state_fields"][field]["equal"] = False
            with pytest.raises(libero_report.ComparisonError):
                libero_report.validate_diagnostics(manifests, bsp, changed)

    def test_diagnostics_allow_mean_reconstruction_error_above_p95_under_the_strict_maximum(self):
        manifests = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        bsp = _bsp_diagnostics()
        bsp["mean_reconstruction_error"] = 0.0009
        bsp["p95_reconstruction_error"] = 0.0008

        result = libero_report.validate_diagnostics(manifests, bsp, _norm_diagnostics())

        assert result["mean_reconstruction_error"] == 0.0009
        assert result["p95_reconstruction_error"] == 0.0008
