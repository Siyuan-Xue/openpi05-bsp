import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from openpi_client import libero_eval
from openpi_client import libero_report


_STEPS = (10000, 20000, 30000)
_BASELINE_NORM = "b" * 64
_BSP_NORM = "c" * 64
_CACHE_SHA = "d" * 64
_CACHE_FINGERPRINT = "e" * 64


def _manifest(variant, step):
    baseline = variant == "baseline"
    return {
        "schema_version": 2,
        "native_control_hz": 10,
        "replan_steps": 8,
        "code_sha": "code-sha",
        "dataset_revision": "v2.1",
        "config_name": "pi05_libero_baseline_h16" if baseline else "pi05_libero_bsp_h16",
        "checkpoint_step": step,
        "bsp_cache_hash": None if baseline else _CACHE_SHA,
        "bsp_cache_manifest_fingerprint": None if baseline else _CACHE_FINGERPRINT,
        "norm_hash": _BASELINE_NORM if baseline else _BSP_NORM,
        "checkpoint": "checkpoint/{}/{}".format(variant, step),
        "container_digest": "sha256:container",
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


def _episode(suite, task_id, init_index, success):
    fingerprint = "state-{}-{}-{}".format(suite, task_id, init_index)
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
        inference_ms=(1.0, 2.0),
    )
    return libero_eval.EpisodeRecord.from_attempt(
        identity,
        42,
        1,
        success=success,
        failure_kind=None if success else "policy",
        result=result,
    )


def _write_run(path, variant, step):
    path.mkdir(parents=True)
    records = []
    for suite in libero_eval.SUPPORTED_SUITES:
        for task_id in range(10):
            for init_index in range(50):
                records.append(_episode(suite, task_id, init_index, variant == "bsp"))
    (path / "manifest.json").write_text(
        json.dumps(_manifest(variant, step), allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (path / "episodes.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    (path / "summary.json").write_text(
        json.dumps(libero_eval.aggregate_records(records), allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_diagnostics(root):
    bsp = root / "bsp-verification.json"
    bsp.write_text(
        json.dumps(
            {
                "verification_passed": True,
                "strict_comparison": True,
                "scipy_version": "1.15.3",
                "episode_count": 1693,
                "frame_count": 273465,
                "strict_max_reconstruction_error": 0.001,
                "mean_reconstruction_error": 0.0002,
                "p95_reconstruction_error": 0.0008,
                "max_error_threshold": 0.002,
                "cache_sha256": _CACHE_SHA,
                "cache_manifest_fingerprint": _CACHE_FINGERPRINT,
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    norm = root / "norm-comparison.json"
    norm.write_text(
        json.dumps(
            {
                "state_stats_equal": True,
                "asset_directories_isolated": True,
                "action_stats_isolated": True,
                "baseline_norm_stats_sha256": _BASELINE_NORM,
                "bsp_norm_stats_sha256": _BSP_NORM,
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bsp, norm


class LiberoPhaseOneReportTest(unittest.TestCase):
    def test_full_12000_rollout_comparison_emits_exactly_six_fixed_milestone_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = []
            # Deliberately shuffled and opaque: classification must come only from manifests.
            identities = [
                ("bsp", 20000),
                ("baseline", 10000),
                ("bsp", 30000),
                ("baseline", 30000),
                ("bsp", 10000),
                ("baseline", 20000),
            ]
            for index, (variant, step) in enumerate(identities):
                path = root / "opaque-run-{}".format(index)
                _write_run(path, variant, step)
                run_dirs.append(path)
            bsp_diagnostics, norm_diagnostics = _write_diagnostics(root)
            output_dir = root / "comparison"

            comparison = libero_report.compare_phase_one(
                run_dirs,
                bsp_diagnostics_path=bsp_diagnostics,
                norm_comparison_path=norm_diagnostics,
                output_dir=output_dir,
            )

            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "task_comparison.csv",
                    "suite_comparison.csv",
                    "learning_curve.csv",
                    "comparison.json",
                    "report.md",
                    "learning_curve.svg",
                },
            )
            self.assertEqual([row["checkpoint_step"] for row in comparison["milestones"]], list(_STEPS))
            for row in comparison["milestones"]:
                self.assertEqual(row["bsp_minus_baseline"], 1.0)
                self.assertEqual(row["bootstrap_95_ci"], [1.0, 1.0])
                self.assertEqual(row["bootstrap_resamples"], 10000)
                self.assertEqual(row["bootstrap_seed"], 42)
            with (output_dir / "task_comparison.csv").open(newline="", encoding="utf-8") as input_file:
                self.assertEqual(len(list(csv.DictReader(input_file))), 120)
            with (output_dir / "suite_comparison.csv").open(newline="", encoding="utf-8") as input_file:
                self.assertEqual(len(list(csv.DictReader(input_file))), 12)
            with (output_dir / "learning_curve.csv").open(newline="", encoding="utf-8") as input_file:
                self.assertEqual(len(list(csv.DictReader(input_file))), 3)
            self.assertNotIn("best", (output_dir / "report.md").read_text(encoding="utf-8").lower())

    def test_manifest_classifier_rejects_missing_duplicate_extra_and_wrong_protocol(self):
        valid = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        cases = []
        cases.append(valid[:-1])
        duplicate = copy.deepcopy(valid)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        cases.append(duplicate)
        extra = copy.deepcopy(valid)
        extra[-1]["checkpoint_step"] = 40000
        cases.append(extra)
        wrong = copy.deepcopy(valid)
        wrong[0]["policy_protocol"] = "baseline_h10_calibration"
        wrong[0]["expected_action_horizon"] = 10
        cases.append(wrong)
        wrong_config = copy.deepcopy(valid)
        wrong_config[0]["config_name"] = "pi05_libero"
        cases.append(wrong_config)
        for manifests in cases:
            with self.subTest(case=cases.index(manifests)), self.assertRaises(libero_report.ComparisonError):
                libero_report.classify_phase_one_manifests(manifests)

    def test_strict_json_rejects_nan_and_truncated_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nan_path = root / "nan.json"
            nan_path.write_text('{"value": NaN}\n', encoding="utf-8")
            truncated_path = root / "truncated.jsonl"
            truncated_path.write_text('{"value": 1}\n{"value":', encoding="utf-8")
            with self.assertRaises(libero_report.ComparisonError):
                libero_report.load_strict_json(nan_path)
            with self.assertRaises(libero_report.ComparisonError):
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
            with self.subTest(invalid=invalid), self.assertRaises(libero_report.ComparisonError):
                libero_report.validate_paired_records(baseline, invalid)

    def test_hierarchical_macro_averages_tasks_then_suites(self):
        paired = {}
        for suite_index, suite in enumerate(libero_eval.SUPPORTED_SUITES):
            for task_id in range(10):
                deltas = [1.0] * 50 if (suite_index == 0 and task_id == 0) else [0.0] * 50
                paired[(suite, task_id)] = deltas

        observed = libero_report.hierarchical_delta(paired)

        self.assertAlmostEqual(observed, 1.0 / 40.0)

    def test_bootstrap_is_reproducible_and_constant_one_has_exact_unit_interval(self):
        paired = {
            (suite, task_id): [1.0] * 50
            for suite in libero_eval.SUPPORTED_SUITES
            for task_id in range(10)
        }

        first = libero_report.stratified_paired_bootstrap(paired)
        second = libero_report.stratified_paired_bootstrap(paired)

        self.assertEqual(first, second)
        self.assertEqual(first, (1.0, 1.0))

    def test_bootstrap_is_reproducible_for_nonconstant_paired_deltas(self):
        paired = {
            (suite, task_id): [0.0] * 50
            for suite in libero_eval.SUPPORTED_SUITES
            for task_id in range(10)
        }
        paired[("libero_spatial", 0)] = [-1.0, 1.0] * 25

        first = libero_report.stratified_paired_bootstrap(paired)
        second = libero_report.stratified_paired_bootstrap(paired)

        self.assertEqual(first, second)
        self.assertLess(first[0], 0.0)
        self.assertGreater(first[1], 0.0)

    def test_run_loader_rejects_artifact_errors_infrastructure_and_summary_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            artifact_run = root / "artifact"
            _write_run(artifact_run, "baseline", 10000)
            (artifact_run / "artifact_errors.jsonl").write_text(
                '{"error": "ffmpeg failed"}\n', encoding="utf-8"
            )
            with self.assertRaises(libero_report.ComparisonError):
                libero_report.load_run(artifact_run)

            infrastructure_run = root / "infrastructure"
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
            with self.assertRaises(libero_report.ComparisonError):
                libero_report.load_run(infrastructure_run)

            summary_run = root / "summary"
            _write_run(summary_run, "baseline", 10000)
            summary = json.loads((summary_run / "summary.json").read_text(encoding="utf-8"))
            summary["successes"] = 1
            (summary_run / "summary.json").write_text(
                json.dumps(summary, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaises(libero_report.ComparisonError):
                libero_report.load_run(summary_run)

    def test_diagnostics_must_match_all_variant_artifact_identities(self):
        manifests = [_manifest(variant, step) for variant in ("baseline", "bsp") for step in _STEPS]
        bsp = {
            "verification_passed": True,
            "strict_comparison": True,
            "scipy_version": "1.15.3",
            "episode_count": 1693,
            "frame_count": 273465,
            "strict_max_reconstruction_error": 0.001,
            "mean_reconstruction_error": 0.0002,
            "p95_reconstruction_error": 0.0008,
            "max_error_threshold": 0.002,
            "cache_sha256": _CACHE_SHA,
            "cache_manifest_fingerprint": _CACHE_FINGERPRINT,
        }
        norm = {
            "state_stats_equal": True,
            "asset_directories_isolated": True,
            "action_stats_isolated": True,
            "baseline_norm_stats_sha256": _BASELINE_NORM,
            "bsp_norm_stats_sha256": _BSP_NORM,
        }
        libero_report.validate_diagnostics(manifests, bsp, norm)
        for payload, key, replacement in (
            (bsp, "cache_sha256", "f" * 64),
            (bsp, "verification_passed", False),
            (bsp, "strict_max_reconstruction_error", 0.002),
            (bsp, "mean_reconstruction_error", -0.1),
            (bsp, "p95_reconstruction_error", 0.0015),
            (norm, "state_stats_equal", False),
            (norm, "baseline_norm_stats_sha256", "f" * 64),
        ):
            changed_bsp = copy.deepcopy(bsp)
            changed_norm = copy.deepcopy(norm)
            target = changed_bsp if payload is bsp else changed_norm
            target[key] = replacement
            with self.subTest(key=key), self.assertRaises(libero_report.ComparisonError):
                libero_report.validate_diagnostics(manifests, changed_bsp, changed_norm)


if __name__ == "__main__":
    unittest.main()
