import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from openpi_client import libero_eval


def _identity(suite: str = "libero_spatial", task_id: int = 0, init_index: int = 0):
    return libero_eval.EpisodeIdentity(
        suite=suite,
        task_id=task_id,
        task_name=f"task {task_id}",
        init_state_index=init_index,
        init_state_fingerprint=f"state-{init_index}",
    )


class LiberoEvaluationTest(unittest.TestCase):
    def test_resolve_suites_accepts_four_named_suites_or_all_but_not_libero_90(self):
        self.assertEqual(libero_eval.resolve_suites("spatial"), ("libero_spatial",))
        self.assertEqual(libero_eval.resolve_suites("libero_10"), ("libero_10",))
        self.assertEqual(libero_eval.resolve_suites("all"), libero_eval.SUPPORTED_SUITES)
        with self.assertRaisesRegex(ValueError, "supported"):
            libero_eval.resolve_suites("libero_90")

    def test_replan_seed_is_stable_and_namespaced_by_full_paired_identity(self):
        identity = _identity()
        seed = libero_eval.stable_replan_seed(42, identity, 3)

        self.assertEqual(seed, libero_eval.stable_replan_seed(42, identity, 3))
        self.assertNotEqual(seed, libero_eval.stable_replan_seed(43, identity, 3))
        self.assertNotEqual(seed, libero_eval.stable_replan_seed(42, identity, 4))
        self.assertNotEqual(
            seed,
            libero_eval.stable_replan_seed(
                42, dataclasses.replace(identity, init_state_fingerprint="different-state"), 3
            ),
        )
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_policy_protocol_defaults_to_phase_one_but_allows_official_h10_calibration(self):
        self.assertEqual(
            libero_eval.resolve_policy_protocol("baseline", None),
            libero_eval.PolicyProtocol(name="baseline_h16", expected_action_horizon=16),
        )
        self.assertEqual(
            libero_eval.resolve_policy_protocol("baseline", 10),
            libero_eval.PolicyProtocol(name="baseline_h10_calibration", expected_action_horizon=10),
        )
        self.assertEqual(
            libero_eval.resolve_policy_protocol("bsp", None),
            libero_eval.PolicyProtocol(name="bsp_decoded_h8", expected_action_horizon=8),
        )
        for variant, horizon in (("baseline", 8), ("baseline", 11), ("bsp", 16)):
            with self.subTest(variant=variant, horizon=horizon), self.assertRaises(ValueError):
                libero_eval.resolve_policy_protocol(variant, horizon)

    def test_episode_identity_is_paired_by_suite_task_and_exact_init_state(self):
        identity = _identity(suite="libero_goal", task_id=4, init_index=9)

        self.assertEqual(identity.paired_key, "libero_goal/task-004/init-009/state-9")
        self.assertEqual(identity.episode_id, "libero_goal-task-004-init-009-state-9")

    def test_infrastructure_failures_retry_twice_with_the_same_identity(self):
        identity = _identity()
        seen = []

        def attempt(attempt_number):
            seen.append((attempt_number, identity.paired_key, libero_eval.stable_replan_seed(42, identity, 0)))
            if attempt_number < 3:
                raise libero_eval.InfrastructureFailure("network", "connection dropped")
            return libero_eval.AttemptResult(success=True, steps=17, replans=3)

        record = libero_eval.run_episode_with_retries(identity, attempt, eval_seed=42)

        self.assertTrue(record.success)
        self.assertTrue(record.include_in_success_rate)
        self.assertEqual(record.attempts, 3)
        self.assertEqual([entry[2] for entry in seen], [seen[0][2]] * 3)

    def test_exhausted_infrastructure_is_incomplete_and_excluded(self):
        def attempt(_attempt_number):
            raise libero_eval.InfrastructureFailure("simulator", "EGL context lost")

        record = libero_eval.run_episode_with_retries(_identity(), attempt, eval_seed=42)

        self.assertEqual(record.status, "infrastructure_incomplete")
        self.assertIsNone(record.success)
        self.assertFalse(record.include_in_success_rate)
        self.assertEqual(record.attempts, 3)
        self.assertEqual(record.infrastructure_kind, "simulator")

    def test_policy_failure_is_counted_once_and_never_retried(self):
        attempts = []

        def attempt(attempt_number):
            attempts.append(attempt_number)
            raise libero_eval.PolicyFailure("non-finite policy actions")

        record = libero_eval.run_episode_with_retries(_identity(), attempt, eval_seed=42)

        self.assertEqual(attempts, [1])
        self.assertEqual(record.status, "policy_failure")
        self.assertFalse(record.success)
        self.assertTrue(record.include_in_success_rate)

    def test_exception_classification_separates_policy_network_container_and_simulator(self):
        self.assertIsInstance(
            libero_eval.classify_exception(ValueError("bad output"), phase="policy_infer"),
            libero_eval.PolicyFailure,
        )
        network = libero_eval.classify_exception(ConnectionError("closed"), phase="policy_infer")
        container = libero_eval.classify_exception(ConnectionRefusedError("refused"), phase="server_connect")
        simulator = libero_eval.classify_exception(RuntimeError("EGL"), phase="environment_step")
        self.assertEqual(network.kind, "network")
        self.assertEqual(container.kind, "container")
        self.assertEqual(simulator.kind, "simulator")

    def test_action_selection_executes_exactly_first_eight_and_rejects_invalid_output(self):
        actions = [[float(row + column) for column in range(7)] for row in range(16)]
        selected = libero_eval.select_replan_actions(actions)

        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, tuple(tuple(row) for row in actions[:8]))
        self.assertEqual(libero_eval.select_replan_actions(actions, expected_horizon=16), selected)
        with self.assertRaises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(actions, expected_horizon=8)
        with self.assertRaises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(actions[:7])
        with self.assertRaises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions([*actions[:7], [0.0] * 6])
        malformed_shape = type("ArrayLike", (), {"shape": (8, 7, 1), "__iter__": lambda self: iter([])})()
        with self.assertRaises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(malformed_shape)
        bad = [row[:] for row in actions]
        bad[12][2] = float("nan")
        with self.assertRaises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(bad)

    def test_aggregation_excludes_infrastructure_and_reports_all_levels(self):
        records = [
            libero_eval.EpisodeRecord.from_attempt(_identity("libero_spatial", 0, 0), 42, 1, success=True),
            libero_eval.EpisodeRecord.from_attempt(
                _identity("libero_spatial", 0, 1), 42, 1, success=False, failure_kind="policy"
            ),
            libero_eval.EpisodeRecord.from_attempt(_identity("libero_goal", 0, 0), 42, 1, success=True),
            libero_eval.EpisodeRecord.infrastructure_incomplete(
                _identity("libero_goal", 0, 1), 42, 3, "network", "dropped"
            ),
        ]

        summary = libero_eval.aggregate_records(records)

        task_rows = {(row["suite"], row["task_id"]): row for row in summary["tasks"]}
        suite_rows = {row["suite"]: row for row in summary["suites"]}
        self.assertEqual(task_rows[("libero_spatial", 0)]["success_rate"], 0.5)
        self.assertEqual(suite_rows["libero_goal"]["success_rate"], 1.0)
        self.assertEqual(summary["suite_macro_success_rate"], 0.75)
        self.assertIsNone(summary["four_suite_macro_success_rate"])
        self.assertFalse(summary["all_four_suites_evaluated"])
        self.assertEqual(summary["incomplete_infrastructure_count"], 1)
        self.assertFalse(summary["acceptance_complete"])

    def test_four_suite_macro_is_only_populated_when_every_supported_suite_is_present(self):
        records = [
            libero_eval.EpisodeRecord.from_attempt(_identity(suite, 0, 0), 42, 1, success=True)
            for suite in libero_eval.SUPPORTED_SUITES
        ]

        summary = libero_eval.aggregate_records(records)

        self.assertTrue(summary["all_four_suites_evaluated"])
        self.assertEqual(summary["four_suite_macro_success_rate"], 1.0)

    def test_video_selector_keeps_only_first_success_and_first_counted_failure_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            selector = libero_eval.VideoSelector(Path(directory))
            success = libero_eval.EpisodeRecord.from_attempt(_identity(init_index=2), 42, 1, success=True)
            later_success = libero_eval.EpisodeRecord.from_attempt(_identity(init_index=3), 42, 1, success=True)
            failure = libero_eval.EpisodeRecord.from_attempt(
                _identity(init_index=4), 42, 1, success=False, failure_kind="timeout"
            )
            infra = libero_eval.EpisodeRecord.infrastructure_incomplete(
                _identity(init_index=5), 42, 3, "network", "dropped"
            )

            success_path = selector.claim(success)
            self.assertIsNotNone(success_path)
            self.assertIn("success-init-002", str(success_path))
            self.assertIsNone(selector.claim(later_success))
            failure_path = selector.claim(failure)
            self.assertIsNotNone(failure_path)
            self.assertIn("failure-init-004", str(failure_path))
            self.assertNotEqual(success_path, failure_path)
            self.assertIsNone(selector.claim(infra))

    def test_manifest_preserves_all_audit_identities_and_bsp_parameters(self):
        manifest = libero_eval.EvaluationManifest(
            code_sha="abc",
            dataset_revision="v2.1",
            bsp_cache_hash="cache",
            norm_hash="norm",
            checkpoint="checkpoint/10000",
            container_digest="sha256:container",
            train_seed=42,
            eval_seed=7,
            policy_variant="bsp",
            bsp_parameters=libero_eval.BSP_PARAMETERS,
            policy_protocol="bsp_decoded_h8",
            expected_action_horizon=8,
            execution_horizon=8,
        )

        payload = manifest.to_dict()
        self.assertEqual(payload["code_sha"], "abc")
        self.assertEqual(payload["bsp_cache_hash"], "cache")
        self.assertEqual(payload["bsp_parameters"]["target_rows"], 16)
        self.assertEqual(payload["policy_protocol"], "bsp_decoded_h8")
        self.assertEqual(payload["expected_action_horizon"], 8)
        self.assertEqual(payload["execution_horizon"], 8)

    def test_artifact_writer_emits_jsonl_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = libero_eval.ArtifactWriter(root)
            record = libero_eval.EpisodeRecord.from_attempt(_identity(), 42, 1, success=True)
            writer.append_episode(record)
            summary = writer.write_summary([record])

            episode = json.loads((root / "episodes.jsonl").read_text())
            self.assertEqual(episode["paired_key"], record.identity.paired_key)
            self.assertTrue((root / "tasks.csv").is_file())
            self.assertTrue((root / "suites.csv").is_file())
            self.assertEqual(json.loads((root / "summary.json").read_text()), summary)

    def test_artifact_failure_is_separately_audited_and_marks_summary_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = libero_eval.ArtifactWriter(root)
            record = libero_eval.EpisodeRecord.from_attempt(_identity(), 42, 1, success=True)
            error = libero_eval.ArtifactError(
                episode_id=record.identity.episode_id,
                artifact_type="video",
                path="videos/example.mp4",
                error="ffmpeg exited 1",
            )

            writer.append_episode(record)
            writer.append_artifact_error(error)
            summary = writer.write_summary([record], artifact_errors=[error])

            persisted_episode = json.loads((root / "episodes.jsonl").read_text())
            persisted_error = json.loads((root / "artifact_errors.jsonl").read_text())
            self.assertTrue(persisted_episode["success"])
            self.assertEqual(persisted_error["artifact_type"], "video")
            self.assertEqual(summary["artifact_error_count"], 1)
            self.assertFalse(summary["acceptance_complete"])


if __name__ == "__main__":
    unittest.main()
