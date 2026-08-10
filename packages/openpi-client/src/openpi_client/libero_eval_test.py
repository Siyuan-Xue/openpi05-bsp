import ast
import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

from openpi_client import libero_eval


def _identity(suite: str = "libero_spatial", task_id: int = 0, init_index: int = 0):
    return libero_eval.EpisodeIdentity(
        suite=suite,
        task_id=task_id,
        task_name=f"task {task_id}",
        init_state_index=init_index,
        init_state_fingerprint=f"state-{init_index}",
    )


class TestLiberoEvaluation:
    def test_task_filter_is_canonical_and_rejects_duplicates_or_out_of_range_ids(self):
        assert libero_eval.resolve_task_ids(None) == tuple(range(10))
        assert libero_eval.resolve_task_ids((7, 0, 3)) == (0, 3, 7)
        for invalid in ((0, 0), (-1,), (10,), ()):
            with pytest.raises(ValueError):
                libero_eval.resolve_task_ids(invalid)

    def test_resolve_suites_accepts_four_named_suites_or_all_but_not_libero_90(self):
        assert libero_eval.resolve_suites("spatial") == ("libero_spatial",)
        assert libero_eval.resolve_suites("libero_10") == ("libero_10",)
        assert libero_eval.resolve_suites("all") == libero_eval.SUPPORTED_SUITES
        with pytest.raises(ValueError, match="supported"):
            libero_eval.resolve_suites("libero_90")

    def test_replan_seed_is_stable_and_namespaced_by_full_paired_identity(self):
        identity = _identity()
        seed = libero_eval.stable_replan_seed(42, identity, 3)

        assert seed == libero_eval.stable_replan_seed(42, identity, 3)
        assert seed != libero_eval.stable_replan_seed(43, identity, 3)
        assert seed != libero_eval.stable_replan_seed(42, identity, 4)
        assert seed != libero_eval.stable_replan_seed(
            42, dataclasses.replace(identity, init_state_fingerprint="different-state"), 3
        )
        assert seed >= 0
        assert seed < 2**32

    def test_policy_protocol_defaults_to_phase_one_but_allows_official_h10_calibration(self):
        assert libero_eval.resolve_policy_protocol("baseline", None) == libero_eval.PolicyProtocol(
            name="baseline_h16", expected_action_horizon=16
        )
        assert libero_eval.resolve_policy_protocol("baseline", 10) == libero_eval.PolicyProtocol(
            name="baseline_h10_calibration", expected_action_horizon=10
        )
        assert libero_eval.resolve_policy_protocol("bsp", None) == libero_eval.PolicyProtocol(
            name="bsp_decoded_h8", expected_action_horizon=8
        )
        for variant, horizon in (("baseline", 8), ("baseline", 11), ("bsp", 16)):
            with pytest.raises(ValueError):
                libero_eval.resolve_policy_protocol(variant, horizon)

    def test_episode_identity_is_paired_by_suite_task_and_exact_init_state(self):
        identity = _identity(suite="libero_goal", task_id=4, init_index=9)

        assert identity.paired_key == "libero_goal/task-004/init-009/state-9"
        assert identity.episode_id == "libero_goal-task-004-init-009-state-9"

    def test_infrastructure_failures_retry_twice_with_the_same_identity(self):
        identity = _identity()
        seen = []

        def attempt(attempt_number):
            seen.append((attempt_number, identity.paired_key, libero_eval.stable_replan_seed(42, identity, 0)))
            if attempt_number < 3:
                raise libero_eval.InfrastructureFailure("network", "connection dropped")
            return libero_eval.AttemptResult(success=True, steps=17, replans=3)

        record = libero_eval.run_episode_with_retries(identity, attempt, eval_seed=42)

        assert record.success
        assert record.include_in_success_rate
        assert record.attempts == 3
        assert [entry[2] for entry in seen] == [seen[0][2]] * 3

    def test_exhausted_infrastructure_is_incomplete_and_excluded(self):
        def attempt(_attempt_number):
            raise libero_eval.InfrastructureFailure("simulator", "EGL context lost")

        record = libero_eval.run_episode_with_retries(_identity(), attempt, eval_seed=42)

        assert record.status == "infrastructure_incomplete"
        assert record.success is None
        assert not record.include_in_success_rate
        assert record.attempts == 3
        assert record.infrastructure_kind == "simulator"

    def test_policy_failure_is_counted_once_and_never_retried(self):
        attempts = []

        def attempt(attempt_number):
            attempts.append(attempt_number)
            raise libero_eval.PolicyFailure("non-finite policy actions")

        record = libero_eval.run_episode_with_retries(_identity(), attempt, eval_seed=42)

        assert attempts == [1]
        assert record.status == "policy_failure"
        assert not record.success
        assert record.include_in_success_rate

    def test_exception_classification_separates_policy_network_container_and_simulator(self):
        assert isinstance(
            libero_eval.classify_exception(ValueError("bad output"), phase="policy_infer"), libero_eval.PolicyFailure
        )
        network = libero_eval.classify_exception(ConnectionError("closed"), phase="policy_infer")
        container = libero_eval.classify_exception(ConnectionRefusedError("refused"), phase="server_connect")
        simulator = libero_eval.classify_exception(RuntimeError("EGL"), phase="environment_step")
        assert network.kind == "network"
        assert container.kind == "container"
        assert simulator.kind == "simulator"

    def test_action_selection_executes_exactly_first_eight_and_rejects_invalid_output(self):
        actions = [[float(row + column) for column in range(7)] for row in range(16)]
        selected = libero_eval.select_replan_actions(actions)

        assert len(selected) == 8
        assert selected == tuple(tuple(row) for row in actions[:8])
        assert libero_eval.select_replan_actions(actions, expected_horizon=16) == selected
        with pytest.raises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(actions, expected_horizon=8)
        with pytest.raises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(actions[:7])
        with pytest.raises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions([*actions[:7], [0.0] * 6])
        malformed_shape = type("ArrayLike", (), {"shape": (8, 7, 1), "__iter__": lambda self: iter([])})()
        with pytest.raises(libero_eval.PolicyFailure):
            libero_eval.select_replan_actions(malformed_shape)
        bad = [row[:] for row in actions]
        bad[12][2] = float("nan")
        with pytest.raises(libero_eval.PolicyFailure):
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
        assert task_rows["libero_spatial", 0]["success_rate"] == 0.5
        assert suite_rows["libero_goal"]["success_rate"] == 1.0
        assert summary["suite_macro_success_rate"] == 0.75
        assert summary["four_suite_macro_success_rate"] is None
        assert not summary["all_four_suites_evaluated"]
        assert summary["incomplete_infrastructure_count"] == 1
        assert not summary["acceptance_complete"]

    def test_four_suite_macro_is_only_populated_when_every_supported_suite_is_present(self):
        records = [
            libero_eval.EpisodeRecord.from_attempt(_identity(suite, 0, 0), 42, 1, success=True)
            for suite in libero_eval.SUPPORTED_SUITES
        ]

        summary = libero_eval.aggregate_records(records)

        assert summary["all_four_suites_evaluated"]
        assert summary["four_suite_macro_success_rate"] == 1.0

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
            assert success_path is not None
            assert "success-init-002" in str(success_path)
            assert selector.claim(later_success) is None
            failure_path = selector.claim(failure)
            assert failure_path is not None
            assert "failure-init-004" in str(failure_path)
            assert success_path != failure_path
            assert selector.claim(infra) is None

    def test_manifest_preserves_all_audit_identities_and_bsp_parameters(self):
        manifest = libero_eval.EvaluationManifest(
            code_sha="a" * 40,
            dataset_revision="v2.0",
            config_name="pi05_libero_bsp_h16",
            checkpoint_step=10000,
            bsp_cache_hash="a" * 64,
            bsp_cache_manifest_fingerprint="c" * 64,
            norm_hash="b" * 64,
            checkpoint="checkpoint/10000",
            container_digest="sha256:" + "d" * 64,
            train_seed=42,
            eval_seed=7,
            policy_variant="bsp",
            bsp_parameters=libero_eval.BSP_PARAMETERS,
            policy_protocol="bsp_decoded_h8",
            expected_action_horizon=8,
            execution_horizon=8,
            suites=libero_eval.SUPPORTED_SUITES,
            task_ids=(0, 3),
        )

        payload = manifest.to_dict()
        assert payload["code_sha"] == "a" * 40
        assert payload["config_name"] == "pi05_libero_bsp_h16"
        assert payload["checkpoint_step"] == 10000
        assert payload["bsp_cache_hash"] == "a" * 64
        assert payload["bsp_cache_manifest_fingerprint"] == "c" * 64
        assert payload["schema_version"] == 2
        assert payload["bsp_parameters"]["target_rows"] == 16
        assert payload["policy_protocol"] == "bsp_decoded_h8"
        assert payload["expected_action_horizon"] == 8
        assert payload["execution_horizon"] == 8
        assert payload["task_ids"] == [0, 3]

    def test_manifest_requires_cache_sha_and_fingerprint_together_only_for_bsp(self):
        shared = dict(
            code_sha="a" * 40,
            dataset_revision="v2.0",
            norm_hash="b" * 64,
            checkpoint="checkpoint/10000",
            checkpoint_step=10000,
            container_digest="sha256:" + "d" * 64,
            train_seed=42,
            eval_seed=42,
            bsp_parameters=libero_eval.BSP_PARAMETERS,
            execution_horizon=8,
        )
        baseline = libero_eval.EvaluationManifest(
            **shared,
            config_name="pi05_libero_baseline_h16",
            bsp_cache_hash=None,
            bsp_cache_manifest_fingerprint=None,
            policy_variant="baseline",
            policy_protocol="baseline_h16",
            expected_action_horizon=16,
        )
        assert baseline.to_dict()["bsp_cache_hash"] is None
        for cache_hash, fingerprint in ((None, "c" * 64), ("a" * 64, None)):
            with pytest.raises(ValueError):
                libero_eval.EvaluationManifest(
                    **shared,
                    config_name="pi05_libero_bsp_h16",
                    bsp_cache_hash=cache_hash,
                    bsp_cache_manifest_fingerprint=fingerprint,
                    policy_variant="bsp",
                    policy_protocol="bsp_decoded_h8",
                    expected_action_horizon=8,
                )

    def test_manifest_rejects_unverifiable_source_identities_or_nonfinite_timeouts(self):
        manifest = libero_eval.EvaluationManifest(
            code_sha="a" * 40,
            dataset_revision="v2.0",
            config_name="pi05_libero_baseline_h16",
            checkpoint_step=10000,
            bsp_cache_hash=None,
            bsp_cache_manifest_fingerprint=None,
            norm_hash="b" * 64,
            checkpoint="checkpoint/baseline/10000",
            container_digest="sha256:" + "d" * 64,
            train_seed=42,
            eval_seed=42,
            policy_variant="baseline",
            bsp_parameters=libero_eval.BSP_PARAMETERS,
            policy_protocol="baseline_h16",
            expected_action_horizon=16,
            execution_horizon=8,
        )
        for field, value in (
            ("code_sha", "abc"),
            ("code_sha", 123),
            ("dataset_revision", "v2.1"),
            ("container_digest", "sha256:container"),
            ("container_digest", 123),
            ("connection_timeout_s", 0.0),
            ("inference_timeout_s", float("inf")),
        ):
            with pytest.raises(ValueError):
                dataclasses.replace(manifest, **{field: value})

    def test_client_evaluation_module_parses_as_python_37(self):
        source = Path(libero_eval.__file__).read_text(encoding="utf-8")
        ast.parse(source, filename=str(libero_eval.__file__), feature_version=(3, 7))

    def test_artifact_writer_emits_jsonl_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = libero_eval.ArtifactWriter(root)
            record = libero_eval.EpisodeRecord.from_attempt(_identity(), 42, 1, success=True)
            writer.append_episode(record)
            summary = writer.write_summary([record])

            episode = json.loads((root / "episodes.jsonl").read_text())
            assert episode["paired_key"] == record.identity.paired_key
            assert (root / "tasks.csv").is_file()
            assert (root / "suites.csv").is_file()
            assert json.loads((root / "summary.json").read_text()) == summary

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
            assert persisted_episode["success"]
            assert persisted_error["artifact_type"] == "video"
            assert summary["artifact_error_count"] == 1
            assert not summary["acceptance_complete"]
