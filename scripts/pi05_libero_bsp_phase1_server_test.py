"""Dependency-free contracts for the phase-one H20 server runbook."""

import ast
from pathlib import Path
import re
import subprocess
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_RUNBOOK = _ROOT / "docs" / "pi05_libero_bsp_phase1_server.md"


def _class_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.target.id.replace("_", "-")
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _function_parameters(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    return {argument.arg.replace("_", "-") for argument in arguments}


class PhaseOneServerRunbookContractTest(unittest.TestCase):
    def setUp(self):
        self.runbook = _RUNBOOK.read_text()
        self.bash_blocks = re.findall(r"```bash\n(.*?)```", self.runbook, flags=re.DOTALL)

    def test_every_bash_block_parses_and_avoids_destructive_commands(self):
        self.assertGreater(len(self.bash_blocks), 30)
        for index, block in enumerate(self.bash_blocks, start=1):
            with self.subTest(block=index):
                result = subprocess.run(
                    ["bash", "-n"],
                    input=block,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(re.search(r"^\s*(?:sudo|rm)\b", block, flags=re.MULTILINE))
                self.assertIsNone(
                    re.search(r"^\s*docker\s+compose\b.*\bdown\b", block, flags=re.MULTILINE)
                )
                self.assertIsNone(
                    re.search(r"^\s*git\b.*\b(?:reset|clean)\b", block, flags=re.MULTILINE)
                )
                self.assertNotIn("--rm", block)
                self.assertNotIn("--overwrite", block)

    def test_evaluator_flags_are_real_nested_args(self):
        fields = _class_fields(_ROOT / "examples" / "libero" / "main.py", "Args")
        flags = set(re.findall(r"--args\.([a-z0-9-]+)", self.runbook))
        required = {
            "task-suite-name",
            "task-ids",
            "policy-variant",
            "expected-action-horizon",
            "num-trials-per-task",
            "output-dir",
            "config-name",
            "checkpoint-step",
            "dataset-revision",
            "norm-hash",
            "checkpoint",
            "container-digest",
            "train-seed",
            "eval-seed",
            "bsp-cache-hash",
            "bsp-cache-manifest-fingerprint",
            "control-freq",
            "video-fps",
            "video-show-inference-waits",
        }

        self.assertTrue(required.issubset(flags))
        self.assertEqual(flags.difference(fields), set())
        self.assertNotIn("--args.code-sha", "\n".join(self.bash_blocks))

    def test_training_overrides_match_current_config_fields(self):
        config_path = _ROOT / "src" / "openpi" / "training" / "config.py"
        train_fields = _class_fields(config_path, "TrainConfig")
        data_fields = _class_fields(config_path, "LeRobotLiberoDataConfig")

        for field in (
            "exp-name",
            "seed",
            "batch-size",
            "micro-batch-size",
            "num-train-steps",
            "save-interval",
            "assets-base-dir",
            "checkpoint-base-dir",
            "resume",
        ):
            with self.subTest(field=field):
                self.assertIn(field, train_fields)
                self.assertIn(f"--{field}", self.runbook)
        for field in ("lerobot-root", "bsp-cache-path"):
            with self.subTest(data_field=field):
                self.assertIn(field, data_fields)
                self.assertIn(f"--data.{field}", self.runbook)

    def test_preparation_and_normalization_flags_match_main_signatures(self):
        commands = {
            "prepare_libero_bsp.py": {
                "mode",
                "dataset-root",
                "cache-path",
                "diagnostics-path",
                "repo-id",
                "revision",
                "action-key",
            },
            "compute_norm_stats.py": {
                "config-name",
                "assets-dir",
                "bsp-cache-path",
                "dataset-root",
                "compare-state-stats-with",
                "norm-comparison-output",
            },
        }
        for script_name, documented in commands.items():
            parameters = _function_parameters(_ROOT / "scripts" / script_name, "main")
            with self.subTest(script=script_name):
                self.assertTrue(documented.issubset(parameters))
                for flag in documented:
                    self.assertIn(f"--{flag}", self.runbook)

    def test_serving_uses_checkpoint_union_and_real_bounded_health_endpoint(self):
        serve_policy = (_ROOT / "scripts" / "serve_policy.py").read_text()
        websocket_server = (
            _ROOT / "src" / "openpi" / "serving" / "websocket_policy_server.py"
        ).read_text()

        self.assertIn("class Checkpoint", serve_policy)
        self.assertIn("config: str", serve_policy)
        self.assertIn("dir: str", serve_policy)
        self.assertIn("policy:checkpoint", self.runbook)
        self.assertIn("--policy.config", self.runbook)
        self.assertIn("--policy.dir", self.runbook)
        self.assertIn("env", _class_fields(_ROOT / "scripts" / "serve_policy.py", "Args"))
        self.assertIn("--env LIBERO", self.runbook)
        self.assertIn("process_request=_health_check", websocket_server)
        self.assertIn('request.path == "/healthz"', websocket_server)
        self.assertIn("http.HTTPStatus.OK", websocket_server)
        self.assertIn('f"http://{host}:{port}/healthz"', self.runbook)
        self.assertIn("127.0.0.1 8000 180", self.runbook)
        self.assertLess(
            self.runbook.index("docker compose -f examples/libero/compose.yml up -d openpi_server"),
            self.runbook.index("--name libero-official-h10-task0-smoke runtime"),
        )

    def test_docker_runtime_proves_automatic_git_identity(self):
        dockerfile = (_ROOT / "examples" / "libero" / "Dockerfile").read_text()
        compose = (_ROOT / "examples" / "libero" / "compose.yml").read_text()
        readme = (_ROOT / "examples" / "libero" / "README.md").read_text()

        apt_install = re.search(
            r"apt-get install -y(?: --no-install-recommends)?(.*?)(?:\n\s*$|\n\s*&&)",
            dockerfile,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(apt_install)
        self.assertRegex(apt_install.group(1), r"(?:^|\s)git(?:\s|$)")

        for value in (
            "GIT_OPTIONAL_LOCKS=0",
            "GIT_CONFIG_COUNT=2",
            "GIT_CONFIG_KEY_0=safe.directory",
            "GIT_CONFIG_VALUE_0=/app",
            "GIT_CONFIG_KEY_1=safe.directory",
            "GIT_CONFIG_VALUE_1=/app/third_party/libero",
        ):
            with self.subTest(compose_environment=value):
                self.assertIn(value, compose)

        for document in (readme, self.runbook):
            with self.subTest(document="README" if document is readme else "runbook"):
                self.assertIn("--name libero-git-identity-preflight", document)
                self.assertIn("git -C /app rev-parse HEAD", document)
                self.assertIn(
                    "git -C /app status --porcelain --untracked-files=all", document
                )
                bash = "\n".join(re.findall(r"```bash\n(.*?)```", document, flags=re.DOTALL))
                self.assertNotIn("--args.code-sha", bash)

    def test_final_audit_names_ten_runs_and_three_diagnostic_artifacts(self):
        self.assertNotIn("六个 h16/BSP 评测输入", self.runbook)
        for fragment in (
            "十个 h16/BSP 评测输入",
            "历史 BSP diagnostics",
            "schema-v3 BSP diagnostics",
            "norm diagnostics",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.runbook)

    def test_fixed_protocol_and_audit_artifacts_are_complete(self):
        required_fragments = (
            "1,693 episodes",
            "273,465 frames",
            "40 tasks",
            "10 fps",
            "physical-intelligence/libero",
            "--revision v2.0",
            "pi05_libero_baseline_h16",
            "pi05_libero_bsp_h16",
            "UV_VERSION=0.11.32",
            "python install 3.11.9",
            'scipy.__version__ == "1.15.3"',
            "DockerRootDir",
            "nvidia-container-cli info",
            "official-h10-task0-smoke",
            "--args.task-suite-name all",
            "--args.num-trials-per-task 50",
            '--bsp-verification "$BSP_VERIFY_SCHEMA3"',
            '--norm-comparison "$NORM_COMPARISON"',
            "20,000 episodes",
            "schema v3",
            "dataset FPS 为 10",
            "评测环境 `control_freq_hz` 固定 20 Hz",
            "MP4 默认 40 FPS",
            "video_audit.jsonl",
            "control_steps / 20 + included_control_stall_seconds",
            "schema v2 结果",
            "十次正式 schema-v3 评测",
            "BSP_VERIFY_SCHEMA3",
            'test ! -e "$BSP_VERIFY_SCHEMA3"',
            '--diagnostics-path "$BSP_VERIFY_SCHEMA3"',
            'old["cache_sha256"] == new["cache_sha256"] == cache_sha',
            'old["cache_manifest_fingerprint"] == new["cache_manifest_fingerprint"]',
            'old["cache_contents_sha256"] == new["cache_contents_sha256"]',
            'new["code_sha"] == code_sha',
            "不重训模型、不重建 sidecar",
            "permanent_checkpoint_steps",
            "0k/1k/2k/5k/10k",
            "phase1-short10k-seed42-baseline",
            "phase1-short10k-seed42-bsp",
            "--num-train-steps 10000",
            "10,000 次",
            "seed 43/44",
            "modified_libero_rlds",
            "B-spline.pdf",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.runbook)

        for variant in ("baseline", "bsp"):
            for step in (0, 1000, 2000, 5000, 10000):
                with self.subTest(variant=variant, step=step):
                    self.assertIn(f'"$EVAL_BASE/{variant}-step-{step}"', self.runbook)

        comparison_command = re.search(
            r'"\$OPENPI_PY" scripts/compare_libero_phase1\.py \\\n(.*?)\n\s*--bsp-verification',
            self.runbook,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(comparison_command)
        self.assertEqual(comparison_command.group(1).count('"$EVAL_BASE/'), 10)

        for artifact in (
            "task_comparison.csv",
            "suite_comparison.csv",
            "learning_curve.csv",
            "comparison.json",
            "report.md",
            "learning_curve.svg",
        ):
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, self.runbook)


if __name__ == "__main__":
    unittest.main()
