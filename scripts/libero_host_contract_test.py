"""Dependency-free contracts for the retained host-only LIBERO surface."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "examples" / "libero" / "README.md"
sys.path.insert(0, str(_ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import libero_eval
from openpi_client import libero_report


def _class_fields(path: Path, class_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {
        node.target.id.replace("_", "-")
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _function_parameters(path: Path, function_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    return {argument.arg.replace("_", "-") for argument in arguments}


class LiberoHostContractTest(unittest.TestCase):
    def test_host_server_python_and_scipy_remain_pinned(self):
        project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
        openpi = next(package for package in lock["package"] if package["name"] == "openpi")
        scipy = next(package for package in lock["package"] if package["name"] == "scipy")

        self.assertIn("scipy==1.15.3", project["project"]["dependencies"])
        self.assertEqual((_ROOT / ".python-version").read_text(encoding="utf-8").strip(), "3.11.9")
        self.assertIn({"name": "scipy"}, openpi["dependencies"])
        self.assertIn({"name": "scipy", "specifier": "==1.15.3"}, openpi["metadata"]["requires-dist"])
        self.assertEqual(scipy["version"], "1.15.3")

    def test_readme_is_a_host_only_dual_python_evaluator_path(self):
        readme = _README.read_text(encoding="utf-8")

        self.assertFalse((_ROOT / ".dockerignore").exists())
        self.assertNotRegex(readme.lower(), r"\b(?:docker|compose|preflight)\b")
        self.assertNotIn("POLICY_CONTAINER_DIGEST", readme)
        self.assertIn("uv sync --python 3.11", readme)
        self.assertIn("uv venv --python 3.8 examples/libero/.venv", readme)
        self.assertIn("HOST_RUNTIME_DIGEST", readme)
        self.assertIn("--args.container-digest ${HOST_RUNTIME_DIGEST}", readme)
        self.assertIn("${EXPERIMENTS_DIR}", readme)

    def test_readme_evaluator_flags_match_retained_args_and_protocols(self):
        readme = _README.read_text(encoding="utf-8")
        flags = set(re.findall(r"--args\.([a-z0-9-]+)", readme))
        args = _class_fields(_ROOT / "examples" / "libero" / "main.py", "Args")

        self.assertTrue(flags)
        self.assertEqual(flags.difference(args), set())
        self.assertIn("--args.expected-action-horizon 10", readme)
        self.assertIn("--args.expected-action-horizon 16", readme)
        self.assertIn("--args.config-name pi05_libero", readme)
        self.assertIn("--args.config-name pi05_libero_baseline_h16", readme)

    def test_retained_train_prepare_and_serving_contracts_remain_live(self):
        train_tree = ast.parse((_ROOT / "scripts" / "train.py").read_text(encoding="utf-8"))
        cache_updates = [
            node
            for node in ast.walk(train_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "jax_compilation_cache_dir"
        ]
        self.assertEqual(len(cache_updates), 1)
        self.assertIsInstance(cache_updates[0].args[1], ast.Call)

        expected_parameters = {
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
        for script_name, required in expected_parameters.items():
            with self.subTest(script=script_name):
                actual = _function_parameters(_ROOT / "scripts" / script_name, "main")
                self.assertTrue(required.issubset(actual))

        serve_policy = (_ROOT / "scripts" / "serve_policy.py").read_text(encoding="utf-8")
        websocket_server = (_ROOT / "src" / "openpi" / "serving" / "websocket_policy_server.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class Checkpoint", serve_policy)
        self.assertIn("config: str", serve_policy)
        self.assertIn("dir: str", serve_policy)
        self.assertIn("env", _class_fields(_ROOT / "scripts" / "serve_policy.py", "Args"))
        self.assertIn("process_request=_health_check", websocket_server)
        self.assertIn('request.path == "/healthz"', websocket_server)
        self.assertIn("http.HTTPStatus.OK", websocket_server)

    def test_audit_protocol_retains_calibration_h16_and_comparison_artifacts(self):
        calibration = libero_eval.resolve_policy_protocol("baseline", 10)
        baseline = libero_eval.resolve_policy_protocol("baseline", 16)

        self.assertEqual(calibration.name, "baseline_h10_calibration")
        self.assertEqual(baseline.name, "baseline_h16")
        self.assertEqual(libero_report.MILESTONES, (0, 1000, 2000, 5000, 10000))
        self.assertEqual(
            set(libero_report.OUTPUT_FILENAMES),
            {
                "task_comparison.csv",
                "suite_comparison.csv",
                "learning_curve.csv",
                "comparison.json",
                "report.md",
                "learning_curve.svg",
            },
        )


if __name__ == "__main__":
    unittest.main()
