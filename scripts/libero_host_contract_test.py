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


def _call_keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Missing {name!r} in configuration call")


def _train_config_calls(module: ast.Module | None = None) -> dict[str, ast.Call]:
    if module is None:
        config_path = _ROOT / "src" / "openpi" / "training" / "config.py"
        module = ast.parse(config_path.read_text(encoding="utf-8"))
    configs = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in node.targets)
    )
    if not isinstance(configs, ast.List):
        raise AssertionError("_CONFIGS must remain a literal list of TrainConfig calls")
    return {
        ast.literal_eval(_call_keyword(config, "name")): config
        for config in configs.elts
        if isinstance(config, ast.Call)
    }


def _literal_keyword(call: ast.Call, name: str):
    return ast.literal_eval(_call_keyword(call, name))


def _assert_shared_lora_model_contract(config_source: str) -> None:
    """Reject a LoRA shape change or a config that stops sharing the common model."""
    module = ast.parse(config_source)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_PI05_LIBERO_LORA_H16_MODEL" for target in node.targets)
    )
    if not isinstance(assignment.value, ast.Call):
        raise AssertionError("_PI05_LIBERO_LORA_H16_MODEL must remain a model constructor call")
    model = assignment.value
    expected = {"pi05": True, "action_dim": 32, "action_horizon": 16}
    actual = {name: _literal_keyword(model, name) for name in expected}
    if actual != expected:
        raise AssertionError(f"shared LoRA model contract changed: {actual}")

    configs = _train_config_calls(module)
    for name in ("pi05_libero_baseline_lora_h16", "pi05_libero_bsp_lora_h16"):
        reference = _call_keyword(configs[name], "model")
        if not isinstance(reference, ast.Name) or reference.id != "_PI05_LIBERO_LORA_H16_MODEL":
            raise AssertionError(f"{name} must reference _PI05_LIBERO_LORA_H16_MODEL")


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
        self.assertNotIn("../../docs/pi05_libero_bsp_phase1_server.md", readme)

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

    def test_host_training_overrides_keep_libero_data_inputs_addressable(self):
        """Fails if a host CLI override or LIBERO data path is removed from its dataclass."""
        config_path = _ROOT / "src" / "openpi" / "training" / "config.py"
        train_fields = _class_fields(config_path, "TrainConfig")
        data_fields = _class_fields(config_path, "LeRobotLiberoDataConfig")

        self.assertTrue(
            {
                "exp-name",
                "seed",
                "batch-size",
                "micro-batch-size",
                "num-train-steps",
                "save-interval",
                "assets-base-dir",
                "checkpoint-base-dir",
                "resume",
            }.issubset(train_fields)
        )
        self.assertTrue({"lerobot-root", "bsp-cache-path"}.issubset(data_fields))

    def test_five_configs_keep_h10_and_h16_phase_one_checkpoint_milestones(self):
        """Fails if a production config loses its fixed checkpoint cadence or H10 default."""
        configs = _train_config_calls()
        phase_one = (
            "pi05_libero_baseline_h16",
            "pi05_libero_bsp_h16",
            "pi05_libero_baseline_lora_h16",
            "pi05_libero_bsp_lora_h16",
        )
        milestones = (0, 1_000, 2_000, 5_000, 10_000)

        self.assertEqual(tuple(configs), ("pi05_libero", *phase_one))
        self.assertEqual(_literal_keyword(configs["pi05_libero"], "num_train_steps"), 30_000)
        for name in phase_one:
            with self.subTest(config=name):
                config = configs[name]
                self.assertEqual(_literal_keyword(config, "num_train_steps"), 10_000)
                self.assertEqual(_literal_keyword(config, "save_interval"), 1_000)
                self.assertEqual(_literal_keyword(config, "keep_period"), 10_000)
                self.assertEqual(_literal_keyword(config, "permanent_checkpoint_steps"), milestones)

        h10_model = _call_keyword(configs["pi05_libero"], "model")
        self.assertIsInstance(h10_model, ast.Call)
        self.assertEqual(_literal_keyword(h10_model, "action_horizon"), 10)

    def test_h16_ab_bindings_and_checkpoint_save_protocol_remain_coupled(self):
        """Fails if report A/B identities or training's checkpoint save inputs drift apart."""
        expected_bindings = {
            ("baseline", "full"): "pi05_libero_baseline_h16",
            ("bsp", "full"): "pi05_libero_bsp_h16",
            ("baseline", "lora"): "pi05_libero_baseline_lora_h16",
            ("bsp", "lora"): "pi05_libero_bsp_lora_h16",
        }
        self.assertEqual(libero_report._PHASE_ONE_CONFIGS, expected_bindings)
        self.assertEqual(libero_eval.resolve_policy_protocol("baseline", 10).name, "baseline_h10_calibration")
        self.assertEqual(libero_eval.resolve_policy_protocol("baseline", 16).name, "baseline_h16")

        configs = _train_config_calls()
        for name in ("pi05_libero_baseline_h16", "pi05_libero_bsp_h16"):
            model = _call_keyword(configs[name], "model")
            self.assertIsInstance(model, ast.Call)
            self.assertEqual(_literal_keyword(model, "action_horizon"), 16)
        for name in ("pi05_libero_bsp_h16", "pi05_libero_bsp_lora_h16"):
            data = _call_keyword(configs[name], "data")
            self.assertIsInstance(data, ast.Call)
            self.assertTrue(_literal_keyword(data, "use_bsp"))

        train_tree = ast.parse((_ROOT / "scripts" / "train.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(train_tree) if isinstance(node, ast.Call)]
        call_names = {
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"initialize_checkpoint_dir", "save_state", "should_save_checkpoint"}.issubset(call_names))
        initialize = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == "initialize_checkpoint_dir"
        )
        initialize_keywords = {keyword.arg for keyword in initialize.keywords}
        self.assertTrue({"keep_period", "permanent_checkpoint_steps", "overwrite", "resume"}.issubset(initialize_keywords))
        should_save = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == "should_save_checkpoint"
        )
        should_save_keywords = {keyword.arg for keyword in should_save.keywords}
        self.assertTrue({"num_train_steps", "save_interval"}.issubset(should_save_keywords))

    def test_shared_lora_model_preserves_h16_shape_and_is_reused_by_both_lora_configs(self):
        """Fails if the shared LoRA model changes shape or either LoRA config inlines another model."""
        source = (_ROOT / "src" / "openpi" / "training" / "config.py").read_text(encoding="utf-8")

        _assert_shared_lora_model_contract(source)


if __name__ == "__main__":
    unittest.main()
