"""Dependency-free contract tests for the slim pi0.5 JAX + LIBERO runtime."""

import ast
import pathlib
import unittest


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "src/openpi/training/config.py"


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"missing assignment {name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _function(nodes: list[ast.stmt], name: str) -> ast.FunctionDef:
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _call_keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"missing keyword {name}")


def _train_configs() -> dict[str, ast.Call]:
    value = _assignment(_tree(_CONFIG), "_CONFIGS")
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise AssertionError("_CONFIGS must be a literal sequence")
    configs = {}
    for element in value.elts:
        if not isinstance(element, ast.Call):
            raise AssertionError("_CONFIGS entries must be constructor calls")
        name = ast.literal_eval(_call_keyword(element, "name"))
        configs[name] = element
    return configs


def _called_name(node: ast.expr) -> str:
    if not isinstance(node, ast.Call):
        raise AssertionError("expected a call")
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    raise AssertionError("unsupported call target")


class CoreRuntimeSlimContractTest(unittest.TestCase):
    def test_registry_contains_only_the_five_pi05_libero_configs(self):
        configs = _train_configs()
        self.assertEqual(
            tuple(configs),
            (
                "pi05_libero",
                "pi05_libero_baseline_h16",
                "pi05_libero_bsp_h16",
                "pi05_libero_baseline_lora_h16",
                "pi05_libero_bsp_lora_h16",
            ),
        )
        for name, config in configs.items():
            self.assertEqual(_called_name(_call_keyword(config, "data")), "LeRobotLiberoDataConfig")
            model = _call_keyword(config, "model")
            if name.endswith("_lora_h16"):
                self.assertIsInstance(model, ast.Name)
                self.assertEqual(model.id, "_PI05_LIBERO_LORA_H16_MODEL")
            else:
                self.assertEqual(_called_name(model), "Pi0Config")
                self.assertTrue(ast.literal_eval(_call_keyword(model, "pi05")))

    def test_h16_full_and_lora_recipes_preserve_short10k_invariants(self):
        configs = _train_configs()
        expected_milestones = (0, 1_000, 2_000, 5_000, 10_000)
        for suffix in ("baseline", "bsp"):
            full = configs[f"pi05_libero_{suffix}_h16"]
            lora = configs[f"pi05_libero_{suffix}_lora_h16"]
            for config in (full, lora):
                self.assertEqual(ast.literal_eval(_call_keyword(config, "seed")), 42)
                self.assertEqual(ast.literal_eval(_call_keyword(config, "batch_size")), 256)
                self.assertEqual(ast.literal_eval(_call_keyword(config, "micro_batch_size")), 1)
                self.assertEqual(ast.literal_eval(_call_keyword(config, "num_train_steps")), 10_000)
                self.assertEqual(ast.literal_eval(_call_keyword(config, "save_interval")), 1_000)
                self.assertEqual(ast.literal_eval(_call_keyword(config, "keep_period")), 10_000)
                self.assertEqual(
                    ast.literal_eval(_call_keyword(config, "permanent_checkpoint_steps")), expected_milestones
                )
            self.assertEqual(ast.literal_eval(_call_keyword(full, "ema_decay")), 0.999)
            self.assertIsNone(ast.literal_eval(_call_keyword(lora, "ema_decay")))

        lora_model = _assignment(_tree(_CONFIG), "_PI05_LIBERO_LORA_H16_MODEL")
        self.assertEqual(_called_name(lora_model), "Pi0Config")
        self.assertTrue(ast.literal_eval(_call_keyword(lora_model, "pi05")))
        self.assertEqual(ast.literal_eval(_call_keyword(lora_model, "action_dim")), 32)
        self.assertEqual(ast.literal_eval(_call_keyword(lora_model, "action_horizon")), 16)
        self.assertFalse(ast.literal_eval(_call_keyword(lora_model, "discrete_state_input")))
        self.assertEqual(ast.literal_eval(_call_keyword(lora_model, "paligemma_variant")), "gemma_2b_lora")
        self.assertEqual(ast.literal_eval(_call_keyword(lora_model, "action_expert_variant")), "gemma_300m_lora")

    def test_policy_and_model_loading_are_jax_only(self):
        policy_tree = _tree(_ROOT / "src/openpi/policies/policy.py")
        policy_config_tree = _tree(_ROOT / "src/openpi/policies/policy_config.py")
        model_tree = _tree(_ROOT / "src/openpi/models/model.py")
        array_typing_tree = _tree(_ROOT / "src/openpi/shared/array_typing.py")
        image_tools_tree = _tree(_ROOT / "src/openpi/shared/image_tools.py")
        config_tree = _tree(_CONFIG)

        imports = {
            alias.name.split(".")[0]
            for tree in (policy_tree, policy_config_tree, model_tree, array_typing_tree, image_tools_tree)
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("torch", imports)
        self.assertNotIn("safetensors", imports)
        model_functions = {node.name for node in ast.walk(model_tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("load_pytorch", model_functions)

        create_policy = _function(policy_config_tree.body, "create_trained_policy")
        self.assertNotIn("pytorch_device", {arg.arg for arg in create_policy.args.args + create_policy.args.kwonlyargs})
        train_config = _class(config_tree, "TrainConfig")
        fields = {
            node.target.id
            for node in train_config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertNotIn("pytorch_weight_path", fields)
        self.assertNotIn("pytorch_training_precision", fields)

    def test_tokenization_surface_is_paligemma_only(self):
        tokenizer_tree = _tree(_ROOT / "src/openpi/models/tokenizer.py")
        tokenizer_classes = {node.name for node in tokenizer_tree.body if isinstance(node, ast.ClassDef)}
        self.assertEqual(tokenizer_classes, {"PaligemmaTokenizer"})
        imports = {
            alias.name.split(".")[0]
            for node in tokenizer_tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("transformers", imports)
        self.assertNotIn("orbax", imports)

    def test_data_loader_supports_only_lerobot_and_bsp_paths(self):
        loader_tree = _tree(_ROOT / "src/openpi/training/data_loader.py")
        names = {node.name for node in ast.walk(loader_tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertIn("create_torch_dataset", names)
        self.assertIn("BspLeRobotDataset", (_ROOT / "src/openpi/training/data_loader.py").read_text())
        self.assertNotIn("create_rlds_dataset", names)
        self.assertNotIn("create_rlds_data_loader", names)
        self.assertNotIn("RLDSDataLoader", names)
        create_loader = _function(loader_tree.body, "create_data_loader")
        self.assertNotIn("framework", {arg.arg for arg in create_loader.args.args + create_loader.args.kwonlyargs})

    def test_removed_runtime_families_are_absent(self):
        removed = (
            "src/openpi/models/pi0_fast.py",
            "src/openpi/models/gemma_fast.py",
            "src/openpi/models/utils/fsq_tokenizer.py",
            "src/openpi/models/vit.py",
            "src/openpi/models_pytorch",
            "src/openpi/policies/aloha_policy.py",
            "src/openpi/policies/droid_policy.py",
            "src/openpi/training/droid_rlds_dataset.py",
            "src/openpi/training/misc",
            "scripts/train_pytorch.py",
        )
        remaining = []
        for path in removed:
            target = _ROOT / path
            if target.is_file() or (target.is_dir() and any(candidate.is_file() for candidate in target.rglob("*"))):
                remaining.append(path)
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
