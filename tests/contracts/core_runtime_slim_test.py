"""Stdlib-only pytest contracts for the slim pi0.5 JAX + LIBERO runtime."""

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "src/openpi/training/config.py"


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


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


def test_policy_and_model_loading_are_jax_only():
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
    assert "torch" not in imports
    assert "safetensors" not in imports
    model_functions = {node.name for node in ast.walk(model_tree) if isinstance(node, ast.FunctionDef)}
    assert "load_pytorch" not in model_functions

    create_policy = _function(policy_config_tree.body, "create_trained_policy")
    assert "pytorch_device" not in {arg.arg for arg in create_policy.args.args + create_policy.args.kwonlyargs}
    train_config = _class(config_tree, "TrainConfig")
    fields = {
        node.target.id
        for node in train_config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "pytorch_weight_path" not in fields
    assert "pytorch_training_precision" not in fields


def test_removed_tokenizer_dependencies_stay_absent():
    tokenizer_tree = _tree(_ROOT / "src/openpi/models/tokenizer.py")
    imports = {
        alias.name.split(".")[0] for node in tokenizer_tree.body if isinstance(node, ast.Import) for alias in node.names
    }
    assert "transformers" not in imports
    assert "orbax" not in imports


def test_removed_rlds_loader_surface_stays_absent():
    loader_tree = _tree(_ROOT / "src/openpi/training/data_loader.py")
    names = {node.name for node in ast.walk(loader_tree) if isinstance(node, ast.ClassDef | ast.FunctionDef)}
    assert "create_rlds_dataset" not in names
    assert "create_rlds_data_loader" not in names
    assert "RLDSDataLoader" not in names


def test_removed_runtime_families_are_absent():
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
    assert remaining == []
