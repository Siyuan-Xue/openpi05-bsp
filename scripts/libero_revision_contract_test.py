"""Dependency-free repository contract for the official LIBERO dataset revision."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

_REPOSITORY = Path(__file__).resolve().parents[1]
_EXPECTED_REVISION = "v2.0"
_PHASE_ONE_CONFIG_NAMES = {
    "pi05_libero_baseline_h16",
    "pi05_libero_bsp_h16",
    "pi05_libero_baseline_lora_h16",
    "pi05_libero_bsp_lora_h16",
}


def _module_assignment(path: Path, name: str):
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} not found in {path}")


def _args_field_default(path: Path, name: str):
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == name
                ):
                    return ast.literal_eval(statement.value)
    raise AssertionError(f"Args field {name!r} not found in {path}")


def _phase_one_config_revisions(path: Path) -> dict[str, str]:
    module = ast.parse(path.read_text())
    revisions = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "TrainConfig":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
        if "name" not in keywords or "data" not in keywords:
            continue
        name = ast.literal_eval(keywords["name"])
        if name not in _PHASE_ONE_CONFIG_NAMES:
            continue
        data_call = keywords["data"]
        if not isinstance(data_call, ast.Call):
            raise AssertionError(f"Unexpected data config expression for {name}")
        data_keywords = {keyword.arg: keyword.value for keyword in data_call.keywords if keyword.arg is not None}
        revisions[name] = ast.literal_eval(data_keywords["lerobot_revision"])
    return revisions


class LiberoRevisionContractTest(unittest.TestCase):
    def test_all_phase_one_entrypoints_use_the_real_hub_revision(self):
        self.assertEqual(
            _module_assignment(_REPOSITORY / "src/openpi/training/bsp_dataset.py", "LIBERO_REVISION"),
            _EXPECTED_REVISION,
        )
        self.assertEqual(
            _phase_one_config_revisions(_REPOSITORY / "src/openpi/training/config.py"),
            {
                "pi05_libero_baseline_h16": _EXPECTED_REVISION,
                "pi05_libero_bsp_h16": _EXPECTED_REVISION,
                "pi05_libero_baseline_lora_h16": _EXPECTED_REVISION,
                "pi05_libero_bsp_lora_h16": _EXPECTED_REVISION,
            },
        )
        self.assertEqual(
            _args_field_default(_REPOSITORY / "examples/libero/main.py", "dataset_revision"),
            _EXPECTED_REVISION,
        )
        self.assertIn(
            f"export LIBERO_DATASET_REVISION={_EXPECTED_REVISION}",
            (_REPOSITORY / "examples/libero/README.md").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
