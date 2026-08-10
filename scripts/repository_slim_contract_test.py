"""Dependency-free contracts for the slim repository configuration and CI surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_REMOVED_DISTRIBUTIONS = {
    "chex",
    "dm-tree",
    "equinox",
    "flatbuffers",
    "gym-aloha",
    "ml-collections",
    "opencv-python",
    "polars",
    "rich",
    "tensorflow-cpu",
    "tensorflow-datasets",
    "transformers",
    "treescope",
}
_IMPORT_TO_DISTRIBUTION = {
    "PIL": "pillow",
    "jax": "jax",
    "lerobot": "lerobot",
    "openpi_client": "openpi-client",
    "orbax": "orbax-checkpoint",
    "tree": "dm-tree",
    "tqdm_loggable": "tqdm-loggable",
}


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _dependency_names(project: dict) -> set[str]:
    return {_distribution_name(requirement) for requirement in project["project"]["dependencies"]}


def _production_imports(paths: tuple[Path, ...]) -> set[str]:
    imports = set()
    for base in paths:
        for path in base.rglob("*.py"):
            if path.name.endswith("_test.py") or path.name == "conftest.py":
                continue
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
    return imports


def _file_imports(paths: tuple[Path, ...]) -> set[str]:
    imports = set()
    for path in paths:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    return imports


def _undeclared_imports(imports: set[str], dependencies: set[str], local_modules: set[str]) -> set[str]:
    third_party = imports.difference(sys.stdlib_module_names, local_modules, {"__future__"})
    return {
        name
        for name in third_party
        if _IMPORT_TO_DISTRIBUTION.get(name, name.lower().replace("_", "-")) not in dependencies
    }


class RepositorySlimContractTest(unittest.TestCase):
    def test_surviving_production_imports_are_declared_directly(self):
        root_project = _project(_ROOT / "pyproject.toml")
        client_project = _project(_ROOT / "packages/openpi-client/pyproject.toml")

        root_imports = _production_imports((_ROOT / "src/openpi", _ROOT / "scripts"))
        client_imports = _production_imports((_ROOT / "packages/openpi-client/src/openpi_client",))

        self.assertEqual(
            _undeclared_imports(root_imports, _dependency_names(root_project), {"openpi", "scripts"}),
            set(),
        )
        self.assertEqual(
            _undeclared_imports(client_imports, _dependency_names(client_project), {"openpi_client"}),
            set(),
        )

    def test_openpi_client_test_imports_are_declared(self):
        client_project = _project(_ROOT / "packages/openpi-client/pyproject.toml")
        dependencies = _dependency_names(client_project)
        dependencies.update(
            _distribution_name(requirement) for requirement in client_project["dependency-groups"]["dev"]
        )
        tests = tuple(sorted((_ROOT / "packages/openpi-client/src/openpi_client").glob("*_test.py")))

        self.assertTrue(tests)
        self.assertEqual(
            _undeclared_imports(_file_imports(tests), dependencies, {"openpi_client"}),
            set(),
        )

    def test_removed_dependency_families_and_rlds_group_stay_absent(self):
        project = _project(_ROOT / "pyproject.toml")
        root_dependencies = _dependency_names(project)
        client_dependencies = _dependency_names(_project(_ROOT / "packages/openpi-client/pyproject.toml"))
        source_names = set(project["tool"]["uv"]["sources"])

        self.assertEqual(root_dependencies.intersection(_REMOVED_DISTRIBUTIONS), set())
        self.assertEqual(client_dependencies.intersection(_REMOVED_DISTRIBUTIONS), set())
        self.assertNotIn("rlds", project["dependency-groups"])
        self.assertNotIn("dlimp", source_names)

    def test_retained_runtime_dependency_families_remain_explicit(self):
        root_dependencies = _dependency_names(_project(_ROOT / "pyproject.toml"))
        client_dependencies = _dependency_names(_project(_ROOT / "packages/openpi-client/pyproject.toml"))

        self.assertTrue(
            {
                "etils",
                "flax",
                "imageio",
                "jax",
                "lerobot",
                "optax",
                "orbax-checkpoint",
                "pillow",
                "pydantic",
                "scipy",
                "sentencepiece",
                "torch",
                "tqdm",
                "tyro",
                "wandb",
                "websockets",
            }.issubset(root_dependencies)
        )
        self.assertTrue({"msgpack", "pillow", "typing-extensions", "websockets"}.issubset(client_dependencies))

    def test_metadata_ownership_and_test_discovery_target_the_specialized_fork(self):
        project = _project(_ROOT / "pyproject.toml")
        vscode_source = (_ROOT / ".vscode/settings.json").read_text(encoding="utf-8")
        vscode = json.loads(re.sub(r",(\s*[}\]])", r"\1", vscode_source))
        codeowners = (_ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")

        self.assertIn("pi0.5", project["project"]["description"].lower())
        self.assertIn("libero", project["project"]["description"].lower())
        self.assertEqual(project["project"]["urls"]["Repository"], "https://github.com/Siyuan-Xue/openpi05-bsp")
        self.assertRegex(codeowners, r"(?m)^\*\s+@Siyuan-Xue\s*$")
        self.assertEqual(
            vscode["python.testing.pytestArgs"],
            ["src/openpi", "scripts", "packages/openpi-client/src/openpi_client"],
        )

    def test_generated_experiment_and_tool_artifacts_are_ignored(self):
        for relative_path in (
            ".ruff_cache/cache.db",
            ".uv-cache/archive-v0/item",
            "artifacts/phase1/verification.json",
            "reports/phase1/report.md",
        ):
            with self.subTest(path=relative_path):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", "--no-index", relative_path],
                    cwd=_ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)

    def test_cpu_ci_runs_only_lightweight_static_and_client_gates(self):
        workflows = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted((_ROOT / ".github/workflows").glob("*.yml"))
        )
        lowered = workflows.lower()

        self.assertNotIn("openpi-verylarge", workflows)
        self.assertNotIn("uv sync --all-extras", workflows)
        self.assertNotIn("uv python install", workflows)
        self.assertNotRegex(lowered, r"\b(?:cuda|mujoco|ffmpeg|apt-get)\b")
        self.assertIn("runs-on: ubuntu-latest", workflows)
        self.assertIn("ruff check", workflows)
        self.assertIn("ruff format --check", workflows)
        self.assertIn("repository_slim_contract_test", workflows)
        self.assertIn("core_runtime_slim_test", workflows)
        self.assertIn("libero_host_contract_test", workflows)
        self.assertIn("packages/openpi-client", workflows)
        self.assertIn("pytest", workflows)


if __name__ == "__main__":
    unittest.main()
