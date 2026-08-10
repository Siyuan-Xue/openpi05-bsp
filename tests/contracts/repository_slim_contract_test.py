"""Stdlib-only pytest contracts for the slim repository configuration and CI surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

_ROOT = Path(__file__).resolve().parents[2]
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


def _unittest_style_findings(paths: tuple[Path, ...]) -> list[str]:
    findings = []
    for path in paths:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import) and any(alias.name == "unittest" for alias in node.names):
                findings.append(f"{path.relative_to(_ROOT)}:{node.lineno}: import unittest")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] == "unittest":
                findings.append(f"{path.relative_to(_ROOT)}:{node.lineno}: from unittest")
            elif isinstance(node, ast.ClassDef) and any(
                (isinstance(base, ast.Name) and base.id == "TestCase")
                or (isinstance(base, ast.Attribute) and base.attr == "TestCase")
                for base in node.bases
            ):
                findings.append(f"{path.relative_to(_ROOT)}:{node.lineno}: TestCase")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and (node.attr.startswith("assert") or node.attr == "subTest")
            ):
                findings.append(f"{path.relative_to(_ROOT)}:{node.lineno}: self.{node.attr}")
    return findings


def _is_pytest_module(path: str | Path) -> bool:
    name = Path(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _tracked_pytest_modules() -> tuple[Path, ...]:
    tracked_paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return tuple(
        _ROOT / relative_path
        for relative_path in tracked_paths
        if _is_pytest_module(relative_path) and (_ROOT / relative_path).is_file()
    )


def _run_pytest_probe(tmp_path: Path, source: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / "marker_probe_test.py"
    probe.write_text(source, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--rootdir",
            str(_ROOT),
            "-c",
            str(_ROOT / "pyproject.toml"),
            *arguments,
            str(probe),
            "-q",
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _external_marker_probe(path: Path, test_names: tuple[str, ...]) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    blocks = ["import pytest"]
    for test_name in test_names:
        decorators = []
        for decorator in functions[test_name].decorator_list:
            marker = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(marker, ast.Attribute)
                and isinstance(marker.value, ast.Attribute)
                and isinstance(marker.value.value, ast.Name)
                and marker.value.value.id == "pytest"
                and marker.value.attr == "mark"
                and marker.attr in {"manual", "network", "data", "gpu"}
            ):
                decorators.append(f"@pytest.mark.{marker.attr}")
        blocks.append("\n".join([*decorators, f"def {test_name}():", f'    raise AssertionError("{test_name} ran")']))
    blocks.append("def test_offline_sentinel():\n    pass")
    return "\n\n".join(blocks)


def test_surviving_production_imports_are_declared_directly():
    root_project = _project(_ROOT / "pyproject.toml")
    client_project = _project(_ROOT / "packages/openpi-client/pyproject.toml")

    root_imports = _production_imports((_ROOT / "src/openpi", _ROOT / "scripts"))
    client_imports = _production_imports((_ROOT / "packages/openpi-client/src/openpi_client",))

    assert _undeclared_imports(root_imports, _dependency_names(root_project), {"openpi", "scripts"}) == set()
    assert _undeclared_imports(client_imports, _dependency_names(client_project), {"openpi_client"}) == set()


def test_openpi_client_test_imports_are_declared():
    client_project = _project(_ROOT / "packages/openpi-client/pyproject.toml")
    dependencies = _dependency_names(client_project)
    dependencies.update(_distribution_name(requirement) for requirement in client_project["dependency-groups"]["dev"])
    test_root = _ROOT / "packages/openpi-client/src/openpi_client"
    tests = tuple(sorted(path for path in test_root.glob("*.py") if _is_pytest_module(path)))

    assert tests
    assert _undeclared_imports(_file_imports(tests), dependencies, {"openpi_client"}) == set()


def test_tracked_tests_use_only_native_pytest_style():
    tracked_tests = _tracked_pytest_modules()

    assert tracked_tests
    assert _unittest_style_findings(tracked_tests) == []


def test_external_test_markers_are_registered_under_strict_checking(tmp_path: Path):
    result = _run_pytest_probe(
        tmp_path,
        """
import pytest

@pytest.mark.manual
def test_manual():
    pass

@pytest.mark.network
def test_network():
    pass

@pytest.mark.data
def test_data():
    pass

@pytest.mark.gpu
def test_gpu():
    pass
""",
        "--strict-markers",
        "-m",
        "",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout


def test_default_pytest_selection_deselects_external_tests(tmp_path: Path):
    result = _run_pytest_probe(
        tmp_path,
        """
import pytest

@pytest.mark.manual
def test_manual():
    raise AssertionError("manual probe ran")

@pytest.mark.network
def test_network():
    raise AssertionError("network probe ran")

@pytest.mark.data
def test_data():
    raise AssertionError("data probe ran")

@pytest.mark.gpu
def test_gpu():
    raise AssertionError("GPU probe ran")

def test_offline():
    pass
""",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 4 deselected" in result.stdout


def test_default_selection_deselects_external_tokenizer_transform_tests(tmp_path: Path):
    probe = _external_marker_probe(
        _ROOT / "src/openpi/transforms_test.py",
        ("test_tokenize_prompt", "test_tokenize_no_prompt"),
    )

    for marker_expression in (None, "not network", "not data"):
        arguments = () if marker_expression is None else ("-m", marker_expression)
        result = _run_pytest_probe(tmp_path, probe, *arguments)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed, 2 deselected" in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "test_example.py",
        "example_test.py",
        "nested/test_example.py",
        "nested/example_test.py",
    ],
)
def test_pytest_module_discovery_covers_both_supported_names(name: str):
    assert _is_pytest_module(name)


@pytest.mark.parametrize("name", ["example.py", "test_example.txt"])
def test_pytest_module_discovery_rejects_non_tests(name: str):
    assert not _is_pytest_module(name)


def test_removed_dependency_families_and_rlds_group_stay_absent():
    project = _project(_ROOT / "pyproject.toml")
    root_dependencies = _dependency_names(project)
    client_dependencies = _dependency_names(_project(_ROOT / "packages/openpi-client/pyproject.toml"))
    source_names = set(project["tool"]["uv"]["sources"])

    assert root_dependencies.intersection(_REMOVED_DISTRIBUTIONS) == set()
    assert client_dependencies.intersection(_REMOVED_DISTRIBUTIONS) == set()
    assert "rlds" not in project["dependency-groups"]
    assert "dlimp" not in source_names


def test_retained_runtime_dependency_families_remain_explicit():
    root_dependencies = _dependency_names(_project(_ROOT / "pyproject.toml"))
    client_dependencies = _dependency_names(_project(_ROOT / "packages/openpi-client/pyproject.toml"))

    assert {
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
    assert {"msgpack", "pillow", "typing-extensions", "websockets"}.issubset(client_dependencies)


def test_metadata_ownership_and_test_discovery_target_the_specialized_fork():
    project = _project(_ROOT / "pyproject.toml")
    vscode_source = (_ROOT / ".vscode/settings.json").read_text(encoding="utf-8")
    vscode = json.loads(re.sub(r",(\s*[}\]])", r"\1", vscode_source))
    codeowners = (_ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")

    assert "pi0.5" in project["project"]["description"].lower()
    assert "libero" in project["project"]["description"].lower()
    assert project["project"]["urls"]["Repository"] == "https://github.com/Siyuan-Xue/openpi05-bsp"
    assert re.search("(?m)^\\*\\s+@Siyuan-Xue\\s*$", codeowners) is not None
    assert vscode["python.testing.pytestArgs"] == [
        "src/openpi",
        "scripts",
        "tests/contracts",
        "packages/openpi-client/src/openpi_client",
    ]
    assert vscode["python.testing.pytestEnabled"] is True
    assert vscode["python.testing.unittestEnabled"] is False


@pytest.mark.parametrize(
    "relative_path",
    [
        ".ruff_cache/cache.db",
        ".uv-cache/archive-v0/item",
        "artifacts/phase1/verification.json",
        "reports/phase1/report.md",
    ],
)
def test_generated_experiment_and_tool_artifacts_are_ignored(relative_path: str):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=_ROOT,
        check=False,
    )
    assert result.returncode == 0, relative_path


def test_cpu_ci_runs_only_lightweight_static_and_client_gates():
    workflows = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((_ROOT / ".github/workflows").glob("*.yml"))
    )
    lowered = workflows.lower()

    assert "openpi-verylarge" not in workflows
    assert "uv sync --all-extras" not in workflows
    assert "uv python install" not in workflows
    assert "python -m unittest" not in workflows
    assert re.search("\\b(?:cuda|mujoco|ffmpeg|apt-get)\\b", lowered) is None
    assert "runs-on: ubuntu-latest" in workflows
    assert "ruff check" in workflows
    assert "ruff format --check" in workflows
    assert "repository_slim_contract_test" in workflows
    assert "core_runtime_slim_test" in workflows
    assert "libero_host_contract_test" in workflows
    assert "runtime_paths_test" in workflows
    assert "train_planning_test" in workflows
    assert "compare_libero_phase1_test" in workflows
    assert "packages/openpi-client" in workflows
    assert "pytest==9.0.3" in workflows
    assert "pytest==8.3.5" in workflows
    assert "PYTEST_DEBUG_TEMPROOT" in workflows
    assert "--basetemp" in workflows
