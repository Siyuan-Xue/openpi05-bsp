"""Stdlib-only pytest contracts for the slim repository configuration and CI surface."""

from __future__ import annotations

import ast
from email.parser import Parser
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import tomllib
import zipfile

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


def _yaml_scalars(path: Path) -> dict[tuple[str, ...], str]:
    scalars: dict[tuple[str, ...], str] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.lstrip()
        if not content or content.startswith(("#", "- ")):
            continue
        key, separator, value = content.partition(":")
        if not separator:
            continue
        indentation = len(raw_line) - len(content)
        while stack and stack[-1][0] >= indentation:
            stack.pop()
        location = (*[item[1] for item in stack], key)
        if value.strip():
            scalars[location] = value.split(" #", 1)[0].strip()
        else:
            stack.append((indentation, key))
    return scalars


def _workflow_steps(path: Path) -> tuple[str, ...]:
    source = path.read_text(encoding="utf-8")
    starts = [match.start() for match in re.finditer(r"(?m)^\s*- (?:name|uses):", source)]
    return tuple(source[start:end] for start, end in zip(starts, [*starts[1:], len(source)], strict=True))


def _step_run(step: str) -> str | None:
    match = re.search(r"(?m)^\s+run:\s*(.*)$", step)
    if match is None:
        return None
    value = match.group(1).strip()
    if value not in {"|", ">", "|-", ">-"}:
        return value
    return " ".join(line.strip() for line in step[match.end() :].splitlines())


def _build_wheels(output_dir: Path) -> dict[str, Path]:
    command = ("uv", "build", "--offline", "--all-packages", "--wheel", "--python", sys.executable, "--out-dir")
    result = subprocess.run(
        [*command, str(output_dir)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {Parser().parsestr(_wheel_metadata(path))["Name"]: path for path in output_dir.glob("*.whl")}


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        return wheel.read(metadata_path).decode()


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

    test_roots = tuple((_ROOT / path).resolve() for path in project["tool"]["pytest"]["ini_options"]["testpaths"])
    assert all(any(path.resolve().is_relative_to(root) for root in test_roots) for path in _tracked_pytest_modules())


def test_project_and_client_versions_remain_in_release_parity():
    root_version = _project(_ROOT / "pyproject.toml")["project"]["version"]
    client_version = _project(_ROOT / "packages/openpi-client/pyproject.toml")["project"]["version"]
    runtime_version = runpy.run_path(_ROOT / "packages/openpi-client/src/openpi_client/__init__.py")["__version__"]

    assert {root_version, client_version, runtime_version} == {"0.1.0"}


def test_built_wheels_exclude_tests_and_publish_complete_client_metadata(tmp_path: Path):
    wheels = _build_wheels(tmp_path)

    assert set(wheels) == {"openpi", "openpi-client"}
    for path in wheels.values():
        with zipfile.ZipFile(path) as wheel:
            assert not [
                name
                for name in wheel.namelist()
                if Path(name).name.endswith("_test.py") or Path(name).name == "conftest.py"
            ]

    client_metadata = Parser().parsestr(_wheel_metadata(wheels["openpi-client"]))
    assert client_metadata["Version"] == "0.1.0"
    assert client_metadata["Requires-Python"] == ">=3.8"
    assert client_metadata["License"] == "Apache-2.0"
    assert client_metadata["Description-Content-Type"] == "text/markdown"
    assert client_metadata.get_payload().strip()


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


def test_pr_workflows_are_read_only_bounded_and_commit_pinned():
    expected_actions = {
        "actions/checkout": ("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7.0.1"),
        "actions/setup-python": ("5fda3b95a4ea91299a34e894583c3862153e4b97", "v7.0.0"),
        "astral-sh/setup-uv": ("c771a70e6277c0a99b617c7a806ffedaca235ff9", "v9.0.0"),
    }
    for workflow in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        scalars = _yaml_scalars(workflow)
        jobs = {path[1] for path in scalars if len(path) >= 2 and path[0] == "jobs"}
        steps = _workflow_steps(workflow)

        assert scalars[("permissions", "contents")] == "read"
        assert scalars[("concurrency", "cancel-in-progress")] == "true"
        assert ("concurrency", "group") in scalars
        assert all(int(scalars[("jobs", job, "timeout-minutes")]) > 0 for job in jobs)
        for step in steps:
            match = re.search(r"(?m)^[ \t]*- uses:[ \t]*([^\s#]+)(?:[ \t]+#[ \t]*(\S.*?))?[ \t]*$", step)
            if match is None:
                continue
            action, commit = match.group(1).split("@", 1)
            assert (commit, match.group(2)) == expected_actions[action]
            if action == "actions/checkout":
                assert re.search(r"(?m)^\s+persist-credentials:\s*false\s*$", step)


def test_default_cpu_ci_invokes_quiet_discovery_without_marker_or_file_whitelists():
    commands = filter(None, (_step_run(step) for step in _workflow_steps(_ROOT / ".github/workflows/test.yml")))
    root_commands = [
        shlex.split(command) for command in commands if command.startswith("uv run") and "pytest" in command
    ]

    assert len(root_commands) == 1
    command = root_commands[0]
    assert "--frozen" in command
    pytest_arguments = command[command.index("pytest") + 1 :]
    assert pytest_arguments == ["-q"]


def test_pre_commit_ci_runs_the_locked_tool_through_uvx():
    workflow = _ROOT / ".github/workflows/pre-commit.yml"
    commands = filter(None, (_step_run(step) for step in _workflow_steps(workflow)))

    assert [shlex.split(command) for command in commands] == [
        ["uvx", "--from", "pre-commit==4.2.0", "pre-commit", "run", "--all-files"]
    ]
