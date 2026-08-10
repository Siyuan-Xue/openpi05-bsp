"""Dependency-light contracts for the retained host-only LIBERO surface."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "examples" / "libero" / "README.md"
sys.path.insert(0, str(_ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import libero_eval
from openpi_client import libero_report


def test_host_server_python_and_scipy_remain_pinned():
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    openpi = next(package for package in lock["package"] if package["name"] == "openpi")
    scipy = next(package for package in lock["package"] if package["name"] == "scipy")

    assert "scipy==1.15.3" in project["project"]["dependencies"]
    assert (_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.9"
    assert {"name": "scipy"} in openpi["dependencies"]
    assert {"name": "scipy", "specifier": "==1.15.3"} in openpi["metadata"]["requires-dist"]
    assert scipy["version"] == "1.15.3"


def test_libero_example_defers_server_setup_to_the_canonical_runbook():
    readme = _README.read_text(encoding="utf-8")

    assert "../../docs/pi05_libero_bsp_phase1_server.md" in readme
    for duplicated_server_detail in (
        "uv sync --python 3.11",
        "uv venv --python 3.8 examples/libero/.venv",
        "HOST_RUNTIME_DIGEST",
        "--args.container-digest",
        "scripts/serve_policy.py",
        "/root/openpi-bsp-work",
    ):
        assert duplicated_server_detail not in readme


def test_audit_protocol_retains_calibration_h16_and_comparison_artifacts():
    calibration = libero_eval.resolve_policy_protocol("baseline", 10)
    baseline = libero_eval.resolve_policy_protocol("baseline", 16)

    assert calibration.name == "baseline_h10_calibration"
    assert baseline.name == "baseline_h16"
    assert libero_report.MILESTONES == (0, 1000, 2000, 5000, 10000)
    assert set(libero_report.OUTPUT_FILENAMES) == {
        "task_comparison.csv",
        "suite_comparison.csv",
        "learning_curve.csv",
        "comparison.json",
        "report.md",
        "learning_curve.svg",
    }
