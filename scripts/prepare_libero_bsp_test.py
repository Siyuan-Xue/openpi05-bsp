"""Focused tests for the BSP preparation workflow helpers."""

import datetime as dt
import json
from pathlib import Path

import pytest

from scripts.prepare_libero_bsp import PreparationMode
from scripts.prepare_libero_bsp import make_verification_diagnostics
from scripts.prepare_libero_bsp import require_preparation_paths
from scripts.prepare_libero_bsp import write_json_atomic


def test_all_preparation_modes_require_an_explicit_dataset_location(tmp_path: Path):
    """A hidden repository-local data default makes large downloads non-portable."""
    for mode in PreparationMode:
        with pytest.raises(ValueError, match="dataset_root"):
            require_preparation_paths(mode, dataset_root=None, cache_path=tmp_path / "cache.npz")


def test_build_and_verify_require_an_explicit_sidecar_location(tmp_path: Path):
    """Implicit cache locations can make training fit or consume an unintended sidecar."""
    for mode in (PreparationMode.BUILD, PreparationMode.VERIFY):
        with pytest.raises(ValueError, match="cache_path"):
            require_preparation_paths(mode, dataset_root=tmp_path / "dataset", cache_path=None)


def test_download_does_not_require_a_cache_location(tmp_path: Path):
    """Dataset acquisition is intentionally separable from expensive BSP fitting."""
    dataset_root, cache_path = require_preparation_paths(
        PreparationMode.DOWNLOAD,
        dataset_root=tmp_path / "dataset",
        cache_path=None,
    )

    assert dataset_root == tmp_path / "dataset"
    assert cache_path is None


def test_verification_diagnostics_are_machine_readable_and_audit_environment_identity(tmp_path: Path):
    """Console-only verification cannot prove which cache, code, or SciPy build was accepted."""
    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"canonical-cache-file")
    verification = {
        "episode_count": 1693,
        "frame_count": 273465,
        "segment_count": 1234,
        "reconstruction_error_max": 0.001,
        "reconstruction_error_mean": 0.0002,
        "reconstruction_error_p95": 0.0008,
        "reconstruction_error_threshold": 0.002,
        "strict_reconstruction_tolerance": True,
        "ground_truth_knots_nondecreasing": True,
        "tail_padding_valid": True,
        "future_segment_mapping_valid": True,
        "target_index_bounds_valid": True,
        "no_cross_episode_mapping": True,
        "all_frames_covered": True,
        "targets_match_rebuild": True,
        "mapping_matches_rebuild": True,
        "cache_contents_deterministic": True,
        "cache_contents_sha256": "content",
        "rebuilt_contents_sha256": "content",
    }

    diagnostics = make_verification_diagnostics(
        verification,
        cache_path=cache_path,
        manifest_fingerprint="fingerprint",
        scipy_version="1.15.3",
        code_sha="abc123",
        verified_at=dt.datetime(2026, 8, 6, 1, 2, 3, tzinfo=dt.UTC),
    )
    output = tmp_path / "verification.json"
    write_json_atomic(output, diagnostics)
    persisted = json.loads(output.read_text())

    assert persisted == diagnostics
    assert persisted["cache_sha256"] == "052858071fdc63a642030e3411817e1fdd453bbbb2ad7db83505abef4a8ba563"
    assert persisted["cache_manifest_fingerprint"] == "fingerprint"
    assert persisted["scipy_version"] == "1.15.3"
    assert persisted["strict_max_reconstruction_error"] == 0.001
    assert persisted["mean_reconstruction_error"] == 0.0002
    assert persisted["p95_reconstruction_error"] == 0.0008
    assert persisted["max_error_threshold"] == 0.002
    assert persisted["strict_comparison"] is True
    assert persisted["code_sha"] == "abc123"
    assert persisted["verified_at_utc"] == "2026-08-06T01:02:03Z"
    assert persisted["verification_passed"] is True
