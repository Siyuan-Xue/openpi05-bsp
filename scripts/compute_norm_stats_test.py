"""Tests for persistent normalization-asset path selection."""

import json
from pathlib import Path

import numpy as np
import pytest

import openpi.shared.normalize as normalize
from scripts.compute_norm_stats import compare_norm_stats_assets
from scripts.compute_norm_stats import norm_stats_output_path
from scripts.compute_norm_stats import write_norm_comparison


def test_norm_stats_output_uses_asset_id_instead_of_repo_id(tmp_path: Path):
    """Baseline and BSP stats for one repo must never overwrite each other."""
    output = norm_stats_output_path(tmp_path, asset_id="libero_bsp_h16")

    assert output == tmp_path / "libero_bsp_h16"


def test_norm_stats_output_refuses_missing_asset_id(tmp_path: Path):
    """Falling back to an ambiguous repo directory can mix incompatible action statistics."""
    with pytest.raises(ValueError, match="asset_id"):
        norm_stats_output_path(tmp_path, asset_id=None)


def _stats(*, state_delta: float = 0.0, action_dim: int) -> dict[str, normalize.NormStats]:
    state = normalize.NormStats(
        mean=np.asarray([1.0 + state_delta, 2.0]),
        std=np.asarray([0.5, 0.25]),
        q01=np.asarray([-1.0, -2.0]),
        q99=np.asarray([3.0, 4.0]),
    )
    action = normalize.NormStats(
        mean=np.arange(action_dim, dtype=np.float64),
        std=np.ones(action_dim, dtype=np.float64),
        q01=-np.ones(action_dim, dtype=np.float64),
        q99=np.ones(action_dim, dtype=np.float64),
    )
    return {"state": state, "actions": action}


def test_norm_comparison_proves_state_fields_equal_and_action_assets_isolated(tmp_path: Path):
    """Comparing dataclass objects can hide array ambiguity and cannot prove separate action assets."""
    baseline_dir = tmp_path / "libero_baseline_h16"
    bsp_dir = tmp_path / "libero_bsp_h16"
    normalize.save(baseline_dir, _stats(action_dim=7))
    normalize.save(bsp_dir, _stats(action_dim=8))

    comparison = compare_norm_stats_assets(
        baseline_dir,
        bsp_dir,
        rtol=1e-7,
        atol=1e-8,
    )

    assert comparison["state_stats_equal"] is True
    assert comparison["asset_directories_isolated"] is True
    assert comparison["action_stats_isolated"] is True
    assert comparison["rtol"] == 1e-7
    assert comparison["atol"] == 1e-8
    assert set(comparison["state_fields"]) == {"mean", "std", "q01", "q99"}
    assert comparison["baseline_norm_stats_sha256"] != comparison["bsp_norm_stats_sha256"]
    assert comparison["baseline_action_stats_sha256"] != comparison["bsp_action_stats_sha256"]
    json.dumps(comparison, allow_nan=False)

    output = tmp_path / "eval" / "norm_comparison.json"
    write_norm_comparison(output, comparison)
    assert json.loads(output.read_text()) == comparison


def test_norm_comparison_flags_state_mismatch_without_numpy_object_equality(tmp_path: Path):
    """Every state mean/std/quantile must independently pass the declared numeric tolerance."""
    baseline_dir = tmp_path / "baseline"
    bsp_dir = tmp_path / "bsp"
    normalize.save(baseline_dir, _stats(action_dim=7))
    normalize.save(bsp_dir, _stats(state_delta=1e-3, action_dim=8))

    comparison = compare_norm_stats_assets(baseline_dir, bsp_dir, rtol=1e-7, atol=1e-8)

    assert comparison["state_stats_equal"] is False
    assert comparison["state_fields"]["mean"]["equal"] is False
    assert comparison["state_fields"]["std"]["equal"] is True


def test_norm_comparison_rejects_collapsed_bsp_knot_quantiles(tmp_path: Path):
    """A zero-width knot interval explodes quantile normalization even when every value is finite."""
    baseline_dir = tmp_path / "baseline"
    bsp_dir = tmp_path / "bsp"
    baseline = _stats(action_dim=7)
    bsp = _stats(action_dim=8)
    bsp["actions"].mean[7] = -273360.0
    bsp["actions"].q01[7] = -273360.0
    bsp["actions"].q99[7] = -273360.0
    normalize.save(baseline_dir, baseline)
    normalize.save(bsp_dir, bsp)

    with pytest.raises(ValueError, match="knot quantile interval"):
        compare_norm_stats_assets(baseline_dir, bsp_dir)


def test_norm_comparison_rejects_shared_asset_directory(tmp_path: Path):
    """A shared output directory can overwrite one experiment even when state values match."""
    shared = tmp_path / "shared"
    normalize.save(shared, _stats(action_dim=7))

    with pytest.raises(ValueError, match="distinct asset directories"):
        compare_norm_stats_assets(shared, shared)
