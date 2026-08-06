"""Tests for persistent normalization-asset path selection."""

from pathlib import Path

import pytest

from scripts.compute_norm_stats import norm_stats_output_path


def test_norm_stats_output_uses_asset_id_instead_of_repo_id(tmp_path: Path):
    """Baseline and BSP stats for one repo must never overwrite each other."""
    output = norm_stats_output_path(tmp_path, asset_id="libero_bsp_h16")

    assert output == tmp_path / "libero_bsp_h16"


def test_norm_stats_output_refuses_missing_asset_id(tmp_path: Path):
    """Falling back to an ambiguous repo directory can mix incompatible action statistics."""
    with pytest.raises(ValueError, match="asset_id"):
        norm_stats_output_path(tmp_path, asset_id=None)
