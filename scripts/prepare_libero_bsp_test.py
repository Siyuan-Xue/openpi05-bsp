"""Focused tests for the BSP preparation workflow helpers."""

from pathlib import Path

import pytest

from scripts.prepare_libero_bsp import PreparationMode
from scripts.prepare_libero_bsp import require_preparation_paths


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
