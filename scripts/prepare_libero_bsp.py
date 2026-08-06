"""Download, build, or verify the precomputed LIBERO BSP sidecar.

Examples:
    python scripts/prepare_libero_bsp.py --mode download --dataset-root /data/lerobot/libero
    python scripts/prepare_libero_bsp.py --mode build --dataset-root /data/lerobot/libero \
        --cache-path /data/openpi/libero-bsp-v1.npz
    python scripts/prepare_libero_bsp.py --mode verify --dataset-root /data/lerobot/libero \
        --cache-path /data/openpi/libero-bsp-v1.npz
"""

from __future__ import annotations

import enum
from pathlib import Path

import tyro

from openpi.training.bsp import load_sidecar_cache
from openpi.training.bsp import write_sidecar_cache
from openpi.training.bsp_dataset import BspLeRobotDataset
from openpi.training.bsp_dataset import LIBERO_REPO_ID
from openpi.training.bsp_dataset import LIBERO_REVISION
from openpi.training.bsp_dataset import build_lerobot_bsp_cache
from openpi.training.bsp_dataset import make_lerobot_cache_manifest
from openpi.training.bsp_dataset import validate_lerobot_dataset


class PreparationMode(enum.StrEnum):
    DOWNLOAD = "download"
    BUILD = "build"
    VERIFY = "verify"


def require_preparation_paths(
    mode: PreparationMode,
    *,
    dataset_root: Path | None,
    cache_path: Path | None,
) -> tuple[Path, Path | None]:
    """Require persistent locations instead of writing large artifacts into the checkout."""
    if dataset_root is None:
        raise ValueError("dataset_root is required for every preparation mode")
    if mode in (PreparationMode.BUILD, PreparationMode.VERIFY) and cache_path is None:
        raise ValueError(f"cache_path is required for {mode.value} mode")
    return dataset_root.expanduser().resolve(), None if cache_path is None else cache_path.expanduser().resolve()


def _load_dataset(repo_id: str, revision: str | None, dataset_root: Path):
    # Keep the command importable for --help and static tooling without importing LeRobot eagerly.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {"root": dataset_root}
    if revision is not None:
        kwargs["revision"] = revision
    return LeRobotDataset(repo_id, **kwargs)


def _require_downloaded_dataset(dataset_root: Path) -> None:
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"No downloaded LeRobot dataset metadata found under {dataset_root}; run download mode first"
        )


def main(
    mode: PreparationMode,
    *,
    dataset_root: Path | None = None,
    cache_path: Path | None = None,
    repo_id: str = LIBERO_REPO_ID,
    revision: str | None = LIBERO_REVISION,
) -> None:
    """Run exactly one explicit dataset/cache preparation phase."""
    dataset_root, cache_path = require_preparation_paths(
        mode,
        dataset_root=dataset_root,
        cache_path=cache_path,
    )
    if mode != PreparationMode.DOWNLOAD:
        _require_downloaded_dataset(dataset_root)

    dataset = _load_dataset(repo_id, revision, dataset_root)
    metadata = validate_lerobot_dataset(dataset)
    manifest = make_lerobot_cache_manifest(dataset, repo_id=repo_id, revision=revision)

    if mode == PreparationMode.DOWNLOAD:
        requested_revision = revision or "LeRobot-compatible default"
        print(f"Downloaded and validated {repo_id}@{requested_revision} at {dataset_root}")
        print(f"Metadata: {metadata}")
        print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")
        return

    assert cache_path is not None
    if mode == PreparationMode.BUILD:
        cache = build_lerobot_bsp_cache(dataset)
        write_sidecar_cache(cache_path, cache, manifest)
        print(f"Built BSP sidecar at {cache_path}")
        print(f"Targets: {cache.targets.shape}; mapped frames: {cache.mapping.shape[0]}")
        print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")
        return

    cache = load_sidecar_cache(cache_path, manifest)
    BspLeRobotDataset(dataset, cache)
    print(f"Verified BSP sidecar at {cache_path}")
    print(f"Targets: {cache.targets.shape}; mapped frames: {cache.mapping.shape[0]}")
    print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")


if __name__ == "__main__":
    tyro.cli(main)
