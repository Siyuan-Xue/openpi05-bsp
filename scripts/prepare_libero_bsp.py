"""Download, build, or verify the precomputed LIBERO BSP sidecar.

Examples:
    python scripts/prepare_libero_bsp.py --mode download --dataset-root /data/lerobot/libero
    python scripts/prepare_libero_bsp.py --mode build --dataset-root /data/lerobot/libero \
        --cache-path /data/openpi/libero-bsp-v2.npz
    python scripts/prepare_libero_bsp.py --mode verify --dataset-root /data/lerobot/libero \
        --cache-path /data/openpi/libero-bsp-v2.npz
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import enum
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import tyro

from openpi.training.bsp import load_sidecar_cache
from openpi.training.bsp import write_sidecar_cache
from openpi.training.bsp_dataset import LIBERO_REPO_ID
from openpi.training.bsp_dataset import LIBERO_REVISION
from openpi.training.bsp_dataset import BspLeRobotDataset
from openpi.training.bsp_dataset import build_lerobot_bsp_cache
from openpi.training.bsp_dataset import make_lerobot_cache_manifest
from openpi.training.bsp_dataset import validate_lerobot_dataset
from openpi.training.bsp_dataset import verify_lerobot_bsp_cache

_REQUIRED_SCIPY_VERSION = "1.15.3"
_VERIFICATION_FLAGS = (
    "strict_reconstruction_tolerance",
    "ground_truth_knots_nondecreasing",
    "tail_padding_valid",
    "future_segment_mapping_valid",
    "target_index_bounds_valid",
    "no_cross_episode_mapping",
    "all_frames_covered",
    "targets_match_rebuild",
    "mapping_matches_rebuild",
    "cache_contents_deterministic",
)


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


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _code_sha() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to resolve code SHA from {repository}") from error
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError(f"Unable to resolve code SHA from {repository}")
    return sha


def make_verification_diagnostics(
    verification: Mapping[str, Any],
    *,
    cache_path: Path,
    manifest_fingerprint: str,
    scipy_version: str,
    code_sha: str,
    verified_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Attach artifact/environment identity and compute the auditable verification gate."""
    missing = [name for name in _VERIFICATION_FLAGS if name not in verification]
    if missing:
        raise ValueError(f"BSP verification diagnostics are missing flags: {missing}")
    timestamp = verified_at or dt.datetime.now(dt.UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("verified_at must be timezone-aware")
    scipy_matches = scipy_version == _REQUIRED_SCIPY_VERSION
    diagnostics = {
        **verification,
        "cache_sha256": _file_sha256(cache_path),
        "cache_manifest_fingerprint": manifest_fingerprint,
        "scipy_version": scipy_version,
        "required_scipy_version": _REQUIRED_SCIPY_VERSION,
        "scipy_version_matches_required": scipy_matches,
        "strict_max_reconstruction_error": verification["reconstruction_error_max"],
        "mean_reconstruction_error": verification["reconstruction_error_mean"],
        "p95_reconstruction_error": verification["reconstruction_error_p95"],
        "max_error_threshold": verification["reconstruction_error_threshold"],
        "strict_comparison": True,
        "code_sha": code_sha,
        "verified_at_utc": timestamp.astimezone(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    diagnostics["verification_passed"] = bool(
        scipy_matches and all(diagnostics[name] is True for name in _VERIFICATION_FLAGS)
    )
    json.dumps(diagnostics, allow_nan=False, sort_keys=True)
    return diagnostics


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON artifact atomically beside its persistent cache."""
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(
    mode: tyro.conf.EnumChoicesFromValues[PreparationMode],
    *,
    dataset_root: Path | None = None,
    cache_path: Path | None = None,
    diagnostics_path: Path | None = None,
    repo_id: str = LIBERO_REPO_ID,
    revision: str | None = LIBERO_REVISION,
    action_key: str = "actions",
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
    manifest = make_lerobot_cache_manifest(
        dataset,
        repo_id=repo_id,
        revision=revision,
        action_key=action_key,
    )

    if mode == PreparationMode.DOWNLOAD:
        requested_revision = revision or "LeRobot-compatible default"
        print(f"Downloaded and validated {repo_id}@{requested_revision} at {dataset_root}")
        print(f"Metadata: {metadata}")
        print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")
        return

    assert cache_path is not None
    if mode == PreparationMode.BUILD:
        cache = build_lerobot_bsp_cache(dataset, action_key=action_key)
        write_sidecar_cache(cache_path, cache, manifest)
        print(f"Built BSP sidecar at {cache_path}")
        print(f"Targets: {cache.targets.shape}; mapped frames: {cache.mapping.shape[0]}")
        print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")
        return

    cache = load_sidecar_cache(cache_path, manifest)
    verification = verify_lerobot_bsp_cache(dataset, cache, action_key=action_key)
    manifest_source = json.loads(manifest.source)
    diagnostics = make_verification_diagnostics(
        verification.diagnostics,
        cache_path=cache_path,
        manifest_fingerprint=manifest.fingerprint,
        scipy_version=manifest_source["scipy_version"],
        code_sha=_code_sha(),
    )
    diagnostics_output = diagnostics_path or Path(f"{cache_path}.verification.json")
    write_json_atomic(diagnostics_output, diagnostics)
    BspLeRobotDataset(dataset, cache, action_key=action_key)
    print(f"Verified BSP sidecar at {cache_path}")
    print(f"Targets: {cache.targets.shape}; mapped frames: {cache.mapping.shape[0]}")
    print(f"Snapshot/cache fingerprint: {manifest.fingerprint}")
    print(f"Verification diagnostics: {diagnostics_output}")
    if not diagnostics["verification_passed"]:
        failed = [name for name in _VERIFICATION_FLAGS if diagnostics[name] is not True]
        if not diagnostics["scipy_version_matches_required"]:
            failed.append("scipy_version_matches_required")
        raise ValueError(f"BSP sidecar verification failed: {failed}")


if __name__ == "__main__":
    tyro.cli(main)
