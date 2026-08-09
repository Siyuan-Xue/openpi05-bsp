"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


_STATE_STATS_RTOL = 1e-7
_STATE_STATS_ATOL = 1e-8
_STATE_FIELDS = ("mean", "std", "q01", "q99")
_BSP_ACTION_DIM = 8
_BSP_KNOT_CHANNEL = 7
_MIN_BSP_KNOT_QUANTILE_SPAN = 1e-6


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def norm_stats_output_path(assets_dir: Path, *, asset_id: str | None) -> Path:
    """Select the same asset namespace that DataConfigFactory loads during training."""
    if asset_id is None:
        raise ValueError("Data config must have an asset_id before normalization stats can be written")
    return assets_dir / asset_id


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _stats_sha256(stats: normalize.NormStats) -> str:
    payload = {}
    for field in _STATE_FIELDS:
        value = getattr(stats, field)
        if value is None:
            raise ValueError(f"Normalization statistics are missing required field {field!r}")
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"Normalization statistics field {field!r} contains non-finite values")
        payload[field] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_bsp_action_stats(stats: normalize.NormStats) -> None:
    """Reject knot statistics that would make BSP quantile normalization unstable."""
    q01 = np.asarray(stats.q01)
    q99 = np.asarray(stats.q99)
    if q01.shape != (_BSP_ACTION_DIM,) or q99.shape != (_BSP_ACTION_DIM,):
        raise ValueError(
            "BSP action normalization statistics must contain exactly "
            f"{_BSP_ACTION_DIM} channels, got q01={q01.shape} and q99={q99.shape}"
        )
    knot_q01 = float(q01[_BSP_KNOT_CHANNEL])
    knot_q99 = float(q99[_BSP_KNOT_CHANNEL])
    knot_span = knot_q99 - knot_q01
    if not np.isfinite(knot_span) or knot_span <= _MIN_BSP_KNOT_QUANTILE_SPAN:
        raise ValueError(
            "BSP knot quantile interval is degenerate: "
            f"q01={knot_q01}, q99={knot_q99}, span={knot_span}"
        )


def compare_norm_stats_assets(
    baseline_dir: Path,
    bsp_dir: Path,
    *,
    rtol: float = _STATE_STATS_RTOL,
    atol: float = _STATE_STATS_ATOL,
) -> dict:
    """Compare all state fields numerically and prove action assets are separate."""
    baseline_dir = baseline_dir.expanduser().resolve()
    bsp_dir = bsp_dir.expanduser().resolve()
    if baseline_dir == bsp_dir:
        raise ValueError("Baseline and BSP normalization stats require distinct asset directories")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Normalization comparison tolerances must be non-negative")

    baseline_path = baseline_dir / "norm_stats.json"
    bsp_path = bsp_dir / "norm_stats.json"
    baseline = normalize.load(baseline_dir)
    bsp = normalize.load(bsp_dir)
    for name, stats in (("baseline", baseline), ("bsp", bsp)):
        missing = {"state", "actions"}.difference(stats)
        if missing:
            raise ValueError(f"{name} normalization stats are missing keys: {sorted(missing)}")
    _validate_bsp_action_stats(bsp["actions"])

    state_fields = {}
    for field in _STATE_FIELDS:
        baseline_value = getattr(baseline["state"], field)
        bsp_value = getattr(bsp["state"], field)
        if baseline_value is None or bsp_value is None:
            raise ValueError(f"State normalization statistics are missing required field {field!r}")
        baseline_array = np.asarray(baseline_value)
        bsp_array = np.asarray(bsp_value)
        same_shape = baseline_array.shape == bsp_array.shape
        equal = bool(
            same_shape
            and np.isfinite(baseline_array).all()
            and np.isfinite(bsp_array).all()
            and np.allclose(baseline_array, bsp_array, rtol=rtol, atol=atol, equal_nan=False)
        )
        state_fields[field] = {
            "equal": equal,
            "shape": list(baseline_array.shape) if same_shape else None,
            "baseline_shape": list(baseline_array.shape),
            "bsp_shape": list(bsp_array.shape),
            "max_abs_difference": (
                float(np.max(np.abs(baseline_array - bsp_array))) if same_shape and baseline_array.size else None
            ),
        }

    baseline_action_hash = _stats_sha256(baseline["actions"])
    bsp_action_hash = _stats_sha256(bsp["actions"])
    action_stats_isolated = baseline_action_hash != bsp_action_hash
    comparison = {
        "baseline_asset_dir": str(baseline_dir),
        "bsp_asset_dir": str(bsp_dir),
        "baseline_norm_stats_sha256": _file_sha256(baseline_path),
        "bsp_norm_stats_sha256": _file_sha256(bsp_path),
        "baseline_action_stats_sha256": baseline_action_hash,
        "bsp_action_stats_sha256": bsp_action_hash,
        "asset_directories_isolated": True,
        "action_stats_isolated": action_stats_isolated,
        "state_stats_equal": all(result["equal"] for result in state_fields.values()),
        "state_fields": state_fields,
        "rtol": rtol,
        "atol": atol,
    }
    json.dumps(comparison, allow_nan=False, sort_keys=True)
    return comparison


def write_norm_comparison(path: Path, comparison: dict) -> None:
    """Write a machine-readable normalization gate artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, allow_nan=False, indent=2, sort_keys=True) + "\n")


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    assets_dir: Path | None = None,
    bsp_cache_path: Path | None = None,
    dataset_root: Path | None = None,
    compare_state_stats_with: Path | None = None,
    norm_comparison_output: Path | None = None,
):
    if norm_comparison_output is not None and compare_state_stats_with is None:
        raise ValueError("norm_comparison_output requires compare_state_stats_with")
    config = _config.get_config(config_name)
    data_factory = config.data
    overrides = {}
    if bsp_cache_path is not None:
        if not hasattr(data_factory, "bsp_cache_path"):
            raise ValueError(f"Config {config_name!r} does not support a BSP cache path")
        overrides["bsp_cache_path"] = str(bsp_cache_path.expanduser().resolve())
    if dataset_root is not None:
        if not hasattr(data_factory, "lerobot_root"):
            raise ValueError(f"Config {config_name!r} does not support a LeRobot dataset root")
        overrides["lerobot_root"] = str(dataset_root.expanduser().resolve())
    if overrides:
        data_factory = dataclasses.replace(data_factory, **overrides)

    persistent_assets_dir = (assets_dir or config.assets_dirs).expanduser().resolve()
    data_config = data_factory.create(persistent_assets_dir, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    if data_config.use_bsp:
        _validate_bsp_action_stats(norm_stats["actions"])

    # Model transforms are intentionally absent above: BSP stats therefore see compact [16, 8]
    # targets, while baseline stats see raw [16, 7] actions, before either representation is padded.
    output_path = norm_stats_output_path(persistent_assets_dir, asset_id=data_config.asset_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    if compare_state_stats_with is not None:
        reference_path = compare_state_stats_with.expanduser().resolve()
        if data_config.asset_id == "libero_bsp_h16":
            baseline_path, bsp_path = reference_path, output_path
        elif data_config.asset_id == "libero_baseline_h16":
            baseline_path, bsp_path = output_path, reference_path
        else:
            raise ValueError("State-stats comparison is only defined for the phase-one LIBERO baseline/BSP assets")
        comparison = compare_norm_stats_assets(baseline_path, bsp_path)
        comparison_path = (
            norm_comparison_output.expanduser().resolve()
            if norm_comparison_output is not None
            else bsp_path / "norm_comparison.json"
        )
        write_norm_comparison(comparison_path, comparison)
        print(f"Writing norm comparison to: {comparison_path}")
        failed = [
            name
            for name in ("state_stats_equal", "asset_directories_isolated", "action_stats_isolated")
            if comparison[name] is not True
        ]
        if failed:
            raise ValueError(f"Baseline/BSP normalization comparison failed: {failed}")


if __name__ == "__main__":
    tyro.cli(main)
