"""LeRobot integration for precomputed LIBERO BSP target sidecars."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import importlib.metadata
import json
import operator
from typing import Any, Protocol, SupportsIndex, TypeVar

import numpy as np

from openpi.training.bsp import BspCache
from openpi.training.bsp import BspCacheManifest
from openpi.training.bsp import BspSettings
from openpi.training.bsp import build_episode_targets
from openpi.training.bsp import build_episode_targets_with_artifacts
from openpi.training.bsp import make_cache_manifest

LIBERO_REPO_ID = "physical-intelligence/libero"
LIBERO_REVISION = "v2.0"


@dataclasses.dataclass(frozen=True)
class LeRobotDatasetMetadata:
    """Metadata that identifies the supported LIBERO dataset contents."""

    episodes: int
    frames: int
    tasks: int
    fps: int


LIBERO_METADATA = LeRobotDatasetMetadata(episodes=1693, frames=273465, tasks=40, fps=10)


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co: ...

    def __len__(self) -> int: ...


def _episode_boundaries(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        episode_index = dataset.episode_data_index
        starts = np.array(episode_index["from"], dtype=np.int64, copy=True)
        ends = np.array(episode_index["to"], dtype=np.int64, copy=True)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("LeRobot dataset has invalid episode_data_index boundaries") from error
    if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
        raise ValueError("LeRobot episode_data_index boundaries must be equal-length vectors")
    if starts.size == 0 or starts[0] != 0 or np.any(ends <= starts):
        raise ValueError("LeRobot episode_data_index contains empty or nonzero-origin episodes")
    if np.any(starts[1:] != ends[:-1]):
        raise ValueError("LeRobot episode_data_index must cover contiguous global frames")
    return starts, ends


def observed_lerobot_metadata(dataset: Any) -> LeRobotDatasetMetadata:
    """Read counts only from APIs exposed by the repository's locked LeRobot."""
    starts, ends = _episode_boundaries(dataset)
    try:
        frame_count = len(dataset.hf_dataset)
        tasks = dataset.meta.tasks
        fps = dataset.meta.fps
    except AttributeError as error:
        raise ValueError("LeRobot dataset does not expose hf_dataset and metadata") from error
    if int(ends[-1]) != frame_count:
        raise ValueError(
            f"LeRobot episode_data_index ends at frame {int(ends[-1])}, but hf_dataset frames={frame_count}"
        )
    return LeRobotDatasetMetadata(
        episodes=int(starts.size),
        frames=int(frame_count),
        tasks=len(tasks),
        fps=int(fps),
    )


def validate_lerobot_dataset(
    dataset: Any,
    expected: LeRobotDatasetMetadata = LIBERO_METADATA,
) -> LeRobotDatasetMetadata:
    """Reject a LeRobot snapshot that is not the expected LIBERO corpus."""
    observed = observed_lerobot_metadata(dataset)
    mismatches = {
        field.name: (getattr(observed, field.name), getattr(expected, field.name))
        for field in dataclasses.fields(expected)
        if getattr(observed, field.name) != getattr(expected, field.name)
    }
    if mismatches:
        details = ", ".join(f"{name}={actual} (expected {wanted})" for name, (actual, wanted) in mismatches.items())
        raise ValueError(f"Unexpected LeRobot LIBERO metadata: {details}")
    return observed


def make_lerobot_cache_manifest(
    dataset: Any,
    *,
    repo_id: str,
    revision: str | None,
    action_key: str = "actions",
    expected_metadata: LeRobotDatasetMetadata = LIBERO_METADATA,
    settings: BspSettings | None = None,
) -> BspCacheManifest:
    """Fingerprint the concrete HF table and its requested Hub revision."""
    observed = validate_lerobot_dataset(dataset, expected_metadata)
    try:
        hf_fingerprint = str(dataset.hf_dataset._fingerprint)  # noqa: SLF001 -- HF snapshot identity field.
    except AttributeError as error:
        raise ValueError("LeRobot hf_dataset does not expose its snapshot fingerprint") from error
    if not hf_fingerprint:
        raise ValueError("LeRobot hf_dataset snapshot fingerprint is empty")
    if not isinstance(action_key, str) or not action_key:
        raise ValueError("LeRobot BSP action_key must be a non-empty string")
    try:
        features = dataset.hf_dataset.features
    except AttributeError as error:
        raise ValueError("LeRobot hf_dataset does not expose its feature schema") from error
    if hasattr(features, "to_dict"):
        features = features.to_dict()
    try:
        feature_schema = json.loads(json.dumps(features, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    except (TypeError, ValueError) as error:
        raise ValueError("LeRobot hf_dataset feature schema is not canonically JSON serializable") from error
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("SciPy must be installed to construct the BSP cache identity") from error
    starts, ends = _episode_boundaries(dataset)
    boundary_hash = hashlib.sha256()
    boundary_hash.update(starts.astype("<i8", copy=False).tobytes())
    boundary_hash.update(ends.astype("<i8", copy=False).tobytes())
    metadata_revision = getattr(dataset.meta, "revision", None)
    source = {
        "repo_id": repo_id,
        "revision": revision,
        "metadata_revision": None if metadata_revision is None else str(metadata_revision),
        "hf_dataset_fingerprint": hf_fingerprint,
        "feature_schema": feature_schema,
        "action_key": action_key,
        "scipy_version": scipy_version,
        "episode_data_index_sha256": boundary_hash.hexdigest(),
        "metadata": dataclasses.asdict(observed),
    }
    return make_cache_manifest(source, settings)


def bsp_cache_contents_sha256(cache: BspCache) -> str:
    """Hash canonical array metadata and bytes, independent of NPZ container encoding."""
    hasher = hashlib.sha256(b"openpi-bsp-cache-contents-v1\0")
    arrays = (
        ("targets", np.asarray(cache.targets, dtype="<f4")),
        ("mapping", np.asarray(cache.mapping, dtype="<u4")),
    )
    for name, array in arrays:
        contiguous = np.ascontiguousarray(array)
        descriptor = json.dumps(
            {"name": name, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        hasher.update(descriptor)
        hasher.update(b"\0")
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _episode_actions(dataset: Any, start: int, end: int, action_key: str) -> np.ndarray:
    try:
        rows = dataset.hf_dataset[start:end]
        raw_actions = rows[action_key]
        try:
            action_rows = iter(raw_actions)
        except TypeError:
            actions = np.asarray(raw_actions, dtype=np.float32)
        else:
            actions = np.stack(
                [np.asarray(action_row, dtype=np.float32) for action_row in action_rows],
                axis=0,
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LeRobot hf_dataset does not contain usable raw '{action_key}' actions") from error
    if actions.shape != (end - start, BspSettings().action_dim):
        raise ValueError(
            f"LeRobot episode actions must have shape ({end - start}, {BspSettings().action_dim}), got {actions.shape}"
        )
    return actions


def build_lerobot_bsp_cache(
    dataset: Any,
    *,
    expected_metadata: LeRobotDatasetMetadata = LIBERO_METADATA,
    action_key: str = "actions",
    settings: BspSettings | None = None,
) -> BspCache:
    """Fit each complete raw-action episode and concatenate a global sidecar."""
    settings = settings or BspSettings()
    validate_lerobot_dataset(dataset, expected_metadata)
    starts, ends = _episode_boundaries(dataset)
    target_parts: list[np.ndarray] = []
    mapping_parts: list[np.ndarray] = []
    target_offset = 0
    for start, end in zip(starts, ends, strict=True):
        episode = build_episode_targets(
            _episode_actions(dataset, int(start), int(end), action_key),
            settings,
        )
        target_parts.append(episode.targets)
        mapping_parts.append(episode.mapping + target_offset)
        target_offset += episode.targets.shape[0]
    return BspCache(
        targets=np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
        mapping=np.concatenate(mapping_parts, axis=0).astype(np.uint32, copy=False),
    )


class BspLeRobotDataset(Dataset[Mapping[str, Any]]):
    """Preserve the standard LeRobot sample and replace only its action value."""

    def __init__(self, dataset: Dataset[Mapping[str, Any]], cache: BspCache, *, action_key: str = "actions"):
        if len(cache.mapping) != len(dataset):
            raise ValueError(
                f"BSP global mapping has {len(cache.mapping)} frames but the LeRobot dataset has {len(dataset)} samples"
            )
        if cache.mapping.size and (cache.targets.shape[0] == 0 or int(cache.mapping.max()) >= cache.targets.shape[0]):
            raise ValueError("BSP global mapping contains an out-of-range target index")
        self._dataset = dataset
        self._targets = cache.targets
        self._mapping = cache.mapping
        self._action_key = action_key
        self._knot_channel = BspSettings().action_dim
        self._episode_starts, self._episode_ends = _episode_boundaries(dataset)
        if int(self._episode_ends[-1]) != len(dataset):
            raise ValueError("BSP episode boundaries must cover every LeRobot sample")

    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Any]:
        frame_index = operator.index(index)
        if frame_index < 0:
            frame_index += len(self)
        if frame_index < 0 or frame_index >= len(self):
            raise IndexError(f"LeRobot sample index {operator.index(index)} is out of range")
        sample = self._dataset[frame_index]
        if self._action_key not in sample:
            raise KeyError(f"LeRobot sample does not contain action key '{self._action_key}'")
        result = dict(sample)
        episode_index = int(np.searchsorted(self._episode_ends, frame_index, side="right"))
        local_frame_index = frame_index - int(self._episode_starts[episode_index])
        target = self._targets[int(self._mapping[frame_index])].copy()
        target[:, self._knot_channel] -= local_frame_index
        result[self._action_key] = target
        return result

    def __len__(self) -> int:
        return len(self._dataset)


@dataclasses.dataclass(frozen=True)
class BspCacheVerification:
    """A deterministic rebuild plus JSON-safe verification measurements."""

    rebuilt_cache: BspCache
    diagnostics: Mapping[str, Any]


def _expected_future_mapping(segment_starts: np.ndarray, episode_length: int) -> np.ndarray:
    """Independently derive nearest-future indices without using the production mapper."""
    expected = np.empty(episode_length, dtype=np.uint32)
    segment_index = 0
    for frame_index in range(episode_length):
        while segment_index < len(segment_starts) - 1 and segment_starts[segment_index] < frame_index:
            segment_index += 1
        expected[frame_index] = segment_index
    return expected


def verify_lerobot_bsp_cache(
    dataset: Any,
    cache: BspCache,
    *,
    expected_metadata: LeRobotDatasetMetadata = LIBERO_METADATA,
    action_key: str = "actions",
    settings: BspSettings | None = None,
) -> BspCacheVerification:
    """Rebuild raw full episodes and audit cache contents outside the training path."""
    settings = settings or BspSettings()
    observed = validate_lerobot_dataset(dataset, expected_metadata)
    starts, ends = _episode_boundaries(dataset)
    target_parts: list[np.ndarray] = []
    mapping_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    target_offset = 0
    ground_truth_knots_nondecreasing = True
    tail_padding_valid = True
    tail_padding_observed = False
    future_segment_mapping_valid = True
    target_index_bounds_valid = cache.mapping.ndim == 1
    no_cross_episode_mapping = True

    for start_value, end_value in zip(starts, ends, strict=True):
        start = int(start_value)
        end = int(end_value)
        actions = _episode_actions(dataset, start, end, action_key)
        episode, artifacts = build_episode_targets_with_artifacts(actions, settings)
        target_count = episode.targets.shape[0]
        target_end = target_offset + target_count
        target_parts.append(episode.targets)
        mapping_parts.append(episode.mapping + target_offset)
        error_parts.append(artifacts.absolute_errors.reshape(-1))

        ground_truth_knots_nondecreasing &= bool(np.all(np.diff(artifacts.full_knots) >= 0.0))
        unique_knots = artifacts.full_knots[settings.degree : -settings.degree]
        for segment_index, source_index in enumerate(range(0, unique_knots.size - 1, settings.stride)):
            target = episode.targets[segment_index]
            knot_rows = min(settings.target_rows, artifacts.full_knots.shape[0] - source_index)
            control_rows = min(settings.target_rows, artifacts.controls.shape[0] - source_index)
            if knot_rows < settings.target_rows:
                tail_padding_observed = True
                tail_padding_valid &= bool(
                    np.all(target[knot_rows:, settings.action_dim] == target[knot_rows - 1, settings.action_dim])
                )
            if control_rows < settings.target_rows:
                tail_padding_observed = True
                tail_padding_valid &= bool(
                    np.all(
                        target[control_rows:, : settings.action_dim] == target[control_rows - 1, : settings.action_dim]
                    )
                )

        segment_starts = episode.targets[:, settings.degree, settings.action_dim].astype(np.float64)
        independently_expected = _expected_future_mapping(segment_starts, end - start)
        future_segment_mapping_valid &= bool(np.array_equal(episode.mapping, independently_expected))

        cached_episode_mapping = np.asarray(cache.mapping[start:end])
        target_index_bounds_valid &= bool(
            cached_episode_mapping.shape == (end - start,) and np.all(cached_episode_mapping < cache.targets.shape[0])
        )
        no_cross_episode_mapping &= bool(
            cached_episode_mapping.shape == (end - start,)
            and np.all(cached_episode_mapping >= target_offset)
            and np.all(cached_episode_mapping < target_end)
        )
        target_offset = target_end

    rebuilt_cache = BspCache(
        targets=np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
        mapping=np.concatenate(mapping_parts, axis=0).astype(np.uint32, copy=False),
    )
    errors = np.concatenate(error_parts).astype(np.float64, copy=False)
    error_max = float(np.max(errors))
    targets_match = bool(np.array_equal(cache.targets, rebuilt_cache.targets))
    mapping_matches = bool(np.array_equal(cache.mapping, rebuilt_cache.mapping))
    cache_hash = bsp_cache_contents_sha256(cache)
    rebuilt_hash = bsp_cache_contents_sha256(rebuilt_cache)
    all_frames_covered = bool(
        cache.mapping.shape == (observed.frames,)
        and sum(int(end - start) for start, end in zip(starts, ends, strict=True)) == observed.frames
    )
    diagnostics = {
        "episode_count": observed.episodes,
        "frame_count": observed.frames,
        "segment_count": int(cache.targets.shape[0]),
        "reconstruction_error_max": error_max,
        "reconstruction_error_mean": float(np.mean(errors)),
        "reconstruction_error_p95": float(np.percentile(errors, 95)),
        "reconstruction_error_threshold": settings.max_abs_error,
        "strict_reconstruction_tolerance": bool(error_max < settings.max_abs_error),
        "ground_truth_knots_nondecreasing": ground_truth_knots_nondecreasing,
        "tail_padding_valid": bool(tail_padding_observed and tail_padding_valid),
        "future_segment_mapping_valid": future_segment_mapping_valid,
        "target_index_bounds_valid": target_index_bounds_valid,
        "no_cross_episode_mapping": no_cross_episode_mapping,
        "all_frames_covered": all_frames_covered,
        "targets_match_rebuild": targets_match,
        "mapping_matches_rebuild": mapping_matches,
        "cache_contents_sha256": cache_hash,
        "rebuilt_contents_sha256": rebuilt_hash,
        "cache_contents_deterministic": bool(targets_match and mapping_matches and cache_hash == rebuilt_hash),
    }
    return BspCacheVerification(rebuilt_cache=rebuilt_cache, diagnostics=diagnostics)
