"""LeRobot integration for precomputed LIBERO BSP target sidecars."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import operator
from typing import Any
from typing import Protocol
from typing import SupportsIndex
from typing import TypeVar

import numpy as np

from openpi.training.bsp import BspCache
from openpi.training.bsp import BspCacheManifest
from openpi.training.bsp import BspSettings
from openpi.training.bsp import build_episode_targets
from openpi.training.bsp import make_cache_manifest


LIBERO_REPO_ID = "physical-intelligence/libero"
LIBERO_REVISION = "v2.1"


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
        starts = np.asarray(episode_index["from"], dtype=np.int64)
        ends = np.asarray(episode_index["to"], dtype=np.int64)
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
    expected_metadata: LeRobotDatasetMetadata = LIBERO_METADATA,
    settings: BspSettings | None = None,
) -> BspCacheManifest:
    """Fingerprint the concrete HF table and its requested Hub revision."""
    observed = validate_lerobot_dataset(dataset, expected_metadata)
    try:
        hf_fingerprint = str(dataset.hf_dataset._fingerprint)
    except AttributeError as error:
        raise ValueError("LeRobot hf_dataset does not expose its snapshot fingerprint") from error
    if not hf_fingerprint:
        raise ValueError("LeRobot hf_dataset snapshot fingerprint is empty")
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
        "episode_data_index_sha256": boundary_hash.hexdigest(),
        "metadata": dataclasses.asdict(observed),
    }
    return make_cache_manifest(source, settings)


def _episode_actions(dataset: Any, start: int, end: int, action_key: str) -> np.ndarray:
    try:
        rows = dataset.hf_dataset[start:end]
        actions = np.asarray(rows[action_key], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LeRobot hf_dataset does not contain usable raw '{action_key}' actions") from error
    if actions.shape != (end - start, BspSettings().action_dim):
        raise ValueError(
            f"LeRobot episode actions must have shape ({end - start}, {BspSettings().action_dim}), "
            f"got {actions.shape}"
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
    for start, end in zip(starts, ends):
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

    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Any]:
        frame_index = operator.index(index)
        sample = self._dataset[frame_index]
        if self._action_key not in sample:
            raise KeyError(f"LeRobot sample does not contain action key '{self._action_key}'")
        result = dict(sample)
        result[self._action_key] = self._targets[int(self._mapping[frame_index])]
        return result

    def __len__(self) -> int:
        return len(self._dataset)
