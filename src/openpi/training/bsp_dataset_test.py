"""Behavior tests for precomputed BSP targets in the LeRobot data path."""

import dataclasses

import numpy as np
import pytest

from openpi.training.bsp import BspCache
from openpi.training.bsp_dataset import BspLeRobotDataset
from openpi.training.bsp_dataset import LeRobotDatasetMetadata
from openpi.training.bsp_dataset import build_lerobot_bsp_cache
from openpi.training.bsp_dataset import make_lerobot_cache_manifest
from openpi.training.bsp_dataset import validate_lerobot_dataset


class TinyHfDataset:
    """Small, real in-memory analogue of the locked dataset's HF table."""

    def __init__(self, actions: np.ndarray, fingerprint: str = "tiny-snapshot"):
        self._actions = actions
        self._fingerprint = fingerprint

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return {"actions": self._actions[index]}
        return {"actions": self._actions[index]}


class TinyLeRobotDataset:
    """Exercises wrapper behavior without mocking LeRobot calls."""

    def __init__(self):
        first_episode = np.stack([np.arange(7, dtype=np.float32) + frame for frame in range(8)])
        second_episode = np.stack([np.arange(7, dtype=np.float32) - frame for frame in range(8)])
        actions = np.concatenate((first_episode, second_episode), axis=0)
        self.hf_dataset = TinyHfDataset(actions)
        self.episode_data_index = {
            "from": np.asarray([0, 8], dtype=np.int64),
            "to": np.asarray([8, 16], dtype=np.int64),
        }
        self.meta = type("Meta", (), {"fps": 10, "tasks": {0: "first", 1: "second"}})()
        self.samples = [
            {
                "image": np.full((2, 2, 3), frame, dtype=np.uint8),
                "state": np.asarray([frame, frame + 1], dtype=np.float32),
                "actions": np.full((16, 7), -frame, dtype=np.float32),
                "task": frame % 2,
                "prompt": f"task {frame % 2}",
            }
            for frame in range(16)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


TINY_METADATA = LeRobotDatasetMetadata(episodes=2, frames=16, tasks=2, fps=10)


def test_wrapper_preserves_standard_sample_and_replaces_only_mapped_actions():
    """Copying observations or using the frame index as a target index breaks the wrapper contract."""
    dataset = TinyLeRobotDataset()
    targets = np.stack(
        [np.full((16, 8), target_index, dtype=np.float32) for target_index in range(3)]
    )
    cache = BspCache(
        targets=targets,
        mapping=np.asarray([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint32),
    )
    wrapped = BspLeRobotDataset(dataset, cache)

    sample = wrapped[0]

    assert sample is not dataset.samples[0]
    assert sample["image"] is dataset.samples[0]["image"]
    assert sample["state"] is dataset.samples[0]["state"]
    assert sample["task"] == dataset.samples[0]["task"]
    assert sample["prompt"] == dataset.samples[0]["prompt"]
    np.testing.assert_array_equal(sample["actions"], targets[2])
    np.testing.assert_array_equal(dataset.samples[0]["actions"], np.zeros((16, 7), dtype=np.float32))


def test_wrapper_refuses_a_mapping_that_does_not_cover_every_standard_sample():
    """A partial global mapping could silently pair observations with the wrong episode."""
    dataset = TinyLeRobotDataset()
    cache = BspCache(
        targets=np.zeros((1, 16, 8), dtype=np.float32),
        mapping=np.zeros(15, dtype=np.uint32),
    )

    with pytest.raises(ValueError, match="16 samples"):
        BspLeRobotDataset(dataset, cache)


def test_snapshot_manifest_changes_with_the_hf_table_fingerprint():
    """Reusing a sidecar after the underlying table changes must fail fingerprint validation."""
    dataset = TinyLeRobotDataset()
    validate_lerobot_dataset(dataset, TINY_METADATA)
    first = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        expected_metadata=TINY_METADATA,
    )
    dataset.hf_dataset._fingerprint = "changed-snapshot"
    second = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        expected_metadata=TINY_METADATA,
    )

    assert first.fingerprint != second.fingerprint


def test_snapshot_manifest_changes_with_episode_boundaries():
    """A sidecar built for different episode splits must be stale even when the table itself is unchanged."""
    dataset = TinyLeRobotDataset()
    first = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision=None,
        expected_metadata=TINY_METADATA,
    )
    dataset.episode_data_index = {
        "from": np.asarray([0, 7], dtype=np.int64),
        "to": np.asarray([7, 16], dtype=np.int64),
    }
    second = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision=None,
        expected_metadata=TINY_METADATA,
    )

    assert first.fingerprint != second.fingerprint


def test_build_uses_raw_seven_dimensional_episode_actions_and_global_offsets():
    """Fitting action windows or forgetting per-episode target offsets corrupts the global mapping."""
    dataset = TinyLeRobotDataset()

    cache = build_lerobot_bsp_cache(dataset, expected_metadata=TINY_METADATA)

    assert cache.targets.dtype == np.float32
    assert cache.targets.shape[1:] == (16, 8)
    assert cache.mapping.dtype == np.uint32
    assert cache.mapping.shape == (16,)
    second_episode_first_target = cache.mapping[8]
    assert second_episode_first_target > cache.mapping[:8].max()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_data_index", {"from": np.asarray([0]), "to": np.asarray([16])}, "episodes"),
        ("hf_dataset", TinyHfDataset(np.zeros((15, 7), dtype=np.float32)), "frames"),
        ("meta", type("Meta", (), {"fps": 30, "tasks": {0: "first", 1: "second"}})(), "fps"),
    ],
)
def test_metadata_validation_rejects_an_unexpected_libero_snapshot(field, value, message):
    """Building from a truncated or incompatible dataset must not publish a plausible sidecar."""
    dataset = TinyLeRobotDataset()
    setattr(dataset, field, value)

    with pytest.raises(ValueError, match=message):
        validate_lerobot_dataset(dataset, TINY_METADATA)


def test_metadata_expectations_are_immutable():
    """Mutation during a multi-mode preparation run could validate different data than was built."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        TINY_METADATA.frames = 15
