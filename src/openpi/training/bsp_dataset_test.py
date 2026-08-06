"""Behavior tests for precomputed BSP targets in the LeRobot data path."""

import dataclasses
import json

import numpy as np
import pytest

from openpi.training.bsp import BspCache
import openpi.training.bsp_dataset as bsp_dataset
from openpi.training.bsp_dataset import BspLeRobotDataset
from openpi.training.bsp_dataset import LIBERO_REVISION
from openpi.training.bsp_dataset import LeRobotDatasetMetadata
from openpi.training.bsp_dataset import build_lerobot_bsp_cache
from openpi.training.bsp_dataset import make_lerobot_cache_manifest
from openpi.training.bsp_dataset import verify_lerobot_bsp_cache
from openpi.training.bsp_dataset import validate_lerobot_dataset


class TinyHfDataset:
    """Small, real in-memory analogue of the locked dataset's HF table."""

    def __init__(self, actions: np.ndarray, fingerprint: str = "tiny-snapshot"):
        self._actions = actions
        self._fingerprint = fingerprint
        self.features = {
            "actions": {"dtype": "float32", "shape": [7]},
            "state": {"dtype": "float32", "shape": [8]},
        }

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


def test_wrapper_materializes_current_frame_knots_without_mutating_compact_cache():
    """Leaving cached episode-local knots absolute makes the policy predict the observation timestamp."""
    dataset = TinyLeRobotDataset()
    targets = np.zeros((2, 16, 8), dtype=np.float32)
    targets[0, :, :7] = 11.0
    targets[0, :, 7] = np.arange(16, dtype=np.float32)
    targets[1, :, :7] = 22.0
    targets[1, :, 7] = np.arange(16, dtype=np.float32) + 4.0
    original_targets = targets.copy()
    cache = BspCache(
        targets=targets,
        mapping=np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32),
    )
    wrapped = BspLeRobotDataset(dataset, cache)

    first_episode = wrapped[5]
    second_episode_start = wrapped[8]
    final_frame = wrapped[-1]

    np.testing.assert_array_equal(first_episode["actions"][:, :7], targets[1, :, :7])
    np.testing.assert_array_equal(first_episode["actions"][:, 7], targets[1, :, 7] - 5.0)
    np.testing.assert_array_equal(second_episode_start["actions"][:, 7], targets[0, :, 7])
    np.testing.assert_array_equal(final_frame["actions"][:, 7], targets[1, :, 7] - 7.0)
    np.testing.assert_array_equal(cache.targets, original_targets)
    assert not np.shares_memory(first_episode["actions"], cache.targets)


def test_wrapper_normalizes_negative_indices_and_rejects_out_of_range_values():
    """Applying episode offsets to raw negative indices can silently use the wrong knot origin."""
    dataset = TinyLeRobotDataset()
    targets = np.zeros((1, 16, 8), dtype=np.float32)
    targets[0, :, 7] = 9.0
    wrapped = BspLeRobotDataset(
        dataset,
        BspCache(targets=targets, mapping=np.zeros(len(dataset), dtype=np.uint32)),
    )

    np.testing.assert_array_equal(wrapped[-len(dataset)]["actions"][:, 7], np.full(16, 9.0))
    np.testing.assert_array_equal(wrapped[-1]["actions"][:, 7], np.full(16, 2.0))
    with pytest.raises(IndexError):
        wrapped[-len(dataset) - 1]
    with pytest.raises(IndexError):
        wrapped[len(dataset)]


def test_wrapper_materialization_never_refits_an_episode(monkeypatch):
    """Worker-time FITPACK calls would make training slow and nondeterministic."""
    dataset = TinyLeRobotDataset()
    cache = BspCache(
        targets=np.zeros((1, 16, 8), dtype=np.float32),
        mapping=np.zeros(len(dataset), dtype=np.uint32),
    )
    wrapped = BspLeRobotDataset(dataset, cache)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("FITPACK fitting entered BspLeRobotDataset.__getitem__")

    monkeypatch.setattr(bsp_dataset, "build_episode_targets", fail_if_called)

    assert wrapped[3]["actions"].shape == (16, 8)


def test_wrapper_refuses_a_mapping_that_does_not_cover_every_standard_sample():
    """A partial global mapping could silently pair observations with the wrong episode."""
    dataset = TinyLeRobotDataset()
    cache = BspCache(
        targets=np.zeros((1, 16, 8), dtype=np.float32),
        mapping=np.zeros(15, dtype=np.uint32),
    )

    with pytest.raises(ValueError, match="16 samples"):
        BspLeRobotDataset(dataset, cache)


def test_snapshot_manifest_changes_with_the_hf_table_fingerprint(monkeypatch):
    """Reusing a sidecar after the underlying table changes must fail fingerprint validation."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")
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


def test_snapshot_manifest_changes_with_episode_boundaries(monkeypatch):
    """A sidecar built for different episode splits must be stale even when the table itself is unchanged."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")
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


def test_manifest_covers_exact_schema_action_key_scipy_and_materialization_semantics(monkeypatch):
    """Omitting a producer or layout input can make an incompatible sidecar look current."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")

    manifest = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="actions",
        expected_metadata=TINY_METADATA,
    )
    source = json.loads(manifest.source)
    protocol = json.loads(manifest.protocol)

    assert manifest.format_version == 2
    assert source["scipy_version"] == "1.15.3"
    assert source["feature_schema"] == dataset.hf_dataset.features
    assert source["action_key"] == "actions"
    assert protocol["time_axis"] == "episode_local_frame_index"
    assert protocol["cached_knot_origin"] == "episode_start"
    assert protocol["materialized_knot_origin"] == "current_episode_local_frame"
    assert protocol["channel_layout"] == "controls[0:7],knot[7]"

    different_action_key = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="action",
        expected_metadata=TINY_METADATA,
    )
    dataset.hf_dataset.features = {
        "actions": {"dtype": "float64", "shape": [7]},
        "state": {"dtype": "float32", "shape": [8]},
    }
    different_schema = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="actions",
        expected_metadata=TINY_METADATA,
    )
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.4")
    different_scipy = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="actions",
        expected_metadata=TINY_METADATA,
    )

    assert len({manifest.fingerprint, different_action_key.fingerprint, different_schema.fingerprint}) == 3
    assert different_scipy.fingerprint != different_schema.fingerprint


def test_official_cache_manifest_records_the_real_requested_revision(monkeypatch):
    """A nonexistent requested revision can silently resolve to another Hub snapshot in locked LeRobot."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")

    manifest = make_lerobot_cache_manifest(
        dataset,
        repo_id="physical-intelligence/libero",
        revision=LIBERO_REVISION,
        expected_metadata=TINY_METADATA,
    )

    assert LIBERO_REVISION == "v2.0"
    assert json.loads(manifest.source)["revision"] == "v2.0"


def test_feature_schema_canonicalization_ignores_mapping_insertion_order(monkeypatch):
    """Equivalent Hugging Face schemas must yield one identity across preparation processes."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")
    first = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="actions",
        expected_metadata=TINY_METADATA,
    )
    dataset.hf_dataset.features = {
        "state": {"shape": [8], "dtype": "float32"},
        "actions": {"shape": [7], "dtype": "float32"},
    }
    second = make_lerobot_cache_manifest(
        dataset,
        repo_id="example/libero",
        revision="snapshot-a",
        action_key="actions",
        expected_metadata=TINY_METADATA,
    )

    assert first.fingerprint == second.fingerprint


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


def test_verification_rebuilds_all_episodes_and_reports_each_independent_invariant(monkeypatch):
    """A shape-only verification can miss tolerance, padding, mapping, and cross-episode corruption."""
    dataset = TinyLeRobotDataset()
    monkeypatch.setattr(bsp_dataset.importlib.metadata, "version", lambda package: "1.15.3")
    cache = build_lerobot_bsp_cache(dataset, expected_metadata=TINY_METADATA)

    verification = verify_lerobot_bsp_cache(
        dataset,
        cache,
        expected_metadata=TINY_METADATA,
    )

    diagnostics = verification.diagnostics
    assert verification.rebuilt_cache.targets is not cache.targets
    assert diagnostics["episode_count"] == 2
    assert diagnostics["frame_count"] == 16
    assert diagnostics["segment_count"] == cache.targets.shape[0]
    assert diagnostics["strict_reconstruction_tolerance"] is True
    assert diagnostics["reconstruction_error_max"] < diagnostics["reconstruction_error_threshold"]
    assert diagnostics["reconstruction_error_mean"] >= 0.0
    assert diagnostics["reconstruction_error_p95"] >= 0.0
    assert diagnostics["ground_truth_knots_nondecreasing"] is True
    assert diagnostics["tail_padding_valid"] is True
    assert diagnostics["future_segment_mapping_valid"] is True
    assert diagnostics["target_index_bounds_valid"] is True
    assert diagnostics["no_cross_episode_mapping"] is True
    assert diagnostics["all_frames_covered"] is True
    assert diagnostics["targets_match_rebuild"] is True
    assert diagnostics["mapping_matches_rebuild"] is True
    assert diagnostics["cache_contents_deterministic"] is True
    assert diagnostics["cache_contents_sha256"] == diagnostics["rebuilt_contents_sha256"]


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
