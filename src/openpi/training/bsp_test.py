"""Focused contract tests for the BSP training primitives."""

import dataclasses

import numpy as np
import pytest

import openpi.training.bsp as bsp
from openpi.training.bsp import BspCache
from openpi.training.bsp import BspCacheValidationError
from openpi.training.bsp import BspSettings
from openpi.training.bsp import build_episode_targets
from openpi.training.bsp import decode_actions
from openpi.training.bsp import load_sidecar_cache
from openpi.training.bsp import make_cache_manifest
from openpi.training.bsp import map_timesteps_to_future_segments
from openpi.training.bsp import pad_segment_rows
from openpi.training.bsp import project_knots
from openpi.training.bsp import write_sidecar_cache


def test_settings_are_fixed_to_the_libero_bsp_protocol():
    """Changing a protocol value would make training and decoding incompatible."""
    settings = BspSettings()

    assert dataclasses.is_dataclass(settings)
    assert settings.degree == 3
    assert settings.chunk_size == 10
    assert settings.target_rows == 16
    assert settings.action_dim == 7
    assert settings.target_channels == 8
    assert settings.max_abs_error == 0.002
    assert settings.smoothing == 1e-12
    assert settings.stride == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.degree = 2


def test_episode_targets_keep_controls_first_and_use_twelve_controls():
    """Swapping the knot channel or treating padded controls as active breaks decode."""
    actions = np.stack([np.arange(7, dtype=np.float64) + frame for frame in range(20)])

    episode = build_episode_targets(actions)

    assert episode.targets.dtype == np.float32
    assert episode.targets.shape[1:] == (16, 8)
    assert episode.mapping.dtype == np.uint32
    assert episode.mapping.shape == (20,)
    assert np.isfinite(episode.targets).all()
    assert np.isfinite(episode.targets[:, :, 7]).all()

    target = np.zeros((16, 8), dtype=np.float32)
    target[:, 7] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    target[:12, :7] = np.arange(12, dtype=np.float32)[:, None]
    expected = decode_actions(target)
    target[12:, :7] = 1000.0
    np.testing.assert_allclose(decode_actions(target), expected)


def test_mapping_selects_the_nearest_future_segment_and_pads_at_episode_end():
    """Mapping a frame to a past segment or dropping final frames loses training data."""
    mapping = map_timesteps_to_future_segments(
        episode_length=11,
        segment_starts=np.asarray([0.0, 4.0, 8.0]),
    )

    np.testing.assert_array_equal(
        mapping,
        np.asarray([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.uint32),
    )
    padded = pad_segment_rows(np.asarray([[1.0], [2.0], [3.0]]), target_rows=16)
    assert padded.shape == (16, 1)
    np.testing.assert_array_equal(padded[-3:], np.asarray([[3.0], [3.0], [3.0]]))


def test_knot_projection_repairs_only_descending_values():
    """A projection that changes valid equal knots would not be code-faithful."""
    projected = project_knots(np.asarray([0.0, 2.0, 1.0, 1.0, 4.0]))

    np.testing.assert_allclose(projected, [0.0, 2.0, 2.000001, 2.000002, 4.0])
    np.testing.assert_array_equal(project_knots(np.asarray([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "target",
    [
        np.zeros((15, 8), dtype=np.float32),
        np.full((16, 8), np.nan, dtype=np.float32),
        np.zeros((16, 8), dtype=np.float32),
    ],
)
def test_decode_rejects_invalid_target_parameters(target):
    """Invalid learned parameters must fail rather than extrapolate an action."""
    with pytest.raises(ValueError):
        decode_actions(target)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((3, 7), dtype=np.float32),
        np.zeros((4, 6), dtype=np.float32),
        np.full((4, 7), np.nan, dtype=np.float32),
    ],
)
def test_episode_build_rejects_short_malformed_and_nonfinite_actions(actions):
    """Bad episodes must be rejected before FITPACK sees ambiguous input."""
    with pytest.raises(ValueError):
        build_episode_targets(actions)


def test_cache_fingerprint_is_stable_and_rejects_stale_sidecars(tmp_path):
    """A cache built for another dataset identity must never be silently reused."""
    manifest = make_cache_manifest({"repo_id": "libero", "revision": "abc"})
    same_manifest = make_cache_manifest({"revision": "abc", "repo_id": "libero"})
    stale_manifest = make_cache_manifest({"repo_id": "libero", "revision": "changed"})
    assert manifest.fingerprint == same_manifest.fingerprint
    assert manifest.fingerprint != stale_manifest.fingerprint

    cache = BspCache(
        targets=np.zeros((1, 16, 8), dtype=np.float32),
        mapping=np.asarray([0, 0], dtype=np.uint32),
    )
    cache_path = tmp_path / "libero.bsp.npz"
    write_sidecar_cache(cache_path, cache, manifest)

    loaded = load_sidecar_cache(cache_path, manifest)
    np.testing.assert_array_equal(loaded.targets, cache.targets)
    np.testing.assert_array_equal(loaded.mapping, cache.mapping)
    with pytest.raises(BspCacheValidationError, match="fingerprint"):
        load_sidecar_cache(cache_path, stale_manifest)


def test_cache_reader_waits_for_writer_publication_under_the_shared_lock(tmp_path, monkeypatch):
    """A first reader must see a writer's atomic publication rather than fail early."""
    manifest = make_cache_manifest({"repo_id": "libero", "revision": "abc"})
    cache = BspCache(
        targets=np.zeros((1, 16, 8), dtype=np.float32),
        mapping=np.asarray([0], dtype=np.uint32),
    )
    cache_path = tmp_path / "published-while-waiting.npz"

    class PublishingLock:
        def __enter__(self):
            with cache_path.open("wb") as output:
                np.savez_compressed(
                    output,
                    targets=cache.targets,
                    mapping=cache.mapping,
                    manifest=np.asarray(manifest.to_json()),
                )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(bsp, "_cache_lock", lambda _path: PublishingLock())

    loaded = load_sidecar_cache(cache_path, manifest)

    np.testing.assert_array_equal(loaded.targets, cache.targets)
    np.testing.assert_array_equal(loaded.mapping, cache.mapping)
