"""B-spline targets and durable sidecar storage for LIBERO training.

This is a small, standalone port of the FITPACK chunking and knot repair
behavior in the source below; the author project is never an OpenPI runtime
dependency.
"""

# Derived source repository: https://github.com/B-spline-policy/bspline-policy
# Derived source revision: 61ed5f42fced971d50a89b46417493790876ccd1
# Derived source path: bspline_policy/bspline_policy/common/bspline_action.py
#
# MIT License
#
# Copyright (c) 2026 Haoyu Xiong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Mapping

import numpy as np


_CACHE_FORMAT_VERSION = 1
_KNOT_PROJECTION_EPSILON = 1e-6


class BspCacheValidationError(ValueError):
    """Raised when a sidecar cannot safely be used for the requested data."""


@dataclasses.dataclass(frozen=True)
class BspSettings:
    """The fixed BSP protocol shared by the LIBERO trainer and policy server."""

    degree: int = 3
    chunk_size: int = 10
    target_rows: int = 16
    action_dim: int = 7
    target_channels: int = 8
    max_abs_error: float = 0.002
    smoothing: float = 1e-12
    stride: int = 1
    relative_knots: bool = False
    decoded_actions: int = 8

    def __post_init__(self) -> None:
        expected = {
            "degree": 3,
            "chunk_size": 10,
            "target_rows": 16,
            "action_dim": 7,
            "target_channels": 8,
            "max_abs_error": 0.002,
            "smoothing": 1e-12,
            "stride": 1,
            "relative_knots": False,
            "decoded_actions": 8,
        }
        actual = dataclasses.asdict(self)
        mismatches = {
            key: (actual[key], value) for key, value in expected.items() if actual[key] != value
        }
        if mismatches:
            raise ValueError(f"BSP protocol settings are fixed; received incompatible values: {mismatches}")
        if self.target_rows != self.chunk_size + 2 * self.degree:
            raise ValueError("BSP target_rows must equal chunk_size + 2 * degree")

    @property
    def control_rows(self) -> int:
        """Number of controls represented by one 16-knot cubic spline."""
        return self.target_rows - self.degree - 1


@dataclasses.dataclass(frozen=True)
class BspEpisode:
    """Unique BSP targets for one episode and its frame-to-target mapping."""

    targets: np.ndarray
    mapping: np.ndarray


@dataclasses.dataclass(frozen=True)
class BspCache:
    """Compact cache payload: unique target rows plus global frame mapping."""

    targets: np.ndarray
    mapping: np.ndarray


@dataclasses.dataclass(frozen=True)
class BspCacheManifest:
    """Versioned, deterministic description of the data used to build a cache."""

    fingerprint: str
    source: str
    protocol: str
    format_version: int = _CACHE_FORMAT_VERSION

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    def validate(self) -> None:
        """Reject malformed or internally inconsistent manifest contents."""
        if self.format_version != _CACHE_FORMAT_VERSION:
            raise BspCacheValidationError(
                f"Unsupported BSP cache format {self.format_version}; expected {_CACHE_FORMAT_VERSION}"
            )
        expected_fingerprint = _fingerprint_manifest(self.source, self.protocol)
        if self.fingerprint != expected_fingerprint:
            raise BspCacheValidationError("BSP cache manifest fingerprint does not match its contents")

    @classmethod
    def from_json(cls, value: str) -> BspCacheManifest:
        try:
            parsed = json.loads(value)
            manifest = cls(
                fingerprint=str(parsed["fingerprint"]),
                source=str(parsed["source"]),
                protocol=str(parsed["protocol"]),
                format_version=int(parsed["format_version"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise BspCacheValidationError("BSP cache manifest is malformed") from error
        manifest.validate()
        return manifest


def _fingerprint_manifest(source_json: str, protocol_json: str) -> str:
    return hashlib.sha256(
        f"bsp-cache-v{_CACHE_FORMAT_VERSION}\n{protocol_json}\n{source_json}".encode("utf-8")
    ).hexdigest()


def make_cache_manifest(source: Mapping[str, Any], settings: BspSettings | None = None) -> BspCacheManifest:
    """Return a stable cache identity for a dataset snapshot and fixed protocol."""
    settings = settings or BspSettings()
    try:
        source_json = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise ValueError("BSP cache source must be JSON serializable") from error
    protocol_json = json.dumps(dataclasses.asdict(settings), sort_keys=True, separators=(",", ":"))
    fingerprint = _fingerprint_manifest(source_json, protocol_json)
    return BspCacheManifest(fingerprint=fingerprint, source=source_json, protocol=protocol_json)


def pad_segment_rows(rows: np.ndarray, target_rows: int = 16) -> np.ndarray:
    """Right-pad a knot/control slice by repeating its final row."""
    rows = np.asarray(rows)
    if rows.ndim < 1 or rows.shape[0] == 0:
        raise ValueError("Cannot pad an empty BSP segment")
    if rows.shape[0] > target_rows:
        raise ValueError(f"BSP segment has {rows.shape[0]} rows; expected at most {target_rows}")
    if rows.shape[0] == target_rows:
        return rows.copy()
    padding = np.repeat(rows[-1:], target_rows - rows.shape[0], axis=0)
    return np.concatenate((rows, padding), axis=0)


def map_timesteps_to_future_segments(episode_length: int, segment_starts: np.ndarray) -> np.ndarray:
    """Map each frame to the segment beginning at or immediately after it."""
    if episode_length < 1:
        raise ValueError("BSP episodes must contain at least one frame")
    starts = np.asarray(segment_starts, dtype=np.float64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("BSP segment starts must be a non-empty one-dimensional array")
    if not np.isfinite(starts).all() or np.any(np.diff(starts) < 0):
        raise ValueError("BSP segment starts must be finite and non-decreasing")
    frame_indices = np.arange(episode_length, dtype=np.float64)
    future_segment = np.searchsorted(starts, frame_indices, side="left")
    return np.minimum(future_segment, starts.size - 1).astype(np.uint32, copy=False)


def project_knots(knots: np.ndarray) -> np.ndarray:
    """Repair descending learned knots using the author policy's exact projection."""
    projected = np.asarray(knots, dtype=np.float64).copy()
    if projected.ndim != 1 or projected.size == 0:
        raise ValueError("BSP knots must be a non-empty one-dimensional array")
    if not np.isfinite(projected).all():
        raise ValueError("BSP knots must be finite")
    for index in range(1, projected.size):
        if projected[index] < projected[index - 1]:
            projected[index] = projected[index - 1] + _KNOT_PROJECTION_EPSILON
    return projected


def _validate_episode_actions(actions: np.ndarray, settings: BspSettings) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != settings.action_dim:
        raise ValueError(
            f"BSP episode actions must have shape (frames, {settings.action_dim}), got {array.shape}"
        )
    if array.shape[0] < settings.degree + 1:
        raise ValueError(f"BSP cubic fitting requires at least {settings.degree + 1} action frames")
    if not np.isfinite(array).all():
        raise ValueError("BSP episode actions must be finite")
    return array


def _fit_full_episode(actions: np.ndarray, settings: BspSettings) -> tuple[np.ndarray, np.ndarray]:
    """Fit the entire episode with FITPACK's adaptive knot generator."""
    # Import lazily so static tooling can inspect this module without SciPy installed.
    from scipy.interpolate import generate_knots
    from scipy.interpolate import make_lsq_spline

    frame_indices = np.arange(actions.shape[0], dtype=np.float64)
    last_error: float | None = None
    for knots in generate_knots(frame_indices, actions, s=settings.smoothing):
        spline = make_lsq_spline(frame_indices, actions, knots, k=settings.degree)
        reconstruction = spline(frame_indices)
        error = float(np.max(np.abs(reconstruction - actions)))
        last_error = error
        if error <= settings.max_abs_error:
            full_knots, controls, _ = spline.tck
            return np.asarray(full_knots, dtype=np.float64), np.asarray(controls, dtype=np.float64)
    if last_error is None:
        raise ValueError("FITPACK did not produce a candidate BSP knot vector")
    raise ValueError(
        "BSP fitting exceeded max_abs_error "
        f"{settings.max_abs_error}: best candidate error was {last_error}"
    )


def build_episode_targets(actions: np.ndarray, settings: BspSettings | None = None) -> BspEpisode:
    """Fit one complete episode and emit compact controls-first BSP targets.

    Each target has shape ``[16, 8]``: channels ``0:7`` are controls and
    channel ``7`` is the frame-index knot vector.  All 16 rows are retained
    for training; cubic decoding uses the first 12 controls.
    """
    settings = settings or BspSettings()
    episode_actions = _validate_episode_actions(actions, settings)
    full_knots, controls = _fit_full_episode(episode_actions, settings)
    unique_knots = full_knots[settings.degree : -settings.degree]
    if unique_knots.size < 2:
        raise ValueError("FITPACK produced too few unique BSP knots for chunking")

    targets: list[np.ndarray] = []
    starts: list[float] = []
    for start_index in range(0, unique_knots.size - 1, settings.stride):
        knot_rows = pad_segment_rows(
            full_knots[start_index : start_index + settings.target_rows], settings.target_rows
        )
        control_rows = pad_segment_rows(
            controls[start_index : start_index + settings.target_rows], settings.target_rows
        )
        target = np.empty((settings.target_rows, settings.target_channels), dtype=np.float32)
        target[:, : settings.action_dim] = control_rows
        target[:, settings.action_dim] = knot_rows
        targets.append(target)
        starts.append(float(knot_rows[settings.degree]))

    target_array = np.stack(targets, axis=0)
    mapping = map_timesteps_to_future_segments(episode_actions.shape[0], np.asarray(starts))
    return BspEpisode(targets=target_array, mapping=mapping)


def decode_actions(target: np.ndarray, settings: BspSettings | None = None) -> np.ndarray:
    """Decode eight actions from one controls-first BSP target without extrapolation."""
    # Import lazily so module import does not make SciPy a transitive requirement.
    from scipy.interpolate import BSpline

    settings = settings or BspSettings()
    parameters = np.asarray(target, dtype=np.float64)
    expected_shape = (settings.target_rows, settings.target_channels)
    if parameters.shape != expected_shape:
        raise ValueError(f"BSP target must have shape {expected_shape}, got {parameters.shape}")
    if not np.isfinite(parameters).all():
        raise ValueError("BSP target parameters must be finite")
    knots = project_knots(parameters[:, settings.action_dim])
    controls = parameters[: settings.control_rows, : settings.action_dim]
    t_min = knots[settings.degree]
    t_max = knots[-(settings.degree + 1)]
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")
    evaluation_times = np.linspace(t_min, t_max, settings.decoded_actions, dtype=np.float64)
    decoded = BSpline(knots, controls, settings.degree, extrapolate=False)(evaluation_times)
    if not np.isfinite(decoded).all():
        raise ValueError("BSP decoding produced non-finite actions")
    return np.asarray(decoded, dtype=np.float32)


def _validate_cache(cache: BspCache, settings: BspSettings) -> None:
    targets = np.asarray(cache.targets)
    mapping = np.asarray(cache.mapping)
    if targets.dtype != np.float32:
        raise BspCacheValidationError(f"BSP cache targets must be float32, got {targets.dtype}")
    if targets.ndim != 3 or targets.shape[1:] != (settings.target_rows, settings.target_channels):
        raise BspCacheValidationError(
            "BSP cache targets must have shape "
            f"(segments, {settings.target_rows}, {settings.target_channels}), got {targets.shape}"
        )
    if not np.isfinite(targets).all():
        raise BspCacheValidationError("BSP cache targets contain non-finite values")
    if mapping.dtype != np.uint32 or mapping.ndim != 1:
        raise BspCacheValidationError("BSP cache mapping must be a one-dimensional uint32 array")
    if mapping.size and targets.shape[0] == 0:
        raise BspCacheValidationError("BSP cache mapping refers to an empty target array")
    if mapping.size and int(mapping.max()) >= targets.shape[0]:
        raise BspCacheValidationError("BSP cache mapping contains an out-of-range target index")


def _cache_lock(path: Path):
    """Construct the runtime lock lazily, keeping importing this module lightweight."""
    from filelock import FileLock

    return FileLock(f"{path}.lock")


def write_sidecar_cache(
    path: str | os.PathLike[str],
    cache: BspCache,
    manifest: BspCacheManifest,
    settings: BspSettings | None = None,
) -> None:
    """Atomically write a validated sidecar under an inter-process file lock."""
    settings = settings or BspSettings()
    _validate_cache(cache, settings)
    manifest.validate()
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock(cache_path):
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.", suffix=".tmp.npz", dir=cache_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                np.savez_compressed(
                    output,
                    targets=cache.targets,
                    mapping=cache.mapping,
                    manifest=np.asarray(manifest.to_json()),
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def load_sidecar_cache(
    path: str | os.PathLike[str],
    expected_manifest: BspCacheManifest,
    settings: BspSettings | None = None,
) -> BspCache:
    """Load a sidecar only when its version, shape, and fingerprint are valid."""
    settings = settings or BspSettings()
    expected_manifest.validate()
    cache_path = Path(path)
    with _cache_lock(cache_path):
        if not cache_path.is_file():
            raise BspCacheValidationError(f"BSP sidecar cache does not exist: {cache_path}")
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                required = {"targets", "mapping", "manifest"}
                missing = required.difference(archive.files)
                if missing:
                    raise BspCacheValidationError(f"BSP cache is missing required arrays: {sorted(missing)}")
                saved_manifest = BspCacheManifest.from_json(str(archive["manifest"].item()))
                cache = BspCache(targets=archive["targets"].copy(), mapping=archive["mapping"].copy())
        except (OSError, ValueError) as error:
            if isinstance(error, BspCacheValidationError):
                raise
            raise BspCacheValidationError(f"Unable to read BSP sidecar cache: {cache_path}") from error
    if saved_manifest.fingerprint != expected_manifest.fingerprint:
        raise BspCacheValidationError(
            "BSP cache fingerprint mismatch: cache was built for a different dataset snapshot or protocol"
        )
    _validate_cache(cache, settings)
    return cache


# Readable aliases for call sites that prefer cache-oriented terminology.
save_sidecar_cache = write_sidecar_cache
load_bsp_cache = load_sidecar_cache
