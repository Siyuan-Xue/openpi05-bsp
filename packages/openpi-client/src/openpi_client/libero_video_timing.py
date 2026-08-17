"""Dependency-free timing primitives for selected LIBERO video artifacts.

The evaluator owns measurement.  This module only records those measurements
and deterministically turns control-rate frames into a video-rate timeline.
It deliberately has no knowledge of dataset FPS or demonstration timestamps.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
import copy
import dataclasses
from typing import TypeVar


CONTROL_HZ = 20
DEFAULT_VIDEO_FPS = 40
NANOSECONDS_PER_SECOND = 1_000_000_000

_Frame = TypeVar("_Frame")


def _require_nonnegative_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_video_frequencies(
    *, control_hz: int = CONTROL_HZ, video_fps: int = DEFAULT_VIDEO_FPS
) -> int:
    """Return output frames per control frame for the fixed 20 Hz protocol."""
    if isinstance(control_hz, bool) or not isinstance(control_hz, int) or control_hz != CONTROL_HZ:
        raise ValueError(f"LIBERO control frequency must be exactly {CONTROL_HZ} Hz")
    if isinstance(video_fps, bool) or not isinstance(video_fps, int) or video_fps <= 0:
        raise ValueError("Video FPS must be a positive integer")
    if video_fps % control_hz:
        raise ValueError("Video FPS must be a positive multiple of the control frequency")
    return video_fps // control_hz


@dataclasses.dataclass(frozen=True)
class InferenceRequest:
    """A measured model-request interval, independent of control blocking."""

    replan_index: int
    started_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.replan_index, name="replan_index")
        _require_nonnegative_integer(self.started_ns, name="started_ns")
        _require_nonnegative_integer(self.completed_ns, name="completed_ns")
        if self.completed_ns < self.started_ns:
            raise ValueError("completed_ns must not precede started_ns")

    @property
    def duration_ns(self) -> int:
        return self.completed_ns - self.started_ns

    def to_dict(self) -> dict[str, int]:
        return {
            "replan_index": self.replan_index,
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "latency_ns": self.duration_ns,
        }


@dataclasses.dataclass(frozen=True)
class ControlStall:
    """A measured interval in which control could not advance for a request."""

    replan_index: int
    started_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.replan_index, name="replan_index")
        _require_nonnegative_integer(self.started_ns, name="started_ns")
        _require_nonnegative_integer(self.completed_ns, name="completed_ns")
        if self.completed_ns < self.started_ns:
            raise ValueError("completed_ns must not precede started_ns")

    @property
    def duration_ns(self) -> int:
        return self.completed_ns - self.started_ns

    def to_dict(self) -> dict[str, int]:
        return {
            "replan_index": self.replan_index,
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "stall_ns": self.duration_ns,
        }


def expand_control_frames(
    control_frames: Iterable[_Frame], *, control_hz: int = CONTROL_HZ, video_fps: int = DEFAULT_VIDEO_FPS
) -> tuple[_Frame, ...]:
    """Hold every control frame for an integral number of video frames."""
    hold_count = validate_video_frequencies(control_hz=control_hz, video_fps=video_fps)
    return tuple(frame for frame in control_frames for _ in range(hold_count))


def quantize_stall_frames(
    stalls: Sequence[ControlStall], *, video_fps: int = DEFAULT_VIDEO_FPS
) -> tuple[int, ...]:
    """Quantize stalls cumulatively, preserving sub-frame time across events."""
    validate_video_frequencies(video_fps=video_fps)
    accumulated_ns = 0
    emitted_frames = 0
    event_frames = []
    for stall in stalls:
        if not isinstance(stall, ControlStall):
            raise TypeError("stalls must contain ControlStall records")
        accumulated_ns += stall.duration_ns
        total_frames = accumulated_ns * video_fps // NANOSECONDS_PER_SECOND
        event_frames.append(total_frames - emitted_frames)
        emitted_frames = total_frames
    return tuple(event_frames)


def render_overlay(
    frame: _Frame,
    lines: Sequence[str],
    *,
    renderer: Callable[[_Frame, tuple[str, ...]], _Frame],
) -> _Frame:
    """Render a timing overlay on a deep copy of a renderer-owned frame."""
    return renderer(copy.deepcopy(frame), tuple(lines))


@dataclasses.dataclass(frozen=True)
class VideoTimingAudit:
    """Exact, dependency-free accounting for one encoded video timeline."""

    control_hz: int
    video_fps: int
    control_frame_count: int
    held_frame_count: int
    stall_frame_count: int
    video_frame_count: int
    control_duration_ns: int
    request_count: int
    request_latency_ns: int
    stall_count: int
    control_stall_ns: int
    video_duration_ns: int
    expected_duration_ns: int
    duration_deviation_ns: int

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def build_video_audit(
    *,
    control_frame_count: int,
    requests: Sequence[InferenceRequest],
    stalls: Sequence[ControlStall],
    control_hz: int = CONTROL_HZ,
    video_fps: int = DEFAULT_VIDEO_FPS,
) -> VideoTimingAudit:
    """Calculate a video audit using integers, never raw latency as a stall proxy."""
    _require_nonnegative_integer(control_frame_count, name="control_frame_count")
    hold_count = validate_video_frequencies(control_hz=control_hz, video_fps=video_fps)
    if any(not isinstance(request, InferenceRequest) for request in requests):
        raise TypeError("requests must contain InferenceRequest records")
    if any(not isinstance(stall, ControlStall) for stall in stalls):
        raise TypeError("stalls must contain ControlStall records")

    held_frame_count = control_frame_count * hold_count
    stall_frame_count = sum(quantize_stall_frames(stalls, video_fps=video_fps))
    video_frame_count = held_frame_count + stall_frame_count
    control_duration_ns = control_frame_count * NANOSECONDS_PER_SECOND // control_hz
    control_stall_ns = sum(stall.duration_ns for stall in stalls)
    video_duration_ns = video_frame_count * NANOSECONDS_PER_SECOND // video_fps
    expected_duration_ns = control_duration_ns + control_stall_ns
    return VideoTimingAudit(
        control_hz=control_hz,
        video_fps=video_fps,
        control_frame_count=control_frame_count,
        held_frame_count=held_frame_count,
        stall_frame_count=stall_frame_count,
        video_frame_count=video_frame_count,
        control_duration_ns=control_duration_ns,
        request_count=len(requests),
        request_latency_ns=sum(request.duration_ns for request in requests),
        stall_count=len(stalls),
        control_stall_ns=control_stall_ns,
        video_duration_ns=video_duration_ns,
        expected_duration_ns=expected_duration_ns,
        duration_deviation_ns=video_duration_ns - expected_duration_ns,
    )
