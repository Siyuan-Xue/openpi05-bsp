"""Dependency-free LIBERO evaluation identities, retry rules, and artifacts.

The simulator entry point deliberately delegates bookkeeping here so these
rules can be tested without importing LIBERO, MuJoCo, NumPy, or a GPU stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from openpi_client import libero_video_timing as _video_timing


SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
_SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    **{suite: suite for suite in SUPPORTED_SUITES},
}
_INFRASTRUCTURE_KINDS = ("simulator", "container", "network")
_STALL_REASONS = (
    _video_timing.STALL_REASON_SYNCHRONOUS_INFERENCE,
    _video_timing.STALL_REASON_ASYNC_ACTION_UNDERFLOW,
)
_OVERLAY_TYPE_BY_STALL_REASON = {
    _video_timing.STALL_REASON_SYNCHRONOUS_INFERENCE: "synchronous_inference",
    _video_timing.STALL_REASON_ASYNC_ACTION_UNDERFLOW: "waiting_for_policy_actions",
}
_OVERLAY_TYPES = tuple(_OVERLAY_TYPE_BY_STALL_REASON.values())
_REPLAN_ACTIONS = 8
_ACTION_DIM = 7
BSP_PARAMETERS = {
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
    "control_rows": 12,
    "control_selection": "first_12_rows",
    "channel_layout": "controls[0:7],knot[7]",
    "time_axis": "episode_local_frame_index",
    "cached_knot_origin": "episode_start",
    "materialized_knot_origin": "current_episode_local_frame",
    "decode_interval": "[knots[3],knots[-4]]",
    "projection_epsilon": 1e-6,
    "model_action_dim": 32,
    "model_action_horizon": 16,
    "executed_actions": 8,
}


def _require_exact_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _require_exact_string_tuple(
    value: Any,
    *,
    name: str,
    allowed_values: tuple[str, ...],
    expected_count: int | None = None,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if expected_count is not None and len(value) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} entries")
    if any(type(entry) is not str or entry not in allowed_values for entry in value):
        raise ValueError(f"{name} contains an unsupported value")
    return value


def _require_exact_nonnegative_integer_tuple(
    value: Any,
    *,
    name: str,
    expected_count: int,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(value) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} entries")
    for entry in value:
        _require_exact_integer(entry, name=f"{name} entry")
    return value


def _validate_stall_source_frames(
    stall_source_frames: Sequence[tuple[int, Any]], *, steps: int
) -> None:
    previous_step = -1
    for entry in stall_source_frames:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("stall_source_frames must contain (control_step, frame) tuples")
        control_step, _ = entry
        _require_exact_integer(control_step, name="stall source control_step")
        if control_step <= previous_step:
            raise ValueError("stall source control steps must be unique and strictly increasing")
        if control_step > steps:
            raise ValueError("stall source control_step cannot exceed executed rollout steps")
        previous_step = control_step


def resolve_suites(selection: str) -> tuple[str, ...]:
    """Resolve a CLI suite selection while explicitly excluding LIBERO-90."""
    normalized = selection.strip().lower()
    if normalized == "all":
        return SUPPORTED_SUITES
    try:
        return (_SUITE_ALIASES[normalized],)
    except KeyError as error:
        supported = ", ".join((*_SUITE_ALIASES, "all"))
        raise ValueError(f"Unsupported LIBERO suite {selection!r}; supported values: {supported}") from error


def resolve_task_ids(selection: Sequence[int] | None) -> tuple[int, ...]:
    """Return a deterministic non-empty subset of the ten tasks in every phase-one suite."""
    if selection is None:
        return tuple(range(10))
    task_ids = tuple(selection)
    if not task_ids:
        raise ValueError("At least one LIBERO task id is required")
    if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in task_ids):
        raise ValueError("LIBERO task ids must be integers")
    if any(task_id < 0 or task_id >= 10 for task_id in task_ids):
        raise ValueError("LIBERO task ids must be in the inclusive range 0..9")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("LIBERO task ids must be unique")
    return tuple(sorted(task_ids))


@dataclasses.dataclass(frozen=True)
class PolicyProtocol:
    """Strict server-output contract for one evaluator run."""

    name: str
    expected_action_horizon: int


def resolve_policy_protocol(policy_variant: str, expected_action_horizon: int | None) -> PolicyProtocol:
    """Resolve phase-one, calibration, or decoded-BSP output without shape relaxation."""
    if policy_variant == "baseline":
        horizon = 16 if expected_action_horizon is None else expected_action_horizon
        if horizon == 16:
            return PolicyProtocol(name="baseline_h16", expected_action_horizon=16)
        if horizon == 10:
            return PolicyProtocol(name="baseline_h10_calibration", expected_action_horizon=10)
        raise ValueError("Baseline expected_action_horizon must be 16 (phase one) or 10 (official calibration)")
    if policy_variant == "bsp":
        horizon = 8 if expected_action_horizon is None else expected_action_horizon
        if horizon != 8:
            raise ValueError("BSP policy output must be the decoded horizon-8 protocol")
        return PolicyProtocol(name="bsp_decoded_h8", expected_action_horizon=8)
    raise ValueError("Policy variant must be baseline or bsp")


def _safe_component(value: str, *, fallback: str) -> str:
    component = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return component or fallback


@dataclasses.dataclass(frozen=True)
class EpisodeIdentity:
    """The exact paired unit shared by baseline and BSP evaluation."""

    suite: str
    task_id: int
    task_name: str
    init_state_index: int
    init_state_fingerprint: str

    def __post_init__(self) -> None:
        if self.suite not in SUPPORTED_SUITES:
            raise ValueError(f"Unsupported LIBERO suite: {self.suite}")
        if self.task_id < 0 or self.init_state_index < 0:
            raise ValueError("Task and initial-state indices must be non-negative")
        if not self.init_state_fingerprint:
            raise ValueError("Initial-state fingerprint is required for paired evaluation")

    @property
    def paired_key(self) -> str:
        fingerprint = _safe_component(self.init_state_fingerprint, fallback="state")
        return f"{self.suite}/task-{self.task_id:03d}/init-{self.init_state_index:03d}/{fingerprint}"

    @property
    def episode_id(self) -> str:
        fingerprint = _safe_component(self.init_state_fingerprint, fallback="state")
        return f"{self.suite}-task-{self.task_id:03d}-init-{self.init_state_index:03d}-{fingerprint}"


def fingerprint_init_state(*, dtype: str, shape: Sequence[int], payload: bytes) -> str:
    """Fingerprint a concrete simulator initial state without importing NumPy."""
    descriptor = json.dumps(
        {"dtype": str(dtype), "shape": [int(value) for value in shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + payload).hexdigest()


def stable_replan_seed(eval_seed: int, identity: EpisodeIdentity, replan_index: int) -> int:
    """Derive an A/B-stable uint32 flow-noise seed from the paired identity."""
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) or eval_seed < 0:
        raise ValueError("Evaluation seed must be a non-negative integer")
    if replan_index < 0:
        raise ValueError("Replan index must be non-negative")
    payload = json.dumps(
        {
            "namespace": "openpi-libero-flow-noise-v1",
            "eval_seed": eval_seed,
            "suite": identity.suite,
            "task_id": identity.task_id,
            "init_state_index": identity.init_state_index,
            "init_state_fingerprint": identity.init_state_fingerprint,
            "replan_index": replan_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


class PolicyFailure(RuntimeError):
    """A model/server output failure that counts as an unsuccessful rollout."""


class InfrastructureFailure(RuntimeError):
    """A retryable failure outside the policy metric denominator."""

    def __init__(self, kind: str, message: str):
        if kind not in _INFRASTRUCTURE_KINDS:
            raise ValueError(f"Unsupported infrastructure failure kind: {kind}")
        super().__init__(message)
        self.kind = kind


def classify_exception(error: Exception, *, phase: str) -> PolicyFailure | InfrastructureFailure:
    """Classify an error from one narrow evaluator phase.

    Callers should catch exceptions around simulator, connection, and policy
    operations separately and pass the corresponding phase; this function is
    intentionally not a catch-all around the full rollout.
    """
    if isinstance(error, (PolicyFailure, InfrastructureFailure)):
        return error
    message = f"{type(error).__name__}: {error}"
    if phase in {"environment_create", "environment_reset", "environment_step"}:
        return InfrastructureFailure("simulator", message)
    if phase == "server_connect":
        return InfrastructureFailure("container", message)
    if phase == "policy_infer":
        websocket_disconnect = type(error).__module__.startswith("websockets") and type(error).__name__.startswith(
            "Connection"
        )
        if isinstance(error, (ConnectionError, TimeoutError, EOFError, OSError)) or websocket_disconnect:
            return InfrastructureFailure("network", message)
        return PolicyFailure(message)
    raise ValueError(f"Unknown evaluation error phase: {phase}")


def select_replan_actions(
    actions: Iterable[Iterable[Any]], *, expected_horizon: int | None = None
) -> tuple[tuple[float, ...], ...]:
    """Validate a native 7-D action chunk and return exactly its first 8 rows."""
    shape = getattr(actions, "shape", None)
    if shape is not None and (len(shape) != 2 or shape[0] < _REPLAN_ACTIONS or shape[1] != _ACTION_DIM):
        raise PolicyFailure(
            f"Policy actions must have two-dimensional shape (at least {_REPLAN_ACTIONS}, {_ACTION_DIM}), got {shape}"
        )
    try:
        rows = [tuple(float(value) for value in row) for row in actions]
    except (TypeError, ValueError, OverflowError) as error:
        raise PolicyFailure("Policy actions must be a finite two-dimensional numeric sequence") from error
    if len(rows) < _REPLAN_ACTIONS:
        raise PolicyFailure(f"Policy returned {len(rows)} actions; at least {_REPLAN_ACTIONS} are required")
    if expected_horizon is not None and len(rows) != expected_horizon:
        raise PolicyFailure(f"Policy returned {len(rows)} actions; expected exactly {expected_horizon}")
    if any(len(row) != _ACTION_DIM for row in rows):
        raise PolicyFailure(f"Policy actions must have exactly {_ACTION_DIM} dimensions")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise PolicyFailure("Policy actions contain non-finite values")
    return tuple(rows[:_REPLAN_ACTIONS])


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    success: bool
    steps: int
    replans: int
    failure_kind: str | None = None
    error: str | None = None
    inference_ms: tuple[float, ...] = ()
    inference_requests: tuple[_video_timing.InferenceRequest, ...] = ()
    control_stalls: tuple[_video_timing.ControlStall, ...] = ()
    replay_frames: tuple[Any, ...] = dataclasses.field(default=(), repr=False, compare=False)
    stall_source_frames: tuple[tuple[int, Any], ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.steps < 0 or self.replans < 0:
            raise ValueError("Rollout steps and replans must be non-negative")
        if self.success and self.failure_kind is not None:
            raise ValueError("A successful rollout cannot have a failure kind")
        if not self.success and self.failure_kind not in {"policy", "timeout"}:
            raise ValueError("A counted failure must be classified as policy or timeout")
        if any(not isinstance(event, _video_timing.InferenceRequest) for event in self.inference_requests):
            raise TypeError("inference_requests must contain InferenceRequest records")
        if any(not isinstance(event, _video_timing.ControlStall) for event in self.control_stalls):
            raise TypeError("control_stalls must contain ControlStall records")
        _validate_stall_source_frames(self.stall_source_frames, steps=self.steps)


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    identity: EpisodeIdentity
    eval_seed: int
    status: str
    success: bool | None
    include_in_success_rate: bool
    attempts: int
    failure_kind: str | None = None
    infrastructure_kind: str | None = None
    error: str | None = None
    steps: int = 0
    replans: int = 0
    inference_ms: tuple[float, ...] = ()
    inference_requests: tuple[_video_timing.InferenceRequest, ...] = ()
    control_stalls: tuple[_video_timing.ControlStall, ...] = ()
    infrastructure_history: tuple[Mapping[str, Any], ...] = ()
    replay_frames: tuple[Any, ...] = dataclasses.field(default=(), repr=False, compare=False)
    stall_source_frames: tuple[tuple[int, Any], ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_stall_source_frames(self.stall_source_frames, steps=self.steps)

    @classmethod
    def from_attempt(
        cls,
        identity: EpisodeIdentity,
        eval_seed: int,
        attempts: int,
        *,
        success: bool,
        failure_kind: str | None = None,
        error: str | None = None,
        result: AttemptResult | None = None,
        infrastructure_history: Sequence[Mapping[str, Any]] = (),
    ) -> "EpisodeRecord":
        if success:
            status = "success"
            failure_kind = None
        else:
            failure_kind = failure_kind or "policy"
            status = f"{failure_kind}_failure"
        return cls(
            identity=identity,
            eval_seed=eval_seed,
            status=status,
            success=success,
            include_in_success_rate=True,
            attempts=attempts,
            failure_kind=failure_kind,
            error=error if error is not None else (result.error if result else None),
            steps=result.steps if result else 0,
            replans=result.replans if result else 0,
            inference_ms=result.inference_ms if result else (),
            inference_requests=result.inference_requests if result else (),
            control_stalls=result.control_stalls if result else (),
            infrastructure_history=tuple(infrastructure_history),
            replay_frames=result.replay_frames if result else (),
            stall_source_frames=result.stall_source_frames if result else (),
        )

    @classmethod
    def infrastructure_incomplete(
        cls,
        identity: EpisodeIdentity,
        eval_seed: int,
        attempts: int,
        kind: str,
        error: str,
        *,
        infrastructure_history: Sequence[Mapping[str, Any]] = (),
    ) -> "EpisodeRecord":
        return cls(
            identity=identity,
            eval_seed=eval_seed,
            status="infrastructure_incomplete",
            success=None,
            include_in_success_rate=False,
            attempts=attempts,
            infrastructure_kind=kind,
            error=error,
            infrastructure_history=tuple(infrastructure_history),
        )

    def to_dict(self) -> dict[str, Any]:
        timings = list(self.inference_ms)
        return {
            "episode_id": self.identity.episode_id,
            "paired_key": self.identity.paired_key,
            "suite": self.identity.suite,
            "task_id": self.identity.task_id,
            "task_name": self.identity.task_name,
            "init_state_index": self.identity.init_state_index,
            "init_state_fingerprint": self.identity.init_state_fingerprint,
            "eval_seed": self.eval_seed,
            "status": self.status,
            "success": self.success,
            "include_in_success_rate": self.include_in_success_rate,
            "attempts": self.attempts,
            "failure_kind": self.failure_kind,
            "infrastructure_kind": self.infrastructure_kind,
            "error": self.error,
            "steps": self.steps,
            "replans": self.replans,
            "inference_ms": timings,
            "mean_inference_ms": sum(timings) / len(timings) if timings else None,
            "inference_requests": [event.to_dict() for event in self.inference_requests],
            "control_stalls": [event.to_dict() for event in self.control_stalls],
            "infrastructure_history": [dict(entry) for entry in self.infrastructure_history],
        }


def run_episode_with_retries(
    identity: EpisodeIdentity,
    attempt: Callable[[int], AttemptResult],
    *,
    eval_seed: int,
    infrastructure_retries: int = 2,
) -> EpisodeRecord:
    """Run at most three identical paired attempts, retrying only infrastructure."""
    if infrastructure_retries < 0:
        raise ValueError("Infrastructure retry count must be non-negative")
    infrastructure_history: list[dict[str, Any]] = []
    for attempt_number in range(1, infrastructure_retries + 2):
        try:
            result = attempt(attempt_number)
        except PolicyFailure as error:
            return EpisodeRecord.from_attempt(
                identity,
                eval_seed,
                attempt_number,
                success=False,
                failure_kind="policy",
                error=str(error),
                infrastructure_history=infrastructure_history,
            )
        except InfrastructureFailure as error:
            infrastructure_history.append(
                {"attempt": attempt_number, "kind": error.kind, "error": str(error)}
            )
            if attempt_number <= infrastructure_retries:
                continue
            return EpisodeRecord.infrastructure_incomplete(
                identity,
                eval_seed,
                attempt_number,
                error.kind,
                str(error),
                infrastructure_history=infrastructure_history,
            )
        return EpisodeRecord.from_attempt(
            identity,
            eval_seed,
            attempt_number,
            success=result.success,
            failure_kind=result.failure_kind,
            result=result,
            infrastructure_history=infrastructure_history,
        )
    raise AssertionError("Unreachable retry state")


def _rate(successes: int, eligible: int) -> float | None:
    return successes / eligible if eligible else None


def aggregate_records(
    records: Sequence[EpisodeRecord], *, artifact_errors: Sequence[ArtifactError] = ()
) -> dict[str, Any]:
    """Aggregate episode results without putting infrastructure into denominators."""
    task_groups: dict[tuple[str, int], list[EpisodeRecord]] = {}
    for record in records:
        task_groups.setdefault((record.identity.suite, record.identity.task_id), []).append(record)

    tasks: list[dict[str, Any]] = []
    for (suite, task_id), group in sorted(task_groups.items()):
        eligible = sum(record.include_in_success_rate for record in group)
        successes = sum(record.success is True for record in group)
        incomplete = sum(not record.include_in_success_rate for record in group)
        tasks.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_name": group[0].identity.task_name,
                "requested_episodes": len(group),
                "eligible_episodes": eligible,
                "successes": successes,
                "failures": eligible - successes,
                "incomplete_infrastructure_count": incomplete,
                "success_rate": _rate(successes, eligible),
            }
        )

    suites: list[dict[str, Any]] = []
    for suite in sorted({record.identity.suite for record in records}):
        suite_records = [record for record in records if record.identity.suite == suite]
        eligible = sum(record.include_in_success_rate for record in suite_records)
        successes = sum(record.success is True for record in suite_records)
        suite_tasks = [row for row in tasks if row["suite"] == suite]
        task_rates = [row["success_rate"] for row in suite_tasks if row["success_rate"] is not None]
        suites.append(
            {
                "suite": suite,
                "tasks": len(suite_tasks),
                "requested_episodes": len(suite_records),
                "eligible_episodes": eligible,
                "successes": successes,
                "failures": eligible - successes,
                "incomplete_infrastructure_count": sum(
                    not record.include_in_success_rate for record in suite_records
                ),
                "success_rate": _rate(successes, eligible),
                "task_macro_success_rate": sum(task_rates) / len(task_rates) if task_rates else None,
            }
        )

    suite_rates = [row["success_rate"] for row in suites if row["success_rate"] is not None]
    all_four_suites_evaluated = {row["suite"] for row in suites} == set(SUPPORTED_SUITES)
    suite_macro = sum(suite_rates) / len(suite_rates) if suite_rates else None
    incomplete_count = sum(not record.include_in_success_rate for record in records)
    return {
        "tasks": tasks,
        "suites": suites,
        "suite_macro_success_rate": suite_macro,
        "four_suite_macro_success_rate": (
            suite_macro if all_four_suites_evaluated and len(suite_rates) == len(SUPPORTED_SUITES) else None
        ),
        "evaluated_suite_count": len(suites),
        "all_four_suites_evaluated": all_four_suites_evaluated,
        "requested_episodes": len(records),
        "eligible_episodes": sum(record.include_in_success_rate for record in records),
        "successes": sum(record.success is True for record in records),
        "incomplete_infrastructure_count": incomplete_count,
        "artifact_error_count": len(artifact_errors),
        "acceptance_complete": incomplete_count == 0 and not artifact_errors,
    }


@dataclasses.dataclass(frozen=True)
class EvaluationManifest:
    code_sha: str
    dataset_revision: str
    config_name: str
    checkpoint_step: int
    bsp_cache_hash: str | None
    bsp_cache_manifest_fingerprint: str | None
    norm_hash: str
    checkpoint: str
    container_digest: str
    train_seed: int
    eval_seed: int
    policy_variant: str
    bsp_parameters: Mapping[str, Any]
    policy_protocol: str
    expected_action_horizon: int
    execution_horizon: int
    suites: Sequence[str] = SUPPORTED_SUITES
    task_ids: Sequence[int] = tuple(range(10))
    trials_per_task: int = 50
    num_steps_wait: int = 10
    max_steps_by_suite: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: dict(_MAX_STEPS_BY_SUITE)
    )
    connection_timeout_s: float = 30.0
    inference_timeout_s: float = 120.0
    infrastructure_retries: int = 2
    dataset_fps: int = 10
    source_demo_control_hz: int = 20
    control_freq_hz: int = _video_timing.CONTROL_HZ
    video_fps: int = _video_timing.DEFAULT_VIDEO_FPS
    video_show_inference_waits: bool = False
    inference_schedule: str = _video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE

    def __post_init__(self) -> None:
        if self.policy_variant not in {"baseline", "bsp"}:
            raise ValueError("Policy variant must be baseline or bsp")
        required_text = {
            "code_sha": self.code_sha,
            "dataset_revision": self.dataset_revision,
            "config_name": self.config_name,
            "norm_hash": self.norm_hash,
            "checkpoint": self.checkpoint,
            "container_digest": self.container_digest,
        }
        missing = sorted(
            key
            for key, value in required_text.items()
            if not isinstance(value, str) or not value
        )
        if missing:
            raise ValueError(f"Evaluation manifest has missing or non-string identities: {missing}")
        if not isinstance(self.code_sha, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.code_sha
        ) is None:
            raise ValueError("Evaluation code_sha must be a lowercase 40- or 64-character Git SHA")
        if self.dataset_revision != "v2.0":
            raise ValueError("Physical Intelligence LIBERO evaluation requires dataset revision v2.0")
        for name, value, expected in (
            ("dataset_fps", self.dataset_fps, 10),
            ("source_demo_control_hz", self.source_demo_control_hz, 20),
            ("control_freq_hz", self.control_freq_hz, _video_timing.CONTROL_HZ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Evaluation {name} must be exactly {expected}")
        _video_timing.validate_video_frequencies(
            control_hz=self.control_freq_hz,
            video_fps=self.video_fps,
        )
        if not isinstance(self.video_show_inference_waits, bool):
            raise ValueError("Evaluation video_show_inference_waits must be a boolean")
        _video_timing.validate_inference_schedule(self.inference_schedule)
        if not isinstance(self.container_digest, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.container_digest
        ) is None:
            raise ValueError("Evaluation container_digest must be a lowercase sha256 digest")
        _require_exact_integer(self.checkpoint_step, name="Evaluation checkpoint_step")
        _require_exact_integer(self.train_seed, name="Evaluation train_seed")
        _require_exact_integer(self.eval_seed, name="Evaluation eval_seed")
        if not isinstance(self.norm_hash, str) or len(self.norm_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.norm_hash
        ):
            raise ValueError("norm_hash must be the lowercase SHA256 of norm_stats.json")
        cache_identities_present = (self.bsp_cache_hash is not None, self.bsp_cache_manifest_fingerprint is not None)
        if self.policy_variant == "bsp" and cache_identities_present != (True, True):
            raise ValueError("BSP evaluation requires both the sidecar NPZ SHA256 and manifest fingerprint")
        if self.policy_variant == "baseline" and cache_identities_present != (False, False):
            raise ValueError("Baseline evaluation must record null BSP cache identities")
        if self.bsp_cache_hash is not None and (
            not isinstance(self.bsp_cache_hash, str)
            or len(self.bsp_cache_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.bsp_cache_hash)
        ):
            raise ValueError("bsp_cache_hash must be the lowercase SHA256 of the actual sidecar NPZ")
        if self.bsp_cache_manifest_fingerprint is not None and (
            not isinstance(self.bsp_cache_manifest_fingerprint, str)
            or len(self.bsp_cache_manifest_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.bsp_cache_manifest_fingerprint
            )
        ):
            raise ValueError("bsp_cache_manifest_fingerprint must be a lowercase SHA256")
        if not isinstance(self.bsp_parameters, dict) or set(self.bsp_parameters) != set(BSP_PARAMETERS):
            raise ValueError("Evaluation manifest must record the exact fixed BSP parameters")
        if any(
            type(self.bsp_parameters[field]) is not type(expected)
            or self.bsp_parameters[field] != expected
            for field, expected in BSP_PARAMETERS.items()
        ):
            raise ValueError("Evaluation manifest must record the exact fixed BSP parameters")
        _require_exact_integer(
            self.expected_action_horizon,
            name="Evaluation expected_action_horizon",
            minimum=1,
        )
        _require_exact_integer(
            self.execution_horizon,
            name="Evaluation execution_horizon",
            minimum=1,
        )
        protocol = resolve_policy_protocol(self.policy_variant, self.expected_action_horizon)
        if self.policy_protocol != protocol.name:
            raise ValueError(
                f"Manifest protocol {self.policy_protocol!r} does not match resolved protocol {protocol.name!r}"
            )
        if self.execution_horizon != _REPLAN_ACTIONS:
            raise ValueError(f"LIBERO evaluation must execute exactly {_REPLAN_ACTIONS} actions per replan")
        if isinstance(self.suites, (str, bytes)) or not isinstance(self.suites, (tuple, list)):
            raise ValueError("Evaluation manifest suites must be a list or tuple")
        suites = tuple(self.suites)
        if suites not in tuple((suite,) for suite in SUPPORTED_SUITES) + (SUPPORTED_SUITES,):
            raise ValueError("Evaluation manifest suites must be one suite or all four in canonical order")
        if not isinstance(self.task_ids, (tuple, list)):
            raise ValueError("Evaluation manifest task_ids must be a list or tuple")
        canonical_task_ids = resolve_task_ids(self.task_ids)
        if tuple(self.task_ids) != canonical_task_ids:
            raise ValueError("Evaluation manifest task_ids must be unique and sorted")
        _require_exact_integer(
            self.trials_per_task,
            name="Evaluation trials_per_task",
            minimum=1,
        )
        _require_exact_integer(self.num_steps_wait, name="Evaluation num_steps_wait")
        if not isinstance(self.max_steps_by_suite, dict) or set(self.max_steps_by_suite) != set(suites):
            raise ValueError("Evaluation max_steps_by_suite must exactly match selected suites")
        for suite, max_steps in self.max_steps_by_suite.items():
            _require_exact_integer(max_steps, name=f"Evaluation max steps for {suite}", minimum=1)
            if max_steps != _MAX_STEPS_BY_SUITE[suite]:
                raise ValueError(f"Evaluation max steps for {suite} must be {_MAX_STEPS_BY_SUITE[suite]}")
        for name, timeout in (
            ("connection_timeout_s", self.connection_timeout_s),
            ("inference_timeout_s", self.inference_timeout_s),
        ):
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError(f"Evaluation {name} must be positive and finite")
        _require_exact_integer(self.infrastructure_retries, name="Evaluation infrastructure_retries")
        if self.infrastructure_retries != 2:
            raise ValueError("LIBERO evaluation protocol requires exactly two infrastructure retries")

    def to_dict(self) -> dict[str, Any]:
        # frozen=True prevents attribute assignment, not mutation of caller-owned
        # list/dict values. Revalidate immediately before schema-v3 serialization.
        self.__post_init__()
        payload = dataclasses.asdict(self)
        payload["suites"] = list(self.suites)
        payload["task_ids"] = list(self.task_ids)
        return {
            **payload,
            "schema_version": 3,
            "replan_steps": _REPLAN_ACTIONS,
        }


class VideoSelector:
    """Deterministically reserve only the first success/failure for each task."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._claimed: set[tuple[str, int, str]] = set()

    def claim(self, record: EpisodeRecord) -> Path | None:
        if record.success is True:
            outcome = "success"
        elif record.include_in_success_rate and record.failure_kind in {"policy", "timeout"}:
            outcome = "failure"
        else:
            return None
        task_key = (record.identity.suite, record.identity.task_id, outcome)
        if task_key in self._claimed:
            return None
        self._claimed.add(task_key)
        task_name = _safe_component(record.identity.task_name, fallback="task")
        directory = self._root / record.identity.suite / f"task-{record.identity.task_id:03d}-{task_name}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{outcome}-init-{record.identity.init_state_index:03d}-{record.identity.episode_id}.mp4"


@dataclasses.dataclass(frozen=True)
class ArtifactError:
    """A non-rollout artifact failure preserved separately from policy metrics."""

    episode_id: str
    artifact_type: str
    path: str
    error: str

    def __post_init__(self) -> None:
        if not all((self.episode_id, self.artifact_type, self.path, self.error)):
            raise ValueError("Artifact error fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class VideoArtifactAudit:
    """Planned and encoded timing for one selected video artifact."""

    episode_id: str
    path: str
    planned: _video_timing.VideoTimingAudit
    inference_schedule: str
    video_show_inference_waits: bool
    measured_stall_count: int
    measured_control_stall_ns: int
    measured_stall_reasons: tuple[str, ...]
    included_stall_reasons: tuple[str, ...]
    included_stall_frame_counts: tuple[int, ...]
    inserted_overlay_types: tuple[str, ...]
    encoded_fps: float
    encoded_frame_count: int
    encoded_duration_ns: int
    encoded_duration_deviation_ns: int
    timing_tolerance_ns: int
    timing_gate_pass: bool
    artifact_padding_frame_count: int = 0
    warning: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.planned, _video_timing.VideoTimingAudit):
            raise TypeError("planned must be a VideoTimingAudit")
        if type(self.video_show_inference_waits) is not bool:
            raise TypeError("video_show_inference_waits must be a boolean")
        _video_timing.validate_inference_schedule(self.inference_schedule)
        _require_exact_integer(self.measured_stall_count, name="video measured_stall_count")
        _require_exact_integer(
            self.measured_control_stall_ns,
            name="video measured_control_stall_ns",
        )
        _require_exact_integer(self.included_stall_count, name="video included_stall_count")
        _require_exact_integer(
            self.included_control_stall_ns,
            name="video included_control_stall_ns",
        )
        _require_exact_integer(self.planned.stall_frame_count, name="video stall_frame_count")
        _require_exact_integer(
            self.artifact_padding_frame_count,
            name="video artifact_padding_frame_count",
        )
        if self.artifact_padding_frame_count not in {0, 1}:
            raise ValueError("Video artifact padding must contain at most one frame")
        if self.planned.video_frame_count == 0:
            if self.artifact_padding_frame_count != 1:
                raise ValueError("An empty planned timeline requires one encoded padding frame")
        elif self.artifact_padding_frame_count:
            raise ValueError("Non-empty planned timelines cannot add artifact padding")
        _require_exact_string_tuple(
            self.measured_stall_reasons,
            name="video measured_stall_reasons",
            allowed_values=_STALL_REASONS,
            expected_count=self.measured_stall_count,
        )
        _require_exact_string_tuple(
            self.included_stall_reasons,
            name="video included_stall_reasons",
            allowed_values=_STALL_REASONS,
            expected_count=self.included_stall_count,
        )
        _require_exact_nonnegative_integer_tuple(
            self.included_stall_frame_counts,
            name="video included_stall_frame_counts",
            expected_count=self.included_stall_count,
        )
        if sum(self.included_stall_frame_counts) != self.planned.stall_frame_count:
            raise ValueError("Included stall frame counts do not match the planned timeline")
        _require_exact_string_tuple(
            self.inserted_overlay_types,
            name="video inserted_overlay_types",
            allowed_values=_OVERLAY_TYPES,
        )
        expected_reason = (
            _video_timing.STALL_REASON_SYNCHRONOUS_INFERENCE
            if self.inference_schedule == _video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE
            else _video_timing.STALL_REASON_ASYNC_ACTION_UNDERFLOW
        )
        if any(reason != expected_reason for reason in self.measured_stall_reasons):
            raise ValueError("Measured stall reason does not match inference schedule")
        if self.video_show_inference_waits:
            if self.included_stall_reasons != self.measured_stall_reasons:
                raise ValueError("Included stall reasons must match measured stall reasons")
        elif self.included_stall_reasons or self.inserted_overlay_types:
            raise ValueError("Disabled inference waits cannot include stalls or overlays")
        expected_overlay_types = tuple(
            _OVERLAY_TYPE_BY_STALL_REASON[reason]
            for reason, frame_count in zip(  # noqa: B905 -- LIBERO client runs on Python 3.8.
                self.included_stall_reasons,
                self.included_stall_frame_counts,
            )
            if frame_count
        )
        if (
            self.artifact_padding_frame_count
            and self.video_show_inference_waits
            and self.included_stall_reasons
        ):
            expected_overlay_types = (
                _OVERLAY_TYPE_BY_STALL_REASON[self.included_stall_reasons[0]],
            )
        if self.inserted_overlay_types != expected_overlay_types:
            raise ValueError("Inserted overlay audit does not match the encoded video timeline")

    @property
    def included_stall_count(self) -> int:
        return self.planned.included_stall_count

    @property
    def included_control_stall_ns(self) -> int:
        return self.planned.included_control_stall_ns

    @property
    def expected_duration_ns(self) -> int:
        return self.planned.expected_duration_ns

    @property
    def expected_encoded_frame_count(self) -> int:
        return self.planned.video_frame_count + self.artifact_padding_frame_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "path": self.path,
            **self.planned.to_dict(),
            "inference_schedule": self.inference_schedule,
            "video_show_inference_waits": self.video_show_inference_waits,
            "measured_stall_count": self.measured_stall_count,
            "measured_control_stall_ns": self.measured_control_stall_ns,
            "measured_stall_reasons": list(self.measured_stall_reasons),
            "included_stall_reasons": list(self.included_stall_reasons),
            "included_stall_frame_counts": list(self.included_stall_frame_counts),
            "inserted_overlay_types": list(self.inserted_overlay_types),
            "encoded_fps": self.encoded_fps,
            "encoded_frame_count": self.encoded_frame_count,
            "encoded_duration_ns": self.encoded_duration_ns,
            "encoded_duration_deviation_ns": self.encoded_duration_deviation_ns,
            "timing_tolerance_ns": self.timing_tolerance_ns,
            "timing_gate_pass": self.timing_gate_pass,
            "artifact_padding_frame_count": self.artifact_padding_frame_count,
            "expected_encoded_frame_count": self.expected_encoded_frame_count,
            "warning": self.warning,
        }


def build_video_artifact_audit(
    *,
    episode_id: str,
    path: str,
    planned: _video_timing.VideoTimingAudit,
    measured_stalls: Sequence[_video_timing.ControlStall],
    included_stalls: Sequence[_video_timing.ControlStall],
    video_show_inference_waits: bool,
    inference_schedule: str,
    artifact_padding_frame_count: int = 0,
    encoded_fps: float,
    encoded_frame_count: int,
    encoded_duration_s: float,
) -> VideoArtifactAudit:
    """Validate MP4 readback and classify duration-only drift as a warning."""
    if not episode_id or not path:
        raise ValueError("Video audit identity and path must be non-empty")
    if not isinstance(planned, _video_timing.VideoTimingAudit):
        raise TypeError("planned must be a VideoTimingAudit")
    if not isinstance(video_show_inference_waits, bool):
        raise ValueError("video_show_inference_waits must be a boolean")
    _video_timing.validate_inference_schedule(inference_schedule)
    measured_stalls = tuple(measured_stalls)
    included_stalls = tuple(included_stalls)
    if any(not isinstance(stall, _video_timing.ControlStall) for stall in measured_stalls):
        raise TypeError("measured_stalls must contain ControlStall records")
    if any(not isinstance(stall, _video_timing.ControlStall) for stall in included_stalls):
        raise TypeError("included_stalls must contain ControlStall records")
    expected_included_stalls = measured_stalls if video_show_inference_waits else ()
    if included_stalls != expected_included_stalls:
        raise ValueError("Included stalls do not match video_show_inference_waits")
    for stall in measured_stalls:
        _video_timing.stall_overlay_lines(stall, inference_schedule=inference_schedule)
    if planned.included_stall_count != len(included_stalls) or planned.included_control_stall_ns != sum(
        stall.duration_ns for stall in included_stalls
    ):
        raise ValueError("Planned audit does not match included control stalls")
    if (
        isinstance(encoded_fps, bool)
        or not isinstance(encoded_fps, (int, float))
        or not math.isfinite(encoded_fps)
        or encoded_fps <= 0
    ):
        raise ValueError("Encoded FPS must be positive and finite")
    if not math.isclose(float(encoded_fps), float(planned.video_fps), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Encoded FPS {encoded_fps} does not match requested video FPS {planned.video_fps}"
        )
    _require_exact_integer(
        artifact_padding_frame_count,
        name="video artifact_padding_frame_count",
    )
    if artifact_padding_frame_count not in {0, 1}:
        raise ValueError("Video artifact padding must contain at most one frame")
    if planned.video_frame_count == 0:
        if artifact_padding_frame_count != 1:
            raise ValueError("An empty planned timeline requires one encoded padding frame")
    elif artifact_padding_frame_count:
        raise ValueError("Non-empty planned timelines cannot add artifact padding")
    if (
        isinstance(encoded_frame_count, bool)
        or not isinstance(encoded_frame_count, int)
        or encoded_frame_count < 0
    ):
        raise ValueError("Encoded frame count must be a non-negative integer")
    expected_encoded_frame_count = planned.video_frame_count + artifact_padding_frame_count
    if encoded_frame_count != expected_encoded_frame_count:
        raise ValueError(
            f"Encoded frame count {encoded_frame_count} does not match planned frame count "
            f"{expected_encoded_frame_count}"
        )
    if (
        isinstance(encoded_duration_s, bool)
        or not isinstance(encoded_duration_s, (int, float))
        or not math.isfinite(encoded_duration_s)
        or encoded_duration_s < 0
    ):
        raise ValueError("Encoded duration must be non-negative and finite")

    encoded_duration_ns = round(float(encoded_duration_s) * _video_timing.NANOSECONDS_PER_SECOND)
    deviation_ns = encoded_duration_ns - planned.expected_duration_ns
    tolerance_ns = (
        _video_timing.NANOSECONDS_PER_SECOND + planned.video_fps - 1
    ) // planned.video_fps
    timing_gate_pass = abs(deviation_ns) <= tolerance_ns
    warning = None
    if not timing_gate_pass:
        warning = (
            "encoded duration deviates from expected duration by "
            f"{abs(deviation_ns) / _video_timing.NANOSECONDS_PER_SECOND:.6f} s "
            f"(tolerance {tolerance_ns / _video_timing.NANOSECONDS_PER_SECOND:.6f} s)"
        )
    measured_stall_reasons = tuple(stall.reason for stall in measured_stalls)
    included_stall_reasons = tuple(stall.reason for stall in included_stalls)
    included_stall_frame_counts = _video_timing.quantize_stall_frames(
        included_stalls,
        video_fps=planned.video_fps,
    )
    inserted_overlay_types = tuple(
        _OVERLAY_TYPE_BY_STALL_REASON[stall.reason]
        for stall, frame_count in zip(
            included_stalls,
            included_stall_frame_counts,
        )
        if frame_count
    )
    if (
        artifact_padding_frame_count
        and video_show_inference_waits
        and included_stall_reasons
    ):
        inserted_overlay_types = (
            _OVERLAY_TYPE_BY_STALL_REASON[included_stall_reasons[0]],
        )
    return VideoArtifactAudit(
        episode_id=episode_id,
        path=path,
        planned=planned,
        inference_schedule=inference_schedule,
        video_show_inference_waits=video_show_inference_waits,
        measured_stall_count=len(measured_stalls),
        measured_control_stall_ns=sum(stall.duration_ns for stall in measured_stalls),
        measured_stall_reasons=measured_stall_reasons,
        included_stall_reasons=included_stall_reasons,
        included_stall_frame_counts=included_stall_frame_counts,
        inserted_overlay_types=inserted_overlay_types,
        encoded_fps=float(encoded_fps),
        encoded_frame_count=encoded_frame_count,
        encoded_duration_ns=encoded_duration_ns,
        encoded_duration_deviation_ns=deviation_ns,
        timing_tolerance_ns=tolerance_ns,
        timing_gate_pass=timing_gate_pass,
        artifact_padding_frame_count=artifact_padding_frame_count,
        warning=warning,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        if not fieldnames:
            return
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ArtifactWriter:
    """Incrementally persist episodes and finalize auditable aggregates."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.artifact_errors_path = self.output_dir / "artifact_errors.jsonl"
        self.video_audit_path = self.output_dir / "video_audit.jsonl"

    def write_manifest(self, manifest: EvaluationManifest) -> None:
        _atomic_json(self.output_dir / "manifest.json", manifest.to_dict())

    def append_episode(self, record: EpisodeRecord) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as output:
            json.dump(record.to_dict(), output, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()

    def append_artifact_error(self, error: ArtifactError) -> None:
        with self.artifact_errors_path.open("a", encoding="utf-8") as output:
            json.dump(error.to_dict(), output, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()

    def append_video_audit(self, audit: VideoArtifactAudit) -> None:
        with self.video_audit_path.open("a", encoding="utf-8") as output:
            json.dump(audit.to_dict(), output, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()

    def write_summary(
        self, records: Sequence[EpisodeRecord], *, artifact_errors: Sequence[ArtifactError] = ()
    ) -> dict[str, Any]:
        summary = aggregate_records(records, artifact_errors=artifact_errors)
        _write_csv(self.output_dir / "tasks.csv", summary["tasks"])
        _write_csv(self.output_dir / "suites.csv", summary["suites"])
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary
