"""Schema-v5 LIBERO evaluation with calibrated single-owner inference."""

# ruff: noqa: SLF001, UP006, UP007, UP035, UP038 -- Python 3.8 evaluator.

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import dataclasses
import logging
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import imageio
import numpy as np
from openpi_client import async_inference as _async
from openpi_client import inference as _inference
from openpi_client import latency_sampling as _latency_sampling
from openpi_client import libero_control_v5 as _control
from openpi_client import libero_eval_v5 as _eval
from openpi_client import libero_video_timing_v5 as _timing
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _websocket
import tqdm
import tyro

from examples.libero import main as _environment_helpers

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
EXPECTED_TASKS_PER_SUITE = 10

# These are image/simulator/video primitives only.  No schema-v3 record,
# writer, timing event, or report loader is called by this module.
_prepare_observation = _environment_helpers._prepare_observation
_get_benchmark_suite = _environment_helpers._get_benchmark_suite
_get_libero_env = _environment_helpers._get_libero_env
_read_encoded_video = _environment_helpers._read_encoded_video


@dataclasses.dataclass
class ArgsV5:
    # Single-owner policy connection.
    host: str = "0.0.0.0"
    port: int = 8000
    connection_timeout_s: float = 30.0
    inference_timeout_s: float = 120.0
    worker_shutdown_timeout_s: float = 5.0
    socket_close_timeout_s: float = 1.0
    recv_poll_interval_s: float = 0.05
    resize_size: int = 224

    # Frozen schema-v5 execution mode.
    execution_mode: str = "baseline_async"

    # Benchmark protocol.  ``all`` selects all four suites.
    task_suite_name: str = "libero_spatial"
    task_ids: Optional[Tuple[int, ...]] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    eval_seed: int = 42
    control_freq: int = 20
    video_fps: int = 40
    video_show_inference_waits: bool = False

    output_dir: str = "data/libero/eval-v5"

    # Audited checkpoint and training identities.
    config_name: str = ""
    checkpoint_step: int = 0
    dataset_revision: str = "v2.0"
    bsp_cache_hash: Optional[str] = None
    bsp_cache_manifest_fingerprint: Optional[str] = None
    norm_hash: str = ""
    checkpoint: str = ""
    container_digest: str = ""
    train_seed: int = 42


class _SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wait_until_ns(self, deadline_ns: int) -> None:
        while True:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            time.sleep(remaining_ns / 1_000_000_000)


class RunCleanupError(RuntimeError):
    """Retain a primary failure and a separately fatal cleanup failure."""

    def __init__(self, primary_error: BaseException, cleanup_error: BaseException):
        super().__init__(
            f"{type(primary_error).__name__}: {primary_error}; cleanup also failed with {type(cleanup_error).__name__}: {cleanup_error}"
        )
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error


class _MultipleCleanupError(RuntimeError):
    def __init__(self, first: BaseException, second: BaseException):
        super().__init__(
            f"cleanup failed with {type(first).__name__}: {first}; subsequent cleanup failed with {type(second).__name__}: {second}"
        )
        self.first_error = first
        self.second_error = second


class _TaskEnvironmentV5:
    """Lazy LIBERO environment with retry-safe reset and visible close failures."""

    def __init__(self, task: Any, resolution: int, seed: int, control_freq: int):
        self._task = task
        self._resolution = resolution
        self._seed = seed
        self._control_freq = control_freq
        self._env = None

    def _get(self) -> Any:
        if self._env is None:
            try:
                self._env = _get_libero_env(
                    self._task,
                    self._resolution,
                    self._seed,
                    control_freq=self._control_freq,
                )
            except Exception as error:
                raise _eval.classify_exception(error, phase="environment_create") from error
        return self._env

    def reset_to(self, initial_state: Any) -> Any:
        env = self._get()
        try:
            env.seed(self._seed)
            env.reset()
            return env.set_init_state(initial_state)
        except Exception as error:
            self._invalidate_after_error(error)
            raise _eval.classify_exception(error, phase="environment_reset") from error

    def step(self, action: Any) -> Any:
        env = self._get()
        try:
            return env.step(action)
        except Exception as error:
            self._invalidate_after_error(error)
            raise _eval.classify_exception(error, phase="environment_step") from error

    def _invalidate_after_error(self, primary_error: BaseException) -> None:
        try:
            self.invalidate()
        except BaseException as cleanup_error:
            raise RunCleanupError(primary_error, cleanup_error) from primary_error

    def invalidate(self) -> None:
        if self._env is None:
            return
        env = self._env
        self._env = None
        env.close()

    def close(self) -> None:
        self.invalidate()


@dataclasses.dataclass
class _RequestTraceV5:
    request_id: int
    observation_control_step: int
    submitted_offset_ns: int
    flow_seed: int
    intent: _control.RequestIntentV5
    source_frame: Any
    latency_sample_key: _latency_sampling.LatencySampleKeyV1
    sampled_target_latency_ns: int
    disposition: Optional[str] = None

    def to_event(self) -> _timing.RequestEventV5:
        if self.disposition is None:
            raise RuntimeError("logical request has no terminal disposition")
        return _timing.RequestEventV5(
            request_id=self.request_id,
            observation_control_step=self.observation_control_step,
            submitted_offset_ns=self.submitted_offset_ns,
            flow_seed=self.flow_seed,
            dispatch=self.intent.dispatch,
            trigger=self.intent.trigger,
            scheduler_context=dict(self.intent.scheduler_context),
            disposition=self.disposition,
            latency_sample_key=self.latency_sample_key,
            sampled_target_latency_ns=self.sampled_target_latency_ns,
        )


@dataclasses.dataclass
class _PendingRequestV5:
    job: Any
    trace: _RequestTraceV5


@dataclasses.dataclass
class _PendingSlotV5:
    value: Optional[_PendingRequestV5] = None
    owns_job: bool = False

    def clear(self) -> None:
        self.value = None
        self.owns_job = False


class _AttemptLedgerV5:
    def __init__(
        self,
        *,
        execution_mode: str,
        identity: _eval.EpisodeIdentity,
        eval_seed: int,
        origin_ns: int,
    ) -> None:
        self.execution_mode = execution_mode
        self.identity = identity
        self.eval_seed = eval_seed
        self.origin_ns = origin_ns
        self.requests = []  # type: List[_RequestTraceV5]
        self.latencies = []  # type: List[_timing.LatencyEventV5]
        self.activations = []  # type: List[_timing.PlanActivationV5]
        self.action_seams = []  # type: List[_timing.ActionSeamV5]
        self.underflows = []  # type: List[_timing.ActionUnderflowV5]
        self.stalls = []  # type: List[_timing.ControlStallV5]
        self.replay_frames = []  # type: List[Any]
        self.stall_source_frames = {}  # type: Dict[int, Any]
        self.steps = 0
        self._pending_seam_activation = None  # type: Optional[_timing.PlanActivationV5]
        self._previous_action = None  # type: Optional[Tuple[float, ...]]

    def offset(self, monotonic_ns: int) -> int:
        offset_ns = monotonic_ns - self.origin_ns
        if offset_ns < 0:
            raise _eval.PolicyFailure("worker timestamp precedes the episode origin")
        return offset_ns

    def result(
        self,
        *,
        success: bool,
        now_ns: int,
        failure_kind: Optional[str] = None,
        error: Optional[str] = None,
    ) -> _eval.AttemptResultV5:
        return _eval.AttemptResultV5(
            execution_mode=self.execution_mode,
            success=success,
            steps=self.steps,
            replans=len(self.activations),
            episode_duration_ns=self.offset(now_ns),
            failure_kind=failure_kind,
            error=error,
            inference_requests=tuple(trace.to_event() for trace in self.requests),
            inference_latencies=tuple(self.latencies),
            plan_activations=tuple(self.activations),
            action_seams=tuple(self.action_seams),
            action_underflows=tuple(self.underflows),
            control_stalls=tuple(self.stalls),
            replay_frames=tuple(self.replay_frames),
            stall_source_frames=tuple(sorted(self.stall_source_frames.items())),
        )

    def policy_failure(self, error: BaseException, *, now_ns: int) -> _eval.AttemptResultV5:
        return self.result(
            success=False,
            now_ns=now_ns,
            failure_kind="policy",
            error=f"{type(error).__name__}: {error}",
        )

    def record_stall_source(self, control_step: int, frame: Any) -> None:
        existing = self.stall_source_frames.get(control_step)
        if existing is not None and existing is not frame:
            raise _eval.PolicyFailure("two control stalls require different frames at one step")
        self.stall_source_frames[control_step] = frame

    def extend_discarded_underflow_to(
        self,
        *,
        discarded_request_id: int,
        submitted_ns: int,
    ) -> bool:
        """Extend a matching stale-response underflow, if one actually occurred."""
        underflow = self.underflows[-1] if self.underflows else None
        stall = self.stalls[-1] if self.stalls else None
        underflow_matches = underflow is not None and underflow.request_id == discarded_request_id
        stall_matches = (
            stall is not None
            and stall.request_id == discarded_request_id
            and stall.reason == _timing.STALL_REASON_ASYNC_ACTION_UNDERFLOW
        )
        if underflow_matches != stall_matches:
            raise _eval.PolicyFailure("stale replan has an incomplete action-underflow audit")
        if not underflow_matches:
            return False
        assert underflow is not None
        assert stall is not None
        if stall.started_offset_ns != underflow.started_offset_ns or stall.duration_ns != underflow.duration_ns:
            raise _eval.PolicyFailure("stale replan does not follow its action underflow")
        submitted_offset_ns = self.offset(submitted_ns)
        existing_end_ns = underflow.started_offset_ns + underflow.duration_ns
        if submitted_offset_ns < existing_end_ns:
            raise _eval.PolicyFailure("stale replan submission precedes its response handoff")
        duration_ns = submitted_offset_ns - underflow.started_offset_ns
        self.underflows[-1] = dataclasses.replace(underflow, duration_ns=duration_ns)
        self.stalls[-1] = dataclasses.replace(stall, duration_ns=duration_ns)
        return True

    def record_activation(self, activation: _timing.PlanActivationV5) -> None:
        self.activations.append(activation)
        if activation.plan_id > 0:
            if self._pending_seam_activation is not None:
                raise _eval.PolicyFailure("a plan activated before its prior seam was observed")
            self._pending_seam_activation = activation

    def record_action(self, action: Any) -> None:
        values = tuple(float(value) for value in np.asarray(action).reshape(-1).tolist())
        if len(values) < 7 or any(not math.isfinite(value) for value in values[:7]):
            raise _eval.PolicyFailure("executed action must contain seven finite values")
        activation = self._pending_seam_activation
        if activation is not None:
            if self._previous_action is None:
                raise _eval.PolicyFailure("non-initial activation has no prior executed action")
            self.action_seams.append(
                _timing.ActionSeamV5.from_actions(
                    plan_id=activation.plan_id,
                    request_id=activation.request_id,
                    control_step=activation.control_step,
                    previous_action=self._previous_action,
                    activated_action=values,
                )
            )
            self._pending_seam_activation = None
        self._previous_action = values


def _require_nonnegative_clock(clock: _control.Clock) -> int:
    value = clock.monotonic_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("clock.monotonic_ns() must return a non-negative integer")
    return value


def _validate_args_v5(
    args: ArgsV5,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], _control.ExecutionModeSpec]:
    if not isinstance(args, ArgsV5):
        raise TypeError("args must be ArgsV5")
    try:
        mode = _control.EXECUTION_MODES[args.execution_mode]
    except (KeyError, TypeError) as error:
        raise ValueError("execution_mode must be one of the seven schema-v5 runtime modes") from error
    suite_aliases = {
        "libero_spatial": "libero_spatial",
        "spatial": "libero_spatial",
        "libero_object": "libero_object",
        "object": "libero_object",
        "libero_goal": "libero_goal",
        "goal": "libero_goal",
        "libero_10": "libero_10",
        "10": "libero_10",
    }
    if not isinstance(args.task_suite_name, str):
        raise ValueError("task_suite_name must be a string")
    suite_selection = args.task_suite_name.strip().lower()
    if suite_selection == "all":
        suites = tuple(_eval.SUPPORTED_SUITES)
    else:
        try:
            suites = (suite_aliases[suite_selection],)
        except (KeyError, TypeError) as error:
            raise ValueError("unsupported LIBERO suite selection") from error
    if args.task_ids is None:
        task_ids = tuple(range(10))
    else:
        task_ids = tuple(args.task_ids)
        if (
            not task_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in task_ids)
            or any(value < 0 or value > 9 for value in task_ids)
            or task_ids != tuple(sorted(set(task_ids)))
        ):
            raise ValueError("task_ids must be a sorted unique subset of 0..9")
    integer_fields = (
        ("port", args.port, 1),
        ("resize_size", args.resize_size, 1),
        ("num_steps_wait", args.num_steps_wait, 0),
        ("num_trials_per_task", args.num_trials_per_task, 1),
        ("eval_seed", args.eval_seed, 0),
        ("train_seed", args.train_seed, 0),
        ("checkpoint_step", args.checkpoint_step, 0),
    )
    for name, value, minimum in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    for name, value in (
        ("connection_timeout_s", args.connection_timeout_s),
        ("inference_timeout_s", args.inference_timeout_s),
        ("worker_shutdown_timeout_s", args.worker_shutdown_timeout_s),
        ("socket_close_timeout_s", args.socket_close_timeout_s),
        ("recv_poll_interval_s", args.recv_poll_interval_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if args.socket_close_timeout_s > args.worker_shutdown_timeout_s:
        raise ValueError("socket_close_timeout_s must not exceed worker_shutdown_timeout_s")
    if (
        isinstance(args.control_freq, bool)
        or not isinstance(args.control_freq, int)
        or args.control_freq != 20
        or isinstance(args.video_fps, bool)
        or not isinstance(args.video_fps, int)
        or args.video_fps != 40
    ):
        raise ValueError("schema-v5 execution requires exactly 20 Hz control and 40 fps video")
    if type(args.video_show_inference_waits) is not bool:
        raise ValueError("video_show_inference_waits must be boolean")
    if args.dataset_revision != "v2.0":
        raise ValueError("dataset_revision must be v2.0")
    return suites, task_ids, mode


def _resolve_code_sha_v5() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to resolve evaluator code identity") from error
    if status.stdout.strip():
        raise RuntimeError("Evaluator Git checkout must be clean before writing a manifest")
    return result.stdout.strip()


def _initial_state_fingerprint_v5(initial_state: Any) -> str:
    state = np.ascontiguousarray(initial_state)
    return _eval.fingerprint_init_state(
        dtype=state.dtype.str,
        shape=state.shape,
        payload=state.tobytes(),
    )


def _fingerprint_connection_v5(
    mode: _control.ExecutionModeSpec,
    connection: Any,
) -> str:
    payload = getattr(connection, "metadata_payload", None)
    if not isinstance(payload, bytes):
        raise _control.CalibrationIdentityError("connection is missing immutable metadata bytes")
    try:
        metadata = msgpack_numpy.unpackb(payload)
        return _control.validate_server_metadata(mode, metadata)
    except (TypeError, ValueError) as error:
        raise _control.CalibrationIdentityError("server metadata is invalid") from error


def _validate_connection_identity_v5(
    mode: _control.ExecutionModeSpec,
    connection: Any,
    *,
    expected_fingerprint: str,
) -> None:
    actual = _fingerprint_connection_v5(mode, connection)
    if actual != expected_fingerprint:
        raise _control.CalibrationIdentityError("server metadata fingerprint changed")


def _make_policy_factory_v5(args: ArgsV5) -> Callable[[Any], Any]:
    def factory(cancel_event: Any) -> Any:
        return _websocket.WebsocketClientPolicy(
            args.host,
            args.port,
            connection_timeout=args.connection_timeout_s,
            inference_timeout=args.inference_timeout_s,
            cancel_event=cancel_event,
            close_timeout=args.socket_close_timeout_s,
        )

    return factory


def _pace_dummy_phase_v5(
    environment: Any,
    obs: Any,
    *,
    num_steps_wait: int,
    clock: _control.Clock,
) -> Any:
    pacer = _control.NoCatchupPacer(clock)
    for _ in range(num_steps_wait):
        now_ns = pacer.wait_until_due()
        pacer.mark_action_started(now_ns)
        obs, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
    return obs


def _classify_worker_exception(error: BaseException) -> BaseException:
    if not isinstance(error, Exception):
        return error
    return _eval.classify_exception(error, phase="policy_infer")


def _cancel_generation_v5(worker: Any, *, timeout_s: float) -> None:
    generation = worker.reset_generation()
    worker.wait_until_ready(generation, timeout=timeout_s)


def _cancel_preserving_v5(
    worker: Any,
    *,
    timeout_s: float,
    primary_error: BaseException,
) -> None:
    try:
        _cancel_generation_v5(worker, timeout_s=timeout_s)
    except BaseException as cleanup_error:
        raise RunCleanupError(primary_error, cleanup_error) from primary_error


def _return_policy_failure_v5(
    error: BaseException,
    *,
    now_ns: int,
    worker: Any,
    pending_slot: _PendingSlotV5,
    ledger: _AttemptLedgerV5,
    cleanup_timeout_s: float,
) -> _eval.AttemptResultV5:
    pending = pending_slot.value
    if pending is not None and pending.trace.disposition is None:
        pending.trace.disposition = "abandoned"
    if pending_slot.owns_job:
        _cancel_preserving_v5(
            worker,
            timeout_s=cleanup_timeout_s,
            primary_error=error,
        )
        pending_slot.clear()
    return ledger.policy_failure(error, now_ns=now_ns)


def _close_worker_v5(
    worker: Any,
    *,
    primary_error: Optional[BaseException],
    timeout_s: float,
) -> None:
    cleanup_error = None  # type: Optional[BaseException]
    try:
        _cancel_generation_v5(worker, timeout_s=timeout_s)
    except BaseException as error:
        cleanup_error = error
    try:
        worker.close()
    except BaseException as error:
        cleanup_error = error if cleanup_error is None else _MultipleCleanupError(cleanup_error, error)
    if cleanup_error is None:
        return
    if primary_error is not None:
        raise RunCleanupError(primary_error, cleanup_error) from primary_error
    raise cleanup_error


def _submit_request_v5(
    *,
    worker: Any,
    intent: _control.RequestIntentV5,
    prepared_observation: Mapping[str, Any],
    source_frame: Any,
    ledger: _AttemptLedgerV5,
    pending_slot: _PendingSlotV5,
) -> _PendingRequestV5:
    for reserved_key in (_inference.INFERENCE_SEED_KEY, _inference.RTC_REQUEST_KEY):
        if reserved_key in prepared_observation:
            raise _eval.PolicyFailure(f"prepared observation contains reserved key {reserved_key}")
    request_id = len(ledger.requests)
    flow_seed = _eval.stable_replan_seed(ledger.eval_seed, ledger.identity, request_id)
    request = dict(prepared_observation)
    request.update(dict(intent.request_overlay))
    request[_inference.INFERENCE_SEED_KEY] = flow_seed
    try:
        sample_key = _latency_sampling.LatencySampleKeyV1(
            namespace="formal",
            seed=ledger.eval_seed,
            suite=ledger.identity.suite,
            task_id=ledger.identity.task_id,
            trial_index=ledger.identity.init_state_index,
            request_ordinal=request_id,
        )
        job = worker.submit(request, latency_sample_key=sample_key)
    except BaseException as error:
        classified = _classify_worker_exception(error)
        raise classified from error
    pending_slot.owns_job = True
    submitted_ns = getattr(job, "submitted_monotonic_ns", None)
    if isinstance(submitted_ns, bool) or not isinstance(submitted_ns, int) or submitted_ns < 0:
        raise _eval.PolicyFailure("worker job is missing a valid submit timestamp")
    sampled_target_ns = getattr(job, "sampled_target_latency_ns", None)
    if (
        getattr(job, "latency_sample_key", None) != sample_key
        or isinstance(sampled_target_ns, bool)
        or not isinstance(sampled_target_ns, int)
        or sampled_target_ns < 0
    ):
        raise _eval.PolicyFailure("worker job is missing its paired latency sample identity")
    trace = _RequestTraceV5(
        request_id=request_id,
        observation_control_step=ledger.steps,
        submitted_offset_ns=ledger.offset(submitted_ns),
        flow_seed=flow_seed,
        intent=intent,
        source_frame=source_frame,
        latency_sample_key=sample_key,
        sampled_target_latency_ns=sampled_target_ns,
    )
    ledger.requests.append(trace)
    pending = _PendingRequestV5(job=job, trace=trace)
    pending_slot.value = pending
    return pending


def _outcome_timing_v5(
    pending: _PendingRequestV5,
    outcome: Any,
    ledger: _AttemptLedgerV5,
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    if getattr(outcome, "job", None) is not pending.job:
        raise _eval.PolicyFailure("worker returned an outcome for a different job")
    if getattr(outcome, "stale", False) or getattr(outcome, "cancelled", False):
        raise _eval.InfrastructureFailure("network", "current inference request was cancelled")
    submitted_ns = getattr(pending.job, "submitted_monotonic_ns", None)
    completed_ns = getattr(outcome, "completed_monotonic_ns", None)
    if (
        isinstance(submitted_ns, bool)
        or not isinstance(submitted_ns, int)
        or isinstance(completed_ns, bool)
        or not isinstance(completed_ns, int)
        or submitted_ns < 0
        or completed_ns < submitted_ns
    ):
        raise _eval.PolicyFailure("worker outcome has invalid monotonic timestamps")
    effective_ns = completed_ns - submitted_ns
    raw_ns = getattr(outcome, "raw_inference_latency_ns", None)
    requested_delay_ns = getattr(outcome, "requested_synthetic_delay_ns", None)
    observed_delay_ns = getattr(outcome, "observed_synthetic_delay_ns", None)
    observed_effective_ns = getattr(outcome, "observed_effective_latency_ns", None)
    overshoot_ns = getattr(outcome, "latency_overshoot_ns", None)
    sampled_target_ns = getattr(outcome, "sampled_target_latency_ns", None)
    if (
        raw_ns is None
        and requested_delay_ns is None
        and observed_delay_ns is None
        and observed_effective_ns is None
        and overshoot_ns is None
    ):
        raw_ns, requested_delay_ns, observed_delay_ns, observed_effective_ns, overshoot_ns = (
            effective_ns,
            0,
            0,
            effective_ns,
            0,
        )
    if sampled_target_ns is None:
        sampled_target_ns = 0
    if (
        isinstance(raw_ns, bool)
        or not isinstance(raw_ns, int)
        or raw_ns < 0
        or isinstance(requested_delay_ns, bool)
        or not isinstance(requested_delay_ns, int)
        or requested_delay_ns < 0
        or isinstance(observed_delay_ns, bool)
        or not isinstance(observed_delay_ns, int)
        or observed_delay_ns < 0
        or isinstance(observed_effective_ns, bool)
        or not isinstance(observed_effective_ns, int)
        or observed_effective_ns != effective_ns
        or raw_ns + observed_delay_ns != observed_effective_ns
        or isinstance(overshoot_ns, bool)
        or not isinstance(overshoot_ns, int)
        or overshoot_ns < 0
        or isinstance(sampled_target_ns, bool)
        or not isinstance(sampled_target_ns, int)
        or sampled_target_ns != pending.trace.sampled_target_latency_ns
        or requested_delay_ns != max(raw_ns, sampled_target_ns) - raw_ns
        or observed_effective_ns < max(raw_ns, sampled_target_ns)
        or overshoot_ns != observed_effective_ns - max(raw_ns, sampled_target_ns)
        or observed_delay_ns != requested_delay_ns + overshoot_ns
    ):
        raise _eval.PolicyFailure("worker outcome has invalid latency breakdown")
    return (
        submitted_ns,
        completed_ns,
        ledger.offset(completed_ns),
        raw_ns,
        requested_delay_ns,
        observed_delay_ns,
        observed_effective_ns,
        overshoot_ns,
        sampled_target_ns,
    )


def _append_blocking_stall_v5(
    *,
    pending: _PendingRequestV5,
    ledger: _AttemptLedgerV5,
    submitted_ns: int,
    completed_ns: int,
    due_ns: Optional[int],
) -> None:
    trace = pending.trace
    full_interval = trace.intent.dispatch == "blocking_initial" or trace.intent.trigger in (
        "bsp_stale_replan",
        "baseline_async_capacity_replan",
    )
    if full_interval:
        stall_started_ns = submitted_ns
    else:
        if due_ns is None or completed_ns <= due_ns:
            return
        stall_started_ns = max(submitted_ns, due_ns)
    ledger.stalls.append(
        _timing.ControlStallV5(
            request_id=trace.request_id,
            control_step=trace.observation_control_step,
            started_offset_ns=ledger.offset(stall_started_ns),
            duration_ns=completed_ns - stall_started_ns,
            reason=_timing.STALL_REASON_SYNCHRONOUS_INFERENCE,
        )
    )
    ledger.record_stall_source(trace.observation_control_step, trace.source_frame)


def _record_blocking_wait_v5(
    *,
    pending: _PendingRequestV5,
    ledger: _AttemptLedgerV5,
    submitted_ns: int,
    completed_ns: int,
    due_ns: Optional[int],
    underflow_extension_end_ns: int,
) -> None:
    if not pending.trace.intent.dispatch.startswith("blocking"):
        return
    if pending.trace.intent.trigger == "bsp_stale_replan" and ledger.extend_discarded_underflow_to(
        discarded_request_id=pending.trace.request_id - 1,
        submitted_ns=underflow_extension_end_ns,
    ):
        return
    _append_blocking_stall_v5(
        pending=pending,
        ledger=ledger,
        submitted_ns=submitted_ns,
        completed_ns=completed_ns,
        due_ns=due_ns,
    )


def _complete_request_v5(
    *,
    pending: _PendingRequestV5,
    outcome: Any,
    scheduler: _control.ModeSchedulerV5,
    ledger: _AttemptLedgerV5,
    mode: _control.ExecutionModeSpec,
    expected_server_metadata_fingerprint: str,
    activation_now_ns: int,
    blocking_due_ns: Optional[int] = None,
    underflow_started_ns: Optional[int] = None,
) -> Optional[_eval.AttemptResultV5]:
    _validate_connection_identity_v5(
        mode,
        getattr(outcome, "connection", None),
        expected_fingerprint=expected_server_metadata_fingerprint,
    )
    (
        submitted_ns,
        completed_ns,
        completed_offset_ns,
        raw_inference_latency_ns,
        requested_synthetic_delay_ns,
        observed_synthetic_delay_ns,
        observed_effective_latency_ns,
        latency_overshoot_ns,
        sampled_target_latency_ns,
    ) = _outcome_timing_v5(pending, outcome, ledger)
    if activation_now_ns < completed_ns:
        raise _eval.PolicyFailure("controller poll time precedes worker completion")
    error = getattr(outcome, "error", None)
    if error is not None:
        classified = _classify_worker_exception(error)
        if isinstance(classified, _eval.InfrastructureFailure) or not isinstance(classified, Exception):
            raise classified
        policy_error = classified
        pending.trace.disposition = "failed"
        ledger.latencies.append(
            _timing.LatencyEventV5(
                request_id=pending.trace.request_id,
                completed_offset_ns=completed_offset_ns,
                duration_ns=completed_ns - submitted_ns,
                outcome="policy_failure",
                raw_inference_latency_ns=raw_inference_latency_ns,
                requested_synthetic_delay_ns=requested_synthetic_delay_ns,
                observed_synthetic_delay_ns=observed_synthetic_delay_ns,
                observed_effective_latency_ns=observed_effective_latency_ns,
                latency_overshoot_ns=latency_overshoot_ns,
                sampled_target_latency_ns=sampled_target_latency_ns,
            )
        )
        _record_blocking_wait_v5(
            pending=pending,
            ledger=ledger,
            submitted_ns=submitted_ns,
            completed_ns=completed_ns,
            due_ns=blocking_due_ns,
            underflow_extension_end_ns=completed_ns,
        )
        if underflow_started_ns is not None:
            _append_underflow_v5(
                pending=pending,
                ledger=ledger,
                started_ns=underflow_started_ns,
                ended_ns=completed_ns,
            )
        return ledger.policy_failure(policy_error, now_ns=max(activation_now_ns, completed_ns))

    response = getattr(outcome, "result", None)
    try:
        if not isinstance(response, Mapping):
            raise ValueError("policy response must be a mapping")
        activation = scheduler.install_response(
            pending.trace.intent,
            response,
            now_ns=activation_now_ns,
            control_step=ledger.steps,
        )
    except Exception as error:
        pending.trace.disposition = "failed"
        ledger.latencies.append(
            _timing.LatencyEventV5(
                request_id=pending.trace.request_id,
                completed_offset_ns=completed_offset_ns,
                duration_ns=completed_ns - submitted_ns,
                outcome="policy_failure",
                raw_inference_latency_ns=raw_inference_latency_ns,
                requested_synthetic_delay_ns=requested_synthetic_delay_ns,
                observed_synthetic_delay_ns=observed_synthetic_delay_ns,
                observed_effective_latency_ns=observed_effective_latency_ns,
                latency_overshoot_ns=latency_overshoot_ns,
                sampled_target_latency_ns=sampled_target_latency_ns,
            )
        )
        _record_blocking_wait_v5(
            pending=pending,
            ledger=ledger,
            submitted_ns=submitted_ns,
            completed_ns=completed_ns,
            due_ns=blocking_due_ns,
            underflow_extension_end_ns=completed_ns,
        )
        if underflow_started_ns is not None:
            _append_underflow_v5(
                pending=pending,
                ledger=ledger,
                started_ns=underflow_started_ns,
                ended_ns=completed_ns,
            )
        return ledger.policy_failure(error, now_ns=max(activation_now_ns, completed_ns))

    pending.trace.disposition = "activated"
    ledger.latencies.append(
        _timing.LatencyEventV5(
            request_id=pending.trace.request_id,
            completed_offset_ns=completed_offset_ns,
            duration_ns=completed_ns - submitted_ns,
            outcome="success",
            raw_inference_latency_ns=raw_inference_latency_ns,
            requested_synthetic_delay_ns=requested_synthetic_delay_ns,
            observed_synthetic_delay_ns=observed_synthetic_delay_ns,
            observed_effective_latency_ns=observed_effective_latency_ns,
            latency_overshoot_ns=latency_overshoot_ns,
            sampled_target_latency_ns=sampled_target_latency_ns,
        )
    )
    _record_blocking_wait_v5(
        pending=pending,
        ledger=ledger,
        submitted_ns=submitted_ns,
        completed_ns=completed_ns,
        due_ns=blocking_due_ns,
        underflow_extension_end_ns=activation_now_ns,
    )
    if activation.activation == "discarded_stale_phase":
        pending.trace.disposition = "discarded_stale_phase"
        if underflow_started_ns is not None:
            _append_underflow_v5(
                pending=pending,
                ledger=ledger,
                started_ns=underflow_started_ns,
                ended_ns=activation_now_ns,
            )
        return None
    ledger.record_activation(
        _timing.PlanActivationV5(
            plan_id=len(ledger.activations),
            request_id=pending.trace.request_id,
            control_step=ledger.steps,
            activated_offset_ns=ledger.offset(activation_now_ns),
            activation=activation.activation,
            activation_context=dict(activation.activation_context),
        )
    )
    if underflow_started_ns is not None:
        _append_underflow_v5(
            pending=pending,
            ledger=ledger,
            started_ns=underflow_started_ns,
            ended_ns=activation_now_ns,
        )
    return None


def _append_underflow_v5(
    *,
    pending: _PendingRequestV5,
    ledger: _AttemptLedgerV5,
    started_ns: int,
    ended_ns: int,
) -> None:
    duration_ns = ended_ns - started_ns
    if duration_ns < 0:
        raise _eval.PolicyFailure("async underflow ended before it began")
    underflow = _timing.ActionUnderflowV5(
        request_id=pending.trace.request_id,
        control_step=ledger.steps,
        started_offset_ns=ledger.offset(started_ns),
        duration_ns=duration_ns,
    )
    ledger.underflows.append(underflow)
    ledger.stalls.append(
        _timing.ControlStallV5(
            request_id=underflow.request_id,
            control_step=underflow.control_step,
            started_offset_ns=underflow.started_offset_ns,
            duration_ns=underflow.duration_ns,
            reason=_timing.STALL_REASON_ASYNC_ACTION_UNDERFLOW,
        )
    )


def _wait_for_request_v5(worker: Any, pending: _PendingRequestV5, *, timeout_s: float) -> Any:
    try:
        return worker.wait(pending.job, timeout=timeout_s)
    except BaseException as error:
        classified = _classify_worker_exception(error)
        raise classified from error


def _poll_request_v5(worker: Any, pending: _PendingRequestV5) -> Any:
    try:
        return worker.poll(pending.job)
    except BaseException as error:
        classified = _classify_worker_exception(error)
        raise classified from error


def _attempt_request_v5(
    *,
    now_ns: int,
    at_due: bool,
    prepared_observation: Mapping[str, Any],
    source_frame: Any,
    pending_slot: _PendingSlotV5,
    worker: Any,
    scheduler: _control.ModeSchedulerV5,
    ledger: _AttemptLedgerV5,
    mode: _control.ExecutionModeSpec,
    expected_server_metadata_fingerprint: str,
    pacer: _control.NoCatchupPacer,
    clock: _control.Clock,
    inference_timeout_s: float,
    cleanup_timeout_s: float,
) -> Optional[_eval.AttemptResultV5]:
    pending = pending_slot.value
    try:
        intent = scheduler.maybe_request(
            now_ns,
            at_due=at_due,
            request_in_flight=pending_slot.owns_job,
            control_step=ledger.steps,
        )
    except Exception as error:
        return _return_policy_failure_v5(
            error,
            now_ns=now_ns,
            worker=worker,
            pending_slot=pending_slot,
            ledger=ledger,
            cleanup_timeout_s=cleanup_timeout_s,
        )
    if intent is None:
        return None
    if pending_slot.owns_job:
        return _return_policy_failure_v5(
            _async.BusyError("scheduler requested a second outstanding job"),
            now_ns=now_ns,
            worker=worker,
            pending_slot=pending_slot,
            ledger=ledger,
            cleanup_timeout_s=cleanup_timeout_s,
        )
    try:
        pending = _submit_request_v5(
            worker=worker,
            intent=intent,
            prepared_observation=prepared_observation,
            source_frame=source_frame,
            ledger=ledger,
            pending_slot=pending_slot,
        )
        if intent.trigger == "bsp_stale_replan":
            ledger.extend_discarded_underflow_to(
                discarded_request_id=pending.trace.request_id - 1,
                submitted_ns=pending.job.submitted_monotonic_ns,
            )
    except _eval.PolicyFailure as error:
        return _return_policy_failure_v5(
            error,
            now_ns=_require_nonnegative_clock(clock),
            worker=worker,
            pending_slot=pending_slot,
            ledger=ledger,
            cleanup_timeout_s=cleanup_timeout_s,
        )
    if intent.dispatch == "background":
        return None
    due_ns = pacer.next_deadline_ns
    outcome = _wait_for_request_v5(worker, pending, timeout_s=inference_timeout_s)
    activation_now_ns = _require_nonnegative_clock(clock)
    result = _complete_request_v5(
        pending=pending,
        outcome=outcome,
        scheduler=scheduler,
        ledger=ledger,
        mode=mode,
        expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
        activation_now_ns=activation_now_ns,
        blocking_due_ns=due_ns,
    )
    pending_slot.clear()
    return result


def _run_attempt_v5(
    *,
    environment: Any,
    worker: Any,
    scheduler: _control.ModeSchedulerV5,
    initial_state: Any,
    identity: _eval.EpisodeIdentity,
    task_description: str,
    args: ArgsV5,
    max_steps: int,
    expected_server_metadata_fingerprint: str,
    clock: _control.Clock,
    prepare_observation: Callable[[Any, str, int], Tuple[Mapping[str, Any], Any]] = _prepare_observation,
) -> _eval.AttemptResultV5:
    """Run one deterministic final-attempt timeline through one worker slot."""
    mode = _control.EXECUTION_MODES[args.execution_mode]
    try:
        generation = worker.reset_generation()
        worker.wait_until_ready(generation, timeout=args.connection_timeout_s)
        connection = worker.connect(timeout=args.connection_timeout_s)
    except Exception as error:
        raise _eval.classify_exception(error, phase="server_connect") from error
    _validate_connection_identity_v5(
        mode,
        connection,
        expected_fingerprint=expected_server_metadata_fingerprint,
    )
    scheduler.reset()
    obs = environment.reset_to(initial_state)
    obs = _pace_dummy_phase_v5(
        environment,
        obs,
        num_steps_wait=args.num_steps_wait,
        clock=clock,
    )
    episode_origin_ns = _require_nonnegative_clock(clock)
    ledger = _AttemptLedgerV5(
        execution_mode=mode.name,
        identity=identity,
        eval_seed=args.eval_seed,
        origin_ns=episode_origin_ns,
    )
    pacer = _control.NoCatchupPacer(clock)
    pending_slot = _PendingSlotV5()

    try:
        while ledger.steps < max_steps:
            pending = pending_slot.value
            if pending is not None:
                outcome = _poll_request_v5(worker, pending)
                if outcome is not None:
                    activation_now_ns = _require_nonnegative_clock(clock)
                    result = _complete_request_v5(
                        pending=pending,
                        outcome=outcome,
                        scheduler=scheduler,
                        ledger=ledger,
                        mode=mode,
                        expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                        activation_now_ns=activation_now_ns,
                    )
                    pending_slot.clear()
                    if result is not None:
                        return result

            try:
                prepared_observation, image = prepare_observation(obs, task_description, args.resize_size)
                if not isinstance(prepared_observation, Mapping):
                    raise ValueError("prepared observation must be a mapping")
            except Exception as error:
                try:
                    environment.invalidate()
                except BaseException as cleanup_error:
                    raise RunCleanupError(error, cleanup_error) from error
                raise _eval.classify_exception(error, phase="environment_step") from error

            now_ns = _require_nonnegative_clock(clock)
            next_deadline_ns = pacer.next_deadline_ns
            result = _attempt_request_v5(
                now_ns=now_ns,
                at_due=next_deadline_ns is None or now_ns >= next_deadline_ns,
                prepared_observation=prepared_observation,
                source_frame=image,
                pending_slot=pending_slot,
                worker=worker,
                scheduler=scheduler,
                ledger=ledger,
                mode=mode,
                expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                pacer=pacer,
                clock=clock,
                inference_timeout_s=args.inference_timeout_s,
                cleanup_timeout_s=args.connection_timeout_s,
            )
            if result is not None:
                return result

            due_now_ns = pacer.wait_until_due()
            pending = pending_slot.value
            if pending is not None:
                outcome = _poll_request_v5(worker, pending)
                if outcome is not None:
                    activation_now_ns = _require_nonnegative_clock(clock)
                    result = _complete_request_v5(
                        pending=pending,
                        outcome=outcome,
                        scheduler=scheduler,
                        ledger=ledger,
                        mode=mode,
                        expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                        activation_now_ns=activation_now_ns,
                    )
                    pending_slot.clear()
                    if result is not None:
                        return result

            boundary_now_ns = _require_nonnegative_clock(clock)
            result = _attempt_request_v5(
                now_ns=boundary_now_ns,
                at_due=True,
                prepared_observation=prepared_observation,
                source_frame=image,
                pending_slot=pending_slot,
                worker=worker,
                scheduler=scheduler,
                ledger=ledger,
                mode=mode,
                expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                pacer=pacer,
                clock=clock,
                inference_timeout_s=args.inference_timeout_s,
                cleanup_timeout_s=args.connection_timeout_s,
            )
            if result is not None:
                return result

            try:
                action_decision = scheduler.take_action(
                    _require_nonnegative_clock(clock),
                    control_step=ledger.steps,
                )
            except Exception as error:
                return _return_policy_failure_v5(
                    error,
                    now_ns=_require_nonnegative_clock(clock),
                    worker=worker,
                    pending_slot=pending_slot,
                    ledger=ledger,
                    cleanup_timeout_s=args.connection_timeout_s,
                )
            if action_decision.underflow:
                pending = pending_slot.value
                if pending is None or pending.trace.intent.dispatch != "background":
                    return _return_policy_failure_v5(
                        _eval.PolicyFailure("action plan underflowed without a background request"),
                        now_ns=_require_nonnegative_clock(clock),
                        worker=worker,
                        pending_slot=pending_slot,
                        ledger=ledger,
                        cleanup_timeout_s=args.connection_timeout_s,
                    )
                underflow_started_ns = _require_nonnegative_clock(clock)
                ledger.record_stall_source(ledger.steps, image)
                outcome = _wait_for_request_v5(worker, pending, timeout_s=args.inference_timeout_s)
                activation_now_ns = _require_nonnegative_clock(clock)
                result = _complete_request_v5(
                    pending=pending,
                    outcome=outcome,
                    scheduler=scheduler,
                    ledger=ledger,
                    mode=mode,
                    expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                    activation_now_ns=activation_now_ns,
                    underflow_started_ns=underflow_started_ns,
                )
                pending_slot.clear()
                if result is not None:
                    return result
                if mode.name in ("bsp_spline_async", "bsp_spline_async_speedup1"):
                    result = _attempt_request_v5(
                        now_ns=_require_nonnegative_clock(clock),
                        at_due=True,
                        prepared_observation=prepared_observation,
                        source_frame=image,
                        pending_slot=pending_slot,
                        worker=worker,
                        scheduler=scheduler,
                        ledger=ledger,
                        mode=mode,
                        expected_server_metadata_fingerprint=expected_server_metadata_fingerprint,
                        pacer=pacer,
                        clock=clock,
                        inference_timeout_s=args.inference_timeout_s,
                        cleanup_timeout_s=args.connection_timeout_s,
                    )
                    if result is not None:
                        return result
                try:
                    action_decision = scheduler.take_action(
                        _require_nonnegative_clock(clock),
                        control_step=ledger.steps,
                    )
                except Exception as error:
                    return _return_policy_failure_v5(
                        error,
                        now_ns=_require_nonnegative_clock(clock),
                        worker=worker,
                        pending_slot=pending_slot,
                        ledger=ledger,
                        cleanup_timeout_s=args.connection_timeout_s,
                    )
                if action_decision.underflow:
                    return _return_policy_failure_v5(
                        _eval.PolicyFailure("installed plan remains underflowed"),
                        now_ns=_require_nonnegative_clock(clock),
                        worker=worker,
                        pending_slot=pending_slot,
                        ledger=ledger,
                        cleanup_timeout_s=args.connection_timeout_s,
                    )

            action_started_ns = _require_nonnegative_clock(clock)
            if action_started_ns < due_now_ns:
                return _return_policy_failure_v5(
                    _eval.PolicyFailure("action start precedes controller due time"),
                    now_ns=action_started_ns,
                    worker=worker,
                    pending_slot=pending_slot,
                    ledger=ledger,
                    cleanup_timeout_s=args.connection_timeout_s,
                )
            try:
                ledger.record_action(action_decision.action)
                pacer.mark_action_started(action_started_ns)
                obs, _, done, _ = environment.step(action_decision.action.tolist())
            except Exception as error:
                try:
                    environment.invalidate()
                except BaseException as cleanup_error:
                    raise RunCleanupError(error, cleanup_error) from error
                raise _eval.classify_exception(error, phase="environment_step") from error
            ledger.replay_frames.append(image)
            ledger.steps += 1
            if bool(done):
                episode_finished_ns = _require_nonnegative_clock(clock)
                pending = pending_slot.value
                if pending is not None:
                    pending.trace.disposition = "abandoned"
                    try:
                        _cancel_generation_v5(worker, timeout_s=args.connection_timeout_s)
                    except Exception as error:
                        raise _eval.classify_exception(error, phase="policy_infer") from error
                    pending_slot.clear()
                return ledger.result(
                    success=True,
                    now_ns=episode_finished_ns,
                )

        episode_finished_ns = _require_nonnegative_clock(clock)
        pending = pending_slot.value
        if pending is not None:
            pending.trace.disposition = "abandoned"
            try:
                _cancel_generation_v5(worker, timeout_s=args.connection_timeout_s)
            except Exception as error:
                raise _eval.classify_exception(error, phase="policy_infer") from error
            pending_slot.clear()
        return ledger.result(
            success=False,
            now_ns=episode_finished_ns,
            failure_kind="timeout",
            error="maximum rollout steps reached",
        )
    except BaseException as primary_error:
        pending = pending_slot.value
        if pending is not None and pending.trace.disposition is None:
            pending.trace.disposition = "abandoned"
        if pending_slot.owns_job:
            _cancel_preserving_v5(
                worker,
                timeout_s=args.connection_timeout_s,
                primary_error=primary_error,
            )
        raise


def _draw_cumulative_wait_overlay_v5(frame: Any, lines: Tuple[str, ...]) -> Any:
    """Draw one persistent line without covering the frame with a solid box."""
    if not isinstance(lines, tuple) or len(lines) != 1 or not isinstance(lines[0], str):
        raise ValueError("cumulative wait overlay requires exactly one text line")
    from PIL import Image
    from PIL import ImageDraw

    image = Image.fromarray(np.asarray(frame).copy())
    draw = ImageDraw.Draw(image)
    draw.text(
        (6, 4),
        lines[0],
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(32, 32, 32),
    )
    return np.asarray(image).copy()


def _iter_video_frames_v5(
    control_frames: Sequence[Any],
    stalls: Sequence[_timing.ControlStallV5],
    *,
    stall_source_frames: Sequence[Tuple[int, Any]] = (),
    include_stalls: bool,
    control_hz: int = _timing.CONTROL_HZ,
    video_fps: int = _timing.DEFAULT_VIDEO_FPS,
    overlay_renderer: Optional[Callable[[Any, Tuple[str, ...]], Any]] = None,
) -> Iterator[Any]:
    renderer = _draw_cumulative_wait_overlay_v5 if overlay_renderer is None else overlay_renderer
    hold_count = _timing.validate_video_frequencies_v5(
        control_hz=control_hz,
        video_fps=video_fps,
    )
    frame_count = len(control_frames)
    source_by_step = {}  # type: Dict[int, Any]
    for value in stall_source_frames:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("stall_source_frames must contain two-tuples")
        step, frame = value
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step > frame_count
            or step in source_by_step
        ):
            raise ValueError("stall source steps must be unique within the replay timeline")
        source_by_step[step] = frame
    included_stalls = tuple(stalls) if include_stalls else ()
    stall_frames_by_step = {}
    for stall, stall_frame_count in zip(  # noqa: B905 -- simulator client runs Python 3.8.
        included_stalls,
        _timing.quantize_stall_frames_v5(included_stalls, video_fps=video_fps),
    ):
        if stall.control_step > frame_count or stall.control_step in stall_frames_by_step:
            raise ValueError("control stall step is duplicate or outside the replay timeline")
        stall_frames_by_step[stall.control_step] = (stall, stall_frame_count)
    if include_stalls and set(source_by_step) - set(stall_frames_by_step):
        raise ValueError("every retained stall source must correspond to an included stall")

    cumulative_wait_ns = 0
    for control_step, frame in enumerate(control_frames):
        stall_entry = stall_frames_by_step.get(control_step)
        if stall_entry is not None:
            stall, stall_frame_count = stall_entry
            if stall_frame_count:
                source = source_by_step.get(control_step, frame)
                for frame_index in range(stall_frame_count):
                    displayed_wait_ns = cumulative_wait_ns + min(
                        frame_index * _timing.NANOSECONDS_PER_SECOND // video_fps,
                        stall.duration_ns,
                    )
                    yield _timing.render_overlay_v5(
                        source,
                        _timing.cumulative_wait_overlay_line_v5(displayed_wait_ns),
                        renderer=renderer,
                    )
                cumulative_wait_ns += stall.duration_ns
        for _ in range(hold_count):
            if include_stalls:
                yield _timing.render_overlay_v5(
                    frame,
                    _timing.cumulative_wait_overlay_line_v5(cumulative_wait_ns),
                    renderer=renderer,
                )
            else:
                yield frame

    trailing = stall_frames_by_step.get(frame_count)
    if trailing is not None:
        stall, stall_frame_count = trailing
        if stall_frame_count:
            if frame_count not in source_by_step:
                raise ValueError("trailing stall requires a request-time source frame")
            for frame_index in range(stall_frame_count):
                displayed_wait_ns = cumulative_wait_ns + min(
                    frame_index * _timing.NANOSECONDS_PER_SECOND // video_fps,
                    stall.duration_ns,
                )
                yield _timing.render_overlay_v5(
                    source_by_step[frame_count],
                    _timing.cumulative_wait_overlay_line_v5(displayed_wait_ns),
                    renderer=renderer,
                )


def _build_video_frames_v5(
    control_frames: Sequence[Any],
    stalls: Sequence[_timing.ControlStallV5],
    *,
    stall_source_frames: Sequence[Tuple[int, Any]] = (),
    include_stalls: bool,
    control_hz: int = _timing.CONTROL_HZ,
    video_fps: int = _timing.DEFAULT_VIDEO_FPS,
    overlay_renderer: Optional[Callable[[Any, Tuple[str, ...]], Any]] = None,
) -> Tuple[Any, ...]:
    """Materialize the streaming path only for focused tests and diagnostics."""
    return tuple(
        _iter_video_frames_v5(
            control_frames,
            stalls,
            stall_source_frames=stall_source_frames,
            include_stalls=include_stalls,
            control_hz=control_hz,
            video_fps=video_fps,
            overlay_renderer=overlay_renderer,
        )
    )


def _persist_episode_artifacts_v5(
    record: _eval.EpisodeRecordV5,
    writer: _eval.ArtifactWriterV5,
    video_selector: _eval.VideoSelectorV5,
    *,
    video_show_inference_waits: bool,
    video_writer_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[_eval.EpisodeRecordV5, Optional[_eval.ArtifactErrorV5]]:
    replay_frames = record.replay_frames
    stall_source_frames = record.stall_source_frames
    persisted = dataclasses.replace(record, replay_frames=(), stall_source_frames=())
    writer.append_episode(persisted)
    video_path = video_selector.claim(persisted)
    if video_path is None:
        return persisted, None

    writer_factory = imageio.get_writer if video_writer_factory is None else video_writer_factory
    artifact_error = None  # type: Optional[_eval.ArtifactErrorV5]
    try:
        planned = _timing.build_video_timing_audit_v5(
            control_frame_count=len(replay_frames),
            requests=persisted.inference_requests,
            latencies=persisted.inference_latencies,
            activations=persisted.plan_activations,
            underflows=persisted.action_underflows,
            stalls=persisted.control_stalls,
            include_stalls=video_show_inference_waits,
        )
        frames = _iter_video_frames_v5(
            replay_frames,
            persisted.control_stalls,
            stall_source_frames=(stall_source_frames if video_show_inference_waits else ()),
            include_stalls=video_show_inference_waits,
        )
        padding = 0
        encoded_input_count = 0
        stream = writer_factory(video_path, fps=_eval.VIDEO_FPS)
        try:
            for frame in frames:
                stream.append_data(np.asarray(frame))
                encoded_input_count += 1
            if encoded_input_count == 0:
                if planned.video_frame_count:
                    raise ValueError("non-empty planned video expanded to zero frames")
                source = dict(stall_source_frames).get(0)
                if source is None:
                    raise ValueError("zero-step selected episode has no request-time frame")
                if video_show_inference_waits:
                    source = _timing.render_overlay_v5(
                        source,
                        _timing.cumulative_wait_overlay_line_v5(0),
                        renderer=_draw_cumulative_wait_overlay_v5,
                    )
                stream.append_data(np.asarray(source))
                encoded_input_count = 1
                padding = 1
            expected_count = planned.video_frame_count + padding
            if encoded_input_count != expected_count:
                raise ValueError("expanded frame count does not match the v5 timing audit")
        finally:
            stream.close()
        encoded_fps, encoded_count, encoded_duration_s = _read_encoded_video(video_path)
        audit = _eval.build_video_artifact_audit_v5(
            episode=persisted,
            path=str(video_path),
            planned=planned,
            video_show_inference_waits=video_show_inference_waits,
            encoded_fps=encoded_fps,
            encoded_frame_count=encoded_count,
            encoded_duration_s=encoded_duration_s,
            artifact_padding_frame_count=padding,
        )
        if audit.warning is not None:
            logging.warning("Video timing warning for %s: %s", persisted.episode_id, audit.warning)
        writer.append_video_audit(audit)
    except Exception as error:
        artifact_error = _eval.ArtifactErrorV5(
            episode_id=persisted.episode_id,
            artifact_type="video",
            path=str(video_path),
            error=f"{type(error).__name__}: {error}",
        )
        writer.append_artifact_error(artifact_error)
    return persisted, artifact_error


def _checkpoint_identity_v5(args: ArgsV5, code_sha: str) -> _control.CheckpointIdentityV1:
    return _control.CheckpointIdentityV1(
        code_sha=code_sha,
        config_name=args.config_name,
        checkpoint_step=args.checkpoint_step,
        checkpoint=args.checkpoint,
        container_digest=args.container_digest,
        norm_hash=args.norm_hash,
        bsp_cache_hash=args.bsp_cache_hash or None,
        bsp_cache_manifest_fingerprint=args.bsp_cache_manifest_fingerprint or None,
    )


def _make_manifest_v5(
    *,
    args: ArgsV5,
    mode: _control.ExecutionModeSpec,
    suites: Tuple[str, ...],
    task_ids: Tuple[int, ...],
    code_sha: str,
    server_metadata_fingerprint: str,
    calibration: Optional[_control.LatencyCalibrationV2],
) -> _eval.EvaluationManifestV5:
    return _eval.EvaluationManifestV5(
        schema_version=5,
        dataset_fps=10,
        source_demo_control_hz=20,
        control_freq_hz=20,
        controller_period_ns=_control.CONTROL_PERIOD_NS,
        video_fps=40,
        video_show_inference_waits=args.video_show_inference_waits,
        latency_distribution={
            "distribution": "normal",
            "mean_ns": _latency_sampling.DEFAULT_MEAN_NS,
            "stddev_ns": _latency_sampling.DEFAULT_STDDEV_NS,
            "seed": _latency_sampling.DEFAULT_SEED,
            "sampler_version": _latency_sampling.SAMPLER_VERSION,
            "negative_policy": _latency_sampling.NEGATIVE_POLICY,
        },
        theoretical_p95_latency_ns=_control.THEORETICAL_P95_LATENCY_NS,
        scheduling_latency_budget_ns=_control.SCHEDULING_LATENCY_BUDGET_NS,
        scheduling_delay_ticks=_control.SCHEDULING_DELAY_TICKS,
        execution_mode=mode.name,
        execution_parameters=mode.to_parameters_dict(),
        latency_calibration=calibration,
        server_metadata_fingerprint=server_metadata_fingerprint,
        code_sha=code_sha,
        dataset_revision=args.dataset_revision,
        config_name=args.config_name,
        checkpoint_step=args.checkpoint_step,
        bsp_cache_hash=args.bsp_cache_hash or None,
        bsp_cache_manifest_fingerprint=args.bsp_cache_manifest_fingerprint or None,
        norm_hash=args.norm_hash,
        checkpoint=args.checkpoint,
        container_digest=args.container_digest,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        policy_variant=mode.policy_variant,
        bsp_parameters=dict(_eval.BSP_PARAMETERS),
        policy_protocol=mode.policy_protocol,
        expected_action_horizon=mode.expected_action_horizon,
        suites=suites,
        task_ids=task_ids,
        trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        max_steps_by_suite={suite: _eval.MAX_STEPS_BY_SUITE[suite] for suite in suites},
        connection_timeout_s=args.connection_timeout_s,
        inference_timeout_s=args.inference_timeout_s,
        infrastructure_retries=2,
    )


def _calibration_request_v5(
    *,
    args: ArgsV5,
    suite_name: str,
    task_id: int,
    clock: _control.Clock,
) -> Tuple[Mapping[str, Any], _control.CalibrationObservationIdentityV1]:
    task_suite = _get_benchmark_suite(suite_name)
    if task_suite.n_tasks != EXPECTED_TASKS_PER_SUITE:
        raise ValueError("calibration suite must contain exactly ten tasks")
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if len(initial_states) == 0:
        raise ValueError("calibration task has no initial state")
    initial_state = initial_states[0]
    environment = _TaskEnvironmentV5(
        task,
        LIBERO_ENV_RESOLUTION,
        args.eval_seed,
        args.control_freq,
    )
    primary_error = None  # type: Optional[BaseException]
    try:
        obs = environment.reset_to(initial_state)
        obs = _pace_dummy_phase_v5(
            environment,
            obs,
            num_steps_wait=args.num_steps_wait,
            clock=clock,
        )
        request, _ = _prepare_observation(obs, str(task.language), args.resize_size)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            environment.close()
        except BaseException as cleanup_error:
            if primary_error is not None:
                raise RunCleanupError(primary_error, cleanup_error) from primary_error
            raise
    request_fingerprint = _control.canonical_fingerprint(request)
    identity = _control.CalibrationObservationIdentityV1(
        suite=suite_name,
        task_id=task_id,
        init_state_index=0,
        init_state_fingerprint=_initial_state_fingerprint_v5(initial_state),
        request_fingerprint=request_fingerprint,
    )
    return request, identity


def _ensure_new_run_directory_v5(output_dir: Path) -> None:
    collisions = sorted(path.name for path in output_dir.iterdir()) if output_dir.is_dir() else []
    if collisions:
        raise FileExistsError(f"Evaluation output directory is not empty ({collisions}); use a unique output_dir")


def _evaluate_run_v5(
    *,
    args: ArgsV5,
    suites: Tuple[str, ...],
    task_ids: Tuple[int, ...],
    mode: _control.ExecutionModeSpec,
    worker: Any,
    clock: _control.Clock,
) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    _ensure_new_run_directory_v5(output_dir)
    code_sha = _resolve_code_sha_v5()
    checkpoint_identity = _checkpoint_identity_v5(args, code_sha)

    connection = worker.connect(timeout=args.connection_timeout_s)
    metadata_fingerprint = _fingerprint_connection_v5(mode, connection)
    calibration = None  # type: Optional[_control.LatencyCalibrationV2]
    if mode.asynchronous:
        calibration_request, calibration_identity = _calibration_request_v5(
            args=args,
            suite_name=suites[0],
            task_id=task_ids[0],
            clock=clock,
        )
        calibration = _control.calibrate_async_mode(
            mode,
            calibration_request,
            calibration_identity,
            worker,
            checkpoint_identity,
            metadata_fingerprint,
        )

    manifest = _make_manifest_v5(
        args=args,
        mode=mode,
        suites=suites,
        task_ids=task_ids,
        code_sha=code_sha,
        server_metadata_fingerprint=metadata_fingerprint,
        calibration=calibration,
    )
    # No output artifact exists before capability, calibration, and manifest
    # validation have all succeeded.
    writer = _eval.ArtifactWriterV5(output_dir)
    writer.write_manifest(manifest)
    video_selector = _eval.VideoSelectorV5(output_dir / "videos")
    records = []  # type: List[_eval.EpisodeRecordV5]
    artifact_errors = []  # type: List[_eval.ArtifactErrorV5]
    np.random.seed(args.eval_seed)

    for suite_name in suites:
        task_suite = _get_benchmark_suite(suite_name)
        if task_suite.n_tasks != EXPECTED_TASKS_PER_SUITE:
            raise ValueError(f"Expected ten tasks in {suite_name}, got {task_suite.n_tasks}")
        for task_id in tqdm.tqdm(task_ids, desc=suite_name):
            task = task_suite.get_task(task_id)
            description = str(task.language)
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) < args.num_trials_per_task:
                raise ValueError("selected task has fewer initial states than requested trials")
            environment = _TaskEnvironmentV5(
                task,
                LIBERO_ENV_RESOLUTION,
                args.eval_seed,
                args.control_freq,
            )
            environment_primary = None  # type: Optional[BaseException]
            try:
                for init_state_index in tqdm.tqdm(
                    range(args.num_trials_per_task),
                    desc=f"task-{task_id:03d}",
                    leave=False,
                ):
                    initial_state = initial_states[init_state_index]
                    identity = _eval.EpisodeIdentity(
                        suite=suite_name,
                        task_id=task_id,
                        task_name=description,
                        init_state_index=init_state_index,
                        init_state_fingerprint=_initial_state_fingerprint_v5(initial_state),
                    )
                    scheduler = _control.make_scheduler_v5(mode, calibration)

                    def attempt(
                        _attempt_number: int,
                        environment: _TaskEnvironmentV5 = environment,
                        scheduler: _control.ModeSchedulerV5 = scheduler,
                        initial_state: Any = initial_state,
                        identity: _eval.EpisodeIdentity = identity,
                        description: str = description,
                        suite_name: str = suite_name,
                    ) -> _eval.AttemptResultV5:
                        return _run_attempt_v5(
                            environment=environment,
                            worker=worker,
                            scheduler=scheduler,
                            initial_state=initial_state,
                            identity=identity,
                            task_description=description,
                            args=args,
                            max_steps=_eval.MAX_STEPS_BY_SUITE[suite_name],
                            expected_server_metadata_fingerprint=metadata_fingerprint,
                            clock=clock,
                        )

                    expected_budget = (
                        calibration.derived_prefetch_budget_ns
                        if mode.name in ("bsp_spline_async", "bsp_spline_async_speedup1") and calibration is not None
                        else None
                    )
                    record = _eval.run_episode_with_retries_v5(
                        identity,
                        attempt,
                        eval_seed=args.eval_seed,
                        execution_mode=mode.name,
                        infrastructure_retries=2,
                        expected_bsp_prefetch_budget_ns=expected_budget,
                    )
                    record, artifact_error = _persist_episode_artifacts_v5(
                        record,
                        writer,
                        video_selector,
                        video_show_inference_waits=args.video_show_inference_waits,
                    )
                    records.append(record)
                    if artifact_error is not None:
                        artifact_errors.append(artifact_error)
                    logging.info(
                        "%s status=%s attempts=%d steps=%d",
                        record.episode_id,
                        record.status,
                        record.attempts,
                        record.steps,
                    )
            except BaseException as error:
                environment_primary = error
                raise
            finally:
                try:
                    environment.close()
                except BaseException as cleanup_error:
                    if environment_primary is not None:
                        raise RunCleanupError(environment_primary, cleanup_error) from environment_primary
                    raise

    summary = writer.write_summary(records, artifact_errors=artifact_errors)
    if not summary["acceptance_complete"]:
        raise RuntimeError(
            "Evaluation acceptance is incomplete: {} infrastructure episodes, {} artifact errors".format(
                summary["incomplete_infrastructure_count"],
                summary["artifact_error_count"],
            )
        )
    return summary


def eval_libero_v5(args: ArgsV5) -> Dict[str, Any]:
    suites, task_ids, mode = _validate_args_v5(args)
    clock = _SystemClock()
    worker = _async.AsyncInferenceWorker(
        _make_policy_factory_v5(args),
        shutdown_timeout_s=args.worker_shutdown_timeout_s,
        recv_poll_interval_s=args.recv_poll_interval_s,
        monotonic_ns=clock.monotonic_ns,
        wait_until_ns=clock.wait_until_ns,
        latency_sampler=_latency_sampling.NormalLatencySamplerV1(),
    )
    primary_error = None  # type: Optional[BaseException]
    try:
        return _evaluate_run_v5(
            args=args,
            suites=suites,
            task_ids=task_ids,
            mode=mode,
            worker=worker,
            clock=clock,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_worker_v5(
            worker,
            primary_error=primary_error,
            timeout_s=args.connection_timeout_s,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_v5)
