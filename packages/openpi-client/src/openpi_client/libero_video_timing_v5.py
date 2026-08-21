"""Strict schema-v5 timing events and deterministic LIBERO video timing.

The controller owns event measurement.  This module owns only immutable event
records, fail-closed cross-event validation, and integer video accounting.  It
does not import schema-v3 timing records or infer stalls from request latency.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import copy
import dataclasses
import math
from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

from openpi_client import latency_sampling


CONTROL_HZ = 20
DEFAULT_VIDEO_FPS = 40
NANOSECONDS_PER_SECOND = 1_000_000_000
EPISODE_MONOTONIC_CLOCK = "episode_monotonic_ns"

STALL_REASON_SYNCHRONOUS_INFERENCE = "synchronous_inference"
STALL_REASON_ASYNC_ACTION_UNDERFLOW = "async_action_underflow"

_DISPATCHES = frozenset(("blocking_initial", "blocking_replan", "background"))
_TRIGGERS = frozenset(
    (
        "initial_plan",
        "baseline_chunk_exhausted",
        "baseline_async_launch",
        "baseline_async_capacity_replan",
        "rtc_launch",
        "bsp_curve_exhausted",
        "bsp_prefetch",
        "bsp_stale_replan",
    )
)
_DISPOSITIONS = frozenset(("activated", "discarded_stale_phase", "failed", "abandoned"))
_LATENCY_OUTCOMES = frozenset(("success", "policy_failure"))
_ACTIVATIONS = frozenset(("initial", "blocking_replace", "immediate_swap"))
_STALL_REASONS = frozenset((STALL_REASON_SYNCHRONOUS_INFERENCE, STALL_REASON_ASYNC_ACTION_UNDERFLOW))
_EXECUTION_MODES = frozenset(
    (
        "baseline_async",
        "baseline_async_recovery",
        "baseline_rtc",
        "bsp_spline_async",
        "bsp_spline_async_speedup1",
        "bsp_spline_async_native",
        "baseline_sync",
        "bsp_spline_sync",
    )
)
_NATIVE_MODES = frozenset(("baseline_async", "baseline_async_recovery", "baseline_rtc", "baseline_sync"))
_ASYNC_MODES = frozenset(
    (
        "baseline_async",
        "baseline_async_recovery",
        "baseline_rtc",
        "bsp_spline_async",
        "bsp_spline_async_speedup1",
        "bsp_spline_async_native",
    )
)

_REQUEST_FIELDS = frozenset(
    (
        "clock",
        "request_id",
        "observation_control_step",
        "submitted_offset_ns",
        "flow_seed",
        "dispatch",
        "trigger",
        "scheduler_context",
        "disposition",
        "latency_sample_key",
        "sampled_target_latency_ns",
    )
)
_LATENCY_FIELDS = frozenset(
    (
        "clock",
        "request_id",
        "completed_offset_ns",
        "duration_ns",
        "outcome",
        "raw_inference_latency_ns",
        "requested_synthetic_delay_ns",
        "observed_synthetic_delay_ns",
        "observed_effective_latency_ns",
        "latency_overshoot_ns",
        "sampled_target_latency_ns",
    )
)
_SEAM_FIELDS = frozenset(
    (
        "clock",
        "plan_id",
        "request_id",
        "control_step",
        "arm_l2_jump",
        "arm_max_abs_jump",
        "gripper_abs_jump",
    )
)
_ACTIVATION_FIELDS = frozenset(
    (
        "clock",
        "plan_id",
        "request_id",
        "control_step",
        "activated_offset_ns",
        "activation",
        "activation_context",
    )
)
_UNDERFLOW_FIELDS = frozenset(("clock", "request_id", "control_step", "started_offset_ns", "duration_ns"))
_STALL_FIELDS = frozenset(("clock", "request_id", "control_step", "started_offset_ns", "duration_ns", "reason"))
_AUDIT_FIELDS = frozenset(
    (
        "control_hz",
        "video_fps",
        "control_frame_count",
        "held_frame_count",
        "request_count",
        "latency_count",
        "activation_count",
        "underflow_count",
        "total_request_latency_ns",
        "total_underflow_ns",
        "measured_stall_count",
        "measured_control_stall_ns",
        "included_stall_count",
        "included_control_stall_ns",
        "included_stall_reasons",
        "included_stall_frame_counts",
        "stall_frame_count",
        "video_frame_count",
        "control_duration_ns",
        "video_duration_ns",
        "expected_duration_ns",
        "duration_deviation_ns",
    )
)

_Frame = TypeVar("_Frame")
_Record = TypeVar("_Record")


def _require_json_object(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(label))
    if any(not isinstance(key, str) for key in value):
        raise ValueError("{} keys must be strings".format(label))
    return value


def _require_exact_fields(value: Any, fields: frozenset, *, label: str) -> Dict[str, Any]:
    payload = _require_json_object(value, label=label)
    if set(payload) != fields:
        missing = sorted(fields.difference(payload))
        extra = sorted(set(payload).difference(fields))
        raise ValueError("{} fields mismatch; missing={}, extra={}".format(label, missing, extra))
    return payload


def _require_integer(
    value: Any,
    *,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    if minimum is not None and value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be at most {}".format(name, maximum))
    return value


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    return _require_integer(value, name=name, minimum=0)


def _require_enum(value: Any, choices: frozenset, *, name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError("Unsupported {}: {}".format(name, value))
    return value


def _validate_clock(payload: Mapping[str, Any], *, label: str) -> None:
    if payload["clock"] != EPISODE_MONOTONIC_CLOCK:
        raise ValueError("{} clock must be {}".format(label, EPISODE_MONOTONIC_CLOCK))


def _context_copy(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(label))
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError("{} keys must be strings".format(label))
    return result


def _validate_scheduler_context(trigger: str, context: Mapping[str, Any]) -> None:
    if trigger in ("initial_plan", "baseline_chunk_exhausted", "bsp_curve_exhausted"):
        if context:
            raise ValueError("{} scheduler context must be empty".format(trigger))
        return
    if trigger in ("baseline_async_launch", "rtc_launch"):
        if set(context) != {"s", "d"}:
            raise ValueError("RTC scheduler context must contain exactly s and d")
        s = _require_integer(context["s"], name="RTC s", minimum=8, maximum=16)
        d = _require_integer(context["d"], name="RTC d", minimum=0)
        if d > s or s + d > 16:
            raise ValueError("RTC scheduler context must satisfy d <= s and s + d <= 16")
        return
    if trigger == "baseline_async_capacity_replan":
        if set(context) != {"action_cursor", "forecast_delay_ticks"}:
            raise ValueError(
                "baseline async capacity replan context must contain exactly action_cursor and forecast_delay_ticks"
            )
        cursor = _require_integer(context["action_cursor"], name="action_cursor", minimum=9, maximum=15)
        delay = _require_integer(context["forecast_delay_ticks"], name="forecast_delay_ticks", minimum=8, maximum=8)
        if cursor + delay <= 16:
            raise ValueError("baseline async capacity replan requires cursor + delay > 16")
        return
    if trigger == "bsp_prefetch":
        if set(context) != {"remaining_plan_ns", "budget_ns", "request_control_step"}:
            raise ValueError(
                "BSP prefetch context must contain exactly remaining_plan_ns, budget_ns, and request_control_step"
            )
        remaining = _require_nonnegative_integer(context["remaining_plan_ns"], name="remaining_plan_ns")
        budget = _require_nonnegative_integer(context["budget_ns"], name="budget_ns")
        _require_nonnegative_integer(context["request_control_step"], name="request_control_step")
        if remaining > budget:
            raise ValueError("BSP prefetch requires remaining_plan_ns <= budget_ns")
        return
    if trigger == "bsp_stale_replan":
        expected = {
            "discarded_request_control_step",
            "discarded_activation_control_step",
            "executed_prefix_steps",
            "phase_offset_microindices",
            "curve_t_max_microindices",
        }
        if set(context) != expected:
            raise ValueError("BSP stale replan context must exactly identify the discarded response")
        request_step = _require_nonnegative_integer(
            context["discarded_request_control_step"],
            name="discarded_request_control_step",
        )
        activation_step = _require_nonnegative_integer(
            context["discarded_activation_control_step"],
            name="discarded_activation_control_step",
        )
        prefix = _require_nonnegative_integer(context["executed_prefix_steps"], name="executed_prefix_steps")
        phase = _require_nonnegative_integer(
            context["phase_offset_microindices"],
            name="phase_offset_microindices",
        )
        t_max = _require_nonnegative_integer(
            context["curve_t_max_microindices"],
            name="curve_t_max_microindices",
        )
        if (
            activation_step - request_step != prefix
            or phase not in (prefix * 500_000, prefix * 1_000_000)
            or phase <= t_max
        ):
            raise ValueError("BSP stale replan phase identity is inconsistent")
        return
    raise ValueError("Unsupported request trigger: {}".format(trigger))


def _validate_activation_context(context: Mapping[str, Any]) -> None:
    if set(context) == {"action_cursor"}:
        _require_integer(context["action_cursor"], name="action_cursor", minimum=0, maximum=15)
        return
    expected = {
        "request_control_step",
        "activation_control_step",
        "executed_prefix_steps",
        "phase_offset_microindices",
        "first_sample_microindices",
        "remaining_curve_microindices",
        "remaining_curve_ns",
        "immediate_prefetch",
    }
    if set(context) in (expected, expected | {"prefetch_budget_ns"}):
        request_step = _require_nonnegative_integer(context["request_control_step"], name="request_control_step")
        activation_step = _require_nonnegative_integer(
            context["activation_control_step"],
            name="activation_control_step",
        )
        prefix = _require_nonnegative_integer(context["executed_prefix_steps"], name="executed_prefix_steps")
        phase = _require_nonnegative_integer(
            context["phase_offset_microindices"],
            name="phase_offset_microindices",
        )
        first_sample = _require_nonnegative_integer(
            context["first_sample_microindices"],
            name="first_sample_microindices",
        )
        remaining_indices = _require_nonnegative_integer(
            context["remaining_curve_microindices"],
            name="remaining_curve_microindices",
        )
        remaining_ns = _require_nonnegative_integer(context["remaining_curve_ns"], name="remaining_curve_ns")
        immediate = _require_integer(context["immediate_prefetch"], name="immediate_prefetch", minimum=0, maximum=1)
        prefetch_budget_ns = (
            _require_nonnegative_integer(context["prefetch_budget_ns"], name="prefetch_budget_ns")
            if "prefetch_budget_ns" in context
            else 400_000_000
        )
        matching_rates = tuple(
            rate
            for rate in (10, 20)
            if phase * 20 == prefix * rate * 1_000_000
            and remaining_ns
            == (remaining_indices * NANOSECONDS_PER_SECOND + rate * 1_000_000 - 1) // (rate * 1_000_000)
        )
        if (
            activation_step - request_step != prefix
            or not matching_rates
            or first_sample < phase
            or (immediate == 1 and remaining_ns > prefetch_budget_ns)
        ):
            raise ValueError("BSP phase-skip activation context is inconsistent")
        return
    raise ValueError("activation_context must contain exactly native or phase-skip BSP fields")


@dataclasses.dataclass(frozen=True)
class RequestEventV5:
    request_id: int
    observation_control_step: int
    submitted_offset_ns: int
    flow_seed: int
    dispatch: str
    trigger: str
    scheduler_context: Mapping[str, int]
    disposition: str
    latency_sample_key: latency_sampling.LatencySampleKeyV1
    sampled_target_latency_ns: int

    def __post_init__(self) -> None:
        context = _context_copy(self.scheduler_context, label="scheduler_context")
        object.__setattr__(self, "scheduler_context", MappingProxyType(context))
        self._validate()

    def _validate(self) -> None:
        _require_nonnegative_integer(self.request_id, name="request_id")
        _require_nonnegative_integer(self.observation_control_step, name="observation_control_step")
        _require_nonnegative_integer(self.submitted_offset_ns, name="submitted_offset_ns")
        _require_integer(self.flow_seed, name="flow_seed", minimum=0, maximum=2**32 - 1)
        _require_enum(self.dispatch, _DISPATCHES, name="request dispatch")
        _require_enum(self.trigger, _TRIGGERS, name="request trigger")
        _require_enum(self.disposition, _DISPOSITIONS, name="request disposition")
        if not isinstance(self.latency_sample_key, latency_sampling.LatencySampleKeyV1):
            raise ValueError("latency_sample_key must be a LatencySampleKeyV1")
        if self.latency_sample_key.request_ordinal != self.request_id:
            raise ValueError("latency sample request_ordinal must equal request_id")
        _require_nonnegative_integer(
            self.sampled_target_latency_ns,
            name="sampled_target_latency_ns",
        )
        expected_dispatch = {
            "initial_plan": "blocking_initial",
            "baseline_chunk_exhausted": "blocking_replan",
            "baseline_async_launch": "background",
            "baseline_async_capacity_replan": "blocking_replan",
            "rtc_launch": "background",
            "bsp_curve_exhausted": "blocking_replan",
            "bsp_prefetch": "background",
            "bsp_stale_replan": "blocking_replan",
        }[self.trigger]
        if self.dispatch != expected_dispatch:
            raise ValueError("request trigger {} requires dispatch {}".format(self.trigger, expected_dispatch))
        _validate_scheduler_context(self.trigger, self.scheduler_context)

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "request_id": self.request_id,
            "observation_control_step": self.observation_control_step,
            "submitted_offset_ns": self.submitted_offset_ns,
            "flow_seed": self.flow_seed,
            "dispatch": self.dispatch,
            "trigger": self.trigger,
            "scheduler_context": dict(self.scheduler_context),
            "disposition": self.disposition,
            "latency_sample_key": self.latency_sample_key.to_dict(),
            "sampled_target_latency_ns": self.sampled_target_latency_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RequestEventV5":
        payload = _require_exact_fields(value, _REQUEST_FIELDS, label="request event")
        _validate_clock(payload, label="request event")
        return cls(
            request_id=payload["request_id"],
            observation_control_step=payload["observation_control_step"],
            submitted_offset_ns=payload["submitted_offset_ns"],
            flow_seed=payload["flow_seed"],
            dispatch=payload["dispatch"],
            trigger=payload["trigger"],
            scheduler_context=_require_json_object(payload["scheduler_context"], label="scheduler_context"),
            disposition=payload["disposition"],
            latency_sample_key=latency_sampling.LatencySampleKeyV1.from_dict(payload["latency_sample_key"]),
            sampled_target_latency_ns=payload["sampled_target_latency_ns"],
        )


@dataclasses.dataclass(frozen=True)
class LatencyEventV5:
    request_id: int
    completed_offset_ns: int
    duration_ns: int
    outcome: str
    raw_inference_latency_ns: Optional[int] = None
    requested_synthetic_delay_ns: Optional[int] = None
    observed_synthetic_delay_ns: Optional[int] = None
    observed_effective_latency_ns: Optional[int] = None
    latency_overshoot_ns: Optional[int] = None
    sampled_target_latency_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.raw_inference_latency_ns is None:
            object.__setattr__(self, "raw_inference_latency_ns", self.duration_ns)
        if self.sampled_target_latency_ns is None:
            object.__setattr__(self, "sampled_target_latency_ns", self.duration_ns)
        scheduled_effective_ns = max(self.raw_inference_latency_ns, self.sampled_target_latency_ns)
        if self.requested_synthetic_delay_ns is None:
            object.__setattr__(
                self,
                "requested_synthetic_delay_ns",
                scheduled_effective_ns - self.raw_inference_latency_ns,
            )
        if self.observed_effective_latency_ns is None:
            object.__setattr__(self, "observed_effective_latency_ns", self.duration_ns)
        if self.observed_synthetic_delay_ns is None:
            object.__setattr__(
                self,
                "observed_synthetic_delay_ns",
                self.observed_effective_latency_ns - self.raw_inference_latency_ns,
            )
        if self.latency_overshoot_ns is None:
            object.__setattr__(
                self,
                "latency_overshoot_ns",
                self.observed_effective_latency_ns - scheduled_effective_ns,
            )
        self._validate()

    def _validate(self) -> None:
        _require_nonnegative_integer(self.request_id, name="request_id")
        _require_nonnegative_integer(self.completed_offset_ns, name="completed_offset_ns")
        _require_nonnegative_integer(self.duration_ns, name="duration_ns")
        raw_ns = _require_nonnegative_integer(
            self.raw_inference_latency_ns,
            name="raw_inference_latency_ns",
        )
        requested_ns = _require_nonnegative_integer(
            self.requested_synthetic_delay_ns,
            name="requested_synthetic_delay_ns",
        )
        observed_synthetic_ns = _require_nonnegative_integer(
            self.observed_synthetic_delay_ns,
            name="observed_synthetic_delay_ns",
        )
        observed_effective_ns = _require_nonnegative_integer(
            self.observed_effective_latency_ns,
            name="observed_effective_latency_ns",
        )
        overshoot_ns = _require_nonnegative_integer(
            self.latency_overshoot_ns,
            name="latency_overshoot_ns",
        )
        sampled_ns = _require_nonnegative_integer(
            self.sampled_target_latency_ns,
            name="sampled_target_latency_ns",
        )
        _require_enum(self.outcome, _LATENCY_OUTCOMES, name="latency outcome")
        scheduled_effective_ns = max(raw_ns, sampled_ns)
        if requested_ns != scheduled_effective_ns - raw_ns:
            raise ValueError("requested synthetic delay must complete the scheduled effective latency")
        if raw_ns + observed_synthetic_ns != observed_effective_ns:
            raise ValueError("raw plus observed synthetic latency must equal observed effective latency")
        if self.duration_ns != observed_effective_ns:
            raise ValueError("duration_ns must equal observed_effective_latency_ns")
        if self.duration_ns > self.completed_offset_ns:
            raise ValueError("latency duration cannot begin before the episode origin")
        if observed_effective_ns < scheduled_effective_ns:
            raise ValueError("observed effective latency cannot be shorter than its scheduled target")
        if overshoot_ns != observed_effective_ns - scheduled_effective_ns:
            raise ValueError("latency overshoot must equal observed latency minus scheduled latency")
        if observed_synthetic_ns != requested_ns + overshoot_ns:
            raise ValueError("observed synthetic delay must equal requested delay plus overshoot")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "request_id": self.request_id,
            "completed_offset_ns": self.completed_offset_ns,
            "duration_ns": self.duration_ns,
            "outcome": self.outcome,
            "raw_inference_latency_ns": self.raw_inference_latency_ns,
            "requested_synthetic_delay_ns": self.requested_synthetic_delay_ns,
            "observed_synthetic_delay_ns": self.observed_synthetic_delay_ns,
            "observed_effective_latency_ns": self.observed_effective_latency_ns,
            "latency_overshoot_ns": self.latency_overshoot_ns,
            "sampled_target_latency_ns": self.sampled_target_latency_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LatencyEventV5":
        payload = _require_exact_fields(value, _LATENCY_FIELDS, label="latency event")
        _validate_clock(payload, label="latency event")
        return cls(
            request_id=payload["request_id"],
            completed_offset_ns=payload["completed_offset_ns"],
            duration_ns=payload["duration_ns"],
            outcome=payload["outcome"],
            raw_inference_latency_ns=payload["raw_inference_latency_ns"],
            requested_synthetic_delay_ns=payload["requested_synthetic_delay_ns"],
            observed_synthetic_delay_ns=payload["observed_synthetic_delay_ns"],
            observed_effective_latency_ns=payload["observed_effective_latency_ns"],
            latency_overshoot_ns=payload["latency_overshoot_ns"],
            sampled_target_latency_ns=payload["sampled_target_latency_ns"],
        )


@dataclasses.dataclass(frozen=True)
class ActionSeamV5:
    plan_id: int
    request_id: int
    control_step: int
    arm_l2_jump: float
    arm_max_abs_jump: float
    gripper_abs_jump: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in ("plan_id", "request_id", "control_step"):
            _require_nonnegative_integer(getattr(self, name), name=name)
        for name in ("arm_l2_jump", "arm_max_abs_jump", "gripper_abs_jump"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{} must be numeric".format(name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))

    @classmethod
    def from_actions(
        cls,
        *,
        plan_id: int,
        request_id: int,
        control_step: int,
        previous_action: Sequence[float],
        activated_action: Sequence[float],
    ) -> "ActionSeamV5":
        if len(previous_action) < 7 or len(activated_action) < 7:
            raise ValueError("seam actions must contain six arm values and one gripper value")
        previous = tuple(float(value) for value in previous_action[:7])
        activated = tuple(float(value) for value in activated_action[:7])
        if any(not math.isfinite(value) for value in previous + activated):
            raise ValueError("seam actions must be finite")
        arm_deltas = tuple(activated[index] - previous[index] for index in range(6))
        return cls(
            plan_id=plan_id,
            request_id=request_id,
            control_step=control_step,
            arm_l2_jump=math.sqrt(sum(value * value for value in arm_deltas)),
            arm_max_abs_jump=max(abs(value) for value in arm_deltas),
            gripper_abs_jump=abs(activated[6] - previous[6]),
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "control_step": self.control_step,
            "arm_l2_jump": float(self.arm_l2_jump),
            "arm_max_abs_jump": float(self.arm_max_abs_jump),
            "gripper_abs_jump": float(self.gripper_abs_jump),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ActionSeamV5":
        payload = _require_exact_fields(value, _SEAM_FIELDS, label="action seam")
        _validate_clock(payload, label="action seam")
        return cls(
            plan_id=payload["plan_id"],
            request_id=payload["request_id"],
            control_step=payload["control_step"],
            arm_l2_jump=payload["arm_l2_jump"],
            arm_max_abs_jump=payload["arm_max_abs_jump"],
            gripper_abs_jump=payload["gripper_abs_jump"],
        )


@dataclasses.dataclass(frozen=True)
class PlanActivationV5:
    plan_id: int
    request_id: int
    control_step: int
    activated_offset_ns: int
    activation: str
    activation_context: Mapping[str, int]

    def __post_init__(self) -> None:
        context = _context_copy(self.activation_context, label="activation_context")
        object.__setattr__(self, "activation_context", MappingProxyType(context))
        self._validate()

    def _validate(self) -> None:
        _require_nonnegative_integer(self.plan_id, name="plan_id")
        _require_nonnegative_integer(self.request_id, name="request_id")
        _require_nonnegative_integer(self.control_step, name="control_step")
        _require_nonnegative_integer(self.activated_offset_ns, name="activated_offset_ns")
        _require_enum(self.activation, _ACTIVATIONS, name="plan activation")
        _validate_activation_context(self.activation_context)

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "control_step": self.control_step,
            "activated_offset_ns": self.activated_offset_ns,
            "activation": self.activation,
            "activation_context": dict(self.activation_context),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PlanActivationV5":
        payload = _require_exact_fields(value, _ACTIVATION_FIELDS, label="plan activation")
        _validate_clock(payload, label="plan activation")
        return cls(
            plan_id=payload["plan_id"],
            request_id=payload["request_id"],
            control_step=payload["control_step"],
            activated_offset_ns=payload["activated_offset_ns"],
            activation=payload["activation"],
            activation_context=_require_json_object(payload["activation_context"], label="activation_context"),
        )


@dataclasses.dataclass(frozen=True)
class ActionUnderflowV5:
    request_id: int
    control_step: int
    started_offset_ns: int
    duration_ns: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_nonnegative_integer(self.request_id, name="request_id")
        _require_nonnegative_integer(self.control_step, name="control_step")
        _require_nonnegative_integer(self.started_offset_ns, name="started_offset_ns")
        _require_nonnegative_integer(self.duration_ns, name="duration_ns")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "request_id": self.request_id,
            "control_step": self.control_step,
            "started_offset_ns": self.started_offset_ns,
            "duration_ns": self.duration_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ActionUnderflowV5":
        payload = _require_exact_fields(value, _UNDERFLOW_FIELDS, label="action underflow")
        _validate_clock(payload, label="action underflow")
        return cls(
            request_id=payload["request_id"],
            control_step=payload["control_step"],
            started_offset_ns=payload["started_offset_ns"],
            duration_ns=payload["duration_ns"],
        )


@dataclasses.dataclass(frozen=True)
class ControlStallV5:
    request_id: int
    control_step: int
    started_offset_ns: int
    duration_ns: int
    reason: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_nonnegative_integer(self.request_id, name="request_id")
        _require_nonnegative_integer(self.control_step, name="control_step")
        _require_nonnegative_integer(self.started_offset_ns, name="started_offset_ns")
        _require_nonnegative_integer(self.duration_ns, name="duration_ns")
        _require_enum(self.reason, _STALL_REASONS, name="control stall reason")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "clock": EPISODE_MONOTONIC_CLOCK,
            "request_id": self.request_id,
            "control_step": self.control_step,
            "started_offset_ns": self.started_offset_ns,
            "duration_ns": self.duration_ns,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ControlStallV5":
        payload = _require_exact_fields(value, _STALL_FIELDS, label="control stall")
        _validate_clock(payload, label="control stall")
        return cls(
            request_id=payload["request_id"],
            control_step=payload["control_step"],
            started_offset_ns=payload["started_offset_ns"],
            duration_ns=payload["duration_ns"],
            reason=payload["reason"],
        )


def _records_tuple(values: Sequence[_Record], record_type: Type[_Record], *, label: str) -> Tuple[_Record, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("{} must be a sequence".format(label))
    records = tuple(values)
    if any(not isinstance(value, record_type) for value in records):
        raise ValueError("{} must contain only {} records".format(label, record_type.__name__))
    for record in records:
        record._validate()  # type: ignore[attr-defined]
    return records


def _stable_flow_seeds(
    requests: Sequence[RequestEventV5],
    *,
    eval_seed: int,
    identity: Any,
) -> Tuple[int, ...]:
    _require_nonnegative_integer(eval_seed, name="eval_seed")
    if identity is None:
        raise ValueError("identity is required for stable flow-seed derivation")
    try:
        from openpi_client.libero_eval import stable_replan_seed

        return tuple(stable_replan_seed(eval_seed, identity, request.request_id) for request in requests)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("identity is invalid for stable flow-seed derivation") from error


def validate_timing_events_v5(
    *,
    requests: Sequence[RequestEventV5],
    latencies: Sequence[LatencyEventV5],
    activations: Sequence[PlanActivationV5],
    underflows: Sequence[ActionUnderflowV5],
    stalls: Sequence[ControlStallV5],
    steps: int,
    episode_duration_ns: int,
    execution_mode: str,
    eval_seed: int,
    identity: Any,
    expected_bsp_prefetch_budget_ns: Optional[int] = None,
    verify_sampled_targets: bool = False,
) -> Tuple[
    Tuple[RequestEventV5, ...],
    Tuple[LatencyEventV5, ...],
    Tuple[PlanActivationV5, ...],
    Tuple[ActionUnderflowV5, ...],
    Tuple[ControlStallV5, ...],
]:
    """Validate one final-attempt event graph.

    Returns:
        The five event sequences as defensive tuples, in argument order.

    Raises:
        ValueError: If a record, mode relation, seed, order, or interval is invalid.
    """
    request_records = _records_tuple(requests, RequestEventV5, label="requests")
    latency_records = _records_tuple(latencies, LatencyEventV5, label="latencies")
    activation_records = _records_tuple(activations, PlanActivationV5, label="activations")
    underflow_records = _records_tuple(underflows, ActionUnderflowV5, label="underflows")
    stall_records = _records_tuple(stalls, ControlStallV5, label="stalls")
    step_count = _require_nonnegative_integer(steps, name="steps")
    duration = _require_nonnegative_integer(episode_duration_ns, name="episode_duration_ns")
    _require_enum(execution_mode, _EXECUTION_MODES, name="execution mode")
    if type(verify_sampled_targets) is not bool:
        raise ValueError("verify_sampled_targets must be boolean")
    if not request_records:
        raise ValueError("timing event graph must contain an initial request")

    seeds = _stable_flow_seeds(
        request_records,
        eval_seed=eval_seed,
        identity=identity,
    )
    expected_later_request = {
        "baseline_async": ("background", "baseline_async_launch"),
        "baseline_async_recovery": ("background", "baseline_async_launch"),
        "baseline_rtc": ("background", "rtc_launch"),
        "bsp_spline_async": ("background", "bsp_prefetch"),
        "bsp_spline_async_speedup1": ("background", "bsp_prefetch"),
        "bsp_spline_async_native": ("background", "bsp_prefetch"),
        "baseline_sync": ("blocking_replan", "baseline_chunk_exhausted"),
        "bsp_spline_sync": ("blocking_replan", "bsp_curve_exhausted"),
    }[execution_mode]
    if execution_mode in ("bsp_spline_async", "bsp_spline_async_speedup1"):
        if expected_bsp_prefetch_budget_ns is None:
            raise ValueError("asynchronous BSP requires its calibrated prefetch budget")
        calibrated_bsp_budget = _require_nonnegative_integer(
            expected_bsp_prefetch_budget_ns,
            name="expected_bsp_prefetch_budget_ns",
        )
    elif execution_mode == "bsp_spline_async_native":
        if expected_bsp_prefetch_budget_ns is not None:
            raise ValueError("native asynchronous BSP records rolling budgets instead of one fixed budget")
        calibrated_bsp_budget = None
    else:
        if expected_bsp_prefetch_budget_ns is not None:
            raise ValueError("only asynchronous BSP accepts a prefetch budget")
        calibrated_bsp_budget = None
    observed_bsp_budgets = set()
    previous_submission = -1
    previous_observation_step = -1
    for expected_id, (request, expected_seed) in enumerate(zip(request_records, seeds)):
        if request.request_id != expected_id:
            raise ValueError("request ids must be contiguous from zero")
        if request.submitted_offset_ns < previous_submission:
            raise ValueError("requests must be chronological")
        if request.observation_control_step < previous_observation_step:
            raise ValueError("request observation steps must be chronological")
        if request.observation_control_step > step_count:
            raise ValueError("request observation_control_step must be within 0..steps")
        if request.submitted_offset_ns > duration:
            raise ValueError("request submission is past episode_duration_ns")
        if request.flow_seed != expected_seed:
            raise ValueError("request flow_seed does not match its stable logical request id")
        expected_sample_key = latency_sampling.LatencySampleKeyV1(
            namespace="formal",
            seed=eval_seed,
            suite=identity.suite,
            task_id=identity.task_id,
            trial_index=identity.init_state_index,
            request_ordinal=request.request_id,
        )
        if request.latency_sample_key != expected_sample_key:
            raise ValueError("request latency sample key does not match its episode identity")
        if verify_sampled_targets:
            expected_target = (
                0
                if execution_mode == "bsp_spline_async_native"
                else latency_sampling.NormalLatencySamplerV1().sample_target_ns(expected_sample_key)
            )
            if request.sampled_target_latency_ns != expected_target:
                raise ValueError("request sampled target does not match the frozen schema-v5 sampler")
        if expected_id == 0:
            if (
                request.dispatch != "blocking_initial"
                or request.trigger != "initial_plan"
                or request.observation_control_step != 0
            ):
                raise ValueError("first request must be the step-zero blocking initial plan")
        elif execution_mode in ("bsp_spline_async", "bsp_spline_async_speedup1", "bsp_spline_async_native"):
            if (request.dispatch, request.trigger) not in (
                expected_later_request,
                ("blocking_replan", "bsp_stale_replan"),
            ):
                raise ValueError("request dispatch/trigger does not match BSP execution mode")
        elif execution_mode == "baseline_async_recovery":
            if (request.dispatch, request.trigger) not in (
                expected_later_request,
                ("blocking_replan", "baseline_async_capacity_replan"),
            ):
                raise ValueError("request dispatch/trigger does not match baseline async recovery mode")
        elif (request.dispatch, request.trigger) != expected_later_request:
            raise ValueError("request dispatch/trigger does not match execution mode")
        if request.trigger == "bsp_prefetch":
            budget = request.scheduler_context["budget_ns"]
            observed_bsp_budgets.add(budget)
            if execution_mode == "bsp_spline_async_native":
                control_period_ns = NANOSECONDS_PER_SECOND // CONTROL_HZ
                if budget < control_period_ns or budget % control_period_ns:
                    raise ValueError("native BSP rolling budget must be a positive whole control period")
            elif budget != calibrated_bsp_budget:
                raise ValueError("BSP prefetch budget does not match calibrated budget")
            if request.scheduler_context["request_control_step"] != request.observation_control_step:
                raise ValueError("BSP prefetch request step must match its observation step")
        previous_submission = request.submitted_offset_ns
        previous_observation_step = request.observation_control_step
    if execution_mode != "bsp_spline_async_native" and len(observed_bsp_budgets) > 1:
        raise ValueError("all BSP prefetch requests must record one calibrated budget")
    for index, request in enumerate(request_records):
        if request.disposition == "discarded_stale_phase":
            if (
                execution_mode not in ("bsp_spline_async", "bsp_spline_async_speedup1", "bsp_spline_async_native")
                or request.trigger != "bsp_prefetch"
                or index + 1 >= len(request_records)
                or request_records[index + 1].trigger != "bsp_stale_replan"
            ):
                raise ValueError("a stale BSP response must be followed by a blocking stale replan")
        if request.trigger == "bsp_stale_replan":
            if index == 0 or request_records[index - 1].disposition != "discarded_stale_phase":
                raise ValueError("a BSP stale replan must immediately follow a discarded response")
            previous = request_records[index - 1]
            context = request.scheduler_context
            expected_phase_multiplier = 500_000 if execution_mode == "bsp_spline_async_speedup1" else 1_000_000
            if (
                context["discarded_request_control_step"] != previous.observation_control_step
                or context["discarded_activation_control_step"] != request.observation_control_step
                or context["phase_offset_microindices"] != context["executed_prefix_steps"] * expected_phase_multiplier
            ):
                raise ValueError("BSP stale replan steps do not match the discarded request")

    requests_by_id = {request.request_id: request for request in request_records}
    latencies_by_id: Dict[int, LatencyEventV5] = {}
    previous_latency_id = -1
    previous_completion = -1
    for latency in latency_records:
        if latency.request_id not in requests_by_id or latency.request_id in latencies_by_id:
            raise ValueError("latency must belong to exactly one serialized request")
        if latency.request_id <= previous_latency_id or latency.completed_offset_ns < previous_completion:
            raise ValueError("latencies must be chronological and in request order")
        if latency.completed_offset_ns > duration:
            raise ValueError("latency completion is past episode_duration_ns")
        latencies_by_id[latency.request_id] = latency
        previous_latency_id = latency.request_id
        previous_completion = latency.completed_offset_ns

    previous_request_end = -1
    for index, request in enumerate(request_records):
        latency = latencies_by_id.get(request.request_id)
        if request.submitted_offset_ns < previous_request_end:
            raise ValueError("request intervals must be chronological and non-overlapping")
        if request.disposition == "abandoned":
            if latency is not None or index != len(request_records) - 1:
                raise ValueError("an abandoned request must be final and have no latency")
            continue
        if latency is None:
            raise ValueError("every non-abandoned request must have exactly one latency")
        if request.submitted_offset_ns + latency.duration_ns != latency.completed_offset_ns:
            raise ValueError("request and latency endpoints are inconsistent")
        previous_request_end = latency.completed_offset_ns
        if request.disposition == "activated" and latency.outcome != "success":
            raise ValueError("activated request must have a successful latency")
        if request.disposition == "discarded_stale_phase" and latency.outcome != "success":
            raise ValueError("discarded stale response must have a successful latency")
        if request.disposition == "failed" and latency.outcome != "policy_failure":
            raise ValueError("failed request must have a policy_failure latency")
        if request.disposition == "failed" and index != len(request_records) - 1:
            raise ValueError("a policy failure must terminate the request sequence")
        if latency.sampled_target_latency_ns != request.sampled_target_latency_ns:
            raise ValueError("request and latency sampled targets must match")

    activations_by_request: Dict[int, PlanActivationV5] = {}
    previous_activation_offset = -1
    previous_activation_step = -1
    previous_activation_request = -1
    for expected_plan_id, activation in enumerate(activation_records):
        if activation.plan_id != expected_plan_id:
            raise ValueError("plan ids must be contiguous from zero")
        if activation.request_id not in requests_by_id or activation.request_id in activations_by_request:
            raise ValueError("activation must belong to exactly one serialized request")
        if activation.request_id <= previous_activation_request:
            raise ValueError("activations must follow request order")
        if activation.activated_offset_ns < previous_activation_offset:
            raise ValueError("activations must be chronological")
        if activation.control_step < previous_activation_step or activation.control_step > step_count:
            raise ValueError("activation control steps must be ordered within 0..steps")
        if activation.activated_offset_ns > duration:
            raise ValueError("activation is past episode_duration_ns")
        request = requests_by_id[activation.request_id]
        if activation.control_step < request.observation_control_step:
            raise ValueError("activation control_step cannot precede its observation step")
        if request.dispatch == "blocking_replan" and activation.control_step != request.observation_control_step:
            raise ValueError("blocking replan must activate at its observation control step")
        latency = latencies_by_id.get(activation.request_id)
        if latency is None or latency.outcome != "success":
            raise ValueError("only a successful request may activate a plan")
        if activation.activated_offset_ns < latency.completed_offset_ns:
            raise ValueError("activation cannot precede request completion")
        expected_activation = (
            "initial"
            if activation.request_id == 0
            else "blocking_replace"
            if request.dispatch == "blocking_replan"
            else "immediate_swap"
        )
        if activation.activation != expected_activation:
            raise ValueError("activation kind does not match request dispatch")
        if execution_mode in _NATIVE_MODES:
            if set(activation.activation_context) != {"action_cursor"}:
                raise ValueError("native activation requires action_cursor context")
            cursor = activation.activation_context["action_cursor"]
            if activation.request_id == 0:
                if cursor != 0:
                    raise ValueError("initial native plans activate at cursor zero")
            else:
                expected_cursor = activation.control_step - request.observation_control_step
                if cursor != expected_cursor or not 0 <= cursor <= 15:
                    raise ValueError("RTC immediate activation action_cursor is inconsistent")
        else:
            context = activation.activation_context
            expected_context_fields = {
                "request_control_step",
                "activation_control_step",
                "executed_prefix_steps",
                "phase_offset_microindices",
                "first_sample_microindices",
                "remaining_curve_microindices",
                "remaining_curve_ns",
                "immediate_prefetch",
            }
            if execution_mode == "bsp_spline_async_native":
                expected_context_fields.add("prefetch_budget_ns")
            if set(context) != expected_context_fields:
                raise ValueError("BSP activation requires phase-skip audit context")
            if (
                context["request_control_step"] != request.observation_control_step
                or context["activation_control_step"] != activation.control_step
            ):
                raise ValueError("BSP activation steps do not match request and activation records")
            if execution_mode == "bsp_spline_sync" and (
                context["executed_prefix_steps"] != 0
                or context["phase_offset_microindices"] != 0
                or context["first_sample_microindices"] != 0
                or context["immediate_prefetch"] != 0
            ):
                raise ValueError("BSP sync activations must start at phase zero without prefetch")
            expected_rate_hz = 10 if execution_mode == "bsp_spline_async_speedup1" else 20
            if execution_mode in ("bsp_spline_async", "bsp_spline_async_speedup1", "bsp_spline_async_native") and (
                context["phase_offset_microindices"] * 20
                != context["executed_prefix_steps"] * expected_rate_hz * 1_000_000
                or context["remaining_curve_ns"]
                != (context["remaining_curve_microindices"] * NANOSECONDS_PER_SECOND + expected_rate_hz * 1_000_000 - 1)
                // (expected_rate_hz * 1_000_000)
                or context["immediate_prefetch"]
                != int(context["remaining_curve_ns"] <= context.get("prefetch_budget_ns", 400_000_000))
            ):
                raise ValueError("BSP async immediate-prefetch audit is inconsistent")
        activations_by_request[activation.request_id] = activation
        previous_activation_offset = activation.activated_offset_ns
        previous_activation_step = activation.control_step
        previous_activation_request = activation.request_id

    for request in request_records:
        activation = activations_by_request.get(request.request_id)
        if request.disposition == "activated" and activation is None:
            raise ValueError("successful activated request must have exactly one plan activation")
        if request.disposition != "activated" and activation is not None:
            raise ValueError("discarded, failed, or abandoned request cannot activate a plan")
    first_activation = activations_by_request.get(0)
    if first_activation is not None and (
        first_activation.plan_id != 0 or first_activation.control_step != 0 or first_activation.activation != "initial"
    ):
        raise ValueError("successful first request must install initial plan zero at step zero")
    for request_index in range(1, len(request_records)):
        previous_request = request_records[request_index - 1]
        previous_activation = activations_by_request.get(previous_request.request_id)
        if (
            previous_activation is not None
            and request_records[request_index].submitted_offset_ns < previous_activation.activated_offset_ns
        ):
            raise ValueError("a request cannot be submitted before the prior plan activates")

    underflows_by_request: Dict[int, ActionUnderflowV5] = {}
    previous_underflow_end = -1
    previous_underflow_step = -1
    for underflow in underflow_records:
        request = requests_by_id.get(underflow.request_id)
        if (
            request is None
            or underflow.request_id in underflows_by_request
            or execution_mode not in _ASYNC_MODES
            or request.dispatch != "background"
        ):
            raise ValueError("underflow must belong to one background request in an async mode")
        if underflow.control_step <= previous_underflow_step or underflow.control_step > step_count:
            raise ValueError("underflow control steps must be strictly ordered within 0..steps")
        if underflow.control_step < request.observation_control_step:
            raise ValueError("underflow control_step cannot precede its observation step")
        if underflow.started_offset_ns < request.submitted_offset_ns:
            raise ValueError("underflow cannot begin before request submission")
        underflow_end = underflow.started_offset_ns + underflow.duration_ns
        if underflow.started_offset_ns < previous_underflow_end or underflow_end > duration:
            raise ValueError("underflows must be chronological, non-overlapping, and within episode")
        latency = latencies_by_id.get(underflow.request_id)
        if latency is None:
            raise ValueError("underflow cannot belong to an abandoned request")
        activation = activations_by_request.get(underflow.request_id)
        if request.disposition == "discarded_stale_phase":
            stale_replan = requests_by_id.get(request.request_id + 1)
            if stale_replan is None or stale_replan.trigger != "bsp_stale_replan":
                raise ValueError("discarded underflow is missing its stale replan")
            stale_latency = latencies_by_id.get(stale_replan.request_id)
            stale_activation = activations_by_request.get(stale_replan.request_id)
            if stale_latency is None:
                raise ValueError("discarded underflow is missing the blocking replan outcome")
            expected_end = (
                stale_activation.activated_offset_ns
                if stale_latency.outcome == "success" and stale_activation is not None
                else stale_latency.completed_offset_ns
            )
        else:
            expected_end = (
                activation.activated_offset_ns
                if latency.outcome == "success" and activation is not None
                else latency.completed_offset_ns
            )
        if underflow_end != expected_end:
            raise ValueError("underflow must end at activation or policy-failure completion")
        if activation is not None and activation.control_step != underflow.control_step:
            raise ValueError("underflow and its activation must use the same control step")
        underflows_by_request[underflow.request_id] = underflow
        previous_underflow_end = underflow_end
        previous_underflow_step = underflow.control_step

    stalls_by_request: Dict[int, ControlStallV5] = {}
    previous_stall_end = -1
    previous_stall_step = -1
    for stall in stall_records:
        request = requests_by_id.get(stall.request_id)
        if request is None or stall.request_id in stalls_by_request:
            raise ValueError("control stall must belong to exactly one serialized request")
        stall_end = stall.started_offset_ns + stall.duration_ns
        if stall.control_step <= previous_stall_step or stall.control_step > step_count:
            raise ValueError("stall control steps must be strictly ordered within 0..steps")
        if stall.started_offset_ns < previous_stall_end or stall_end > duration:
            raise ValueError("stalls must be chronological, non-overlapping, and within episode")
        if stall.reason == STALL_REASON_ASYNC_ACTION_UNDERFLOW:
            underflow = underflows_by_request.get(stall.request_id)
            if underflow is None or (
                stall.control_step,
                stall.started_offset_ns,
                stall.duration_ns,
            ) != (
                underflow.control_step,
                underflow.started_offset_ns,
                underflow.duration_ns,
            ):
                raise ValueError("async stall must exactly equal its action underflow")
        else:
            latency = latencies_by_id.get(stall.request_id)
            if request.dispatch not in ("blocking_initial", "blocking_replan") or latency is None:
                raise ValueError("synchronous stall must belong to a completed blocking request")
            if stall.started_offset_ns < request.submitted_offset_ns or stall_end > latency.completed_offset_ns:
                raise ValueError("synchronous stall must lie within its request interval")
            if stall.control_step != request.observation_control_step:
                raise ValueError("blocking stall control_step must match its request step")
            full_interval_required = request.dispatch == "blocking_initial" or request.trigger == "bsp_stale_replan"
            if full_interval_required and (
                stall.started_offset_ns != request.submitted_offset_ns or stall.duration_ns != latency.duration_ns
            ):
                raise ValueError("initial and stale-replan stalls must equal the full request")
            if not full_interval_required and stall_end != latency.completed_offset_ns:
                raise ValueError("blocking-replan stall must end at request completion")
            if not full_interval_required and (stall.duration_ns == 0 or stall_end != latency.completed_offset_ns):
                raise ValueError("later baseline synchronous stall must be an exact positive suffix")
        stalls_by_request[stall.request_id] = stall
        previous_stall_end = stall_end
        previous_stall_step = stall.control_step

    for request in request_records:
        stall = stalls_by_request.get(request.request_id)
        if request.dispatch == "blocking_initial":
            if stall is None or stall.reason != STALL_REASON_SYNCHRONOUS_INFERENCE:
                raise ValueError("initial request must have a full synchronous stall")
        elif request.dispatch == "background":
            underflow = underflows_by_request.get(request.request_id)
            if (underflow is None) != (stall is None):
                raise ValueError("background stall exists exactly when its request underflows")
            if stall is not None and stall.reason != STALL_REASON_ASYNC_ACTION_UNDERFLOW:
                raise ValueError("background request stalls only for async action underflow")
        else:
            if request.trigger == "bsp_stale_replan":
                previous = requests_by_id.get(request.request_id - 1)
                previous_underflow = (
                    previous is not None
                    and previous.disposition == "discarded_stale_phase"
                    and previous.request_id in underflows_by_request
                )
                if not previous_underflow and stall is None:
                    raise ValueError("a stale replan without prior underflow requires a full synchronous stall")
            if stall is not None and stall.reason != STALL_REASON_SYNCHRONOUS_INFERENCE:
                raise ValueError("blocking replan stalls use synchronous_inference")

    return (
        request_records,
        latency_records,
        activation_records,
        underflow_records,
        stall_records,
    )


def validate_action_seams_v5(
    *,
    seams: Sequence[ActionSeamV5],
    activations: Sequence[PlanActivationV5],
) -> Tuple[ActionSeamV5, ...]:
    seam_records = _records_tuple(seams, ActionSeamV5, label="action seams")
    activation_records = _records_tuple(
        activations,
        PlanActivationV5,
        label="activations",
    )
    expected_activations = activation_records[1:]
    if len(seam_records) != len(expected_activations):
        raise ValueError("every non-initial plan activation requires one action seam")
    for seam, activation in zip(seam_records, expected_activations):
        if (seam.plan_id, seam.request_id, seam.control_step) != (
            activation.plan_id,
            activation.request_id,
            activation.control_step,
        ):
            raise ValueError("action seam identity must match its plan activation")
    return seam_records


def validate_video_frequencies_v5(*, control_hz: int = CONTROL_HZ, video_fps: int = DEFAULT_VIDEO_FPS) -> int:
    control = _require_nonnegative_integer(control_hz, name="control_hz")
    video = _require_nonnegative_integer(video_fps, name="video_fps")
    if control != CONTROL_HZ or video != DEFAULT_VIDEO_FPS:
        raise ValueError("schema-v5 LIBERO video timing requires exactly 20 Hz control and 40 fps")
    return video // control


def cumulative_wait_overlay_line_v5(cumulative_wait_ns: int) -> Tuple[str]:
    """Return the single persistent schema-v5 wait annotation."""
    wait_ns = _require_nonnegative_integer(cumulative_wait_ns, name="cumulative_wait_ns")
    centiseconds = (wait_ns + 5_000_000) // 10_000_000
    return ("Cumulative inference wait: {}.{:02d} s".format(centiseconds // 100, centiseconds % 100),)


def expand_control_frames_v5(
    control_frames: Iterable[_Frame],
    *,
    control_hz: int = CONTROL_HZ,
    video_fps: int = DEFAULT_VIDEO_FPS,
) -> Tuple[_Frame, ...]:
    """Hold every 20 Hz control frame exactly twice for 40 fps video."""
    hold_count = validate_video_frequencies_v5(control_hz=control_hz, video_fps=video_fps)
    return tuple(frame for frame in control_frames for _ in range(hold_count))


def quantize_stall_frames_v5(
    stalls: Sequence[ControlStallV5], *, video_fps: int = DEFAULT_VIDEO_FPS
) -> Tuple[int, ...]:
    """Cumulatively quantize measured stalls, carrying fractional frames."""
    validate_video_frequencies_v5(video_fps=video_fps)
    stall_records = _records_tuple(stalls, ControlStallV5, label="stalls")
    accumulated_ns = 0
    emitted_frames = 0
    event_frames: List[int] = []
    previous_step = -1
    previous_end = -1
    for stall in stall_records:
        if stall.control_step <= previous_step:
            raise ValueError("stalls must be in strictly increasing control-step order")
        if stall.started_offset_ns < previous_end:
            raise ValueError("stalls must be chronological and non-overlapping")
        accumulated_ns += stall.duration_ns
        total_frames = accumulated_ns * video_fps // NANOSECONDS_PER_SECOND
        event_frames.append(total_frames - emitted_frames)
        emitted_frames = total_frames
        previous_step = stall.control_step
        previous_end = stall.started_offset_ns + stall.duration_ns
    return tuple(event_frames)


def render_overlay_v5(
    frame: _Frame,
    lines: Sequence[str],
    *,
    renderer: Callable[[_Frame, Tuple[str, ...]], _Frame],
) -> _Frame:
    """Render timing text onto a copy, never a caller-owned control frame."""
    if isinstance(lines, (str, bytes)) or not isinstance(lines, Sequence):
        raise ValueError("overlay lines must be a sequence of strings")
    copied_lines = tuple(lines)
    if any(not isinstance(line, str) for line in copied_lines):
        raise ValueError("overlay lines must be strings")
    return renderer(copy.deepcopy(frame), copied_lines)


@dataclasses.dataclass(frozen=True)
class VideoTimingAuditV5:
    control_hz: int
    video_fps: int
    control_frame_count: int
    held_frame_count: int
    request_count: int
    latency_count: int
    activation_count: int
    underflow_count: int
    total_request_latency_ns: int
    total_underflow_ns: int
    measured_stall_count: int
    measured_control_stall_ns: int
    included_stall_count: int
    included_control_stall_ns: int
    included_stall_reasons: Tuple[str, ...]
    included_stall_frame_counts: Tuple[int, ...]
    stall_frame_count: int
    video_frame_count: int
    control_duration_ns: int
    video_duration_ns: int
    expected_duration_ns: int
    duration_deviation_ns: int

    def __post_init__(self) -> None:
        if isinstance(self.included_stall_reasons, (str, bytes)) or not isinstance(
            self.included_stall_reasons, Sequence
        ):
            raise ValueError("included_stall_reasons must be a sequence")
        if isinstance(self.included_stall_frame_counts, (str, bytes)) or not isinstance(
            self.included_stall_frame_counts, Sequence
        ):
            raise ValueError("included_stall_frame_counts must be a sequence")
        object.__setattr__(self, "included_stall_reasons", tuple(self.included_stall_reasons))
        object.__setattr__(self, "included_stall_frame_counts", tuple(self.included_stall_frame_counts))
        self._validate()

    def _validate(self) -> None:
        validate_video_frequencies_v5(control_hz=self.control_hz, video_fps=self.video_fps)
        nonnegative_fields = (
            "control_frame_count",
            "held_frame_count",
            "request_count",
            "latency_count",
            "activation_count",
            "underflow_count",
            "total_request_latency_ns",
            "total_underflow_ns",
            "measured_stall_count",
            "measured_control_stall_ns",
            "included_stall_count",
            "included_control_stall_ns",
            "stall_frame_count",
            "video_frame_count",
            "control_duration_ns",
            "video_duration_ns",
            "expected_duration_ns",
        )
        for field in nonnegative_fields:
            _require_nonnegative_integer(getattr(self, field), name=field)
        _require_integer(self.duration_deviation_ns, name="duration_deviation_ns")
        if any(reason not in _STALL_REASONS for reason in self.included_stall_reasons):
            raise ValueError("included_stall_reasons contains an invalid reason")
        for count in self.included_stall_frame_counts:
            _require_nonnegative_integer(count, name="included stall frame count")
        if self.held_frame_count != self.control_frame_count * 2:
            raise ValueError("held_frame_count must be exactly twice control_frame_count")
        if self.latency_count > self.request_count:
            raise ValueError("latency_count cannot exceed request_count")
        if self.activation_count > self.latency_count:
            raise ValueError("activation_count cannot exceed latency_count")
        if self.underflow_count > self.measured_stall_count:
            raise ValueError("underflow_count cannot exceed measured_stall_count")
        if self.total_underflow_ns > self.measured_control_stall_ns:
            raise ValueError("total_underflow_ns cannot exceed measured stall duration")
        if self.latency_count == 0 and self.total_request_latency_ns != 0:
            raise ValueError("zero latency_count requires zero total_request_latency_ns")
        if self.underflow_count == 0 and self.total_underflow_ns != 0:
            raise ValueError("zero underflow_count requires zero total_underflow_ns")
        if self.measured_stall_count == 0 and self.measured_control_stall_ns != 0:
            raise ValueError("zero measured_stall_count requires zero measured stall duration")
        if self.included_stall_count != len(self.included_stall_reasons) or self.included_stall_count != len(
            self.included_stall_frame_counts
        ):
            raise ValueError("included stall arrays must match included_stall_count")
        if self.included_stall_count not in (0, self.measured_stall_count):
            raise ValueError("video must include either every measured stall or none")
        if self.included_stall_count == 0:
            if self.included_control_stall_ns != 0:
                raise ValueError("disabled stall overlays must include zero stall duration")
        elif self.included_control_stall_ns != self.measured_control_stall_ns:
            raise ValueError("included stall duration must equal all measured stall duration")
        if self.included_stall_count and self.underflow_count != sum(
            reason == STALL_REASON_ASYNC_ACTION_UNDERFLOW for reason in self.included_stall_reasons
        ):
            raise ValueError("included async stall reasons must match underflow_count")
        if self.stall_frame_count != sum(self.included_stall_frame_counts):
            raise ValueError("stall_frame_count must equal included per-stall frames")
        cumulative_stall_frame_count = self.included_control_stall_ns * self.video_fps // NANOSECONDS_PER_SECOND
        if self.stall_frame_count != cumulative_stall_frame_count:
            raise ValueError("stall_frame_count must equal cumulative stall quantization")
        if self.video_frame_count != self.held_frame_count + self.stall_frame_count:
            raise ValueError("video_frame_count must equal held plus stall frames")
        expected_control_duration = self.control_frame_count * NANOSECONDS_PER_SECOND // self.control_hz
        if self.control_duration_ns != expected_control_duration:
            raise ValueError("control_duration_ns is inconsistent with control frames")
        expected_video_duration = self.video_frame_count * NANOSECONDS_PER_SECOND // self.video_fps
        if self.video_duration_ns != expected_video_duration:
            raise ValueError("video_duration_ns is inconsistent with video frames")
        if self.expected_duration_ns != self.control_duration_ns + self.included_control_stall_ns:
            raise ValueError("expected_duration_ns is inconsistent with included stalls")
        if self.duration_deviation_ns != self.video_duration_ns - self.expected_duration_ns:
            raise ValueError("duration_deviation_ns is inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "control_hz": self.control_hz,
            "video_fps": self.video_fps,
            "control_frame_count": self.control_frame_count,
            "held_frame_count": self.held_frame_count,
            "request_count": self.request_count,
            "latency_count": self.latency_count,
            "activation_count": self.activation_count,
            "underflow_count": self.underflow_count,
            "total_request_latency_ns": self.total_request_latency_ns,
            "total_underflow_ns": self.total_underflow_ns,
            "measured_stall_count": self.measured_stall_count,
            "measured_control_stall_ns": self.measured_control_stall_ns,
            "included_stall_count": self.included_stall_count,
            "included_control_stall_ns": self.included_control_stall_ns,
            "included_stall_reasons": list(self.included_stall_reasons),
            "included_stall_frame_counts": list(self.included_stall_frame_counts),
            "stall_frame_count": self.stall_frame_count,
            "video_frame_count": self.video_frame_count,
            "control_duration_ns": self.control_duration_ns,
            "video_duration_ns": self.video_duration_ns,
            "expected_duration_ns": self.expected_duration_ns,
            "duration_deviation_ns": self.duration_deviation_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "VideoTimingAuditV5":
        payload = _require_exact_fields(value, _AUDIT_FIELDS, label="video timing audit")
        if not isinstance(payload["included_stall_reasons"], list):
            raise ValueError("included_stall_reasons must be a JSON list")
        if not isinstance(payload["included_stall_frame_counts"], list):
            raise ValueError("included_stall_frame_counts must be a JSON list")
        return cls(
            control_hz=payload["control_hz"],
            video_fps=payload["video_fps"],
            control_frame_count=payload["control_frame_count"],
            held_frame_count=payload["held_frame_count"],
            request_count=payload["request_count"],
            latency_count=payload["latency_count"],
            activation_count=payload["activation_count"],
            underflow_count=payload["underflow_count"],
            total_request_latency_ns=payload["total_request_latency_ns"],
            total_underflow_ns=payload["total_underflow_ns"],
            measured_stall_count=payload["measured_stall_count"],
            measured_control_stall_ns=payload["measured_control_stall_ns"],
            included_stall_count=payload["included_stall_count"],
            included_control_stall_ns=payload["included_control_stall_ns"],
            included_stall_reasons=tuple(payload["included_stall_reasons"]),
            included_stall_frame_counts=tuple(payload["included_stall_frame_counts"]),
            stall_frame_count=payload["stall_frame_count"],
            video_frame_count=payload["video_frame_count"],
            control_duration_ns=payload["control_duration_ns"],
            video_duration_ns=payload["video_duration_ns"],
            expected_duration_ns=payload["expected_duration_ns"],
            duration_deviation_ns=payload["duration_deviation_ns"],
        )


def build_video_timing_audit_v5(
    *,
    control_frame_count: int,
    requests: Sequence[RequestEventV5],
    latencies: Sequence[LatencyEventV5],
    activations: Sequence[PlanActivationV5],
    underflows: Sequence[ActionUnderflowV5],
    stalls: Sequence[ControlStallV5],
    include_stalls: bool,
    control_hz: int = CONTROL_HZ,
    video_fps: int = DEFAULT_VIDEO_FPS,
) -> VideoTimingAuditV5:
    """Build exact accounting while keeping measured and included stalls distinct.

    Returns:
        A fully validated immutable timing audit.

    Raises:
        ValueError: If counts, event types, frequencies, or inclusion state are invalid.
    """
    frame_count = _require_nonnegative_integer(control_frame_count, name="control_frame_count")
    hold_count = validate_video_frequencies_v5(control_hz=control_hz, video_fps=video_fps)
    if not isinstance(include_stalls, bool):
        raise ValueError("include_stalls must be boolean")
    request_records = _records_tuple(requests, RequestEventV5, label="requests")
    latency_records = _records_tuple(latencies, LatencyEventV5, label="latencies")
    activation_records = _records_tuple(activations, PlanActivationV5, label="activations")
    underflow_records = _records_tuple(underflows, ActionUnderflowV5, label="underflows")
    stall_records = _records_tuple(stalls, ControlStallV5, label="stalls")

    included_stalls = stall_records if include_stalls else ()
    included_frame_counts = quantize_stall_frames_v5(included_stalls, video_fps=video_fps)
    held_frame_count = frame_count * hold_count
    stall_frame_count = sum(included_frame_counts)
    video_frame_count = held_frame_count + stall_frame_count
    control_duration_ns = frame_count * NANOSECONDS_PER_SECOND // control_hz
    measured_stall_ns = sum(stall.duration_ns for stall in stall_records)
    included_stall_ns = sum(stall.duration_ns for stall in included_stalls)
    video_duration_ns = video_frame_count * NANOSECONDS_PER_SECOND // video_fps
    expected_duration_ns = control_duration_ns + included_stall_ns
    return VideoTimingAuditV5(
        control_hz=control_hz,
        video_fps=video_fps,
        control_frame_count=frame_count,
        held_frame_count=held_frame_count,
        request_count=len(request_records),
        latency_count=len(latency_records),
        activation_count=len(activation_records),
        underflow_count=len(underflow_records),
        total_request_latency_ns=sum(latency.duration_ns for latency in latency_records),
        total_underflow_ns=sum(underflow.duration_ns for underflow in underflow_records),
        measured_stall_count=len(stall_records),
        measured_control_stall_ns=measured_stall_ns,
        included_stall_count=len(included_stalls),
        included_control_stall_ns=included_stall_ns,
        included_stall_reasons=tuple(stall.reason for stall in included_stalls),
        included_stall_frame_counts=included_frame_counts,
        stall_frame_count=stall_frame_count,
        video_frame_count=video_frame_count,
        control_duration_ns=control_duration_ns,
        video_duration_ns=video_duration_ns,
        expected_duration_ns=expected_duration_ns,
        duration_deviation_ns=video_duration_ns - expected_duration_ns,
    )


build_video_audit_v5 = build_video_timing_audit_v5
