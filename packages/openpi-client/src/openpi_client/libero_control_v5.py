"""Pure schema-v5 LIBERO control, calibration, and scheduling primitives."""

import dataclasses
import hashlib
import json
import math
import numbers
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from openpi_client import bsp_spline
from openpi_client import inference
from openpi_client import latency_sampling
from openpi_client import msgpack_numpy
from openpi_client import rtc


CONTROL_PERIOD_NS = 50_000_000
LATENCY_DISTRIBUTION_MEAN_NS = 300_000_000
LATENCY_DISTRIBUTION_STDDEV_NS = 60_000_000
THEORETICAL_P95_LATENCY_NS = 398_691_218
SCHEDULING_LATENCY_BUDGET_NS = 400_000_000
SCHEDULING_DELAY_TICKS = 8
CALIBRATION_WARMUP_COUNT = 5
CALIBRATION_MEASUREMENT_COUNT = 20
CALIBRATION_INFRASTRUCTURE_RETRIES = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CODE_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# An exact single-key envelope is reserved regardless of payload validity; maps
# with any additional key remain ordinary JSON mappings and cannot collide.
_CANONICAL_TYPE_TAGS = frozenset(("__float_hex__", "__bytes__", "__ndarray__"))


_EXECUTION_PARAMETERS_BY_NAME = {
    "baseline_async": {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "forecast_delay_ticks": 8,
        "num_inference_steps": 5,
        "continuity_guidance": False,
        "activation_policy": "immediate_skip_elapsed_prefix",
    },
    "baseline_async_recovery": {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial_with_blocking_capacity_recovery",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "forecast_delay_ticks": 8,
        "num_inference_steps": 5,
        "continuity_guidance": False,
        "activation_policy": "immediate_skip_elapsed_prefix",
        "capacity_recovery": "blocking_replan_from_latest_observation",
    },
    "baseline_rtc": {
        "action_representation": "native",
        "dispatch": "asynchronous_after_initial",
        "model_action_horizon": 16,
        "model_action_dim": 32,
        "minimum_launch_cursor": 8,
        "num_inference_steps": 5,
        "guidance_beta": 5,
        "forecast_delay_ticks": 8,
        "continuity_guidance": True,
        "activation_policy": "immediate",
    },
    "bsp_spline_async": {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
    },
    "bsp_spline_async_speedup1": {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 1,
        "effective_curve_rate_hz": 10,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
    },
    "bsp_spline_async_native": {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
        "latency_injection": "disabled_native",
        "prefetch_budget_policy": "rolling_raw_p95_20_ceil_control_period",
        "prefetch_window_size": 20,
        "prefetch_budget_rounding_ns": 50_000_000,
    },
    "bsp_spline_async_native_speedup4": {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 4,
        "effective_curve_rate_hz": 40,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
        "latency_injection": "disabled_native",
        "prefetch_budget_policy": "rolling_raw_p95_20_ceil_control_period",
        "prefetch_window_size": 20,
        "prefetch_budget_rounding_ns": 50_000_000,
    },
    "bsp_spline_async_native_speedup8": {
        "action_representation": "bsp",
        "dispatch": "asynchronous_after_initial",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 8,
        "effective_curve_rate_hz": 80,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "phase_skip_executed_prefix",
        "prefetch_comparison": "remaining_lte_budget",
        "latency_injection": "disabled_native",
        "prefetch_budget_policy": "rolling_raw_p95_20_ceil_control_period",
        "prefetch_window_size": 20,
        "prefetch_budget_rounding_ns": 50_000_000,
    },
    "baseline_sync": {
        "action_representation": "native",
        "dispatch": "synchronous",
        "model_action_horizon": 16,
        "execution_horizon": 16,
        "num_inference_steps": 5,
        "activation_policy": "blocking_replace_after_full_chunk",
    },
    "bsp_spline_sync": {
        "action_representation": "bsp",
        "dispatch": "synchronous",
        "parameter_shape": [16, 8],
        "origin_hz": 10,
        "degree": 3,
        "speedup": 2,
        "effective_curve_rate_hz": 20,
        "control_freq_hz": 20,
        "alignment": "disabled_delta_eff",
        "activation_policy": "blocking_replace_from_curve_start",
        "replan_policy": "after_closed_curve_endpoint",
    },
}

_MODE_IDENTITIES = {
    "baseline_async": ("baseline", "baseline_async_h16_v1", 16, True, "baseline_async"),
    "baseline_async_recovery": (
        "baseline",
        "baseline_async_h16_blocking_recovery_v2",
        16,
        True,
        "baseline_async",
    ),
    "baseline_rtc": ("baseline", "baseline_rtc_h16_v1", 16, True, "rtc"),
    "bsp_spline_async": (
        "bsp",
        "bsp_spline_async_phase_skip_speedup2_v2",
        8,
        True,
        "bsp",
    ),
    "bsp_spline_async_speedup1": (
        "bsp",
        "bsp_spline_async_phase_skip_speedup1_v1",
        8,
        True,
        "bsp",
    ),
    "bsp_spline_async_native": (
        "bsp",
        "bsp_spline_async_phase_skip_speedup2_v2",
        8,
        True,
        "bsp",
    ),
    "bsp_spline_async_native_speedup4": (
        "bsp",
        "bsp_spline_async_phase_skip_speedup4_v1",
        8,
        True,
        "bsp",
    ),
    "bsp_spline_async_native_speedup8": (
        "bsp",
        "bsp_spline_async_phase_skip_speedup8_v1",
        8,
        True,
        "bsp",
    ),
    "baseline_sync": ("baseline", "baseline_sync_n5_h16_full_v2", 16, False, None),
    "bsp_spline_sync": ("bsp", "bsp_spline_sync_speedup2_phase0_v2", 8, False, None),
}


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("mapping keys must be strings")
            copied[key] = _copy_json_value(item)
        return copied
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_json_value(item) for item in value]
    return value


class _FrozenDict(dict):
    """A msgpack-compatible immutable dictionary used in frozen adapter records."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen mapping cannot be mutated")

    __delitem__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("mapping keys must be strings")
            frozen[key] = _freeze_value(item)
        return _FrozenDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.ascontiguousarray(value).copy()
        copied.setflags(write=False)
        return copied
    return value


def _require_nonbool_int(value: Any, *, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("{} must be an integer".format(label))
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError("{} must be at least {}".format(label, minimum))
    return result


def _to_microindices(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("spline phase must be finite and non-negative")
    return int(round(numeric * 1_000_000))


def _remaining_ns_from_microindices(remaining_microindices: int, *, effective_curve_rate_hz: int = 20) -> int:
    remaining = _require_nonbool_int(
        remaining_microindices,
        label="remaining_microindices",
        minimum=0,
    )
    rate = _require_nonbool_int(
        effective_curve_rate_hz,
        label="effective_curve_rate_hz",
        minimum=1,
    )
    denominator = rate * 1_000_000
    return (remaining * 1_000_000_000 + denominator - 1) // denominator


def _bsp_phase_rate_hz(executed_prefix_steps: int, phase_offset_microindices: int) -> int:
    prefix = _require_nonbool_int(executed_prefix_steps, label="executed_prefix_steps", minimum=0)
    phase = _require_nonbool_int(phase_offset_microindices, label="phase_offset_microindices", minimum=0)
    matches = [rate for rate in (10, 20, 40, 80) if phase * 20 == prefix * rate * 1_000_000]
    if not matches:
        raise ValueError("BSP phase offset does not match a supported execution protocol")
    return matches[0]


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA256".format(label))
    return value


def _require_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty string".format(label))
    return value


@dataclasses.dataclass(frozen=True)
class ExecutionModeSpec:
    name: str
    policy_variant: str
    policy_protocol: str
    expected_action_horizon: int
    asynchronous: bool
    calibration_kind: Optional[str]

    def __post_init__(self) -> None:
        expected = _MODE_IDENTITIES.get(self.name)
        actual = (
            self.policy_variant,
            self.policy_protocol,
            self.expected_action_horizon,
            self.asynchronous,
            self.calibration_kind,
        )
        if expected is None or any(
            type(value) is not type(expected_value) or value != expected_value
            for value, expected_value in zip(actual, expected)
        ):
            raise ValueError("Execution mode does not match the frozen schema-v5 mode table")

    def to_parameters_dict(self) -> Dict[str, Any]:
        return _copy_json_value(_EXECUTION_PARAMETERS_BY_NAME[self.name])


EXECUTION_MODES = MappingProxyType(
    {
        name: ExecutionModeSpec(
            name=name,
            policy_variant=identity[0],
            policy_protocol=identity[1],
            expected_action_horizon=identity[2],
            asynchronous=identity[3],
            calibration_kind=identity[4],
        )
        for name, identity in _MODE_IDENTITIES.items()
    }
)


def is_async_bsp_mode_v5(execution_mode: str) -> bool:
    mode = EXECUTION_MODES.get(execution_mode)
    return mode is not None and mode.asynchronous and mode.policy_variant == "bsp"


def is_native_latency_mode_v5(execution_mode: str) -> bool:
    if not is_async_bsp_mode_v5(execution_mode):
        return False
    return _EXECUTION_PARAMETERS_BY_NAME[execution_mode].get("latency_injection") == "disabled_native"


def bsp_speedup_for_mode_v5(execution_mode: str) -> int:
    if not is_async_bsp_mode_v5(execution_mode):
        raise ValueError("execution_mode must be an asynchronous BSP mode")
    return int(_EXECUTION_PARAMETERS_BY_NAME[execution_mode]["speedup"])


@dataclasses.dataclass(frozen=True)
class RequestIntentV5:
    dispatch: str
    trigger: str
    scheduler_context: Mapping[str, int]
    request_overlay: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.dispatch not in ("blocking_initial", "blocking_replan", "background"):
            raise ValueError("Unsupported request dispatch")
        if self.trigger not in (
            "initial_plan",
            "baseline_chunk_exhausted",
            "baseline_async_launch",
            "baseline_async_capacity_replan",
            "rtc_launch",
            "bsp_curve_exhausted",
            "bsp_prefetch",
            "bsp_stale_replan",
        ):
            raise ValueError("Unsupported request trigger")
        if not isinstance(self.scheduler_context, Mapping):
            raise ValueError("scheduler_context must be a mapping")
        if any(not isinstance(key, str) for key in self.scheduler_context):
            raise ValueError("scheduler_context keys must be strings")
        scheduler_context = _FrozenDict(
            {
                key: _require_nonbool_int(
                    value,
                    label="scheduler_context.{}".format(key),
                    minimum=0,
                )
                for key, value in self.scheduler_context.items()
            }
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
            raise ValueError("request dispatch does not match its trigger")
        if self.trigger in ("initial_plan", "baseline_chunk_exhausted", "bsp_curve_exhausted"):
            if scheduler_context:
                raise ValueError("request trigger requires an empty scheduler context")
        elif self.trigger == "baseline_async_capacity_replan":
            if set(scheduler_context) != {"action_cursor", "forecast_delay_ticks"}:
                raise ValueError("baseline async capacity recovery requires cursor and delay")
            cursor = scheduler_context["action_cursor"]
            delay = scheduler_context["forecast_delay_ticks"]
            if not 9 <= cursor <= 15 or delay != 8 or cursor + delay <= 16:
                raise ValueError("baseline async capacity recovery requires an infeasible cursor")
        elif self.trigger in ("baseline_async_launch", "rtc_launch"):
            if set(scheduler_context) != {"s", "d"}:
                raise ValueError("RTC launch context must contain exactly s and d")
            start = scheduler_context["s"]
            delay = scheduler_context["d"]
            if not 8 <= start <= 16 or not 0 <= delay <= start or start + delay > 16:
                raise ValueError("RTC launch context violates the action horizon")
        elif self.trigger == "bsp_prefetch":
            if set(scheduler_context) != {
                "remaining_plan_ns",
                "budget_ns",
                "request_control_step",
            }:
                raise ValueError("BSP prefetch context must contain exactly remaining time, budget, and request step")
            if scheduler_context["remaining_plan_ns"] > scheduler_context["budget_ns"]:
                raise ValueError("BSP prefetch remaining time must not exceed its budget")
        else:
            if set(scheduler_context) != {
                "discarded_request_control_step",
                "discarded_activation_control_step",
                "executed_prefix_steps",
                "phase_offset_microindices",
                "curve_t_max_microindices",
            }:
                raise ValueError("BSP stale replan context must exactly identify the discarded response")
            request_step = scheduler_context["discarded_request_control_step"]
            activation_step = scheduler_context["discarded_activation_control_step"]
            if (
                activation_step < request_step
                or scheduler_context["executed_prefix_steps"] != activation_step - request_step
                or scheduler_context["phase_offset_microindices"] <= scheduler_context["curve_t_max_microindices"]
            ):
                raise ValueError("BSP stale replan phase identity is inconsistent")
            _bsp_phase_rate_hz(
                scheduler_context["executed_prefix_steps"],
                scheduler_context["phase_offset_microindices"],
            )
        object.__setattr__(self, "scheduler_context", scheduler_context)
        if not isinstance(self.request_overlay, Mapping):
            raise ValueError("request_overlay must be a mapping")
        object.__setattr__(self, "request_overlay", _freeze_value(self.request_overlay))


@dataclasses.dataclass(frozen=True)
class ActivationDecisionV5:
    activation: str
    activation_context: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.activation not in (
            "initial",
            "blocking_replace",
            "immediate_swap",
            "discarded_stale_phase",
        ):
            raise ValueError("Unsupported plan activation")
        if not isinstance(self.activation_context, Mapping):
            raise ValueError("activation_context must be a mapping")
        if any(not isinstance(key, str) for key in self.activation_context):
            raise ValueError("activation_context keys must be strings")
        activation_context = _FrozenDict(
            {
                key: _require_nonbool_int(
                    value,
                    label="activation_context.{}".format(key),
                    minimum=0,
                )
                for key, value in self.activation_context.items()
            }
        )
        if set(activation_context) == {"action_cursor"}:
            if self.activation == "discarded_stale_phase" or activation_context["action_cursor"] > 15:
                raise ValueError("native activation action_cursor must be in 0..15")
        elif set(activation_context) in (
            {
                "request_control_step",
                "activation_control_step",
                "executed_prefix_steps",
                "phase_offset_microindices",
                "first_sample_microindices",
                "remaining_curve_microindices",
                "remaining_curve_ns",
                "immediate_prefetch",
            },
            {
                "request_control_step",
                "activation_control_step",
                "executed_prefix_steps",
                "phase_offset_microindices",
                "first_sample_microindices",
                "remaining_curve_microindices",
                "remaining_curve_ns",
                "immediate_prefetch",
                "prefetch_budget_ns",
            },
        ):
            matching_rates = tuple(
                rate
                for rate in (10, 20, 40, 80)
                if activation_context["phase_offset_microindices"] * 20
                == activation_context["executed_prefix_steps"] * rate * 1_000_000
                and activation_context["remaining_curve_ns"]
                == _remaining_ns_from_microindices(
                    activation_context["remaining_curve_microindices"],
                    effective_curve_rate_hz=rate,
                )
            )
            if (
                self.activation == "discarded_stale_phase"
                or activation_context["activation_control_step"] < activation_context["request_control_step"]
                or activation_context["executed_prefix_steps"]
                != activation_context["activation_control_step"] - activation_context["request_control_step"]
                or not matching_rates
                or activation_context["first_sample_microindices"] < activation_context["phase_offset_microindices"]
                or activation_context["immediate_prefetch"] not in (0, 1)
                or (
                    activation_context["immediate_prefetch"] == 1
                    and activation_context["remaining_curve_ns"]
                    > activation_context.get("prefetch_budget_ns", SCHEDULING_LATENCY_BUDGET_NS)
                )
            ):
                raise ValueError("BSP phase-skip activation context is inconsistent")
        elif set(activation_context) == {
            "request_control_step",
            "activation_control_step",
            "executed_prefix_steps",
            "phase_offset_microindices",
            "first_sample_microindices",
            "remaining_curve_microindices",
            "remaining_curve_ns",
            "curve_t_max_microindices",
        }:
            try:
                _bsp_phase_rate_hz(
                    activation_context["executed_prefix_steps"],
                    activation_context["phase_offset_microindices"],
                )
            except ValueError as error:
                raise ValueError("only a stale response may use the discarded BSP context") from error
            if (
                self.activation != "discarded_stale_phase"
                or activation_context["activation_control_step"] < activation_context["request_control_step"]
                or activation_context["executed_prefix_steps"]
                != activation_context["activation_control_step"] - activation_context["request_control_step"]
                or activation_context["first_sample_microindices"] != activation_context["phase_offset_microindices"]
                or activation_context["phase_offset_microindices"] <= activation_context["curve_t_max_microindices"]
                or activation_context["remaining_curve_microindices"] != 0
                or activation_context["remaining_curve_ns"] != 0
            ):
                raise ValueError("only a stale response may use the discarded BSP context")
        else:
            raise ValueError("activation context must be exactly native or BSP")
        object.__setattr__(self, "activation_context", activation_context)


@dataclasses.dataclass(frozen=True)
class ActionDecisionV5:
    action: Optional[np.ndarray]
    underflow: bool

    def __post_init__(self) -> None:
        if not isinstance(self.underflow, bool):
            raise ValueError("underflow must be a boolean")
        if self.underflow:
            if self.action is not None:
                raise ValueError("an underflow cannot carry an action")
            return
        if not isinstance(self.action, np.ndarray):
            raise ValueError("an actionable decision must carry a numpy array")
        if self.action.shape != (7,) or not np.isfinite(self.action).all():
            raise ValueError("an action must be a finite seven-dimensional vector")
        action = np.asarray(self.action, dtype=np.float32).copy()
        action.setflags(write=False)
        object.__setattr__(self, "action", action)


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    def wait_until_ns(self, deadline_ns: int) -> None: ...


class NoCatchupPacer:
    """A deadline pacer that reanchors after every real action start."""

    def __init__(self, clock: Clock, *, period_ns: int = CONTROL_PERIOD_NS) -> None:
        self._clock = clock
        self._period_ns = _require_nonbool_int(period_ns, label="period_ns", minimum=1)
        self._next_deadline_ns = None  # type: Optional[int]
        self._current_due_ns = None  # type: Optional[int]
        self._last_observed_ns = None  # type: Optional[int]

    @property
    def next_deadline_ns(self) -> Optional[int]:
        return self._next_deadline_ns

    def wait_until_due(self) -> int:
        while True:
            now_ns = self._read_clock()
            if self._next_deadline_ns is None or now_ns >= self._next_deadline_ns:
                self._current_due_ns = now_ns if self._next_deadline_ns is None else self._next_deadline_ns
                return now_ns
            self._clock.wait_until_ns(self._next_deadline_ns)

    def mark_action_started(self, started_ns: int) -> int:
        started = _require_nonbool_int(started_ns, label="started_ns", minimum=0)
        if self._next_deadline_ns is None:
            due_ns = started if self._current_due_ns is None else self._current_due_ns
        else:
            due_ns = self._next_deadline_ns
        if started < due_ns:
            raise ValueError("action start must be at or after the due time")
        if self._last_observed_ns is not None and started < self._last_observed_ns:
            raise ValueError("action start must not move the monotonic clock backwards")
        self._last_observed_ns = started
        self._current_due_ns = None
        self._next_deadline_ns = started + self._period_ns
        return self._next_deadline_ns

    def _read_clock(self) -> int:
        now_ns = _require_nonbool_int(self._clock.monotonic_ns(), label="clock time", minimum=0)
        if self._last_observed_ns is not None and now_ns < self._last_observed_ns:
            raise ValueError("clock must be nondecreasing")
        self._last_observed_ns = now_ns
        return now_ns


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("canonical floats must be finite")
        return {"__float_hex__": numeric.hex()}
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("canonical arrays cannot have object dtype")
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
            raise ValueError("canonical arrays must contain only finite values")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical mapping keys must be strings")
        if len(value) == 1 and next(iter(value)) in _CANONICAL_TYPE_TAGS:
            raise ValueError("mapping uses a reserved canonical type tag envelope")
        canonical = {}
        for key in sorted(value):
            canonical[key] = _canonical_value(value[key])
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError("unsupported canonical fingerprint value: {}".format(type(value).__name__))


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value with the strict schema-v5 canonical fingerprint format."""
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclasses.dataclass(frozen=True)
class CalibrationObservationIdentityV1:
    suite: str
    task_id: int
    init_state_index: int
    init_state_fingerprint: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.suite, label="suite")
        object.__setattr__(
            self,
            "task_id",
            _require_nonbool_int(self.task_id, label="task_id", minimum=0),
        )
        object.__setattr__(
            self,
            "init_state_index",
            _require_nonbool_int(self.init_state_index, label="init_state_index", minimum=0),
        )
        _require_sha256(self.init_state_fingerprint, label="init_state_fingerprint")
        _require_sha256(self.request_fingerprint, label="request_fingerprint")

    def to_dict(self) -> Dict[str, Any]:
        self.__post_init__()
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "init_state_index": self.init_state_index,
            "init_state_fingerprint": self.init_state_fingerprint,
            "request_fingerprint": self.request_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationObservationIdentityV1":
        fields = {
            "suite",
            "task_id",
            "init_state_index",
            "init_state_fingerprint",
            "request_fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("canonical observation identity fields must match schema version 1")
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class CheckpointIdentityV1:
    code_sha: str
    config_name: str
    checkpoint_step: int
    checkpoint: str
    container_digest: str
    norm_hash: str
    bsp_cache_hash: Optional[str]
    bsp_cache_manifest_fingerprint: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.code_sha, str) or _CODE_SHA_PATTERN.fullmatch(self.code_sha) is None:
            raise ValueError("code_sha must be a lowercase 40- or 64-character Git SHA")
        _require_nonempty_text(self.config_name, label="config_name")
        checkpoint_step = _require_nonbool_int(self.checkpoint_step, label="checkpoint_step", minimum=0)
        object.__setattr__(self, "checkpoint_step", checkpoint_step)
        checkpoint = _require_nonempty_text(self.checkpoint, label="checkpoint").rstrip("/")
        if not checkpoint or checkpoint.rsplit("/", 1)[-1] != str(checkpoint_step):
            raise ValueError("checkpoint terminal component must equal checkpoint_step")
        object.__setattr__(self, "checkpoint", checkpoint)
        if (
            not isinstance(self.container_digest, str)
            or _CONTAINER_DIGEST_PATTERN.fullmatch(self.container_digest) is None
        ):
            raise ValueError("container_digest must be sha256: followed by 64 lowercase hex")
        _require_sha256(self.norm_hash, label="norm_hash")
        cache_presence = (
            self.bsp_cache_hash is not None,
            self.bsp_cache_manifest_fingerprint is not None,
        )
        if cache_presence not in ((False, False), (True, True)):
            raise ValueError("BSP cache identities must both be null or both be present")
        if self.bsp_cache_hash is not None:
            _require_sha256(self.bsp_cache_hash, label="bsp_cache_hash")
            _require_sha256(
                self.bsp_cache_manifest_fingerprint,
                label="bsp_cache_manifest_fingerprint",
            )

    def to_dict(self) -> Dict[str, Any]:
        self.__post_init__()
        return {
            "code_sha": self.code_sha,
            "config_name": self.config_name,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint": self.checkpoint,
            "container_digest": self.container_digest,
            "norm_hash": self.norm_hash,
            "bsp_cache_hash": self.bsp_cache_hash,
            "bsp_cache_manifest_fingerprint": self.bsp_cache_manifest_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointIdentityV1":
        fields = {
            "code_sha",
            "config_name",
            "checkpoint_step",
            "checkpoint",
            "container_digest",
            "norm_hash",
            "bsp_cache_hash",
            "bsp_cache_manifest_fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("checkpoint identity fields must match schema version 1")
        return cls(**dict(value))


_CALIBRATION_FIELDS = {
    "schema_version",
    "execution_mode",
    "clock",
    "checkpoint_identity_fingerprint",
    "server_metadata_fingerprint",
    "canonical_observation_identity",
    "seed_namespace",
    "bootstrap_request_fingerprint",
    "warmup_request_fingerprints",
    "measurement_request_fingerprints",
    "warmup_latency_ns",
    "measurement_latency_ns",
    "p95_method",
    "p95_rank",
    "p95_latency_ns",
    "control_period_ns",
    "derived_delay_ticks",
    "derived_prefetch_budget_ns",
    "fingerprint",
}


def nearest_rank_p95_ns(measurements_ns: Sequence[int]) -> Tuple[int, int]:
    if isinstance(measurements_ns, (str, bytes)) or not isinstance(measurements_ns, Sequence):
        raise ValueError("p95 measurements must be a sequence")
    if len(measurements_ns) != CALIBRATION_MEASUREMENT_COUNT:
        raise ValueError("p95 requires exactly 20 measurements")
    measurements = [_require_nonbool_int(value, label="latency measurement", minimum=0) for value in measurements_ns]
    rank = 19
    return rank, sorted(measurements)[rank - 1]


def native_prefetch_budget_ns(measurements_ns: Sequence[int]) -> int:
    """Round the latest twenty native raw latencies up to one control period."""
    _, p95_ns = nearest_rank_p95_ns(measurements_ns)
    ticks = max(1, (p95_ns + CONTROL_PERIOD_NS - 1) // CONTROL_PERIOD_NS)
    return ticks * CONTROL_PERIOD_NS


@dataclasses.dataclass(frozen=True)
class LatencyCalibrationV1:
    schema_version: int
    execution_mode: str
    clock: str
    checkpoint_identity_fingerprint: str
    server_metadata_fingerprint: str
    canonical_observation_identity: CalibrationObservationIdentityV1
    seed_namespace: str
    bootstrap_request_fingerprint: Optional[str]
    warmup_request_fingerprints: Tuple[str, ...]
    measurement_request_fingerprints: Tuple[str, ...]
    warmup_latency_ns: Tuple[int, ...]
    measurement_latency_ns: Tuple[int, ...]
    p95_method: str
    p95_rank: int
    p95_latency_ns: int
    control_period_ns: int
    derived_delay_ticks: Optional[int]
    derived_prefetch_budget_ns: Optional[int]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_request_fingerprints", tuple(self.warmup_request_fingerprints))
        object.__setattr__(
            self,
            "measurement_request_fingerprints",
            tuple(self.measurement_request_fingerprints),
        )
        object.__setattr__(
            self,
            "warmup_latency_ns",
            tuple(_require_nonbool_int(value, label="warmup latency", minimum=0) for value in self.warmup_latency_ns),
        )
        object.__setattr__(
            self,
            "measurement_latency_ns",
            tuple(
                _require_nonbool_int(value, label="measurement latency", minimum=0)
                for value in self.measurement_latency_ns
            ),
        )
        self._validate()

    @classmethod
    def create(
        cls,
        *,
        execution_mode: str,
        checkpoint_identity_fingerprint: str,
        server_metadata_fingerprint: str,
        canonical_observation_identity: CalibrationObservationIdentityV1,
        seed_namespace: str,
        bootstrap_request_fingerprint: Optional[str],
        warmup_request_fingerprints: Sequence[str],
        measurement_request_fingerprints: Sequence[str],
        warmup_latency_ns: Sequence[int],
        measurement_latency_ns: Sequence[int],
    ) -> "LatencyCalibrationV1":
        if not isinstance(canonical_observation_identity, CalibrationObservationIdentityV1):
            raise ValueError("canonical_observation_identity must use schema version 1")
        rank, p95_ns = nearest_rank_p95_ns(measurement_latency_ns)
        delay_ticks = None  # type: Optional[int]
        budget_ns = None  # type: Optional[int]
        rounded_ticks = (p95_ns + CONTROL_PERIOD_NS - 1) // CONTROL_PERIOD_NS
        if execution_mode == "baseline_rtc":
            if rounded_ticks > 8:
                raise ValueError("RTC calibrated delay exceeds eight control ticks")
            delay_ticks = rounded_ticks
        elif execution_mode in ("bsp_spline_async", "bsp_spline_async_speedup1"):
            budget_ns = rounded_ticks * CONTROL_PERIOD_NS
        else:
            raise ValueError("latency calibration is defined only for asynchronous modes")
        values = {
            "schema_version": 1,
            "execution_mode": execution_mode,
            "clock": "monotonic_ns",
            "checkpoint_identity_fingerprint": checkpoint_identity_fingerprint,
            "server_metadata_fingerprint": server_metadata_fingerprint,
            "canonical_observation_identity": canonical_observation_identity,
            "seed_namespace": seed_namespace,
            "bootstrap_request_fingerprint": bootstrap_request_fingerprint,
            "warmup_request_fingerprints": tuple(warmup_request_fingerprints),
            "measurement_request_fingerprints": tuple(measurement_request_fingerprints),
            "warmup_latency_ns": tuple(warmup_latency_ns),
            "measurement_latency_ns": tuple(measurement_latency_ns),
            "p95_method": "nearest_rank",
            "p95_rank": rank,
            "p95_latency_ns": p95_ns,
            "control_period_ns": CONTROL_PERIOD_NS,
            "derived_delay_ticks": delay_ticks,
            "derived_prefetch_budget_ns": budget_ns,
        }
        fingerprint = canonical_fingerprint(_calibration_payload(values))
        return cls(fingerprint=fingerprint, **values)

    def _validate(self) -> None:
        if _require_nonbool_int(self.schema_version, label="calibration schema_version") != 1:
            raise ValueError("calibration schema_version must be integer 1")
        if self.execution_mode not in ("baseline_rtc", "bsp_spline_async", "bsp_spline_async_speedup1"):
            raise ValueError("calibration execution_mode must be asynchronous")
        if self.clock != "monotonic_ns":
            raise ValueError("calibration clock must be monotonic_ns")
        _require_sha256(
            self.checkpoint_identity_fingerprint,
            label="checkpoint_identity_fingerprint",
        )
        _require_sha256(self.server_metadata_fingerprint, label="server_metadata_fingerprint")
        if not isinstance(self.canonical_observation_identity, CalibrationObservationIdentityV1):
            raise ValueError("canonical_observation_identity must use schema version 1")
        self.canonical_observation_identity.__post_init__()
        expected_namespace = "openpi-libero-calibration-v1/{}/{}".format(
            self.execution_mode,
            self.checkpoint_identity_fingerprint,
        )
        if self.seed_namespace != expected_namespace:
            raise ValueError("calibration seed_namespace does not match its identities")
        if len(self.warmup_request_fingerprints) != CALIBRATION_WARMUP_COUNT:
            raise ValueError("calibration requires exactly five warmup request fingerprints")
        if len(self.measurement_request_fingerprints) != CALIBRATION_MEASUREMENT_COUNT:
            raise ValueError("calibration requires exactly twenty measurement request fingerprints")
        for value in self.warmup_request_fingerprints + self.measurement_request_fingerprints:
            _require_sha256(value, label="request fingerprint")
        warmups = tuple(
            _require_nonbool_int(value, label="warmup latency", minimum=0) for value in self.warmup_latency_ns
        )
        if len(warmups) != CALIBRATION_WARMUP_COUNT:
            raise ValueError("calibration requires exactly five warmup latencies")
        measurements = tuple(
            _require_nonbool_int(value, label="measurement latency", minimum=0) for value in self.measurement_latency_ns
        )
        if len(measurements) != CALIBRATION_MEASUREMENT_COUNT:
            raise ValueError("calibration requires exactly twenty measurement latencies")
        rank, p95_ns = nearest_rank_p95_ns(measurements)
        if self.p95_method != "nearest_rank" or _require_nonbool_int(self.p95_rank, label="p95_rank") != rank:
            raise ValueError("calibration must use nearest-rank p95 at rank 19")
        if _require_nonbool_int(self.p95_latency_ns, label="p95_latency_ns", minimum=0) != p95_ns:
            raise ValueError("calibration p95_latency_ns does not match raw measurements")
        if _require_nonbool_int(self.control_period_ns, label="control_period_ns", minimum=1) != CONTROL_PERIOD_NS:
            raise ValueError("calibration control_period_ns must be 50000000")
        rounded_ticks = (p95_ns + CONTROL_PERIOD_NS - 1) // CONTROL_PERIOD_NS
        if self.execution_mode == "baseline_rtc":
            if self.bootstrap_request_fingerprint is None:
                raise ValueError("RTC calibration requires a bootstrap request fingerprint")
            _require_sha256(
                self.bootstrap_request_fingerprint,
                label="bootstrap_request_fingerprint",
            )
            if rounded_ticks > 8:
                raise ValueError("RTC calibrated delay exceeds eight control ticks")
            if (
                self.derived_delay_ticks is None
                or _require_nonbool_int(
                    self.derived_delay_ticks,
                    label="derived_delay_ticks",
                    minimum=0,
                )
                != rounded_ticks
                or self.derived_prefetch_budget_ns is not None
            ):
                raise ValueError("RTC calibration derived values do not match p95")
        else:
            expected_budget = rounded_ticks * CONTROL_PERIOD_NS
            if self.bootstrap_request_fingerprint is not None:
                raise ValueError("BSP calibration bootstrap fingerprint must be null")
            if self.derived_delay_ticks is not None:
                raise ValueError("BSP calibration delay ticks must be null")
            if (
                self.derived_prefetch_budget_ns is None
                or _require_nonbool_int(
                    self.derived_prefetch_budget_ns,
                    label="derived_prefetch_budget_ns",
                    minimum=0,
                )
                != expected_budget
            ):
                raise ValueError("BSP calibration budget does not match p95")
        _require_sha256(self.fingerprint, label="calibration fingerprint")
        expected_fingerprint = canonical_fingerprint(self._payload_without_fingerprint())
        if self.fingerprint != expected_fingerprint:
            raise ValueError("calibration fingerprint does not bind its serialized fields")

    def _payload_without_fingerprint(self) -> Dict[str, Any]:
        return _calibration_payload(
            {
                "schema_version": self.schema_version,
                "execution_mode": self.execution_mode,
                "clock": self.clock,
                "checkpoint_identity_fingerprint": self.checkpoint_identity_fingerprint,
                "server_metadata_fingerprint": self.server_metadata_fingerprint,
                "canonical_observation_identity": self.canonical_observation_identity,
                "seed_namespace": self.seed_namespace,
                "bootstrap_request_fingerprint": self.bootstrap_request_fingerprint,
                "warmup_request_fingerprints": self.warmup_request_fingerprints,
                "measurement_request_fingerprints": self.measurement_request_fingerprints,
                "warmup_latency_ns": self.warmup_latency_ns,
                "measurement_latency_ns": self.measurement_latency_ns,
                "p95_method": self.p95_method,
                "p95_rank": self.p95_rank,
                "p95_latency_ns": self.p95_latency_ns,
                "control_period_ns": self.control_period_ns,
                "derived_delay_ticks": self.derived_delay_ticks,
                "derived_prefetch_budget_ns": self.derived_prefetch_budget_ns,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        payload = self._payload_without_fingerprint()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LatencyCalibrationV1":
        if not isinstance(value, Mapping) or set(value) != _CALIBRATION_FIELDS:
            raise ValueError("latency calibration fields must exactly match schema version 1")
        values = dict(value)
        values["canonical_observation_identity"] = CalibrationObservationIdentityV1.from_dict(
            values["canonical_observation_identity"]
        )
        for field in (
            "warmup_request_fingerprints",
            "measurement_request_fingerprints",
            "warmup_latency_ns",
            "measurement_latency_ns",
        ):
            if not isinstance(values[field], list):
                raise ValueError("{} must be a JSON list".format(field))
            values[field] = tuple(values[field])
        return cls(**values)


def _calibration_payload(values: Mapping[str, Any]) -> Dict[str, Any]:
    identity = values["canonical_observation_identity"]
    if isinstance(identity, CalibrationObservationIdentityV1):
        identity = identity.to_dict()
    return {
        "schema_version": values["schema_version"],
        "execution_mode": values["execution_mode"],
        "clock": values["clock"],
        "checkpoint_identity_fingerprint": values["checkpoint_identity_fingerprint"],
        "server_metadata_fingerprint": values["server_metadata_fingerprint"],
        "canonical_observation_identity": dict(identity),
        "seed_namespace": values["seed_namespace"],
        "bootstrap_request_fingerprint": values["bootstrap_request_fingerprint"],
        "warmup_request_fingerprints": list(values["warmup_request_fingerprints"]),
        "measurement_request_fingerprints": list(values["measurement_request_fingerprints"]),
        "warmup_latency_ns": list(values["warmup_latency_ns"]),
        "measurement_latency_ns": list(values["measurement_latency_ns"]),
        "p95_method": values["p95_method"],
        "p95_rank": values["p95_rank"],
        "p95_latency_ns": values["p95_latency_ns"],
        "control_period_ns": values["control_period_ns"],
        "derived_delay_ticks": values["derived_delay_ticks"],
        "derived_prefetch_budget_ns": values["derived_prefetch_budget_ns"],
    }


_CALIBRATION_V2_SERIES_FIELDS = (
    "warmup_raw_inference_latency_ns",
    "warmup_sampled_target_latency_ns",
    "warmup_requested_synthetic_delay_ns",
    "warmup_observed_synthetic_delay_ns",
    "warmup_observed_effective_latency_ns",
    "warmup_latency_overshoot_ns",
    "measurement_raw_inference_latency_ns",
    "measurement_sampled_target_latency_ns",
    "measurement_requested_synthetic_delay_ns",
    "measurement_observed_synthetic_delay_ns",
    "measurement_observed_effective_latency_ns",
    "measurement_latency_overshoot_ns",
)
_CALIBRATION_V2_FIELDS = {
    "schema_version",
    "execution_mode",
    "clock",
    "checkpoint_identity_fingerprint",
    "server_metadata_fingerprint",
    "canonical_observation_identity",
    "seed_namespace",
    "bootstrap_request_fingerprint",
    "warmup_request_fingerprints",
    "measurement_request_fingerprints",
    *_CALIBRATION_V2_SERIES_FIELDS,
    "p95_method",
    "p95_rank",
    "empirical_raw_p95_ns",
    "empirical_sampled_target_p95_ns",
    "empirical_observed_effective_p95_ns",
    "theoretical_p95_latency_ns",
    "control_period_ns",
    "scheduling_latency_budget_ns",
    "derived_delay_ticks",
    "derived_prefetch_budget_ns",
    "empirical_p95_exceeds_budget",
    "fingerprint",
}


def _latency_series(values: Sequence[int], *, label: str, count: int) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("{} must be a sequence".format(label))
    result = tuple(_require_nonbool_int(value, label=label, minimum=0) for value in values)
    if len(result) != count:
        raise ValueError("{} must contain exactly {} values".format(label, count))
    return result


def _calibration_v2_payload(values: Mapping[str, Any]) -> Dict[str, Any]:
    identity = values["canonical_observation_identity"]
    if isinstance(identity, CalibrationObservationIdentityV1):
        identity = identity.to_dict()
    payload = {
        "schema_version": values["schema_version"],
        "execution_mode": values["execution_mode"],
        "clock": values["clock"],
        "checkpoint_identity_fingerprint": values["checkpoint_identity_fingerprint"],
        "server_metadata_fingerprint": values["server_metadata_fingerprint"],
        "canonical_observation_identity": dict(identity),
        "seed_namespace": values["seed_namespace"],
        "bootstrap_request_fingerprint": values["bootstrap_request_fingerprint"],
        "warmup_request_fingerprints": list(values["warmup_request_fingerprints"]),
        "measurement_request_fingerprints": list(values["measurement_request_fingerprints"]),
        "p95_method": values["p95_method"],
        "p95_rank": values["p95_rank"],
        "empirical_raw_p95_ns": values["empirical_raw_p95_ns"],
        "empirical_sampled_target_p95_ns": values["empirical_sampled_target_p95_ns"],
        "empirical_observed_effective_p95_ns": values["empirical_observed_effective_p95_ns"],
        "theoretical_p95_latency_ns": values["theoretical_p95_latency_ns"],
        "control_period_ns": values["control_period_ns"],
        "scheduling_latency_budget_ns": values["scheduling_latency_budget_ns"],
        "derived_delay_ticks": values["derived_delay_ticks"],
        "derived_prefetch_budget_ns": values["derived_prefetch_budget_ns"],
        "empirical_p95_exceeds_budget": values["empirical_p95_exceeds_budget"],
    }
    for field in _CALIBRATION_V2_SERIES_FIELDS:
        payload[field] = list(values[field])
    return payload


@dataclasses.dataclass(frozen=True)
class LatencyCalibrationV2:
    schema_version: int
    execution_mode: str
    clock: str
    checkpoint_identity_fingerprint: str
    server_metadata_fingerprint: str
    canonical_observation_identity: CalibrationObservationIdentityV1
    seed_namespace: str
    bootstrap_request_fingerprint: Optional[str]
    warmup_request_fingerprints: Tuple[str, ...]
    measurement_request_fingerprints: Tuple[str, ...]
    warmup_raw_inference_latency_ns: Tuple[int, ...]
    warmup_sampled_target_latency_ns: Tuple[int, ...]
    warmup_requested_synthetic_delay_ns: Tuple[int, ...]
    warmup_observed_synthetic_delay_ns: Tuple[int, ...]
    warmup_observed_effective_latency_ns: Tuple[int, ...]
    warmup_latency_overshoot_ns: Tuple[int, ...]
    measurement_raw_inference_latency_ns: Tuple[int, ...]
    measurement_sampled_target_latency_ns: Tuple[int, ...]
    measurement_requested_synthetic_delay_ns: Tuple[int, ...]
    measurement_observed_synthetic_delay_ns: Tuple[int, ...]
    measurement_observed_effective_latency_ns: Tuple[int, ...]
    measurement_latency_overshoot_ns: Tuple[int, ...]
    p95_method: str
    p95_rank: int
    empirical_raw_p95_ns: int
    empirical_sampled_target_p95_ns: int
    empirical_observed_effective_p95_ns: int
    theoretical_p95_latency_ns: int
    control_period_ns: int
    scheduling_latency_budget_ns: int
    derived_delay_ticks: Optional[int]
    derived_prefetch_budget_ns: Optional[int]
    empirical_p95_exceeds_budget: bool
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_request_fingerprints", tuple(self.warmup_request_fingerprints))
        object.__setattr__(
            self,
            "measurement_request_fingerprints",
            tuple(self.measurement_request_fingerprints),
        )
        for field in _CALIBRATION_V2_SERIES_FIELDS:
            count = CALIBRATION_WARMUP_COUNT if field.startswith("warmup_") else CALIBRATION_MEASUREMENT_COUNT
            object.__setattr__(
                self,
                field,
                _latency_series(getattr(self, field), label=field, count=count),
            )
        self._validate()

    @classmethod
    def create(
        cls,
        *,
        execution_mode: str,
        checkpoint_identity_fingerprint: str,
        server_metadata_fingerprint: str,
        canonical_observation_identity: CalibrationObservationIdentityV1,
        seed_namespace: str,
        bootstrap_request_fingerprint: Optional[str],
        warmup_request_fingerprints: Sequence[str],
        measurement_request_fingerprints: Sequence[str],
        warmup_raw_inference_latency_ns: Sequence[int],
        warmup_sampled_target_latency_ns: Sequence[int],
        warmup_requested_synthetic_delay_ns: Sequence[int],
        warmup_observed_synthetic_delay_ns: Sequence[int],
        warmup_observed_effective_latency_ns: Sequence[int],
        warmup_latency_overshoot_ns: Sequence[int],
        measurement_raw_inference_latency_ns: Sequence[int],
        measurement_sampled_target_latency_ns: Sequence[int],
        measurement_requested_synthetic_delay_ns: Sequence[int],
        measurement_observed_synthetic_delay_ns: Sequence[int],
        measurement_observed_effective_latency_ns: Sequence[int],
        measurement_latency_overshoot_ns: Sequence[int],
    ) -> "LatencyCalibrationV2":
        if not isinstance(canonical_observation_identity, CalibrationObservationIdentityV1):
            raise ValueError("canonical_observation_identity must use schema version 1")
        raw = _latency_series(
            measurement_raw_inference_latency_ns,
            label="measurement_raw_inference_latency_ns",
            count=CALIBRATION_MEASUREMENT_COUNT,
        )
        sampled = _latency_series(
            measurement_sampled_target_latency_ns,
            label="measurement_sampled_target_latency_ns",
            count=CALIBRATION_MEASUREMENT_COUNT,
        )
        observed_effective = _latency_series(
            measurement_observed_effective_latency_ns,
            label="measurement_observed_effective_latency_ns",
            count=CALIBRATION_MEASUREMENT_COUNT,
        )
        rank, raw_p95 = nearest_rank_p95_ns(raw)
        _, sampled_p95 = nearest_rank_p95_ns(sampled)
        _, observed_effective_p95 = nearest_rank_p95_ns(observed_effective)
        if execution_mode in ("baseline_async", "baseline_async_recovery", "baseline_rtc"):
            delay_ticks = SCHEDULING_DELAY_TICKS
            prefetch_budget_ns = None
        elif is_async_bsp_mode_v5(execution_mode) and not is_native_latency_mode_v5(execution_mode):
            delay_ticks = None
            prefetch_budget_ns = SCHEDULING_LATENCY_BUDGET_NS
        elif is_native_latency_mode_v5(execution_mode):
            delay_ticks = None
            prefetch_budget_ns = native_prefetch_budget_ns(raw)
        else:
            raise ValueError("latency calibration is defined only for schema-v5 modes")
        values = {
            "schema_version": 2,
            "execution_mode": execution_mode,
            "clock": "monotonic_ns",
            "checkpoint_identity_fingerprint": checkpoint_identity_fingerprint,
            "server_metadata_fingerprint": server_metadata_fingerprint,
            "canonical_observation_identity": canonical_observation_identity,
            "seed_namespace": seed_namespace,
            "bootstrap_request_fingerprint": bootstrap_request_fingerprint,
            "warmup_request_fingerprints": tuple(warmup_request_fingerprints),
            "measurement_request_fingerprints": tuple(measurement_request_fingerprints),
            "warmup_raw_inference_latency_ns": tuple(warmup_raw_inference_latency_ns),
            "warmup_sampled_target_latency_ns": tuple(warmup_sampled_target_latency_ns),
            "warmup_requested_synthetic_delay_ns": tuple(warmup_requested_synthetic_delay_ns),
            "warmup_observed_synthetic_delay_ns": tuple(warmup_observed_synthetic_delay_ns),
            "warmup_observed_effective_latency_ns": tuple(warmup_observed_effective_latency_ns),
            "warmup_latency_overshoot_ns": tuple(warmup_latency_overshoot_ns),
            "measurement_raw_inference_latency_ns": raw,
            "measurement_sampled_target_latency_ns": sampled,
            "measurement_requested_synthetic_delay_ns": tuple(measurement_requested_synthetic_delay_ns),
            "measurement_observed_synthetic_delay_ns": tuple(measurement_observed_synthetic_delay_ns),
            "measurement_observed_effective_latency_ns": observed_effective,
            "measurement_latency_overshoot_ns": tuple(measurement_latency_overshoot_ns),
            "p95_method": "nearest_rank",
            "p95_rank": rank,
            "empirical_raw_p95_ns": raw_p95,
            "empirical_sampled_target_p95_ns": sampled_p95,
            "empirical_observed_effective_p95_ns": observed_effective_p95,
            "theoretical_p95_latency_ns": (
                0 if is_native_latency_mode_v5(execution_mode) else THEORETICAL_P95_LATENCY_NS
            ),
            "control_period_ns": CONTROL_PERIOD_NS,
            "scheduling_latency_budget_ns": (
                prefetch_budget_ns if is_native_latency_mode_v5(execution_mode) else SCHEDULING_LATENCY_BUDGET_NS
            ),
            "derived_delay_ticks": delay_ticks,
            "derived_prefetch_budget_ns": prefetch_budget_ns,
            "empirical_p95_exceeds_budget": observed_effective_p95
            > (prefetch_budget_ns if is_native_latency_mode_v5(execution_mode) else SCHEDULING_LATENCY_BUDGET_NS),
        }
        return cls(
            fingerprint=canonical_fingerprint(_calibration_v2_payload(values)),
            **values,
        )

    def _validate(self) -> None:
        if _require_nonbool_int(self.schema_version, label="calibration schema_version") != 2:
            raise ValueError("calibration schema_version must be integer 2")
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError("calibration execution_mode must be a schema-v5 mode")
        if self.clock != "monotonic_ns":
            raise ValueError("calibration clock must be monotonic_ns")
        _require_sha256(self.checkpoint_identity_fingerprint, label="checkpoint_identity_fingerprint")
        _require_sha256(self.server_metadata_fingerprint, label="server_metadata_fingerprint")
        if not isinstance(self.canonical_observation_identity, CalibrationObservationIdentityV1):
            raise ValueError("canonical_observation_identity must use schema version 1")
        self.canonical_observation_identity.__post_init__()
        expected_namespace = "openpi-libero-calibration-v2/{}/{}".format(
            self.execution_mode,
            self.checkpoint_identity_fingerprint,
        )
        if self.seed_namespace != expected_namespace:
            raise ValueError("calibration seed_namespace does not match its identities")
        if len(self.warmup_request_fingerprints) != CALIBRATION_WARMUP_COUNT:
            raise ValueError("calibration requires exactly five warmup request fingerprints")
        if len(self.measurement_request_fingerprints) != CALIBRATION_MEASUREMENT_COUNT:
            raise ValueError("calibration requires exactly twenty measurement request fingerprints")
        for value in self.warmup_request_fingerprints + self.measurement_request_fingerprints:
            _require_sha256(value, label="request fingerprint")
        if self.execution_mode == "baseline_rtc":
            _require_sha256(self.bootstrap_request_fingerprint, label="bootstrap_request_fingerprint")
        elif self.bootstrap_request_fingerprint is not None:
            raise ValueError("only RTC calibration may carry a bootstrap fingerprint")
        for prefix, count in (
            ("warmup", CALIBRATION_WARMUP_COUNT),
            ("measurement", CALIBRATION_MEASUREMENT_COUNT),
        ):
            raw = _latency_series(
                getattr(self, prefix + "_raw_inference_latency_ns"),
                label=prefix + " raw latency",
                count=count,
            )
            sampled = _latency_series(
                getattr(self, prefix + "_sampled_target_latency_ns"),
                label=prefix + " sampled target",
                count=count,
            )
            requested = _latency_series(
                getattr(self, prefix + "_requested_synthetic_delay_ns"),
                label=prefix + " requested synthetic delay",
                count=count,
            )
            observed_synthetic = _latency_series(
                getattr(self, prefix + "_observed_synthetic_delay_ns"),
                label=prefix + " observed synthetic delay",
                count=count,
            )
            observed_effective = _latency_series(
                getattr(self, prefix + "_observed_effective_latency_ns"),
                label=prefix + " observed effective latency",
                count=count,
            )
            overshoot = _latency_series(
                getattr(self, prefix + "_latency_overshoot_ns"),
                label=prefix + " latency overshoot",
                count=count,
            )
            if any(
                requested_value != max(raw_value, sampled_value) - raw_value
                or raw_value + observed_synthetic_value != observed_effective_value
                or observed_effective_value < max(raw_value, sampled_value)
                or overshoot_value != observed_effective_value - max(raw_value, sampled_value)
                or observed_synthetic_value != requested_value + overshoot_value
                for raw_value, sampled_value, requested_value, observed_synthetic_value, observed_effective_value, overshoot_value in zip(
                    raw,
                    sampled,
                    requested,
                    observed_synthetic,
                    observed_effective,
                    overshoot,
                )
            ):
                raise ValueError("calibration latency breakdown is inconsistent")
        rank, raw_p95 = nearest_rank_p95_ns(self.measurement_raw_inference_latency_ns)
        _, sampled_p95 = nearest_rank_p95_ns(self.measurement_sampled_target_latency_ns)
        _, observed_effective_p95 = nearest_rank_p95_ns(self.measurement_observed_effective_latency_ns)
        native_budget_ns = native_prefetch_budget_ns(self.measurement_raw_inference_latency_ns)
        expected_budget_ns = (
            native_budget_ns if is_native_latency_mode_v5(self.execution_mode) else SCHEDULING_LATENCY_BUDGET_NS
        )
        expected_scalars = {
            "p95_rank": rank,
            "empirical_raw_p95_ns": raw_p95,
            "empirical_sampled_target_p95_ns": sampled_p95,
            "empirical_observed_effective_p95_ns": observed_effective_p95,
            "theoretical_p95_latency_ns": (
                0 if is_native_latency_mode_v5(self.execution_mode) else THEORETICAL_P95_LATENCY_NS
            ),
            "control_period_ns": CONTROL_PERIOD_NS,
            "scheduling_latency_budget_ns": expected_budget_ns,
        }
        if self.p95_method != "nearest_rank":
            raise ValueError("calibration must use nearest-rank p95")
        for field, expected in expected_scalars.items():
            if _require_nonbool_int(getattr(self, field), label=field, minimum=0) != expected:
                raise ValueError("{} does not match calibration measurements".format(field))
        if self.execution_mode in ("baseline_async", "baseline_async_recovery", "baseline_rtc"):
            if self.derived_delay_ticks != SCHEDULING_DELAY_TICKS or self.derived_prefetch_budget_ns is not None:
                raise ValueError("baseline calibration must use the fixed eight-tick budget")
        elif is_native_latency_mode_v5(self.execution_mode):
            if self.derived_delay_ticks is not None or self.derived_prefetch_budget_ns != native_budget_ns:
                raise ValueError("native BSP calibration must derive its initial raw-latency budget")
            for field in _CALIBRATION_V2_SERIES_FIELDS:
                if "sampled_target" in field or "synthetic_delay" in field or "overshoot" in field:
                    if any(getattr(self, field)):
                        raise ValueError("native BSP calibration must not inject synthetic latency")
        elif self.derived_delay_ticks is not None or self.derived_prefetch_budget_ns != SCHEDULING_LATENCY_BUDGET_NS:
            raise ValueError("BSP calibration must use the fixed prefetch budget")
        if type(self.empirical_p95_exceeds_budget) is not bool or self.empirical_p95_exceeds_budget != (
            observed_effective_p95 > expected_budget_ns
        ):
            raise ValueError("empirical_p95_exceeds_budget does not match the measurements")
        _require_sha256(self.fingerprint, label="calibration fingerprint")
        if self.fingerprint != canonical_fingerprint(self._payload_without_fingerprint()):
            raise ValueError("calibration fingerprint does not bind its serialized fields")

    def _payload_without_fingerprint(self) -> Dict[str, Any]:
        return _calibration_v2_payload(dataclasses.asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        payload = self._payload_without_fingerprint()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LatencyCalibrationV2":
        if not isinstance(value, Mapping) or set(value) != _CALIBRATION_V2_FIELDS:
            raise ValueError("latency calibration fields must exactly match schema version 2")
        values = dict(value)
        values["canonical_observation_identity"] = CalibrationObservationIdentityV1.from_dict(
            values["canonical_observation_identity"]
        )
        for field in (
            "warmup_request_fingerprints",
            "measurement_request_fingerprints",
            *_CALIBRATION_V2_SERIES_FIELDS,
        ):
            if not isinstance(values[field], list):
                raise ValueError("{} must be a JSON list".format(field))
            values[field] = tuple(values[field])
        return cls(**values)


def validate_server_metadata(mode: ExecutionModeSpec, metadata: Mapping[str, Any]) -> str:
    """Validate the exact family capability and fingerprint all server metadata."""
    _require_known_mode(mode)
    if not isinstance(metadata, Mapping):
        raise ValueError("server metadata must be a mapping")
    if inference.INFERENCE_CAPABILITIES_KEY not in metadata:
        raise ValueError("server metadata is missing inference capabilities")
    capabilities = metadata[inference.INFERENCE_CAPABILITIES_KEY]
    fields = {
        "schema_version",
        "action_representation",
        "model_action_horizon",
        "model_action_dim",
        "supported_protocols",
    }
    if not isinstance(capabilities, Mapping) or set(capabilities) != fields:
        raise ValueError("server inference capabilities must have the exact schema-one fields")
    if mode.policy_variant == "baseline":
        expected = {
            "schema_version": 1,
            "action_representation": "native",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": [
                "baseline_h16_n5_v1",
                "baseline_sync_n5_h16_full_v2",
                "baseline_async_h16_v1",
                "baseline_async_h16_blocking_recovery_v2",
                "baseline_rtc_h16_v1",
            ],
        }
    else:
        expected = {
            "schema_version": 1,
            "action_representation": "bsp",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": [
                "bsp_spline_sync_speedup2_phase0_v2",
                "bsp_spline_async_phase_skip_speedup2_v2",
                "bsp_spline_async_phase_skip_speedup1_v1",
                "bsp_spline_async_phase_skip_speedup4_v1",
                "bsp_spline_async_phase_skip_speedup8_v1",
            ],
        }
    for field, expected_value in expected.items():
        actual = capabilities[field]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError("server inference capabilities do not match the required protocol")
    return canonical_fingerprint(metadata)


def calibration_seed(namespace: str, phase: str, index: int) -> int:
    _require_nonempty_text(namespace, label="namespace")
    _require_nonempty_text(phase, label="phase")
    seed_index = _require_nonbool_int(index, label="seed index", minimum=0)
    payload = canonical_json_bytes({"namespace": namespace, "phase": phase, "index": seed_index})
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


class CalibrationError(RuntimeError):
    """Base class for fatal schema-v5 calibration failures."""


class CalibrationInfrastructureError(CalibrationError):
    """An infrastructure failure after calibration retries are exhausted."""


class CalibrationPolicyError(CalibrationError):
    """A malformed response or current policy failure during calibration."""


class CalibrationIdentityError(CalibrationError):
    """The connected server does not match the calibrated run identity."""


class CalibrationWorker(Protocol):
    def connect(self, timeout: Optional[float] = None) -> Any: ...

    def submit(
        self,
        observation: Dict[str, Any],
        *,
        latency_sample_key: latency_sampling.LatencySampleKeyV1,
    ) -> Any: ...

    def wait(self, job: Any, timeout: Optional[float] = None) -> Any: ...

    def reset_generation(self) -> int: ...

    def wait_until_ready(self, generation: int, timeout: Optional[float] = None) -> None: ...


@dataclasses.dataclass(frozen=True)
class _ProbeResult:
    response: Mapping[str, Any]
    request_fingerprint: str
    raw_inference_latency_ns: int
    sampled_target_latency_ns: int
    requested_synthetic_delay_ns: int
    observed_synthetic_delay_ns: int
    observed_effective_latency_ns: int
    latency_overshoot_ns: int


class _RetryCalibration(Exception):
    pass


def calibrate_async_mode(
    mode: ExecutionModeSpec,
    canonical_request: Mapping[str, Any],
    canonical_identity: CalibrationObservationIdentityV1,
    worker: CalibrationWorker,
    checkpoint_identity: CheckpointIdentityV1,
    server_metadata_fingerprint: str,
) -> LatencyCalibrationV2:
    """Run a complete deterministic 5+20 calibration through the public worker API."""
    _require_known_mode(mode)
    if mode.calibration_kind not in ("baseline_async", "rtc", "bsp"):
        raise ValueError("calibration is defined only for asynchronous modes")
    if not isinstance(canonical_request, Mapping):
        raise ValueError("canonical_request must be a mapping")
    if not isinstance(canonical_identity, CalibrationObservationIdentityV1):
        raise ValueError("canonical_identity must use schema version 1")
    canonical_identity.__post_init__()
    if not isinstance(checkpoint_identity, CheckpointIdentityV1):
        raise ValueError("checkpoint_identity must use schema version 1")
    checkpoint_identity.__post_init__()
    expected_request_fingerprint = canonical_fingerprint(canonical_request)
    if canonical_identity.request_fingerprint != expected_request_fingerprint:
        raise CalibrationIdentityError("canonical request fingerprint does not match its identity")
    _require_sha256(server_metadata_fingerprint, label="server_metadata_fingerprint")
    for reserved_key in (
        inference.INFERENCE_SEED_KEY,
        inference.RTC_REQUEST_KEY,
        inference.BSP_EXECUTION_KEY,
    ):
        if reserved_key in canonical_request:
            raise ValueError("canonical_request must not contain reserved inference envelopes")
    if mode.policy_variant == "baseline":
        if checkpoint_identity.bsp_cache_hash is not None:
            raise CalibrationIdentityError("baseline calibration requires null BSP cache identities")
    elif checkpoint_identity.bsp_cache_hash is None:
        raise CalibrationIdentityError("BSP calibration requires BSP cache identities")

    checkpoint_fingerprint = checkpoint_identity.fingerprint
    namespace = "openpi-libero-calibration-v2/{}/{}".format(
        mode.name,
        checkpoint_fingerprint,
    )
    last_infrastructure_error = None  # type: Optional[BaseException]
    for attempt in range(CALIBRATION_INFRASTRUCTURE_RETRIES + 1):
        if attempt:
            try:
                generation = worker.reset_generation()
                worker.wait_until_ready(generation)
            except Exception as error:
                if not _is_infrastructure_error(error):
                    raise
                last_infrastructure_error = error
                continue
        try:
            connection = worker.connect()
            _validate_connection_identity(
                mode,
                connection,
                expected_fingerprint=server_metadata_fingerprint,
            )
            return _calibrate_once(
                mode,
                canonical_request,
                canonical_identity,
                worker,
                checkpoint_fingerprint,
                server_metadata_fingerprint,
                namespace,
            )
        except _RetryCalibration as retry:
            last_infrastructure_error = retry.__cause__ or retry
            if attempt == CALIBRATION_INFRASTRUCTURE_RETRIES:
                break
        except Exception as error:
            if not _is_infrastructure_error(error):
                raise
            last_infrastructure_error = error
            if attempt == CALIBRATION_INFRASTRUCTURE_RETRIES:
                break
    message = "Calibration infrastructure failed after three complete attempts"
    raise CalibrationInfrastructureError(message) from last_infrastructure_error


def _calibrate_once(
    mode: ExecutionModeSpec,
    canonical_request: Mapping[str, Any],
    canonical_identity: CalibrationObservationIdentityV1,
    worker: CalibrationWorker,
    checkpoint_fingerprint: str,
    server_metadata_fingerprint: str,
    namespace: str,
) -> LatencyCalibrationV2:
    bootstrap_fingerprint = None  # type: Optional[str]
    previous_response = None  # type: Optional[Mapping[str, Any]]
    if mode.calibration_kind == "rtc":
        bootstrap_overlay = rtc.RtcPlan(d_init=8).begin_bootstrap()
        bootstrap = _run_probe(
            mode,
            canonical_request,
            worker,
            namespace=namespace,
            phase="bootstrap",
            index=0,
            overlay=bootstrap_overlay,
            expected_metadata_fingerprint=server_metadata_fingerprint,
        )
        _validate_rtc_calibration_response(bootstrap.response)
        bootstrap_fingerprint = bootstrap.request_fingerprint
        previous_response = bootstrap.response

    warmup_fingerprints = []  # type: List[str]
    measurement_fingerprints = []  # type: List[str]
    warmup_breakdown = {field: [] for field in _CALIBRATION_V2_SERIES_FIELDS[:6]}
    measurement_breakdown = {field: [] for field in _CALIBRATION_V2_SERIES_FIELDS[6:]}
    for phase, count, fingerprints, breakdown in (
        (
            "warmup",
            CALIBRATION_WARMUP_COUNT,
            warmup_fingerprints,
            warmup_breakdown,
        ),
        (
            "measurement",
            CALIBRATION_MEASUREMENT_COUNT,
            measurement_fingerprints,
            measurement_breakdown,
        ),
    ):
        for index in range(count):
            overlay = {}  # type: Mapping[str, Any]
            if mode.calibration_kind == "rtc":
                if previous_response is None:
                    raise AssertionError("RTC calibration lost its chained bootstrap response")
                plan = rtc.RtcPlan(d_init=8)
                plan.begin_bootstrap()
                plan.install_result(previous_response)
                for _ in range(8):
                    plan.consume_action()
                overlay = plan.begin_guided()
            elif mode.calibration_kind == "baseline_async":
                overlay = {
                    inference.RTC_REQUEST_KEY: {
                        "schema_version": inference.RTC_SCHEMA_VERSION,
                    }
                }
            elif is_async_bsp_mode_v5(mode.name) and bsp_speedup_for_mode_v5(mode.name) != 2:
                overlay = {
                    inference.BSP_EXECUTION_KEY: {
                        "schema_version": inference.BSP_EXECUTION_SCHEMA_VERSION,
                        "speedup": bsp_speedup_for_mode_v5(mode.name),
                    }
                }
            probe = _run_probe(
                mode,
                canonical_request,
                worker,
                namespace=namespace,
                phase=phase,
                index=index,
                overlay=overlay,
                expected_metadata_fingerprint=server_metadata_fingerprint,
            )
            if mode.calibration_kind in ("baseline_async", "rtc"):
                _validate_rtc_calibration_response(probe.response)
                if mode.calibration_kind == "rtc":
                    previous_response = probe.response
            else:
                _validate_bsp_calibration_response(
                    probe.response,
                    expected_speedup=bsp_speedup_for_mode_v5(mode.name),
                )
            fingerprints.append(probe.request_fingerprint)
            prefix = phase + "_"
            breakdown[prefix + "raw_inference_latency_ns"].append(probe.raw_inference_latency_ns)
            breakdown[prefix + "sampled_target_latency_ns"].append(probe.sampled_target_latency_ns)
            breakdown[prefix + "requested_synthetic_delay_ns"].append(probe.requested_synthetic_delay_ns)
            breakdown[prefix + "observed_synthetic_delay_ns"].append(probe.observed_synthetic_delay_ns)
            breakdown[prefix + "observed_effective_latency_ns"].append(probe.observed_effective_latency_ns)
            breakdown[prefix + "latency_overshoot_ns"].append(probe.latency_overshoot_ns)

    return LatencyCalibrationV2.create(
        execution_mode=mode.name,
        checkpoint_identity_fingerprint=checkpoint_fingerprint,
        server_metadata_fingerprint=server_metadata_fingerprint,
        canonical_observation_identity=canonical_identity,
        seed_namespace=namespace,
        bootstrap_request_fingerprint=bootstrap_fingerprint,
        warmup_request_fingerprints=warmup_fingerprints,
        measurement_request_fingerprints=measurement_fingerprints,
        **warmup_breakdown,
        **measurement_breakdown,
    )


def _run_probe(
    mode: ExecutionModeSpec,
    canonical_request: Mapping[str, Any],
    worker: CalibrationWorker,
    *,
    namespace: str,
    phase: str,
    index: int,
    overlay: Mapping[str, Any],
    expected_metadata_fingerprint: str,
) -> _ProbeResult:
    request = dict(canonical_request)
    request.update(overlay)
    request[inference.INFERENCE_SEED_KEY] = calibration_seed(namespace, phase, index)
    request_fingerprint = canonical_fingerprint(request)
    latency_sample_key = latency_sampling.LatencySampleKeyV1(
        namespace="calibration/{}".format(phase),
        seed=latency_sampling.DEFAULT_SEED,
        suite="calibration",
        task_id=0,
        trial_index=0,
        request_ordinal=index,
    )
    try:
        job = worker.submit(request, latency_sample_key=latency_sample_key)
        outcome = worker.wait(job)
    except Exception as error:
        if _is_infrastructure_error(error):
            raise _RetryCalibration() from error
        raise CalibrationPolicyError(
            "Calibration policy call failed: {}: {}".format(type(error).__name__, error)
        ) from error
    if getattr(outcome, "job", None) is not job:
        raise CalibrationPolicyError("Calibration worker returned an outcome for a different job")
    if getattr(outcome, "stale", False) or getattr(outcome, "cancelled", False):
        raise _RetryCalibration() from CalibrationInfrastructureError("Calibration request became stale or cancelled")
    _validate_connection_identity(
        mode,
        getattr(outcome, "connection", None),
        expected_fingerprint=expected_metadata_fingerprint,
    )
    error = getattr(outcome, "error", None)
    if error is not None:
        if not isinstance(error, Exception):
            raise error
        if _is_infrastructure_error(error):
            raise _RetryCalibration() from error
        raise CalibrationPolicyError("Calibration policy failed: {}: {}".format(type(error).__name__, error)) from error
    submitted_ns = getattr(job, "submitted_monotonic_ns", None)
    completed_ns = getattr(outcome, "completed_monotonic_ns", None)
    try:
        submitted_ns = _require_nonbool_int(submitted_ns, label="probe submit time", minimum=0)
        completed_ns = _require_nonbool_int(completed_ns, label="probe completion time", minimum=0)
    except ValueError as error:
        raise CalibrationPolicyError(str(error)) from error
    if completed_ns < submitted_ns:
        raise CalibrationPolicyError("Calibration completion precedes submission")
    measured_latency_ns = completed_ns - submitted_ns
    breakdown = (
        getattr(outcome, "sampled_target_latency_ns", None),
        getattr(outcome, "raw_inference_latency_ns", None),
        getattr(outcome, "requested_synthetic_delay_ns", None),
        getattr(outcome, "observed_synthetic_delay_ns", None),
        getattr(outcome, "observed_effective_latency_ns", None),
        getattr(outcome, "latency_overshoot_ns", None),
    )
    if any(value is None for value in breakdown):
        raise CalibrationPolicyError("Calibration latency breakdown must be complete")
    try:
        sampled_target_latency_ns = _require_nonbool_int(breakdown[0], label="sampled target latency", minimum=0)
        raw_latency_ns = _require_nonbool_int(breakdown[1], label="raw inference latency", minimum=0)
        requested_delay_ns = _require_nonbool_int(breakdown[2], label="requested synthetic delay", minimum=0)
        observed_delay_ns = _require_nonbool_int(breakdown[3], label="observed synthetic delay", minimum=0)
        observed_effective_ns = _require_nonbool_int(breakdown[4], label="observed effective latency", minimum=0)
        overshoot_ns = _require_nonbool_int(breakdown[5], label="latency overshoot", minimum=0)
    except ValueError as error:
        raise CalibrationPolicyError(str(error)) from error
    scheduled_effective_ns = max(raw_latency_ns, sampled_target_latency_ns)
    if (
        requested_delay_ns != scheduled_effective_ns - raw_latency_ns
        or raw_latency_ns + observed_delay_ns != observed_effective_ns
        or observed_effective_ns != measured_latency_ns
        or observed_effective_ns < scheduled_effective_ns
        or overshoot_ns != observed_effective_ns - scheduled_effective_ns
        or observed_delay_ns != requested_delay_ns + overshoot_ns
        or getattr(job, "sampled_target_latency_ns", None) != sampled_target_latency_ns
        or getattr(job, "latency_sample_key", None) != latency_sample_key
    ):
        raise CalibrationPolicyError("Calibration effective latency does not match its measured breakdown")
    response = getattr(outcome, "result", None)
    if not isinstance(response, Mapping):
        raise CalibrationPolicyError("Calibration response must be a mapping")
    return _ProbeResult(
        response=response,
        request_fingerprint=request_fingerprint,
        raw_inference_latency_ns=raw_latency_ns,
        sampled_target_latency_ns=sampled_target_latency_ns,
        requested_synthetic_delay_ns=requested_delay_ns,
        observed_synthetic_delay_ns=observed_delay_ns,
        observed_effective_latency_ns=observed_effective_ns,
        latency_overshoot_ns=overshoot_ns,
    )


def _validate_connection_identity(
    mode: ExecutionModeSpec,
    connection: Any,
    *,
    expected_fingerprint: str,
) -> None:
    payload = getattr(connection, "metadata_payload", None)
    if not isinstance(payload, bytes):
        raise CalibrationIdentityError("Calibration connection is missing metadata bytes")
    try:
        metadata = msgpack_numpy.unpackb(payload)
        actual_fingerprint = validate_server_metadata(mode, metadata)
    except (TypeError, ValueError) as error:
        raise CalibrationIdentityError("Calibration server metadata is invalid") from error
    if actual_fingerprint != expected_fingerprint:
        raise CalibrationIdentityError("Calibration server metadata fingerprint changed")


def _is_infrastructure_error(error: BaseException) -> bool:
    websocket_disconnect = type(error).__module__.startswith("websockets") and type(error).__name__.startswith(
        "Connection"
    )
    return isinstance(error, (ConnectionError, TimeoutError, EOFError, OSError)) or websocket_disconnect


def _validate_rtc_calibration_response(response: Mapping[str, Any]) -> None:
    try:
        rtc.RtcActionChunk.from_response(response)
    except (TypeError, ValueError) as error:
        raise CalibrationPolicyError("Malformed RTC calibration response") from error


def _validate_bsp_calibration_response(response: Mapping[str, Any], *, expected_speedup: int = 2) -> None:
    try:
        validate_bsp_response(response, expected_speedup=expected_speedup)
    except (TypeError, ValueError) as error:
        raise CalibrationPolicyError("Malformed BSP calibration response") from error


class ModeSchedulerV5:
    _pending_intent: Optional[RequestIntentV5]

    def reset(self) -> None:
        raise NotImplementedError

    def maybe_request(
        self,
        now_ns: int,
        *,
        at_due: bool,
        request_in_flight: bool,
        control_step: int = 0,
    ) -> Optional[RequestIntentV5]:
        raise NotImplementedError

    def install_response(
        self,
        intent: RequestIntentV5,
        response: Mapping[str, Any],
        *,
        now_ns: int,
        control_step: int,
    ) -> ActivationDecisionV5:
        raise NotImplementedError

    def take_action(self, now_ns: int, *, control_step: int = 0) -> ActionDecisionV5:
        raise NotImplementedError

    def observe_raw_inference_latency(self, raw_inference_latency_ns: int) -> None:
        raise ValueError("scheduler does not use a rolling raw-latency budget")

    def _reuse_pending(
        self,
        *,
        request_in_flight: bool,
    ) -> Optional[RequestIntentV5]:
        if self._pending_intent is None:
            return None
        if request_in_flight:
            return None
        return self._pending_intent

    def _set_pending(self, intent: RequestIntentV5) -> RequestIntentV5:
        if self._pending_intent is not None:
            raise RuntimeError("a scheduler request transition is already pending")
        self._pending_intent = intent
        return intent

    def _require_pending(self, intent: RequestIntentV5) -> None:
        if intent is not self._pending_intent:
            raise ValueError("response intent does not match the pending scheduler transition")

    def _complete_pending(self) -> None:
        if self._pending_intent is None:
            raise RuntimeError("scheduler has no pending transition to complete")
        self._pending_intent = None


class _BaselineSyncScheduler(ModeSchedulerV5):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunk = None  # type: Optional[rtc.RtcActionChunk]
        self._cursor = 0
        self._installed_count = 0
        self._pending_intent = None

    def maybe_request(
        self,
        now_ns: int,
        *,
        at_due: bool,
        request_in_flight: bool,
        control_step: int = 0,
    ) -> Optional[RequestIntentV5]:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_bool(at_due, label="at_due")
        _require_bool(request_in_flight, label="request_in_flight")
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        pending = self._reuse_pending(request_in_flight=request_in_flight)
        if pending is not None or self._pending_intent is not None:
            return pending
        if request_in_flight:
            return None
        if self._chunk is None:
            dispatch = "blocking_initial"
            trigger = "initial_plan"
        elif self._cursor >= 16:
            dispatch = "blocking_replan"
            trigger = "baseline_chunk_exhausted"
        else:
            return None
        return self._set_pending(
            RequestIntentV5(
                dispatch=dispatch,
                trigger=trigger,
                scheduler_context={},
                request_overlay={
                    inference.RTC_REQUEST_KEY: {
                        "schema_version": inference.RTC_SCHEMA_VERSION,
                    }
                },
            )
        )

    def install_response(
        self,
        intent: RequestIntentV5,
        response: Mapping[str, Any],
        *,
        now_ns: int,
        control_step: int,
    ) -> ActivationDecisionV5:
        _validate_install_inputs(intent, now_ns=now_ns, control_step=control_step)
        self._require_pending(intent)
        self._chunk = rtc.RtcActionChunk.from_response(response)
        self._cursor = 0
        activation = "initial" if self._installed_count == 0 else "blocking_replace"
        self._installed_count += 1
        self._complete_pending()
        return ActivationDecisionV5(
            activation=activation,
            activation_context={"action_cursor": 0},
        )

    def take_action(self, now_ns: int, *, control_step: int = 0) -> ActionDecisionV5:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        if self._chunk is None:
            raise rtc.RtcPlanExhaustedError("baseline sync has no installed chunk")
        if self._cursor >= 16:
            return ActionDecisionV5(action=None, underflow=True)
        action = self._chunk.actions[self._cursor]
        self._cursor += 1
        return ActionDecisionV5(action=action, underflow=False)


class _BaselineAsyncScheduler(ModeSchedulerV5):
    def __init__(self, d_init: int, *, blocking_capacity_recovery: bool = False) -> None:
        self._d_init = d_init
        self._blocking_capacity_recovery = blocking_capacity_recovery
        self._plan = rtc.RawAsyncPlan(d_init=d_init)
        self._installed_count = 0
        self._pending_intent = None

    def reset(self) -> None:
        self._plan.reset(d_init=self._d_init)
        self._installed_count = 0
        self._pending_intent = None

    def maybe_request(
        self,
        now_ns: int,
        *,
        at_due: bool,
        request_in_flight: bool,
        control_step: int = 0,
    ) -> Optional[RequestIntentV5]:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_bool(at_due, label="at_due")
        _require_bool(request_in_flight, label="request_in_flight")
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        pending = self._reuse_pending(request_in_flight=request_in_flight)
        if pending is not None or self._pending_intent is not None:
            return pending
        if request_in_flight:
            return None
        state = self._plan.state
        if state is rtc.RtcPlanState.BOOTSTRAP_REQUIRED:
            return self._set_pending(
                RequestIntentV5(
                    dispatch="blocking_initial",
                    trigger="initial_plan",
                    scheduler_context={},
                    request_overlay=self._plan.begin_bootstrap(),
                )
            )
        if state is rtc.RtcPlanState.READY_TO_LAUNCH:
            context = {"s": self._plan.cursor, "d": self._plan.forecast_delay}
            return self._set_pending(
                RequestIntentV5(
                    dispatch="background",
                    trigger="baseline_async_launch",
                    scheduler_context=context,
                    request_overlay=self._plan.begin_background(),
                )
            )
        if state is rtc.RtcPlanState.INFEASIBLE:
            if self._blocking_capacity_recovery:
                return self._set_pending(
                    RequestIntentV5(
                        dispatch="blocking_replan",
                        trigger="baseline_async_capacity_replan",
                        scheduler_context={
                            "action_cursor": self._plan.cursor,
                            "forecast_delay_ticks": self._plan.forecast_delay,
                        },
                        request_overlay=self._plan.begin_blocking_replan(),
                    )
                )
            self._plan.begin_background()
        return None

    def install_response(
        self,
        intent: RequestIntentV5,
        response: Mapping[str, Any],
        *,
        now_ns: int,
        control_step: int,
    ) -> ActivationDecisionV5:
        _validate_install_inputs(intent, now_ns=now_ns, control_step=control_step)
        self._require_pending(intent)
        self._plan.install_result(response)
        if self._installed_count == 0:
            activation = "initial"
        elif intent.dispatch == "blocking_replan":
            activation = "blocking_replace"
        else:
            activation = "immediate_swap"
        self._installed_count += 1
        self._complete_pending()
        return ActivationDecisionV5(
            activation=activation,
            activation_context={"action_cursor": self._plan.cursor},
        )

    def take_action(self, now_ns: int, *, control_step: int = 0) -> ActionDecisionV5:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        if self._plan.state is rtc.RtcPlanState.EXHAUSTED:
            return ActionDecisionV5(action=None, underflow=True)
        return ActionDecisionV5(action=self._plan.consume_action(), underflow=False)


class _RtcScheduler(ModeSchedulerV5):
    def __init__(self, d_init: int) -> None:
        self._d_init = d_init
        self._plan = rtc.RtcPlan(d_init=d_init, fixed_delay=True)
        self._installed_count = 0
        self._pending_intent = None

    def reset(self) -> None:
        self._plan.reset(d_init=self._d_init)
        self._installed_count = 0
        self._pending_intent = None

    def maybe_request(
        self,
        now_ns: int,
        *,
        at_due: bool,
        request_in_flight: bool,
        control_step: int = 0,
    ) -> Optional[RequestIntentV5]:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_bool(at_due, label="at_due")
        _require_bool(request_in_flight, label="request_in_flight")
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        pending = self._reuse_pending(request_in_flight=request_in_flight)
        if pending is not None or self._pending_intent is not None:
            return pending
        if request_in_flight:
            return None
        state = self._plan.state
        if state is rtc.RtcPlanState.BOOTSTRAP_REQUIRED:
            return self._set_pending(
                RequestIntentV5(
                    dispatch="blocking_initial",
                    trigger="initial_plan",
                    scheduler_context={},
                    request_overlay=self._plan.begin_bootstrap(),
                )
            )
        if state is rtc.RtcPlanState.READY_TO_LAUNCH:
            overlay = self._plan.begin_guided()
            context = overlay[inference.RTC_REQUEST_KEY]
            return self._set_pending(
                RequestIntentV5(
                    dispatch="background",
                    trigger="rtc_launch",
                    scheduler_context={"s": context["s"], "d": context["d"]},
                    request_overlay=overlay,
                )
            )
        if state is rtc.RtcPlanState.INFEASIBLE:
            self._plan.begin_guided()
        return None

    def install_response(
        self,
        intent: RequestIntentV5,
        response: Mapping[str, Any],
        *,
        now_ns: int,
        control_step: int,
    ) -> ActivationDecisionV5:
        _validate_install_inputs(intent, now_ns=now_ns, control_step=control_step)
        self._require_pending(intent)
        self._plan.install_result(response)
        activation = "initial" if self._installed_count == 0 else "immediate_swap"
        self._installed_count += 1
        self._complete_pending()
        return ActivationDecisionV5(
            activation=activation,
            activation_context={"action_cursor": self._plan.cursor},
        )

    def take_action(self, now_ns: int, *, control_step: int = 0) -> ActionDecisionV5:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        _require_nonbool_int(control_step, label="control_step", minimum=0)
        if self._plan.state is rtc.RtcPlanState.EXHAUSTED:
            return ActionDecisionV5(action=None, underflow=True)
        return ActionDecisionV5(action=self._plan.consume_action(), underflow=False)


class _BspScheduler(ModeSchedulerV5):
    def __init__(
        self,
        *,
        asynchronous: bool,
        budget_ns: Optional[int],
        expected_speedup: int = 2,
        rolling_raw_latencies_ns: Optional[Sequence[int]] = None,
    ) -> None:
        self._asynchronous = asynchronous
        self._budget_ns = budget_ns
        if isinstance(expected_speedup, bool) or expected_speedup not in (1, 2, 4, 8):
            raise ValueError("expected_speedup must be integer 1, 2, 4, or 8")
        self._expected_speedup = expected_speedup
        self._initial_rolling_raw_latencies_ns = (
            None
            if rolling_raw_latencies_ns is None
            else _latency_series(
                rolling_raw_latencies_ns,
                label="rolling_raw_latencies_ns",
                count=CALIBRATION_MEASUREMENT_COUNT,
            )
        )
        if self._initial_rolling_raw_latencies_ns is not None:
            expected_budget = native_prefetch_budget_ns(self._initial_rolling_raw_latencies_ns)
            if budget_ns != expected_budget:
                raise ValueError("rolling raw-latency history does not match the initial prefetch budget")
        self._request_overlay = (
            {
                inference.BSP_EXECUTION_KEY: {
                    "schema_version": inference.BSP_EXECUTION_SCHEMA_VERSION,
                    "speedup": expected_speedup,
                }
            }
            if expected_speedup != 2
            else {}
        )
        self.reset()

    def reset(self) -> None:
        self._plan = bsp_spline.BspControlActionPlan(expected_speedup=self._expected_speedup)
        self._installed_count = 0
        self._pending_intent = None
        self._pending_request_control_step = None
        self._stale_replan_context = None
        self._rolling_raw_latencies_ns = (
            None if self._initial_rolling_raw_latencies_ns is None else list(self._initial_rolling_raw_latencies_ns)
        )

    def observe_raw_inference_latency(self, raw_inference_latency_ns: int) -> None:
        if self._rolling_raw_latencies_ns is None:
            return super().observe_raw_inference_latency(raw_inference_latency_ns)
        raw_ns = _require_nonbool_int(
            raw_inference_latency_ns,
            label="raw_inference_latency_ns",
            minimum=0,
        )
        self._rolling_raw_latencies_ns.append(raw_ns)
        del self._rolling_raw_latencies_ns[:-CALIBRATION_MEASUREMENT_COUNT]
        self._budget_ns = native_prefetch_budget_ns(self._rolling_raw_latencies_ns)

    def _set_bsp_pending(
        self,
        intent: RequestIntentV5,
        *,
        request_control_step: int,
    ) -> RequestIntentV5:
        self._pending_request_control_step = request_control_step
        return self._set_pending(intent)

    def maybe_request(
        self,
        now_ns: int,
        *,
        at_due: bool,
        request_in_flight: bool,
        control_step: int = 0,
    ) -> Optional[RequestIntentV5]:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        step = _require_nonbool_int(control_step, label="control_step", minimum=0)
        _require_bool(at_due, label="at_due")
        _require_bool(request_in_flight, label="request_in_flight")
        pending = self._reuse_pending(request_in_flight=request_in_flight)
        if pending is not None or self._pending_intent is not None:
            return pending
        if request_in_flight:
            return None
        if self._stale_replan_context is not None:
            return self._set_bsp_pending(
                RequestIntentV5(
                    dispatch="blocking_replan",
                    trigger="bsp_stale_replan",
                    scheduler_context=self._stale_replan_context,
                    request_overlay=self._request_overlay,
                ),
                request_control_step=step,
            )
        if self._plan.spline is None:
            return self._set_bsp_pending(
                RequestIntentV5(
                    dispatch="blocking_initial",
                    trigger="initial_plan",
                    scheduler_context={},
                    request_overlay=self._request_overlay,
                ),
                request_control_step=step,
            )
        if self._asynchronous:
            if self._budget_ns is None:
                raise AssertionError("asynchronous BSP scheduler is missing its budget")
            decision = self._plan.prefetch_decision(step, lead_time_ns=self._budget_ns)
            if not decision.should_prefetch:
                return None
            return self._set_bsp_pending(
                RequestIntentV5(
                    dispatch="background",
                    trigger="bsp_prefetch",
                    scheduler_context={
                        "remaining_plan_ns": decision.remaining_time_ns,
                        "budget_ns": self._budget_ns,
                        "request_control_step": step,
                    },
                    request_overlay=self._request_overlay,
                ),
                request_control_step=step,
            )
        if not at_due:
            return None
        sample = self._plan.sample(step)
        if not sample.underflow:
            return None
        return self._set_bsp_pending(
            RequestIntentV5(
                dispatch="blocking_replan",
                trigger="bsp_curve_exhausted",
                scheduler_context={},
                request_overlay=self._request_overlay,
            ),
            request_control_step=step,
        )

    def install_response(
        self,
        intent: RequestIntentV5,
        response: Mapping[str, Any],
        *,
        now_ns: int,
        control_step: int,
    ) -> ActivationDecisionV5:
        _, step = _validate_install_inputs(intent, now_ns=now_ns, control_step=control_step)
        self._require_pending(intent)
        if self._pending_request_control_step is None:
            raise RuntimeError("BSP scheduler is missing its request control step")
        validate_bsp_response(response, expected_speedup=self._expected_speedup)
        request_step = self._pending_request_control_step
        installed = self._plan.install(
            response["bsp"],
            request_control_step=request_step,
            activation_control_step=step,
            control_freq_hz=20,
        )
        phase_microindices = _to_microindices(installed.phase_offset_indices)
        first_microindices = _to_microindices(installed.first_sample_time)
        curve_t_max_microindices = _to_microindices(
            bsp_spline.BspSpline.from_response(
                response["bsp"],
                expected_speedup=self._expected_speedup,
            ).t_max
        )
        remaining_microindices = max(0, curve_t_max_microindices - phase_microindices)
        remaining_curve_ns = _remaining_ns_from_microindices(
            remaining_microindices,
            effective_curve_rate_hz=10 * self._expected_speedup,
        )
        if installed.stale:
            self._complete_pending()
            self._pending_request_control_step = None
            self._stale_replan_context = {
                "discarded_request_control_step": request_step,
                "discarded_activation_control_step": step,
                "executed_prefix_steps": installed.executed_prefix_steps,
                "phase_offset_microindices": phase_microindices,
                "curve_t_max_microindices": curve_t_max_microindices,
            }
            return ActivationDecisionV5(
                activation="discarded_stale_phase",
                activation_context={
                    "request_control_step": request_step,
                    "activation_control_step": step,
                    "executed_prefix_steps": installed.executed_prefix_steps,
                    "phase_offset_microindices": phase_microindices,
                    "first_sample_microindices": first_microindices,
                    "remaining_curve_microindices": 0,
                    "remaining_curve_ns": 0,
                    "curve_t_max_microindices": curve_t_max_microindices,
                },
            )
        if self._installed_count == 0:
            activation = "initial"
        elif intent.dispatch == "blocking_replan":
            activation = "blocking_replace"
        elif self._asynchronous:
            activation = "immediate_swap"
        else:
            activation = "blocking_replace"
        self._installed_count += 1
        self._complete_pending()
        self._pending_request_control_step = None
        self._stale_replan_context = None
        immediate_prefetch = int(
            self._asynchronous and self._budget_ns is not None and remaining_curve_ns <= self._budget_ns
        )
        activation_context = {
            "request_control_step": request_step,
            "activation_control_step": step,
            "executed_prefix_steps": installed.executed_prefix_steps,
            "phase_offset_microindices": phase_microindices,
            "first_sample_microindices": first_microindices,
            "remaining_curve_microindices": remaining_microindices,
            "remaining_curve_ns": remaining_curve_ns,
            "immediate_prefetch": immediate_prefetch,
        }
        if self._rolling_raw_latencies_ns is not None:
            if self._budget_ns is None:
                raise AssertionError("native BSP scheduler lost its rolling budget")
            activation_context["prefetch_budget_ns"] = self._budget_ns
        return ActivationDecisionV5(
            activation=activation,
            activation_context=activation_context,
        )

    def take_action(self, now_ns: int, *, control_step: int = 0) -> ActionDecisionV5:
        _require_nonbool_int(now_ns, label="now_ns", minimum=0)
        step = _require_nonbool_int(control_step, label="control_step", minimum=0)
        sample = self._plan.sample(step)
        if sample.underflow:
            return ActionDecisionV5(action=None, underflow=True)
        return ActionDecisionV5(action=sample.action, underflow=False)


def make_scheduler_v5(
    mode: ExecutionModeSpec,
    calibration: Optional["LatencyCalibrationV2"],
) -> ModeSchedulerV5:
    _require_known_mode(mode)
    _require_matching_calibration(mode, calibration)
    if mode.name == "baseline_sync":
        return _BaselineSyncScheduler()
    if mode.name == "bsp_spline_sync":
        return _BspScheduler(asynchronous=False, budget_ns=None, expected_speedup=2)
    if mode.name in ("baseline_async", "baseline_async_recovery"):
        assert calibration is not None
        if calibration.derived_delay_ticks is None:
            raise ValueError("baseline async calibration is missing derived delay ticks")
        return _BaselineAsyncScheduler(
            calibration.derived_delay_ticks,
            blocking_capacity_recovery=mode.name == "baseline_async_recovery",
        )
    if mode.name == "baseline_rtc":
        assert calibration is not None
        if calibration.derived_delay_ticks is None:
            raise ValueError("RTC calibration is missing derived delay ticks")
        return _RtcScheduler(calibration.derived_delay_ticks)
    assert calibration is not None
    if calibration.derived_prefetch_budget_ns is None:
        raise ValueError("BSP calibration is missing its prefetch budget")
    return _BspScheduler(
        asynchronous=True,
        budget_ns=calibration.derived_prefetch_budget_ns,
        expected_speedup=bsp_speedup_for_mode_v5(mode.name),
        rolling_raw_latencies_ns=(
            calibration.measurement_raw_inference_latency_ns if is_native_latency_mode_v5(mode.name) else None
        ),
    )


def _require_known_mode(mode: ExecutionModeSpec) -> None:
    if not isinstance(mode, ExecutionModeSpec) or EXECUTION_MODES.get(mode.name) != mode:
        raise ValueError("mode must be one of the frozen schema-v5 execution modes")


def _require_matching_calibration(
    mode: ExecutionModeSpec,
    calibration: Optional["LatencyCalibrationV2"],
) -> None:
    if not mode.asynchronous:
        if calibration is not None:
            raise ValueError("synchronous modes must not carry latency calibration")
        return
    if not isinstance(calibration, LatencyCalibrationV2):
        raise ValueError("asynchronous modes require latency calibration")
    calibration._validate()
    if calibration.execution_mode != mode.name:
        raise ValueError("latency calibration execution mode does not match scheduler mode")


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("{} must be a boolean".format(label))
    return value


def _validate_install_inputs(
    intent: RequestIntentV5,
    *,
    now_ns: int,
    control_step: int,
) -> Tuple[int, int]:
    if not isinstance(intent, RequestIntentV5):
        raise ValueError("intent must be a RequestIntentV5")
    now = _require_nonbool_int(now_ns, label="now_ns", minimum=0)
    step = _require_nonbool_int(control_step, label="control_step", minimum=0)
    return now, step


def validate_bsp_response(response: Mapping[str, Any], *, expected_speedup: int = 2) -> bsp_spline.BspSpline:
    """Validate the legacy eight actions and exact continuous BSP sidecar together."""
    if not isinstance(response, Mapping) or "actions" not in response or "bsp" not in response:
        raise ValueError("BSP policy response must contain legacy actions and a bsp sidecar")
    try:
        actions = np.asarray(response["actions"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("BSP legacy actions must be a numeric array") from error
    if actions.shape != (8, 7):
        raise ValueError("BSP legacy actions must have exact shape (8, 7)")
    if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(actions.dtype, np.complexfloating):
        raise ValueError("BSP legacy actions must be real numeric values")
    if not np.isfinite(actions).all():
        raise ValueError("BSP legacy actions must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        float32_actions = np.asarray(actions, dtype=np.float32)
    if not np.isfinite(float32_actions).all():
        raise ValueError("BSP legacy actions must be representable as finite float32")
    return bsp_spline.BspSpline.from_response(response["bsp"], expected_speedup=expected_speedup)
