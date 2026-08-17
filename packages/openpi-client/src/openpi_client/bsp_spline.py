"""Pure NumPy BSP decoding and clock-driven action plan state."""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any, Optional

import numpy as np


_SCHEMA_FIELDS = {
    "schema_version",
    "parameters",
    "origin_hz",
    "degree",
    "speedup",
    "alignment",
}
_SCHEMA_VERSION = 1
_ORIGIN_HZ = 10
_DEGREE = 3
_SPEEDUP = 1
_ALIGNMENT = "disabled_delta_eff"
_PARAMETER_SHAPE = (16, 8)
_ACTION_DIM = 7
_CONTROL_ROWS = 12
_KNOT_EPSILON = 1e-6
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _require_exact_integer(value: Any, *, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError("{} must be the integer {}".format(name, expected))
    return value


def _require_nonnegative_ns(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(name))
    return value


def _read_only_copy(array: np.ndarray, dtype: Any) -> np.ndarray:
    copied = np.asarray(array, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclasses.dataclass(frozen=True)
class BspSpline:
    """Validated schema-one cubic B-spline with immutable retained arrays."""

    parameters: np.ndarray
    knots: np.ndarray
    controls: np.ndarray
    degree: int
    origin_hz: int
    speedup: int
    alignment: str
    t_min: float
    t_max: float

    @classmethod
    def from_response(cls, bsp_mapping: Mapping) -> "BspSpline":
        if not isinstance(bsp_mapping, Mapping):
            raise ValueError("BSP response must be a mapping")
        if set(bsp_mapping) != _SCHEMA_FIELDS:
            raise ValueError("BSP response fields must exactly match schema version 1")

        _require_exact_integer(
            bsp_mapping["schema_version"], name="schema_version", expected=_SCHEMA_VERSION
        )
        origin_hz = _require_exact_integer(
            bsp_mapping["origin_hz"], name="origin_hz", expected=_ORIGIN_HZ
        )
        degree = _require_exact_integer(bsp_mapping["degree"], name="degree", expected=_DEGREE)
        speedup = _require_exact_integer(bsp_mapping["speedup"], name="speedup", expected=_SPEEDUP)
        alignment = bsp_mapping["alignment"]
        if not isinstance(alignment, str) or alignment != _ALIGNMENT:
            raise ValueError("alignment must be {!r}".format(_ALIGNMENT))

        try:
            converted_parameters = np.asarray(bsp_mapping["parameters"], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("BSP parameters must be convertible to float32") from error
        if converted_parameters.shape != _PARAMETER_SHAPE:
            raise ValueError(
                "BSP parameters must have shape {}, got {}".format(_PARAMETER_SHAPE, converted_parameters.shape)
            )
        if not np.isfinite(converted_parameters).all():
            raise ValueError("BSP parameters must be finite")

        parameters = _read_only_copy(converted_parameters, np.float32)
        working = np.asarray(parameters, dtype=np.float64)
        knots = working[:, _ACTION_DIM].copy()
        for index in range(1, knots.size):
            if knots[index] < knots[index - 1]:
                knots[index] = knots[index - 1] + _KNOT_EPSILON
        controls = working[:_CONTROL_ROWS, :_ACTION_DIM].copy()
        t_min = float(knots[degree])
        t_max = float(knots[_CONTROL_ROWS])
        if t_max <= t_min:
            raise ValueError("Invalid B-spline range: [{}, {}]".format(t_min, t_max))

        knots.setflags(write=False)
        controls.setflags(write=False)
        return cls(
            parameters=parameters,
            knots=knots,
            controls=controls,
            degree=degree,
            origin_hz=origin_hz,
            speedup=speedup,
            alignment=alignment,
            t_min=t_min,
            t_max=t_max,
        )

    def evaluate(self, times: Any) -> np.ndarray:
        """Evaluate scalar or arbitrarily shaped times inside the closed valid interval."""
        try:
            query = np.asarray(times, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("BSP evaluation times must be convertible to float64") from error
        if not np.isfinite(query).all():
            raise ValueError("BSP evaluation times must be finite")
        if np.any(query < self.t_min) or np.any(query > self.t_max):
            raise ValueError("BSP evaluation times must stay within the closed spline interval")

        flattened = query.reshape(-1)
        evaluated = np.empty((flattened.size, _ACTION_DIM), dtype=np.float64)
        for index, value in enumerate(flattened):
            evaluated[index] = self._evaluate_scalar(float(value))
        reshaped = evaluated.reshape(query.shape + (_ACTION_DIM,))
        return np.asarray(reshaped, dtype=np.float32)

    def decode_eight(self) -> np.ndarray:
        """Decode the legacy eight inclusive samples from the continuous curve."""
        times = np.linspace(self.t_min, self.t_max, 8, dtype=np.float64)
        return self.evaluate(times)

    def _evaluate_scalar(self, value: float) -> np.ndarray:
        if value == self.t_max:
            span = _CONTROL_ROWS - 1
        else:
            span = int(np.searchsorted(self.knots, value, side="right") - 1)
        if span < self.degree or span >= _CONTROL_ROWS:
            raise ValueError("BSP evaluation times do not identify a valid spline span")

        values = self.controls[span - self.degree : span + 1].copy()
        for recurrence in range(1, self.degree + 1):
            for local_index in range(self.degree, recurrence - 1, -1):
                knot_index = span - self.degree + local_index
                denominator = (
                    self.knots[knot_index + self.degree - recurrence + 1] - self.knots[knot_index]
                )
                if denominator == 0.0:
                    alpha = 0.0
                else:
                    alpha = (value - self.knots[knot_index]) / denominator
                values[local_index] = (
                    (1.0 - alpha) * values[local_index - 1] + alpha * values[local_index]
                )
        return values[self.degree]


@dataclasses.dataclass(frozen=True)
class BspPlanSample:
    """One actionable spline sample or an explicit exhausted-plan result."""

    action: Optional[np.ndarray]
    spline_time: float
    underflow: bool


@dataclasses.dataclass(frozen=True)
class BspPrefetchDecision:
    """Pure remaining-time decision for a caller-owned inference schedule."""

    remaining_time_ns: int
    should_prefetch: bool
    underflow: bool


class BspActionPlan:
    """Own one active curve and its activation clock, but no transport or clock source."""

    def __init__(self) -> None:
        self._spline = None  # type: Optional[BspSpline]
        self._activation_time_ns = None  # type: Optional[int]

    @property
    def spline(self) -> Optional[BspSpline]:
        return self._spline

    @property
    def activation_time_ns(self) -> Optional[int]:
        return self._activation_time_ns

    def install(self, bsp_mapping: Mapping, *, activation_time_ns: int) -> BspSpline:
        """Validate then immediately replace the curve and reset its clock."""
        validated_activation = _require_nonnegative_ns(activation_time_ns, name="activation_time_ns")
        candidate = BspSpline.from_response(bsp_mapping)
        self._spline = candidate
        self._activation_time_ns = validated_activation
        return candidate

    def sample(self, now_ns: int) -> BspPlanSample:
        """Sample at wall-clock time, clamping only below the spline's lower bound."""
        validated_now = _require_nonnegative_ns(now_ns, name="now_ns")
        spline, activation_time_ns = self._require_active()
        spline_time = self._spline_time(validated_now, spline, activation_time_ns)
        if spline_time > spline.t_max:
            return BspPlanSample(action=None, spline_time=spline_time, underflow=True)
        evaluation_time = max(spline_time, spline.t_min)
        return BspPlanSample(
            action=spline.evaluate(evaluation_time),
            spline_time=evaluation_time,
            underflow=False,
        )

    def remaining_time_ns(self, now_ns: int) -> int:
        """Return wall-clock nanoseconds until the closed right endpoint."""
        validated_now = _require_nonnegative_ns(now_ns, name="now_ns")
        spline, activation_time_ns = self._require_active()
        return self._remaining_time_ns(validated_now, spline, activation_time_ns)

    def prefetch_decision(self, now_ns: int, *, lead_time_ns: int) -> BspPrefetchDecision:
        """Decide whether a caller-supplied calibrated lead time has been reached."""
        validated_now = _require_nonnegative_ns(now_ns, name="now_ns")
        validated_lead = _require_nonnegative_ns(lead_time_ns, name="lead_time_ns")
        spline, activation_time_ns = self._require_active()
        spline_time = self._spline_time(validated_now, spline, activation_time_ns)
        remaining_time_ns = self._remaining_time_ns(validated_now, spline, activation_time_ns)
        return BspPrefetchDecision(
            remaining_time_ns=remaining_time_ns,
            should_prefetch=remaining_time_ns <= validated_lead,
            underflow=spline_time > spline.t_max,
        )

    def _require_active(self):
        if self._spline is None or self._activation_time_ns is None:
            raise RuntimeError("BSP action plan has no installed curve")
        return self._spline, self._activation_time_ns

    @staticmethod
    def _spline_time(now_ns: int, spline: BspSpline, activation_time_ns: int) -> float:
        elapsed_ns = BspActionPlan._elapsed_ns(now_ns, activation_time_ns)
        return elapsed_ns * spline.origin_hz * spline.speedup / _NANOSECONDS_PER_SECOND

    @staticmethod
    def _remaining_time_ns(now_ns: int, spline: BspSpline, activation_time_ns: int) -> int:
        elapsed_ns = BspActionPlan._elapsed_ns(now_ns, activation_time_ns)
        end_offset_ns = spline.t_max * _NANOSECONDS_PER_SECOND / (spline.origin_hz * spline.speedup)
        return max(0, int(math.ceil(end_offset_ns - elapsed_ns)))

    @staticmethod
    def _elapsed_ns(now_ns: int, activation_time_ns: int) -> int:
        if now_ns < activation_time_ns:
            raise ValueError("now_ns must not precede activation_time_ns")
        return now_ns - activation_time_ns
