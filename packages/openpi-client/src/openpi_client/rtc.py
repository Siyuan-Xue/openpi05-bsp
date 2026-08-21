"""Pure client-side state for baseline real-time chunking (RTC)."""

from collections import deque
import dataclasses
import enum
import numbers
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import numpy as np

from openpi_client import inference


ACTION_HORIZON = 16
NATIVE_ACTION_DIM = 7
MODEL_ACTION_DIM = 32
MIN_START = 8
DELAY_HISTORY_SIZE = 10
MAX_INITIAL_DELAY = 8


class RtcError(RuntimeError):
    """Base class for RTC plan-state errors."""


class RtcRequestInFlightError(RtcError):
    pass


class RtcNoRequestInFlightError(RtcError):
    pass


class RtcLaunchNotReadyError(RtcError):
    pass


class RtcLaunchInfeasibleError(RtcError):
    pass


class RtcPlanExhaustedError(RtcError):
    pass


class RtcInvalidDelayError(RtcError):
    pass


class RtcPlanState(enum.Enum):
    BOOTSTRAP_REQUIRED = "bootstrap_required"
    EXECUTING = "executing"
    READY_TO_LAUNCH = "ready_to_launch"
    IN_FLIGHT = "in_flight"
    INFEASIBLE = "infeasible"
    EXHAUSTED = "exhausted"


@dataclasses.dataclass(frozen=True)
class RtcActionChunk:
    """Synchronized native and opaque normalized representations of one chunk."""

    actions: np.ndarray
    model_actions: np.ndarray

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> "RtcActionChunk":
        if not isinstance(response, Mapping):
            raise ValueError("RTC response must be a mapping")
        if "actions" not in response or "rtc" not in response:
            raise ValueError("RTC response must contain actions and rtc")
        rtc_sidecar = response["rtc"]
        if not isinstance(rtc_sidecar, Mapping):
            raise ValueError("RTC response rtc sidecar must be a mapping")
        if set(rtc_sidecar) != {"schema_version", "model_actions"}:
            raise ValueError("RTC response sidecar must contain exactly schema_version and model_actions")
        schema_version = rtc_sidecar["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, numbers.Integral)
            or int(schema_version) != inference.RTC_SCHEMA_VERSION
        ):
            raise ValueError("RTC response schema_version must be integer 1")

        actions = _validated_float32_copy(
            response["actions"],
            shape=(ACTION_HORIZON, NATIVE_ACTION_DIM),
            label="RTC native actions",
        )
        model_actions = _validated_float32_copy(
            rtc_sidecar["model_actions"],
            shape=(ACTION_HORIZON, MODEL_ACTION_DIM),
            label="RTC normalized model_actions",
        )
        return cls(actions=actions, model_actions=model_actions)


class RtcPlan:
    """Own action-chunk timing while leaving transport and clocks to the caller."""

    def __init__(self, *, d_init: int, fixed_delay: bool = False):
        self._d_init = _validate_initial_delay(d_init)
        if not isinstance(fixed_delay, bool):
            raise ValueError("fixed_delay must be a boolean")
        self._fixed_delay = fixed_delay
        self.reset()

    @property
    def state(self) -> RtcPlanState:
        if self._chunk is None:
            if self._request_kind is not None:
                return RtcPlanState.IN_FLIGHT
            return RtcPlanState.BOOTSTRAP_REQUIRED
        if self._cursor >= ACTION_HORIZON:
            return RtcPlanState.EXHAUSTED
        if self._request_kind is not None:
            return RtcPlanState.IN_FLIGHT
        delay = self.forecast_delay
        if self._cursor + delay > ACTION_HORIZON:
            return RtcPlanState.INFEASIBLE
        if self._cursor >= max(MIN_START, delay):
            return RtcPlanState.READY_TO_LAUNCH
        return RtcPlanState.EXECUTING

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def delay_history(self) -> Tuple[int, ...]:
        return tuple(self._delay_history)

    @property
    def forecast_delay(self) -> int:
        if self._fixed_delay:
            return self._d_init
        return max(self._delay_history)

    @property
    def request_in_flight(self) -> bool:
        return self._request_kind is not None

    @property
    def model_actions(self) -> np.ndarray:
        if self._chunk is None:
            raise RtcPlanExhaustedError("RTC plan has no installed chunk")
        return self._chunk.model_actions

    def reset(self, *, d_init: Optional[int] = None) -> None:
        """Discard active state and return to the synchronous bootstrap seam."""
        if d_init is not None:
            self._d_init = _validate_initial_delay(d_init)
        self._delay_history: Deque[int] = deque([self._d_init], maxlen=DELAY_HISTORY_SIZE)
        self._chunk: Optional[RtcActionChunk] = None
        self._cursor = 0
        self._request_kind: Optional[str] = None
        self._request_start_cursor: Optional[int] = None

    def begin_bootstrap(self) -> Dict[str, Dict[str, int]]:
        """Mark the schema-only n=5 bootstrap request as in flight."""
        if self._request_kind is not None:
            raise RtcRequestInFlightError("an RTC request is already in flight")
        if self._chunk is not None:
            raise RtcError("bootstrap is only valid before a chunk is installed")
        self._request_kind = "bootstrap"
        self._request_start_cursor = 0
        return {
            inference.RTC_REQUEST_KEY: {
                "schema_version": inference.RTC_SCHEMA_VERSION,
            }
        }

    def consume_action(self) -> np.ndarray:
        """Return the next native action and advance the installed-chunk cursor."""
        if self._chunk is None or self._cursor >= ACTION_HORIZON:
            raise RtcPlanExhaustedError("RTC action plan is exhausted")
        action = self._chunk.actions[self._cursor]
        self._cursor += 1
        return action

    def begin_guided(self) -> Dict[str, Dict[str, Any]]:
        """Capture a feasible guided request from the current synchronized chunk."""
        if self._request_kind is not None:
            raise RtcRequestInFlightError("an RTC request is already in flight")
        if self._chunk is None:
            raise RtcLaunchNotReadyError("RTC bootstrap has not installed a chunk")
        if self._cursor >= ACTION_HORIZON:
            raise RtcPlanExhaustedError("RTC action plan is exhausted")

        delay = self.forecast_delay
        if self._cursor + delay > ACTION_HORIZON:
            raise RtcLaunchInfeasibleError("RTC launch cannot satisfy cursor + delay <= 16")
        if self._cursor < max(MIN_START, delay):
            raise RtcLaunchNotReadyError("RTC launch threshold has not been reached")

        previous = self._chunk.model_actions.copy()
        previous.setflags(write=False)
        self._request_kind = "guided"
        self._request_start_cursor = self._cursor
        return {
            inference.RTC_REQUEST_KEY: {
                "schema_version": inference.RTC_SCHEMA_VERSION,
                "previous_model_actions": previous,
                "s": self._cursor,
                "d": delay,
            }
        }

    def install_result(self, response: Mapping[str, Any]) -> None:
        """Validate and immediately install the response for the outstanding request."""
        if self._request_kind is None or self._request_start_cursor is None:
            raise RtcNoRequestInFlightError("there is no RTC request in flight")
        chunk = RtcActionChunk.from_response(response)

        if self._request_kind == "bootstrap":
            self._chunk = chunk
            self._cursor = 0
            self._request_kind = None
            self._request_start_cursor = None
            return

        actual_delay = self._cursor - self._request_start_cursor
        if actual_delay < 0 or actual_delay >= ACTION_HORIZON:
            raise RtcInvalidDelayError("RTC actual delay must satisfy 0 <= q < 16")
        self._chunk = chunk
        self._cursor = actual_delay
        self._delay_history.append(actual_delay)
        self._request_kind = None
        self._request_start_cursor = None


class RawAsyncPlan(RtcPlan):
    """Baseline async timing with no RTC continuity guidance."""

    def __init__(self, *, d_init: int):
        super().__init__(d_init=d_init, fixed_delay=True)

    def begin_background(self) -> Dict[str, Dict[str, int]]:
        if self._request_kind is not None:
            raise RtcRequestInFlightError("an async request is already in flight")
        if self._chunk is None:
            raise RtcLaunchNotReadyError("async bootstrap has not installed a chunk")
        if self._cursor >= ACTION_HORIZON:
            raise RtcPlanExhaustedError("async action plan is exhausted")

        delay = self.forecast_delay
        if self._cursor + delay > ACTION_HORIZON:
            raise RtcLaunchInfeasibleError("async launch cannot satisfy cursor + delay <= 16")
        if self._cursor < max(MIN_START, delay):
            raise RtcLaunchNotReadyError("async launch threshold has not been reached")

        self._request_kind = "raw_background"
        self._request_start_cursor = self._cursor
        return {
            inference.RTC_REQUEST_KEY: {
                "schema_version": inference.RTC_SCHEMA_VERSION,
            }
        }

    def begin_blocking_replan(self) -> Dict[str, Dict[str, int]]:
        """Replace an infeasible raw chunk without advancing the environment."""
        if self._request_kind is not None:
            raise RtcRequestInFlightError("an async request is already in flight")
        if self.state is not RtcPlanState.INFEASIBLE:
            raise RtcLaunchNotReadyError("blocking async recovery requires an infeasible chunk")
        self._request_kind = "raw_blocking_replan"
        self._request_start_cursor = self._cursor
        return {
            inference.RTC_REQUEST_KEY: {
                "schema_version": inference.RTC_SCHEMA_VERSION,
            }
        }

    def install_result(self, response: Mapping[str, Any]) -> None:
        if self._request_kind != "raw_blocking_replan":
            super().install_result(response)
            return
        if self._request_start_cursor is None or self._cursor != self._request_start_cursor:
            raise RtcInvalidDelayError("blocking async recovery cannot advance the action cursor")
        chunk = RtcActionChunk.from_response(response)
        self._chunk = chunk
        self._cursor = 0
        self._request_kind = None
        self._request_start_cursor = None


def _validated_float32_copy(value: Any, *, shape: Tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.asarray(array, dtype=np.float32).copy()
    if not np.isfinite(copied).all():
        raise ValueError(f"{label} must be representable as finite float32 values")
    copied.setflags(write=False)
    return copied


def _validate_initial_delay(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("d_init must be an integer")
    delay = int(value)
    if delay < 0 or delay > MAX_INITIAL_DELAY:
        raise ValueError("d_init must satisfy 0 <= d_init <= 8")
    return delay
