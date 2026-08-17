"""Small, dependency-free helpers shared by policy clients and servers."""

import dataclasses
import numbers
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


# These keys are part of the websocket envelope, not a model observation.
# Policy implementations must remove them before running observation transforms.
INFERENCE_SEED_KEY = "__openpi_inference_seed"
RTC_REQUEST_KEY = "__openpi_rtc"
INFERENCE_CAPABILITIES_KEY = "__openpi_inference_capabilities"

RTC_SCHEMA_VERSION = 1
RTC_ACTION_HORIZON = 16
RTC_MODEL_ACTION_DIM = 32
RTC_MIN_START = 8


@dataclasses.dataclass(frozen=True)
class RtcInferenceContext:
    """Validated RTC request state in normalized OpenPI action space."""

    schema_version: int
    previous_model_actions: Optional[np.ndarray] = None
    s: Optional[int] = None
    d: Optional[int] = None

    @property
    def is_bootstrap(self) -> bool:
        return self.previous_model_actions is None

    @property
    def is_guided(self) -> bool:
        return self.previous_model_actions is not None


def pop_inference_seed(observation: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[int]]:
    """Copy a request and remove its optional uint32 stateless sampling seed."""
    inputs = dict(observation)
    value = inputs.pop(INFERENCE_SEED_KEY, None)
    if value is None:
        return inputs, None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{INFERENCE_SEED_KEY} must be an integer in [0, 2**32)")
    seed = int(value)
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"{INFERENCE_SEED_KEY} must be an integer in [0, 2**32)")
    return inputs, seed


def pop_rtc_context(
    observation: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[RtcInferenceContext]]:
    """Copy a request, remove its RTC envelope, and validate it strictly."""
    inputs = dict(observation)
    if RTC_REQUEST_KEY not in inputs:
        return inputs, None
    value = inputs.pop(RTC_REQUEST_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{RTC_REQUEST_KEY} must be a mapping")

    keys = set(value)
    bootstrap_keys = {"schema_version"}
    guided_keys = {"schema_version", "previous_model_actions", "s", "d"}
    if keys not in (bootstrap_keys, guided_keys):
        raise ValueError(f"{RTC_REQUEST_KEY} must be exactly a bootstrap or guided RTC context")

    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, numbers.Integral)
        or int(schema_version) != RTC_SCHEMA_VERSION
    ):
        raise ValueError(f"{RTC_REQUEST_KEY}.schema_version must be integer 1")

    if keys == bootstrap_keys:
        return inputs, RtcInferenceContext(schema_version=RTC_SCHEMA_VERSION)

    previous = value["previous_model_actions"]
    if not isinstance(previous, np.ndarray):
        raise ValueError(f"{RTC_REQUEST_KEY}.previous_model_actions must be a numpy ndarray")
    if previous.dtype != np.float32:
        raise ValueError(f"{RTC_REQUEST_KEY}.previous_model_actions must have dtype float32")
    if previous.shape != (RTC_ACTION_HORIZON, RTC_MODEL_ACTION_DIM):
        raise ValueError(
            f"{RTC_REQUEST_KEY}.previous_model_actions must have shape "
            f"({RTC_ACTION_HORIZON}, {RTC_MODEL_ACTION_DIM})"
        )
    if not np.isfinite(previous).all():
        raise ValueError(f"{RTC_REQUEST_KEY}.previous_model_actions must be finite")

    s = _require_nonbool_integer(value["s"], label=f"{RTC_REQUEST_KEY}.s")
    d = _require_nonbool_integer(value["d"], label=f"{RTC_REQUEST_KEY}.d")
    if not RTC_MIN_START <= s <= RTC_ACTION_HORIZON:
        raise ValueError(f"{RTC_REQUEST_KEY}.s must satisfy 8 <= s <= 16")
    if not 0 <= d <= s:
        raise ValueError(f"{RTC_REQUEST_KEY}.d must satisfy 0 <= d <= s")
    if s + d > RTC_ACTION_HORIZON:
        raise ValueError(f"{RTC_REQUEST_KEY} must satisfy s + d <= 16")

    previous_copy = previous.copy()
    previous_copy.setflags(write=False)
    return inputs, RtcInferenceContext(
        schema_version=RTC_SCHEMA_VERSION,
        previous_model_actions=previous_copy,
        s=s,
        d=d,
    )


def _require_nonbool_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)
