"""Small, dependency-free helpers shared by policy clients and servers."""

from __future__ import annotations

from collections.abc import Mapping
import numbers
from typing import Any


# This key is part of the websocket request envelope, not a model observation.
# Policy implementations must remove it before running observation transforms.
INFERENCE_SEED_KEY = "__openpi_inference_seed"


def pop_inference_seed(observation: Mapping[str, Any]) -> tuple[dict[str, Any], int | None]:
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
