from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import inference as _inference
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy
_MISSING = object()


def _select_jax_inference_rng(
    stateful_rng: at.KeyArrayLike, inference_seed: int | None
) -> tuple[at.KeyArrayLike, at.KeyArrayLike]:
    """Select sampling RNG without advancing state for an explicit request seed."""
    if inference_seed is not None:
        return stateful_rng, jax.random.key(inference_seed)
    next_rng, sample_rng = jax.random.split(stateful_rng)
    return next_rng, sample_rng


def _inference_capabilities(
    *,
    action_representation: str,
    model_action_horizon: int,
    model_action_dim: int,
    has_rtc_hook: bool,
) -> dict[str, Any]:
    if (
        action_representation == "native"
        and (model_action_horizon, model_action_dim) == (16, 32)
        and has_rtc_hook
    ):
        supported_protocols = ["baseline_h16_n5_v1", "baseline_rtc_h16_v1"]
    elif action_representation == "bsp":
        supported_protocols = ["bsp_spline_h8_v1"]
    else:
        supported_protocols = []
    return {
        "schema_version": 1,
        "action_representation": action_representation,
        "model_action_horizon": model_action_horizon,
        "model_action_dim": model_action_dim,
        "supported_protocols": supported_protocols,
    }


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        action_representation: str,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX sampling.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            action_representation: Explicitly identifies native actions versus BSP parameters.
        """
        if action_representation not in ("native", "bsp"):
            raise ValueError("action_representation must be 'native' or 'bsp'")
        metadata = dict(metadata or {})
        if _inference.INFERENCE_CAPABILITIES_KEY in metadata:
            raise ValueError(f"{_inference.INFERENCE_CAPABILITIES_KEY} is reserved policy metadata")

        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._action_representation = action_representation
        self._model_action_horizon = int(model.action_horizon)
        self._model_action_dim = int(model.action_dim)
        has_rtc_hook = bool(getattr(model, "supports_rtc", False)) and callable(
            getattr(model, "sample_actions_rtc", None)
        )
        self._sample_actions_rtc = nnx_utils.module_jit(model.sample_actions_rtc) if has_rtc_hook else None
        metadata[_inference.INFERENCE_CAPABILITIES_KEY] = _inference_capabilities(
            action_representation=action_representation,
            model_action_horizon=self._model_action_horizon,
            model_action_dim=self._model_action_dim,
            has_rtc_hook=has_rtc_hook,
        )
        self._metadata = metadata
        self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # The seed belongs to the request envelope, not the model observation.
        # pop_inference_seed copies the top-level mapping so the caller is not mutated.
        inputs, inference_seed = _inference.pop_inference_seed(obs)
        inputs, rtc_context = _inference.pop_rtc_context(inputs)
        rtc_sample_call = None
        if rtc_context is not None:
            self._validate_rtc_capability()
            rtc_sample_call = self._prepare_rtc_sample_call(rtc_context, noise=noise)
        # Make a tree copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = _select_jax_inference_rng(self._rng, inference_seed)

        if rtc_sample_call is None:
            # Preserve the legacy callable and kwargs preparation path exactly.
            sample_actions = self._sample_actions
            sample_kwargs = dict(self._sample_kwargs)
            if noise is not None:
                noise = jnp.asarray(noise)
                if noise.ndim == 2:
                    noise = noise[None, ...]
                sample_kwargs["noise"] = noise
        else:
            sample_actions, sample_kwargs = rtc_sample_call

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": sample_actions(sample_rng, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        rtc_model_actions = None
        if rtc_context is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                rtc_model_actions = np.asarray(outputs["actions"], dtype=np.float32).copy()
            if rtc_model_actions.shape != (16, 32):
                raise ValueError(f"RTC model output must have shape (16, 32), got {rtc_model_actions.shape}")
            if not np.isfinite(rtc_model_actions).all():
                raise ValueError("RTC model output must be representable as finite float32 values")

        outputs = self._output_transform(outputs)
        if rtc_model_actions is not None:
            outputs["rtc"] = {
                "schema_version": _inference.RTC_SCHEMA_VERSION,
                "model_actions": rtc_model_actions,
            }
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _validate_rtc_capability(self) -> None:
        if getattr(self, "_action_representation", None) != "native":
            raise ValueError("RTC requests require native action representation")
        model_shape = (
            getattr(self, "_model_action_horizon", None),
            getattr(self, "_model_action_dim", None),
        )
        if model_shape != (16, 32):
            raise ValueError("RTC requests require model action shape (16, 32)")
        if getattr(self, "_sample_actions_rtc", None) is None:
            raise ValueError("RTC requests require a model RTC sampling hook")

    def _prepare_rtc_sample_call(self, rtc_context, *, noise):
        """Validate RTC-only kwargs and targets before advancing stateful RNG."""
        sample_kwargs = dict(self._sample_kwargs)
        configured_noise = sample_kwargs.pop("noise", _MISSING)
        rtc_noise = noise if noise is not None else configured_noise
        if rtc_noise is not _MISSING:
            sample_kwargs["noise"] = _canonicalize_rtc_noise(rtc_noise)

        unsupported_kwargs = set(sample_kwargs) - {"num_steps", "noise"}
        if unsupported_kwargs:
            names = ", ".join(sorted(unsupported_kwargs))
            raise ValueError(f"RTC does not support configured sampler kwargs: {names}")
        if rtc_context.is_bootstrap:
            sample_kwargs["num_steps"] = _pi0.RTC_NUM_STEPS
            return self._sample_actions, sample_kwargs

        assert rtc_context.previous_model_actions is not None
        assert rtc_context.s is not None
        assert rtc_context.d is not None
        target, weights = _pi0.make_rtc_target_and_weights(
            rtc_context.previous_model_actions,
            s=rtc_context.s,
            d=rtc_context.d,
        )
        sample_kwargs.pop("num_steps", None)
        sample_kwargs["target"] = target[None, ...]
        sample_kwargs["weights"] = weights
        assert self._sample_actions_rtc is not None
        return self._sample_actions_rtc, sample_kwargs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _canonicalize_rtc_noise(noise) -> jax.Array:
    """Validate RTC noise without changing the legacy request path."""
    try:
        array = np.asarray(noise)
    except (TypeError, ValueError) as error:
        raise ValueError("RTC noise must be a finite numeric array") from error
    if array.shape not in ((16, 32), (1, 16, 32)):
        raise ValueError("RTC noise must have shape (16, 32) or (1, 16, 32)")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("RTC noise must be a finite numeric array")
    if not np.isfinite(array).all():
        raise ValueError("RTC noise must be finite")
    if array.ndim == 2:
        array = array[None, ...]
    return jnp.asarray(array)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
