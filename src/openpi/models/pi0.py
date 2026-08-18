import logging
import numbers

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

RTC_ACTION_HORIZON = 16
RTC_MODEL_ACTION_DIM = 32
RTC_NATIVE_ACTION_DIM = 7
RTC_NUM_STEPS = 5
RTC_BETA = 5.0
RTC_MIN_START = 8


def rtc_sample_times():
    """Return the fixed OpenPI ``t=1 -> 0`` RTC evaluation schedule."""
    return jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2], dtype=jnp.float32)


def rtc_guidance_scales(time):
    """Map RTC's paper-time coefficient onto OpenPI time and clip by beta."""
    time = jnp.asarray(time)
    denominator = time * (1.0 - time)
    at_noise_endpoint = time == 1.0
    safe_denominator = jnp.where(at_noise_endpoint, 1.0, denominator)
    unclipped = ((1.0 - time) ** 2 + time**2) / safe_denominator
    return jnp.where(at_noise_endpoint, RTC_BETA, jnp.minimum(RTC_BETA, unclipped))


def make_rtc_target_and_weights(previous_model_actions, *, s: int, d: int):
    """Prepare Algorithm 1's right-padded target and Equation 5 mask."""
    previous = jnp.asarray(previous_model_actions)
    if previous.shape != (RTC_ACTION_HORIZON, RTC_MODEL_ACTION_DIM):
        raise ValueError("previous_model_actions must have shape (16, 32)")
    if isinstance(s, bool) or not isinstance(s, numbers.Integral):
        raise ValueError("s must be a non-boolean integer")
    if isinstance(d, bool) or not isinstance(d, numbers.Integral):
        raise ValueError("d must be a non-boolean integer")
    s = int(s)
    d = int(d)
    if not RTC_MIN_START <= s <= RTC_ACTION_HORIZON:
        raise ValueError("s must satisfy 8 <= s <= 16")
    if not 0 <= d <= s or s + d > RTC_ACTION_HORIZON:
        raise ValueError("d must satisfy 0 <= d <= s and s + d <= 16")

    indices = jnp.arange(RTC_ACTION_HORIZON)
    overlap = RTC_ACTION_HORIZON - s
    source_indices = jnp.minimum(indices + s, RTC_ACTION_HORIZON - 1)
    target = previous[source_indices]
    target = jnp.where((indices < overlap)[:, None], target, jnp.zeros_like(target))

    denominator = overlap - d + 1
    c = (overlap - indices) / denominator
    soft_weights = c * jnp.expm1(c) / jnp.expm1(jnp.asarray(1.0, dtype=previous.dtype))
    time_weights = jnp.where(indices < d, 1.0, jnp.where(indices < overlap, soft_weights, 0.0))
    dimension_mask = jnp.arange(RTC_MODEL_ACTION_DIM) < RTC_NATIVE_ACTION_DIM
    weights = time_weights[:, None] * dimension_mask[None, :]
    return target, weights.astype(previous.dtype)


def rtc_guided_velocity(velocity_fn, x_t, time, target, weights, gamma):
    """Apply the mapped RTC VJP correction without truncating input gradients."""

    def denoised(current):
        velocity = velocity_fn(current, time)
        return current - time * velocity, velocity

    x_hat_0, pullback, velocity = jax.vjp(denoised, x_t, has_aux=True)
    cotangent = (target - x_hat_0) * weights
    guidance = pullback(cotangent)[0]
    return velocity - gamma * guidance


def rtc_euler_step(x_t, guided_velocity):
    """Take one of the five fixed negative OpenPI Euler steps."""
    return x_t + (-1.0 / RTC_NUM_STEPS) * guided_velocity


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @property
    @override
    def supports_rtc(self) -> bool:
        return self.action_horizon == RTC_ACTION_HORIZON and self.action_dim == RTC_MODEL_ACTION_DIM

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation, noise, prefix_tokens, prefix_mask, kv_cache = self._prepare_sampling(
            rng, observation, noise=noise
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            v_t = self._action_velocity(
                observation,
                x_t,
                time,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        target: at.Float[at.Array, "b 16 32"],
        weights: at.Float[at.Array, "16 32"],
        noise: at.Float[at.Array, "b 16 32"] | None = None,
    ) -> _model.Actions:
        """Sample with fixed n=5, beta=5 RTC guidance in normalized flow space."""
        if not self.supports_rtc:
            raise ValueError("RTC sampling requires model action shape (16, 32)")
        observation, noise, prefix_tokens, prefix_mask, kv_cache = self._prepare_sampling(
            rng, observation, noise=noise
        )
        if target.shape != noise.shape:
            raise ValueError(f"RTC target must have shape {noise.shape}, got {target.shape}")
        if weights.shape != (RTC_ACTION_HORIZON, RTC_MODEL_ACTION_DIM):
            raise ValueError("RTC weights must have shape (16, 32)")
        times = rtc_sample_times()
        gammas = rtc_guidance_scales(times)

        def step(index, x_t):
            time = times[index]

            def velocity_fn(current, current_time):
                return self._action_velocity(
                    observation,
                    current,
                    current_time,
                    prefix_tokens=prefix_tokens,
                    prefix_mask=prefix_mask,
                    kv_cache=kv_cache,
                )

            guided_velocity = rtc_guided_velocity(
                velocity_fn,
                x_t,
                time,
                target,
                weights,
                gammas[index],
            )
            return rtc_euler_step(x_t, guided_velocity)

        return jax.lax.fori_loop(0, RTC_NUM_STEPS, step, noise)

    def _prepare_sampling(self, rng, observation, *, noise):
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill the prefix KV cache once and share it across every denoising step.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return observation, noise, prefix_tokens, prefix_mask, kv_cache

    def _action_velocity(
        self,
        observation,
        x_t,
        time,
        *,
        prefix_tokens,
        prefix_mask,
        kv_cache,
    ):
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])
