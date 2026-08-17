import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.models import pi0
from openpi.shared import nnx_utils


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_rtc_uses_the_five_openpi_times_and_mapped_gamma_literals():
    times = pi0.rtc_sample_times()
    gammas = pi0.rtc_guidance_scales(times)

    np.testing.assert_allclose(times, [1.0, 0.8, 0.6, 0.4, 0.2], rtol=0, atol=1e-7)
    np.testing.assert_allclose(1 - times, [0.0, 0.2, 0.4, 0.6, 0.8], rtol=0, atol=1e-7)
    np.testing.assert_allclose(gammas, [5.0, 4.25, 13 / 6, 13 / 6, 4.25], rtol=0, atol=1e-6)


def test_rtc_negative_guidance_sign_moves_an_openpi_step_toward_the_target():
    x = jnp.asarray([[[0.0]]])
    target = jnp.asarray([[[1.0]]])
    weights = jnp.ones_like(x)

    velocity = pi0.rtc_guided_velocity(
        lambda current, _time: jnp.zeros_like(current),
        x,
        jnp.asarray(1.0),
        target,
        weights,
        jnp.asarray(1.0),
    )
    updated = pi0.rtc_euler_step(x, velocity)

    np.testing.assert_allclose(velocity, [[[-1.0]]])
    np.testing.assert_allclose(updated, [[[0.2]]])


def test_rtc_vjp_keeps_cross_coordinate_guidance_from_a_masked_output_dimension():
    x = jnp.asarray([[[0.0, 0.0]]])
    target = jnp.asarray([[[2.0, 0.0]]])
    weights = jnp.asarray([[[1.0, 0.0]]])

    def coupled_velocity(current, _time):
        return jnp.stack((current[..., 0] + current[..., 1], jnp.zeros_like(current[..., 1])), axis=-1)

    velocity = pi0.rtc_guided_velocity(
        coupled_velocity,
        x,
        jnp.asarray(0.5),
        target,
        weights,
        jnp.asarray(1.0),
    )

    np.testing.assert_allclose(velocity, [[[-1.0, 1.0]]])
    assert velocity[0, 0, 1] != 0


def test_rtc_zero_cotangent_preserves_the_ordinary_euler_update():
    x = jnp.asarray([[[2.0, -1.0]]])
    ordinary_velocity = jnp.asarray([[[0.25, -0.5]]])
    target = jnp.asarray([[[99.0, 99.0]]])

    guided_velocity = pi0.rtc_guided_velocity(
        lambda _current, _time: ordinary_velocity,
        x,
        jnp.asarray(0.6),
        target,
        jnp.zeros_like(x),
        jnp.asarray(13 / 6),
    )

    np.testing.assert_allclose(guided_velocity, ordinary_velocity)
    np.testing.assert_allclose(pi0.rtc_euler_step(x, guided_velocity), x - ordinary_velocity / 5)


def test_rtc_target_right_pads_the_full_previous_chunk_and_masks_only_first_seven_dimensions():
    previous = jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)

    target, weights = pi0.make_rtc_target_and_weights(previous, s=10, d=2)

    np.testing.assert_array_equal(target[:6], np.asarray(previous[10:16]))
    np.testing.assert_array_equal(target[6:], np.zeros((10, 32), dtype=np.float32))
    expected_time_weights = np.asarray(
        [
            1.0,
            1.0,
            0.8 * np.expm1(0.8) / np.expm1(1.0),
            0.6 * np.expm1(0.6) / np.expm1(1.0),
            0.4 * np.expm1(0.4) / np.expm1(1.0),
            0.2 * np.expm1(0.2) / np.expm1(1.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(weights[:, 0], expected_time_weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(weights[:, 6], expected_time_weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(weights[:, 7:], np.zeros((16, 25), dtype=np.float32))


def test_rtc_target_and_mask_cover_horizon_boundaries():
    previous = jnp.ones((16, 32), dtype=jnp.float32)

    target_full, weights_full = pi0.make_rtc_target_and_weights(previous, s=8, d=8)
    target_empty, weights_empty = pi0.make_rtc_target_and_weights(previous, s=16, d=0)

    np.testing.assert_array_equal(target_full[:8], np.ones((8, 32), dtype=np.float32))
    np.testing.assert_array_equal(target_full[8:], np.zeros((8, 32), dtype=np.float32))
    np.testing.assert_array_equal(weights_full[:8, :7], np.ones((8, 7), dtype=np.float32))
    np.testing.assert_array_equal(weights_full[8:], np.zeros((8, 32), dtype=np.float32))
    np.testing.assert_array_equal(target_empty, np.zeros((16, 32), dtype=np.float32))
    np.testing.assert_array_equal(weights_empty, np.zeros((16, 32), dtype=np.float32))


class _TinyImageEncoder(nnx.Module):
    """Minimal image encoder used to exercise Pi0's real sampling internals."""

    def __init__(self, width):
        self.width = width

    def __call__(self, image, *, train=False):
        del train
        image_mean = jnp.mean(image, axis=(1, 2, 3), keepdims=False)
        tokens = jnp.broadcast_to(image_mean[:, None, None], (image.shape[0], 1, self.width))
        return tokens, None


class _TinyPrefixSuffixLlm(nnx.Module):
    """Stateless prefix/cache/suffix surface matching the Pi0 LLM calls."""

    def __init__(self, width):
        self.width = width

    def __call__(
        self,
        inputs,
        *,
        mask=None,
        positions=None,
        kv_cache=None,
        adarms_cond=None,
        method=None,
    ):
        del mask, positions, adarms_cond
        if method == "embed":
            token_values = jnp.asarray(inputs, dtype=jnp.float32) / 100.0
            return jnp.broadcast_to(token_values[..., None], (*token_values.shape, self.width))

        prefix, suffix = inputs
        if suffix is None:
            cache = jnp.mean(prefix, axis=1, keepdims=True)
            return (prefix, None), cache
        assert prefix is None
        assert kv_cache is not None
        return (None, suffix + jnp.tanh(kv_cache)), kv_cache


class _TinySamplerPi0(pi0.Pi0):
    """H16/D32 Pi0 with tiny components but the production sampler methods."""

    def __init__(self, rng):
        _model.BaseModel.__init__(self, action_dim=32, action_horizon=16, max_token_len=2)
        self.pi05 = True
        width = 8
        rngs = nnx.Rngs(rng)
        self.PaliGemma = nnx.Dict(
            llm=_TinyPrefixSuffixLlm(width),
            img=_TinyImageEncoder(width),
        )
        self.action_in_proj = nnx.Linear(32, width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(width, width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(width, width, rngs=rngs)
        self.action_out_proj = nnx.Linear(width, 32, rngs=rngs)
        self.deterministic = True


def _tiny_sampler_observation():
    batch_size = 1
    images = {
        key: jnp.full((batch_size, 224, 224, 3), index / 10.0, dtype=jnp.float32)
        for index, key in enumerate(_model.IMAGE_KEYS, start=1)
    }
    return _model.Observation(
        images=images,
        image_masks={key: jnp.ones((batch_size,), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
        state=jnp.zeros((batch_size, 32), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray([[3, 7]], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 2), dtype=jnp.bool_),
    )


def _pre_refactor_sample_actions_reference(model, rng, observation, *, num_steps, noise):
    """Frozen test-local copy of Pi0.sample_actions before velocity factoring."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation,
            x_t,
            jnp.broadcast_to(time, batch_size),
        )
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        return x_t + dt * velocity, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    result, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return result


def _five_step_rtc_reference(model, rng, observation, *, target, weights, noise):
    observation, x_t, prefix_tokens, prefix_mask, kv_cache = model._prepare_sampling(
        rng,
        observation,
        noise=noise,
    )
    for time, gamma in zip(
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [5.0, 4.25, 13 / 6, 13 / 6, 4.25],
        strict=True,
    ):
        time = jnp.asarray(time, dtype=jnp.float32)

        def velocity_fn(current, current_time):
            return model._action_velocity(
                observation,
                current,
                current_time,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )

        velocity = pi0.rtc_guided_velocity(
            velocity_fn,
            x_t,
            time,
            target,
            weights,
            jnp.asarray(gamma, dtype=jnp.float32),
        )
        x_t = pi0.rtc_euler_step(x_t, velocity)
    return x_t


@pytest.mark.gpu
def test_actual_jitted_pi0_legacy_sampler_matches_frozen_pre_refactor_reference():
    key = jax.random.key(314)
    model = _TinySamplerPi0(key)
    observation = _tiny_sampler_observation()
    noise = jnp.linspace(-0.5, 0.5, 16 * 32, dtype=jnp.float32).reshape(1, 16, 32)

    expected = _pre_refactor_sample_actions_reference(
        model,
        key,
        observation,
        num_steps=4,
        noise=noise,
    )
    actual = nnx_utils.module_jit(model.sample_actions)(
        key,
        observation,
        num_steps=4,
        noise=noise,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_actual_jitted_pi0_rtc_sampler_runs_fixed_five_step_vjp_with_shared_prefix_cache():
    key = jax.random.key(2718)
    model = _TinySamplerPi0(key)
    observation = _tiny_sampler_observation()
    noise = jnp.linspace(-0.25, 0.75, 16 * 32, dtype=jnp.float32).reshape(1, 16, 32)
    previous = jnp.linspace(-1.0, 1.0, 16 * 32, dtype=jnp.float32).reshape(16, 32)
    target, weights = pi0.make_rtc_target_and_weights(previous, s=9, d=3)
    target = target[None, ...]

    expected = _five_step_rtc_reference(
        model,
        key,
        observation,
        target=target,
        weights=weights,
        noise=noise,
    )
    actual = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        observation,
        target=target,
        weights=weights,
        noise=noise,
    )

    assert actual.shape == (1, 16, 32)
    assert jnp.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
