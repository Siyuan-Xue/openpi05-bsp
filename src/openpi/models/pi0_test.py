import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as _pi0_config
from openpi.models import pi0


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
