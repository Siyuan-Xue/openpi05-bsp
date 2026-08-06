import dataclasses
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import train_planning
from openpi.training import utils as training_utils

from . import train


class _DeterministicLinearModel(_model.BaseModel):
    """Tiny RNG-independent model for exact accumulation semantics."""

    def __init__(self):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([0.25, -0.5], dtype=jnp.float32))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state @ self.kernel.value
        target = actions[:, 0, 0]
        return jnp.square(prediction - target)[:, None]

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return (observation.state @ self.kernel.value)[:, None, None]


def _make_deterministic_train_state(config):
    model = _DeterministicLinearModel()
    model_def, params = nnx.split(model)
    tx = optax.adam(learning_rate=0.05)
    return training_utils.TrainState(
        step=jnp.asarray(7, dtype=jnp.int32),
        params=params,
        model_def=model_def,
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=0.8,
        ema_params=params,
    )


def _assert_trees_allclose(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        micro_batch_size=1,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)
    checkpoint_dir = tmp_path / "checkpoint" / config_name / "test"
    assert (checkpoint_dir / "2").is_dir()
    assert not (checkpoint_dir / "1").exists()

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
    assert (checkpoint_dir / "4").is_dir()
    assert not (checkpoint_dir / "3").exists()


def test_none_and_equal_micro_batch_training_are_equivalent(tmp_path: pathlib.Path):
    params = []
    for exp_name, micro_batch_size in (("implicit", None), ("explicit", 2)):
        config = dataclasses.replace(
            _config._CONFIGS_DICT["debug"],  # noqa: SLF001
            batch_size=2,
            micro_batch_size=micro_batch_size,
            checkpoint_base_dir=str(tmp_path / "checkpoint"),
            exp_name=exp_name,
            overwrite=True,
            resume=False,
            num_train_steps=1,
            log_interval=1,
            save_interval=1,
        )
        train.main(config)
        checkpoint = tmp_path / "checkpoint" / "debug" / exp_name / "1" / "params"
        params.append(_model.restore_params(checkpoint, restore_type=np.ndarray))

    for implicit, explicit in zip(jax.tree.leaves(params[0]), jax.tree.leaves(params[1]), strict=True):
        np.testing.assert_array_equal(implicit, explicit)


def test_two_micro_batches_match_one_direct_batch_and_advance_state_once():
    config = _config._CONFIGS_DICT["debug"]  # noqa: SLF001
    accumulation_plan = train_planning.plan_gradient_accumulation(
        batch_size=4,
        micro_batch_size=2,
        process_count=1,
        device_count=1,
    )
    direct_state = _make_deterministic_train_state(config)
    accumulated_state = _make_deterministic_train_state(config)
    old_params = jax.tree.map(jnp.array, accumulated_state.params)
    old_ema = jax.tree.map(jnp.array, accumulated_state.ema_params)
    assert int(accumulated_state.opt_state[0].count) == 0
    rng = jax.random.key(123)
    states = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0.5, -1.0, 0.25, 1.5], dtype=jnp.float32)[:, None, None]
    observation = _model.Observation(images={}, image_masks={}, state=states)

    direct_loss, direct_grads = train.compute_microbatch_grad(
        config,
        rng,
        direct_state,
        (observation, actions),
        jnp.asarray(0, dtype=jnp.uint32),
        accumulation_steps=1,
    )
    direct_next, direct_info = train.apply_optimizer_step(
        config,
        direct_state,
        direct_grads,
        direct_loss,
        accumulation_steps=1,
    )

    accumulated_result = None
    for accumulation_index in accumulation_plan.accumulation_indices:
        start = accumulation_index * accumulation_plan.micro_batch_size
        micro_observation = _model.Observation(
            images={},
            image_masks={},
            state=states[start : start + accumulation_plan.micro_batch_size],
        )
        micro_result = train.compute_microbatch_grad(
            config,
            rng,
            accumulated_state,
            (micro_observation, actions[start : start + accumulation_plan.micro_batch_size]),
            jnp.asarray(accumulation_index, dtype=jnp.uint32),
            accumulation_steps=accumulation_plan.accumulation_steps,
        )
        accumulated_result = (
            micro_result
            if accumulated_result is None
            else train.add_microbatch_results(accumulated_result, micro_result)
        )

    assert accumulated_result is not None
    accumulated_loss_sum, accumulated_gradient_sum = accumulated_result
    accumulated_grads = train_planning.average_tree_sum(
        accumulated_gradient_sum,
        accumulation_plan.accumulation_steps,
        tree_map=jax.tree.map,
    )
    accumulated_next, accumulated_info = train.apply_optimizer_step(
        config,
        accumulated_state,
        accumulated_gradient_sum,
        accumulated_loss_sum,
        accumulation_steps=accumulation_plan.accumulation_steps,
    )

    _assert_trees_allclose(accumulated_grads, direct_grads)
    _assert_trees_allclose(accumulated_next.params, direct_next.params)
    _assert_trees_allclose(accumulated_next.opt_state, direct_next.opt_state)
    _assert_trees_allclose(accumulated_next.ema_params, direct_next.ema_params)
    np.testing.assert_allclose(accumulated_info["loss"], direct_info["loss"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(accumulated_info["grad_norm"], direct_info["grad_norm"], rtol=1e-6, atol=1e-6)

    assert int(accumulated_next.step) == 8
    assert int(accumulated_next.opt_state[0].count) == 1
    assert float(accumulated_info["grad_norm"]) > 0.0
    assert any(
        not np.allclose(old_leaf, new_leaf)
        for old_leaf, new_leaf in zip(
            jax.tree.leaves(old_params),
            jax.tree.leaves(accumulated_next.params),
            strict=True,
        )
    )
    expected_ema = jax.tree.map(
        lambda old, new: 0.8 * old + 0.2 * new,
        old_ema,
        accumulated_next.params,
    )
    _assert_trees_allclose(accumulated_next.ema_params, expected_ema)
