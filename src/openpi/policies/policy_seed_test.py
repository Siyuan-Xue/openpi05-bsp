import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import inference

from openpi.policies import policy


def _assert_key_equal(actual, expected):
    np.testing.assert_array_equal(jax.random.key_data(actual), jax.random.key_data(expected))


def test_explicit_inference_seed_is_stateless_but_absent_seed_preserves_split_behavior():
    stateful = jax.random.key(91)

    explicit_state, explicit_key = policy._select_jax_inference_rng(stateful, 7)
    split_state, split_key = policy._select_jax_inference_rng(stateful, None)
    expected_state, expected_key = jax.random.split(stateful)

    _assert_key_equal(explicit_state, stateful)
    _assert_key_equal(explicit_key, jax.random.key(7))
    _assert_key_equal(split_state, expected_state)
    _assert_key_equal(split_key, expected_key)


def test_policy_infer_pops_reserved_seed_before_input_transforms(monkeypatch):
    seen_inputs = []
    policy_instance = object.__new__(policy.Policy)
    policy_instance._rng = jax.random.key(5)
    policy_instance._sample_kwargs = {}
    policy_instance._input_transform = lambda inputs: seen_inputs.append(inputs) or {"state": np.asarray([1.0])}
    policy_instance._output_transform = lambda outputs: outputs
    policy_instance._sample_actions = lambda rng, observation, **kwargs: jnp.zeros((1, 16, 32))
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    observation = {"raw": np.asarray([3.0]), inference.INFERENCE_SEED_KEY: 123}

    policy_instance.infer(observation)

    assert inference.INFERENCE_SEED_KEY not in seen_inputs[0]
    assert observation[inference.INFERENCE_SEED_KEY] == 123
