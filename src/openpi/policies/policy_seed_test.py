import jax
import numpy as np

from openpi.policies import policy


def test_explicit_inference_seed_is_stateless_but_absent_seed_preserves_split_behavior():
    stateful = jax.random.key(91)

    explicit_state, explicit_key = policy._select_jax_inference_rng(stateful, 7)
    split_state, split_key = policy._select_jax_inference_rng(stateful, None)
    expected_state, expected_key = jax.random.split(stateful)

    np.testing.assert_array_equal(explicit_state, stateful)
    np.testing.assert_array_equal(explicit_key, jax.random.key(7))
    np.testing.assert_array_equal(split_state, expected_state)
    np.testing.assert_array_equal(split_key, expected_key)
