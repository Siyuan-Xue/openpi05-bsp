import unittest

from openpi_client import inference


class InferenceRequestTest(unittest.TestCase):
    def test_reserved_seed_is_popped_without_mutating_the_callers_observation(self):
        observation = {"state": [1.0], inference.INFERENCE_SEED_KEY: 123}

        inputs, seed = inference.pop_inference_seed(observation)

        self.assertEqual(seed, 123)
        self.assertNotIn(inference.INFERENCE_SEED_KEY, inputs)
        self.assertEqual(observation[inference.INFERENCE_SEED_KEY], 123)

    def test_seed_must_be_a_non_boolean_uint32(self):
        for invalid in (-1, 2**32, True, 1.5, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inference.pop_inference_seed({inference.INFERENCE_SEED_KEY: invalid})

    def test_bsp_execution_context_is_exact_speedup_one_and_removed_from_model_inputs(self):
        observation = {
            "state": [0.0] * 8,
            inference.BSP_EXECUTION_KEY: {"schema_version": 1, "speedup": 1},
        }

        inputs, speedup = inference.pop_bsp_execution(observation)

        self.assertEqual(speedup, 1)
        self.assertEqual(set(inputs), {"state"})
        self.assertIn(inference.BSP_EXECUTION_KEY, observation)

    def test_bsp_execution_context_rejects_nonexact_payloads(self):
        invalid_values = (
            None,
            {"schema_version": True, "speedup": 1},
            {"schema_version": 1, "speedup": True},
            {"schema_version": 1, "speedup": 2},
            {"schema_version": 1, "speedup": 1, "extra": 0},
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bsp_execution"):
                inference.pop_bsp_execution({inference.BSP_EXECUTION_KEY: value})


if __name__ == "__main__":
    unittest.main()
