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


if __name__ == "__main__":
    unittest.main()
