import pytest

from openpi_client import inference


class TestInferenceRequest:
    def test_reserved_seed_is_popped_without_mutating_the_callers_observation(self):
        observation = {"state": [1.0], inference.INFERENCE_SEED_KEY: 123}

        inputs, seed = inference.pop_inference_seed(observation)

        assert seed == 123
        assert inference.INFERENCE_SEED_KEY not in inputs
        assert observation[inference.INFERENCE_SEED_KEY] == 123

    @pytest.mark.parametrize("invalid", [-1, 2**32, True, 1.5, "1"])
    def test_seed_must_be_a_non_boolean_uint32(self, invalid):
        with pytest.raises(ValueError):
            inference.pop_inference_seed({inference.INFERENCE_SEED_KEY: invalid})
