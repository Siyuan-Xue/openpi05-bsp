import numpy as np
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


class TestRtcInferenceRequest:
    def test_absent_context_preserves_the_request(self):
        observation = {"state": [1.0]}

        inputs, context = inference.pop_rtc_context(observation)

        assert inputs == observation
        assert inputs is not observation
        assert context is None

    def test_schema_only_context_is_an_exact_bootstrap_marker(self):
        observation = {
            "state": [1.0],
            inference.RTC_REQUEST_KEY: {"schema_version": 1},
        }

        inputs, context = inference.pop_rtc_context(observation)

        assert inference.RTC_REQUEST_KEY not in inputs
        assert inference.RTC_REQUEST_KEY in observation
        assert context is not None
        assert context.is_bootstrap
        assert not context.is_guided
        assert context.previous_model_actions is None
        assert context.s is None
        assert context.d is None

    def test_guided_context_copies_normalized_model_actions_read_only(self):
        previous = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
        envelope = {
            "schema_version": 1,
            "previous_model_actions": previous,
            "s": 9,
            "d": 3,
        }
        observation = {"state": [1.0], inference.RTC_REQUEST_KEY: envelope}

        inputs, context = inference.pop_rtc_context(observation)
        previous[0, 0] = -1

        assert inference.RTC_REQUEST_KEY not in inputs
        assert observation[inference.RTC_REQUEST_KEY] is envelope
        assert context is not None
        assert context.is_guided
        assert context.s == 9
        assert context.d == 3
        assert context.previous_model_actions.dtype == np.float32
        assert context.previous_model_actions.shape == (16, 32)
        assert context.previous_model_actions[0, 0] == 0
        assert not context.previous_model_actions.flags.writeable

    @pytest.mark.parametrize(
        "context",
        [
            {},
            {"schema_version": 1, "unexpected": 1},
            {"schema_version": 1, "s": 8},
            {"schema_version": 1, "previous_model_actions": np.zeros((16, 32), dtype=np.float32)},
            {
                "schema_version": 1,
                "previous_model_actions": np.zeros((16, 32), dtype=np.float32),
                "s": 8,
            },
        ],
    )
    def test_context_rejects_nonexact_bootstrap_and_partial_guided_forms(self, context):
        with pytest.raises(ValueError):
            inference.pop_rtc_context({inference.RTC_REQUEST_KEY: context})

    @pytest.mark.parametrize("schema_version", [True, 1.0, "1", 0, 2])
    def test_schema_version_is_exactly_integer_one(self, schema_version):
        with pytest.raises(ValueError):
            inference.pop_rtc_context(
                {inference.RTC_REQUEST_KEY: {"schema_version": schema_version}}
            )

    @pytest.mark.parametrize(
        ("s", "d"),
        [
            (True, 0),
            (8, False),
            (7, 0),
            (17, 0),
            (8, -1),
            (8, 9),
            (9, 8),
            (8.0, 0),
            (8, 0.0),
        ],
    )
    def test_guided_context_rejects_invalid_integer_and_horizon_constraints(self, s, d):
        with pytest.raises(ValueError):
            inference.pop_rtc_context(
                {
                    inference.RTC_REQUEST_KEY: {
                        "schema_version": 1,
                        "previous_model_actions": np.zeros((16, 32), dtype=np.float32),
                        "s": s,
                        "d": d,
                    }
                }
            )

    @pytest.mark.parametrize("s,d", [(8, 0), (8, 8), (16, 0), (11, 5)])
    def test_guided_context_accepts_boundary_values(self, s, d):
        _, context = inference.pop_rtc_context(
            {
                inference.RTC_REQUEST_KEY: {
                    "schema_version": 1,
                    "previous_model_actions": np.zeros((16, 32), dtype=np.float32),
                    "s": s,
                    "d": d,
                }
            }
        )

        assert context is not None
        assert (context.s, context.d) == (s, d)

    @pytest.mark.parametrize(
        "previous",
        [
            [[0.0] * 32 for _ in range(16)],
            np.zeros((16, 32), dtype=np.float64),
            np.zeros((15, 32), dtype=np.float32),
            np.zeros((16, 31), dtype=np.float32),
            np.full((16, 32), np.nan, dtype=np.float32),
            np.full((16, 32), np.inf, dtype=np.float32),
        ],
    )
    def test_guided_context_requires_exact_finite_float32_ndarray(self, previous):
        with pytest.raises(ValueError):
            inference.pop_rtc_context(
                {
                    inference.RTC_REQUEST_KEY: {
                        "schema_version": 1,
                        "previous_model_actions": previous,
                        "s": 8,
                        "d": 0,
                    }
                }
            )
