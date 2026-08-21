# ruff: noqa: SLF001 -- policy contract tests intentionally exercise private seams.

import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import inference
import pytest

from openpi import transforms
from openpi.policies import policy
from openpi.policies import policy_config


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
    policy_instance._action_representation = "native"
    policy_instance._model_action_horizon = 16
    policy_instance._model_action_dim = 32
    policy_instance._sample_actions_rtc = lambda *args, **kwargs: None
    policy_instance._input_transform = lambda inputs: seen_inputs.append(inputs) or {"state": np.asarray([1.0])}
    policy_instance._output_transform = lambda outputs: outputs
    policy_instance._sample_actions = lambda rng, observation, **kwargs: jnp.zeros((1, 16, 32))
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    observation = {"raw": np.asarray([3.0]), inference.INFERENCE_SEED_KEY: 123}

    policy_instance.infer(observation)

    assert inference.INFERENCE_SEED_KEY not in seen_inputs[0]
    assert observation[inference.INFERENCE_SEED_KEY] == 123


def _fake_policy(*, action_representation="native", model_horizon=16, model_dim=32):
    seen = {"input": [], "legacy": [], "rtc": [], "output": []}
    policy_instance = object.__new__(policy.Policy)
    policy_instance._rng = jax.random.key(17)
    policy_instance._sample_kwargs = {"num_steps": 11}
    policy_instance._action_representation = action_representation
    policy_instance._model_action_horizon = model_horizon
    policy_instance._model_action_dim = model_dim
    policy_instance._input_transform = lambda inputs: seen["input"].append(inputs) or {"state": np.asarray([1.0])}

    def output_transform(outputs):
        seen["output"].append(np.asarray(outputs["actions"]).copy())
        return {"actions": np.asarray(outputs["actions"])[:, :7] + 1000}

    def legacy_sampler(rng, observation, **kwargs):
        del observation
        seen["legacy"].append((rng, kwargs))
        return jnp.arange(16 * 32, dtype=jnp.float32).reshape(1, 16, 32)

    def rtc_sampler(rng, observation, **kwargs):
        del observation
        seen["rtc"].append((rng, kwargs))
        return jnp.arange(16 * 32, dtype=jnp.float32).reshape(1, 16, 32) + 5000

    policy_instance._output_transform = output_transform
    policy_instance._sample_actions = legacy_sampler
    policy_instance._sample_actions_rtc = rtc_sampler
    return policy_instance, seen


def _guided_observation(*, s=8, d=2):
    return {
        "raw": np.asarray([3.0]),
        inference.RTC_REQUEST_KEY: {
            "schema_version": 1,
            "previous_model_actions": np.arange(16 * 32, dtype=np.float32).reshape(16, 32),
            "s": s,
            "d": d,
        },
    }


def test_legacy_request_preserves_callable_kwargs_response_shape_and_rng_semantics(monkeypatch):
    policy_instance, seen = _fake_policy()
    policy_instance._sample_kwargs = {"num_steps": 13, "legacy_option": "preserved"}
    original_rng = policy_instance._rng
    expected_next_rng, expected_sample_rng = jax.random.split(original_rng)
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    result = policy_instance.infer({"raw": np.asarray([3.0])})

    assert len(seen["legacy"]) == 1
    assert not seen["rtc"]
    _assert_key_equal(seen["legacy"][0][0], expected_sample_rng)
    _assert_key_equal(policy_instance._rng, expected_next_rng)
    assert seen["legacy"][0][1] == {"num_steps": 13, "legacy_option": "preserved"}
    assert set(result) == {"actions", "policy_timing"}
    assert "rtc" not in result


@pytest.mark.parametrize("requested_speedup", [1, 4, 8])
def test_bsp_execution_envelope_is_removed_before_transforms_and_relabels_the_sidecar(monkeypatch, requested_speedup):
    policy_instance, seen = _fake_policy(action_representation="bsp")

    def bsp_output_transform(outputs):
        seen["output"].append(np.asarray(outputs["actions"]).copy())
        return {
            "actions": np.zeros((8, 7), dtype=np.float32),
            "bsp": {
                "schema_version": 1,
                "parameters": np.zeros((16, 8), dtype=np.float32),
                "origin_hz": 10,
                "degree": 3,
                "speedup": 2,
                "alignment": "disabled_delta_eff",
            },
        }

    policy_instance._output_transform = bsp_output_transform
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    request = {
        "raw": np.asarray([3.0]),
        inference.BSP_EXECUTION_KEY: {"schema_version": 1, "speedup": requested_speedup},
    }

    result = policy_instance.infer(request)

    assert inference.BSP_EXECUTION_KEY not in seen["input"][0]
    assert request[inference.BSP_EXECUTION_KEY] == {"schema_version": 1, "speedup": requested_speedup}
    assert result["bsp"]["speedup"] == requested_speedup


def test_native_policy_rejects_bsp_speedup_one_execution_envelope_before_sampling(monkeypatch):
    policy_instance, seen = _fake_policy(action_representation="native")
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    with pytest.raises(ValueError, match="BSP action representation"):
        policy_instance.infer(
            {
                "raw": np.asarray([3.0]),
                inference.BSP_EXECUTION_KEY: {"schema_version": 1, "speedup": 1},
            }
        )

    assert not seen["legacy"]


def test_bootstrap_pops_both_envelopes_forces_n5_and_captures_before_output_transform(monkeypatch):
    policy_instance, seen = _fake_policy()
    policy_instance._sample_kwargs = {"num_steps": 99}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    observation = {
        "raw": np.asarray([3.0]),
        inference.INFERENCE_SEED_KEY: 23,
        inference.RTC_REQUEST_KEY: {"schema_version": 1},
    }

    result = policy_instance.infer(observation)

    assert inference.INFERENCE_SEED_KEY not in seen["input"][0]
    assert inference.RTC_REQUEST_KEY not in seen["input"][0]
    assert inference.INFERENCE_SEED_KEY in observation
    assert inference.RTC_REQUEST_KEY in observation
    assert len(seen["legacy"]) == 1
    assert not seen["rtc"]
    _assert_key_equal(seen["legacy"][0][0], jax.random.key(23))
    assert seen["legacy"][0][1] == {"num_steps": 5}
    np.testing.assert_array_equal(result["rtc"]["model_actions"], seen["output"][0])
    assert result["rtc"]["model_actions"].dtype == np.float32
    assert result["rtc"]["model_actions"].shape == (16, 32)
    assert result["rtc"]["schema_version"] == 1
    assert np.all(result["actions"] == seen["output"][0][:, :7] + 1000)


def test_schema_only_bootstrap_can_be_repeated_as_the_baseline_sync_n5_protocol(monkeypatch):
    policy_instance, seen = _fake_policy()
    policy_instance._sample_kwargs = {"num_steps": 99}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    request = {"raw": np.asarray([3.0]), inference.RTC_REQUEST_KEY: {"schema_version": 1}}

    first = policy_instance.infer(request)
    second = policy_instance.infer(request)

    assert [kwargs for _, kwargs in seen["legacy"]] == [{"num_steps": 5}, {"num_steps": 5}]
    assert not seen["rtc"]
    assert first["rtc"]["model_actions"].shape == (16, 32)
    assert second["rtc"]["model_actions"].shape == (16, 32)
    assert request[inference.RTC_REQUEST_KEY] == {"schema_version": 1}


def test_guided_request_routes_fixed_sampler_with_prepared_target_mask_and_noise(monkeypatch):
    policy_instance, seen = _fake_policy()
    policy_instance._sample_kwargs = {"num_steps": 71}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))
    noise = np.ones((16, 32), dtype=np.float32)

    result = policy_instance.infer(_guided_observation(s=10, d=2), noise=noise)

    assert not seen["legacy"]
    assert len(seen["rtc"]) == 1
    kwargs = seen["rtc"][0][1]
    assert set(kwargs) == {"target", "weights", "noise"}
    assert kwargs["noise"].shape == (1, 16, 32)
    np.testing.assert_array_equal(
        kwargs["target"][0, :6],
        np.arange(16 * 32, dtype=np.float32).reshape(16, 32)[10:],
    )
    np.testing.assert_array_equal(kwargs["target"][0, 6:], np.zeros((10, 32), dtype=np.float32))
    assert np.all(kwargs["weights"][:2, :7] == 1)
    assert np.all(kwargs["weights"][:, 7:] == 0)
    assert result["rtc"]["model_actions"].shape == (16, 32)


@pytest.mark.parametrize(
    ("action_representation", "model_horizon", "model_dim", "has_hook"),
    [
        ("bsp", 16, 32, True),
        (None, 16, 32, True),
        ("native", 15, 32, True),
        ("native", 16, 31, True),
        ("native", 16, 32, False),
    ],
)
def test_rtc_request_rejects_bsp_unknown_wrong_shape_or_missing_hook(
    monkeypatch, action_representation, model_horizon, model_dim, has_hook
):
    policy_instance, _ = _fake_policy(
        action_representation=action_representation,
        model_horizon=model_horizon,
        model_dim=model_dim,
    )
    if not has_hook:
        policy_instance._sample_actions_rtc = None
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    with pytest.raises(ValueError, match="RTC"):
        policy_instance.infer({"raw": np.asarray([3.0]), inference.RTC_REQUEST_KEY: {"schema_version": 1}})


def test_guided_request_rejects_unrelated_sampler_kwargs(monkeypatch):
    policy_instance, _ = _fake_policy()
    policy_instance._sample_kwargs = {"num_steps": 5, "temperature": 0.4}
    initial_rng = policy_instance._rng
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    with pytest.raises(ValueError, match="sampler kwargs"):
        policy_instance.infer(_guided_observation())

    _assert_key_equal(policy_instance._rng, initial_rng)


@pytest.mark.parametrize("mode", ["bootstrap", "guided"])
@pytest.mark.parametrize("source", ["explicit", "configured"])
@pytest.mark.parametrize(
    "invalid_noise",
    [
        np.zeros((16, 32), dtype=np.bool_),
        np.full((16, 32), "noise", dtype=object),
        np.full((16, 32), 1 + 2j, dtype=np.complex64),
        np.full((16, 32), np.nan, dtype=np.float32),
        np.full((1, 16, 32), np.inf, dtype=np.float32),
        np.full((16, 32), np.finfo(np.float64).max, dtype=np.float64),
        np.zeros((15, 32), dtype=np.float32),
        np.zeros((1, 16, 31), dtype=np.float32),
        np.zeros((2, 16, 32), dtype=np.float32),
    ],
)
def test_rtc_noise_is_strictly_validated_before_rng_advance(
    monkeypatch,
    mode,
    source,
    invalid_noise,
):
    policy_instance, _ = _fake_policy()
    initial_rng = policy_instance._rng
    request = (
        {"raw": np.asarray([3.0]), inference.RTC_REQUEST_KEY: {"schema_version": 1}}
        if mode == "bootstrap"
        else _guided_observation()
    )
    call_kwargs = {}
    if source == "explicit":
        call_kwargs["noise"] = invalid_noise
    else:
        policy_instance._sample_kwargs = {"num_steps": 5, "noise": invalid_noise}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    with pytest.raises(ValueError, match="RTC noise"):
        policy_instance.infer(request, **call_kwargs)

    _assert_key_equal(policy_instance._rng, initial_rng)


@pytest.mark.parametrize("shape", [(16, 32), (1, 16, 32)])
@pytest.mark.parametrize("source", ["explicit", "configured"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_rtc_noise_accepts_real_numeric_shapes_as_defensive_float32_batch(
    monkeypatch,
    shape,
    source,
    dtype,
):
    policy_instance, seen = _fake_policy()
    supplied = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    expected = supplied.astype(np.float32).reshape(1, 16, 32)
    call_kwargs = {}
    if source == "explicit":
        call_kwargs["noise"] = supplied
    else:
        policy_instance._sample_kwargs = {"num_steps": 99, "noise": supplied}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    policy_instance.infer(
        {"raw": np.asarray([3.0]), inference.RTC_REQUEST_KEY: {"schema_version": 1}},
        **call_kwargs,
    )

    canonical = seen["legacy"][0][1]["noise"]
    assert canonical.shape == (1, 16, 32)
    assert canonical.dtype == np.float32
    supplied[...] = -1
    np.testing.assert_array_equal(canonical, expected)


def test_absent_rtc_preserves_legacy_configured_noise_without_new_validation(monkeypatch):
    policy_instance, seen = _fake_policy()
    configured_noise = object()
    policy_instance._sample_kwargs = {"noise": configured_noise, "legacy_option": True}
    monkeypatch.setattr(policy._model.Observation, "from_dict", staticmethod(lambda inputs: object()))

    policy_instance.infer({"raw": np.asarray([3.0])})

    assert seen["legacy"][0][1] == {"noise": configured_noise, "legacy_option": True}


class _FakeModel:
    def __init__(self, *, supports_rtc=True, horizon=16, action_dim=32):
        self.supports_rtc = supports_rtc
        self.action_horizon = horizon
        self.action_dim = action_dim

    def sample_actions(self, *args, **kwargs):
        raise AssertionError("not called")

    def sample_actions_rtc(self, *args, **kwargs):
        raise AssertionError("not called")


def test_policy_computes_exact_reserved_capability_metadata_and_rejects_collision(monkeypatch):
    monkeypatch.setattr(policy.nnx_utils, "module_jit", lambda method: method)
    native = policy.Policy(
        _FakeModel(),
        action_representation="native",
        metadata={"robot": "libero"},
    )
    bsp = policy.Policy(
        _FakeModel(supports_rtc=False),
        action_representation="bsp",
        metadata={"robot": "libero"},
    )

    assert native.metadata == {
        "robot": "libero",
        inference.INFERENCE_CAPABILITIES_KEY: {
            "schema_version": 1,
            "action_representation": "native",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": [
                "baseline_h16_n5_v1",
                "baseline_sync_n5_h16_full_v2",
                "baseline_async_h16_v1",
                "baseline_async_h16_blocking_recovery_v2",
                "baseline_rtc_h16_v1",
            ],
        },
    }
    assert bsp.metadata == {
        "robot": "libero",
        inference.INFERENCE_CAPABILITIES_KEY: {
            "schema_version": 1,
            "action_representation": "bsp",
            "model_action_horizon": 16,
            "model_action_dim": 32,
            "supported_protocols": [
                "bsp_spline_sync_speedup2_phase0_v2",
                "bsp_spline_async_phase_skip_speedup2_v2",
                "bsp_spline_async_phase_skip_speedup1_v1",
                "bsp_spline_async_phase_skip_speedup4_delta_accum_v2",
                "bsp_spline_async_phase_skip_speedup8_delta_accum_v2",
            ],
        },
    }
    with pytest.raises(ValueError, match="reserved"):
        policy.Policy(
            _FakeModel(),
            action_representation="native",
            metadata={inference.INFERENCE_CAPABILITIES_KEY: {}},
        )


def test_non_pi0_model_does_not_advertise_or_install_rtc_hook(monkeypatch):
    monkeypatch.setattr(policy.nnx_utils, "module_jit", lambda method: method)
    policy_instance = policy.Policy(
        _FakeModel(supports_rtc=False),
        action_representation="native",
    )

    assert policy_instance._sample_actions_rtc is None
    assert policy_instance.metadata[inference.INFERENCE_CAPABILITIES_KEY]["supported_protocols"] == []


@pytest.mark.parametrize(("use_bsp", "expected"), [(False, "native"), (True, "bsp")])
def test_policy_config_passes_explicit_action_representation_from_data_config(monkeypatch, use_bsp, expected):
    captured = {}
    model = types.SimpleNamespace(action_horizon=16, action_dim=32)
    model_config = types.SimpleNamespace(load=lambda params: model)
    data_config = types.SimpleNamespace(
        use_bsp=use_bsp,
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
        use_quantile_norm=False,
    )
    train_config = types.SimpleNamespace(
        model=model_config,
        data=types.SimpleNamespace(create=lambda assets_dirs, config: data_config),
        assets_dirs=pathlib.Path("assets"),
        policy_metadata={"robot": "libero"},
    )
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: pathlib.Path(path))
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: object())

    def capture_policy(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(policy_config._policy, "Policy", capture_policy)

    result = policy_config.create_trained_policy(
        train_config,
        pathlib.Path("checkpoint"),
        norm_stats={},
    )

    assert result is model
    assert captured["action_representation"] == expected
