import numpy as np
import pytest

from openpi import transforms
from openpi.policies import libero_policy
from openpi.training import bsp as training_bsp


def _valid_bsp_output() -> np.ndarray:
    target = np.zeros((16, 32), dtype=np.float32)
    target[:, 7] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    target[:12, :7] = np.arange(12, dtype=np.float32)[:, None]
    return target


def test_bsp_libero_outputs_preserve_legacy_decode_and_capture_raw_parameters_before_repair(monkeypatch):
    target = _valid_bsp_output().astype(np.float64)
    target[:12, :7] += np.linspace(0.0, 0.01, 12, dtype=np.float64)[:, None]
    target[:, 7] = [0, 0, 0, 0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    target[:, 8:] = 1234.5
    expected_parameters = target[:, :8].copy()
    expected_actions = training_bsp.decode_actions(expected_parameters)
    decoded_parameters = {}
    real_decode_actions = training_bsp.decode_actions

    def recording_decode_actions(parameters, settings):
        decoded_parameters["value"] = np.asarray(parameters).copy()
        return real_decode_actions(parameters, settings)

    monkeypatch.setattr(training_bsp, "decode_actions", recording_decode_actions)

    output = libero_policy.BspLiberoOutputs()({"actions": target})

    assert set(output) == {"actions", "bsp"}
    np.testing.assert_array_equal(output["actions"], expected_actions)
    np.testing.assert_array_equal(decoded_parameters["value"], expected_parameters)
    assert decoded_parameters["value"].dtype == np.float64

    sidecar = output["bsp"]
    assert set(sidecar) == {"schema_version", "parameters", "origin_hz", "degree", "speedup", "alignment"}
    assert sidecar["schema_version"] == 1
    assert sidecar["origin_hz"] == 10
    assert sidecar["degree"] == 3
    assert sidecar["speedup"] == 2
    assert sidecar["alignment"] == "disabled_delta_eff"
    assert sidecar["parameters"].shape == (16, 8)
    assert sidecar["parameters"].dtype == np.float32
    np.testing.assert_array_equal(sidecar["parameters"], expected_parameters.astype(np.float32))
    assert sidecar["parameters"][5, 7] < sidecar["parameters"][4, 7]

    captured_parameters = sidecar["parameters"].copy()
    target[:, :8] = -999.0
    np.testing.assert_array_equal(sidecar["parameters"], captured_parameters)


def test_bsp_libero_outputs_ignore_only_inactive_controls_and_reject_invalid_parameters():
    target = _valid_bsp_output()
    expected = libero_policy.BspLiberoOutputs()({"actions": target})["actions"]
    target[12:, :7] = 10_000
    np.testing.assert_allclose(libero_policy.BspLiberoOutputs()({"actions": target})["actions"], expected)

    nonfinite_padding = _valid_bsp_output()
    nonfinite_padding[0, 31] = np.nan
    for invalid in (
        np.zeros((15, 32)),
        np.full((16, 32), np.nan),
        nonfinite_padding,
        np.zeros((16, 7)),
        np.zeros((16, 9)),
        np.zeros((16, 31)),
    ):
        with pytest.raises(ValueError, match=r"BSP policy output (?:must have padded shape|contains non-finite)"):
            libero_policy.BspLiberoOutputs()({"actions": invalid})


def test_bsp_libero_outputs_reject_parameters_that_overflow_the_float32_sidecar():
    target = _valid_bsp_output().astype(np.float64)
    target[0, 0] = np.finfo(np.float64).max

    with pytest.raises(ValueError, match="finite float32"):
        libero_policy.BspLiberoOutputs()({"actions": target})


def test_composed_output_transforms_capture_parameters_after_quantile_unnormalize():
    unnormalized = _valid_bsp_output().astype(np.float64)
    q01 = np.linspace(-2.0, 1.5, 8, dtype=np.float64)
    q99 = q01 + np.linspace(2.0, 5.5, 8, dtype=np.float64)
    stats = transforms.NormStats(
        mean=np.zeros(8, dtype=np.float64),
        std=np.ones(8, dtype=np.float64),
        q01=q01,
        q99=q99,
    )
    normalized = np.zeros((16, 32), dtype=np.float64)
    normalized[:, :8] = (unnormalized[:, :8] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    output_transform = transforms.compose(
        (
            transforms.Unnormalize({"actions": stats}, use_quantiles=True),
            libero_policy.BspLiberoOutputs(),
        )
    )

    output = output_transform({"actions": normalized})

    np.testing.assert_allclose(
        output["bsp"]["parameters"],
        unnormalized[:, :8].astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(output["bsp"]["parameters"], normalized[:, :8])
    np.testing.assert_allclose(
        output["actions"],
        training_bsp.decode_actions(unnormalized[:, :8]),
        rtol=1e-6,
        atol=1e-6,
    )
