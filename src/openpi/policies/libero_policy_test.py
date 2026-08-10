import numpy as np
import pytest

from openpi.policies import libero_policy


def _valid_bsp_output() -> np.ndarray:
    target = np.zeros((16, 32), dtype=np.float32)
    target[:, 7] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    target[:12, :7] = np.arange(12, dtype=np.float32)[:, None]
    return target


def test_bsp_libero_outputs_decode_eight_actions_after_model_padding_is_removed():
    output = libero_policy.BspLiberoOutputs()({"actions": _valid_bsp_output()})

    assert output["actions"].shape == (8, 7)
    assert np.isfinite(output["actions"]).all()


def test_bsp_libero_outputs_ignore_only_inactive_controls_and_reject_invalid_parameters():
    target = _valid_bsp_output()
    expected = libero_policy.BspLiberoOutputs()({"actions": target})["actions"]
    target[12:, :7] = 10_000
    np.testing.assert_allclose(libero_policy.BspLiberoOutputs()({"actions": target})["actions"], expected)

    for invalid in (
        np.zeros((15, 32)),
        np.full((16, 32), np.nan),
        np.zeros((16, 7)),
        np.zeros((16, 9)),
        np.zeros((16, 31)),
    ):
        with pytest.raises(ValueError, match=r"BSP policy output (?:must have padded shape|contains non-finite)"):
            libero_policy.BspLiberoOutputs()({"actions": invalid})
