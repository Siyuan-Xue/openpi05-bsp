import pytest

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi05_libero")

    example = libero_policy.make_libero_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 7)
