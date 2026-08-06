import dataclasses
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        micro_batch_size=1,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)
    checkpoint_dir = tmp_path / "checkpoint" / config_name / "test"
    assert (checkpoint_dir / "2").is_dir()
    assert not (checkpoint_dir / "1").exists()

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
    assert (checkpoint_dir / "4").is_dir()
    assert not (checkpoint_dir / "3").exists()


def test_none_and_equal_micro_batch_training_are_equivalent(tmp_path: pathlib.Path):
    params = []
    for exp_name, micro_batch_size in (("implicit", None), ("explicit", 2)):
        config = dataclasses.replace(
            _config._CONFIGS_DICT["debug"],  # noqa: SLF001
            batch_size=2,
            micro_batch_size=micro_batch_size,
            checkpoint_base_dir=str(tmp_path / "checkpoint"),
            exp_name=exp_name,
            overwrite=True,
            resume=False,
            num_train_steps=1,
            log_interval=1,
            save_interval=1,
        )
        train.main(config)
        checkpoint = tmp_path / "checkpoint" / "debug" / exp_name / "1" / "params"
        params.append(_model.restore_params(checkpoint, restore_type=np.ndarray))

    for implicit, explicit in zip(jax.tree.leaves(params[0]), jax.tree.leaves(params[1]), strict=True):
        np.testing.assert_array_equal(implicit, explicit)
