import dataclasses

import jax
import pytest

from openpi.models import pi0_config
from openpi.policies import libero_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_jax_data_loader_emits_the_global_micro_batch():
    config = dataclasses.replace(_config.get_config("debug"), batch_size=4, micro_batch_size=2)

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1)
    batch = next(iter(loader))

    assert all(x.shape[0] == 2 for x in jax.tree.leaves(batch))


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_libero_h16_configs_keep_baseline_and_bsp_assets_separate():
    """Sharing action stats would normalize raw actions and BSP parameters with incompatible distributions."""
    baseline = _config.get_config("pi05_libero_baseline_h16")
    bsp = _config.get_config("pi05_libero_bsp_h16")

    assert baseline.model.action_horizon == 16
    assert bsp.model.action_horizon == 16
    assert baseline.model.pi05 is True
    assert bsp.model.pi05 is True
    assert baseline.model.action_dim == 32
    assert bsp.model.action_dim == 32
    assert baseline.model.discrete_state_input is False
    assert bsp.model.discrete_state_input is False
    assert baseline.data.assets.asset_id == "libero_baseline_h16"
    assert bsp.data.assets.asset_id == "libero_bsp_h16"
    assert baseline.data.lerobot_revision == "v2.1"
    assert bsp.data.lerobot_revision == "v2.1"
    assert baseline.data.bsp_cache_path is None
    assert bsp.data.use_bsp is True

    baseline_data = baseline.data.create(baseline.assets_dirs, baseline.model)
    bsp_data = bsp.data.create(bsp.assets_dirs, bsp.model)
    assert isinstance(baseline_data.data_transforms.outputs[0], libero_policy.LiberoOutputs)
    assert isinstance(bsp_data.data_transforms.outputs[0], libero_policy.BspLiberoOutputs)


def test_libero_h16_configs_use_the_same_jax_full_finetuning_recipe():
    for name in ("pi05_libero_baseline_h16", "pi05_libero_bsp_h16"):
        config = _config.get_config(name)

        assert config.seed == 42
        assert config.batch_size == 256
        assert config.micro_batch_size == 1
        assert config.num_train_steps == 30_000
        assert config.save_interval == 1_000
        assert config.keep_period == 10_000
        assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
        assert config.pytorch_weight_path is None
        assert config.lr_schedule.warmup_steps == 10_000
        assert config.lr_schedule.peak_lr == 5e-5
        assert config.lr_schedule.decay_steps == 1_000_000
        assert config.lr_schedule.decay_lr == 5e-5
        assert config.optimizer.clip_gradient_norm == 1.0
        assert config.ema_decay == 0.999


def test_bsp_training_refuses_to_create_a_dataset_without_an_explicit_sidecar(tmp_path):
    """Falling back to worker-time fitting would make training nondeterministic and prohibitively slow."""
    config = _config.get_config("pi05_libero_bsp_h16")
    data_config = config.data.create(tmp_path, config.model)

    with pytest.raises(ValueError, match="explicit precomputed cache path"):
        _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
