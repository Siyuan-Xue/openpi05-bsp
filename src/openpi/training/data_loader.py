"""JAX data loading for fake and LeRobot LIBERO datasets."""

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training.bsp import load_sidecar_cache
from openpi.training.bsp_dataset import BspLeRobotDataset
from openpi.training.bsp_dataset import make_lerobot_cache_manifest
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DataLoader(Protocol[T_co]):
    def data_config(self) -> _config.DataConfig:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        return {**observation.to_dict(), "actions": action}

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create the map-style dataset consumed by the JAX loader."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if data_config.use_bsp and data_config.bsp_cache_path is None:
        raise ValueError(
            "BSP training requires an explicit precomputed cache path. "
            "Run scripts/prepare_libero_bsp.py in build mode, then set bsp_cache_path."
        )

    lerobot_kwargs = {}
    if data_config.lerobot_root is not None:
        lerobot_kwargs["root"] = data_config.lerobot_root
    if data_config.lerobot_revision is not None:
        lerobot_kwargs["revision"] = data_config.lerobot_revision

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, **lerobot_kwargs)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [step / dataset_meta.fps for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        **lerobot_kwargs,
    )
    if data_config.use_bsp:
        if len(data_config.action_sequence_keys) != 1:
            raise ValueError("BSP training requires exactly one LeRobot action sequence key")
        action_key = data_config.action_sequence_keys[0]
        expected_manifest = make_lerobot_cache_manifest(
            dataset,
            repo_id=repo_id,
            revision=data_config.lerobot_revision,
            action_key=action_key,
        )
        cache = load_sidecar_cache(data_config.bsp_cache_path, expected_manifest)
        dataset = BspLeRobotDataset(dataset, cache, action_key=action_key)
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)
    batch_size = config.micro_batch_size if config.micro_batch_size is not None else config.batch_size
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    local_batch_size = batch_size // jax.process_count()
    logging.info("local_batch_size: %s", local_batch_size)
    return DataLoaderImpl(
        data_config,
        TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=seed,
        ),
    )


class TorchDataLoader:
    """Use LeRobot's torch dataset machinery while emitting JAX-sharded arrays."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")
        self._sharding = sharding
        if self._sharding is None:
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield jax.tree.map(lambda value: jax.make_array_from_process_local_data(self._sharding, value), batch)


def _collate_fn(items):
    return jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
