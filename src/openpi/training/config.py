"""Training configuration registry for the pi0.5 JAX + LIBERO runtime."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Location of normalization assets used by the data pipeline."""

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    lerobot_root: str | None = None
    lerobot_revision: str | None = None
    use_bsp: bool = False
    bsp_cache_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a transform group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Create the PaliGemma transforms shared by pi0 and pi0.5."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                discrete_state_input = False
            case _model.ModelType.PI05:
                if not isinstance(model_config, pi0_config.Pi0Config):
                    raise TypeError("pi0.5 requires Pi0Config")
                discrete_state_input = model_config.discrete_state_input
            case _:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")

        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logging.info("Norm stats not found in %s, skipping.", data_assets_dir)
            return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """Small dependency-free dataset config used by focused loader tests."""

    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs, model_config
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """Transforms and storage options for the LeRobot LIBERO dataset."""

    extra_delta_transform: bool = False
    lerobot_root: str | None = None
    lerobot_revision: str | None = None
    use_bsp: bool = False
    bsp_cache_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        output_transform = libero_policy.BspLiberoOutputs() if self.use_bsp else libero_policy.LiberoOutputs()
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[output_transform],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            lerobot_root=self.lerobot_root,
            lerobot_revision=self.lerobot_revision,
            use_bsp=self.use_bsp,
            bsp_cache_path=self.bsp_cache_path,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    micro_batch_size: int | None = None
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int | None = 5_000
    permanent_checkpoint_steps: tuple[int, ...] = ()
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in self.permanent_checkpoint_steps
        ):
            raise ValueError("permanent_checkpoint_steps must contain only nonnegative integers")
        if tuple(sorted(self.permanent_checkpoint_steps)) != self.permanent_checkpoint_steps or len(
            set(self.permanent_checkpoint_steps)
        ) != len(self.permanent_checkpoint_steps):
            raise ValueError("permanent_checkpoint_steps must be unique and in ascending order")


_PI05_LIBERO_LORA_H16_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

_PHASE_ONE_SCHEDULE = _optimizer.CosineDecaySchedule(
    warmup_steps=10_000,
    peak_lr=5e-5,
    decay_steps=1_000_000,
    decay_lr=5e-5,
)
_PHASE_ONE_OPTIMIZER = _optimizer.AdamW(clip_gradient_norm=1.0)
_PI05_BASE_WEIGHTS = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
_CONFIGS = [
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_PHASE_ONE_SCHEDULE,
        optimizer=_PHASE_ONE_OPTIMIZER,
        ema_decay=0.999,
        weight_loader=_PI05_BASE_WEIGHTS,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_baseline_h16",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero_baseline_h16"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            lerobot_revision="v2.0",
        ),
        seed=42,
        batch_size=256,
        micro_batch_size=1,
        lr_schedule=_PHASE_ONE_SCHEDULE,
        optimizer=_PHASE_ONE_OPTIMIZER,
        ema_decay=0.999,
        weight_loader=_PI05_BASE_WEIGHTS,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=10_000,
        permanent_checkpoint_steps=(0, 1_000, 2_000, 5_000, 10_000),
    ),
    TrainConfig(
        name="pi05_libero_bsp_h16",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero_bsp_h16"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            lerobot_revision="v2.0",
            use_bsp=True,
            bsp_cache_path=None,
        ),
        seed=42,
        batch_size=256,
        micro_batch_size=1,
        lr_schedule=_PHASE_ONE_SCHEDULE,
        optimizer=_PHASE_ONE_OPTIMIZER,
        ema_decay=0.999,
        weight_loader=_PI05_BASE_WEIGHTS,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=10_000,
        permanent_checkpoint_steps=(0, 1_000, 2_000, 5_000, 10_000),
    ),
    TrainConfig(
        name="pi05_libero_baseline_lora_h16",
        model=_PI05_LIBERO_LORA_H16_MODEL,
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero_baseline_h16"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            lerobot_revision="v2.0",
        ),
        seed=42,
        batch_size=256,
        micro_batch_size=1,
        lr_schedule=_PHASE_ONE_SCHEDULE,
        optimizer=_PHASE_ONE_OPTIMIZER,
        ema_decay=None,
        freeze_filter=_PI05_LIBERO_LORA_H16_MODEL.get_freeze_filter(),
        weight_loader=_PI05_BASE_WEIGHTS,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=10_000,
        permanent_checkpoint_steps=(0, 1_000, 2_000, 5_000, 10_000),
    ),
    TrainConfig(
        name="pi05_libero_bsp_lora_h16",
        model=_PI05_LIBERO_LORA_H16_MODEL,
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero_bsp_h16"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            lerobot_revision="v2.0",
            use_bsp=True,
            bsp_cache_path=None,
        ),
        seed=42,
        batch_size=256,
        micro_batch_size=1,
        lr_schedule=_PHASE_ONE_SCHEDULE,
        optimizer=_PHASE_ONE_OPTIMIZER,
        ema_decay=None,
        freeze_filter=_PI05_LIBERO_LORA_H16_MODEL.get_freeze_filter(),
        weight_loader=_PI05_BASE_WEIGHTS,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=10_000,
        permanent_checkpoint_steps=(0, 1_000, 2_000, 5_000, 10_000),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({key: (key, value) for key, value in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
