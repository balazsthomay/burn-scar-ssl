"""Configuration management for burn scar segmentation training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_path: str = "data/hls_burn_scars"
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""

    backbone: Literal["prithvi_eo_v2_300", "prithvi_eo_v2_300_tl"] = "prithvi_eo_v2_300"
    pretrained: bool = True
    img_size: int = 512
    decoder_channels: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    head_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.05
    max_epochs: int = 100
    early_stopping_patience: int = 15

    # Loss
    loss: Literal["ce", "focal", "dice"] = "ce"
    class_weights: list[float] | None = None

    # Freezing
    freeze_backbone: bool = False
    freeze_decoder: bool = False

    # Hardware
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: int = 1

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "prithvi-300m-baseline"
    project: str = "burn-scar-ssl"
    output_dir: str = "outputs"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Loaded ExperimentConfig.
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Build nested dataclasses
    data_config = DataConfig(**raw_config.get("data", {}))
    model_config = ModelConfig(**raw_config.get("model", {}))
    training_config = TrainingConfig(**raw_config.get("training", {}))

    # Extract top-level fields
    experiment_fields = {
        "name": raw_config.get("name", "prithvi-300m-baseline"),
        "project": raw_config.get("project", "burn-scar-ssl"),
        "output_dir": raw_config.get("output_dir", "outputs"),
        "seed": raw_config.get("seed", 42),
        "data": data_config,
        "model": model_config,
        "training": training_config,
    }

    return ExperimentConfig(**experiment_fields)


def save_config(config: ExperimentConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: ExperimentConfig to save.
        config_path: Path to save YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    config_dict = {
        "name": config.name,
        "project": config.project,
        "output_dir": config.output_dir,
        "seed": config.seed,
        "data": {
            "dataset_path": config.data.dataset_path,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
        },
        "model": {
            "backbone": config.model.backbone,
            "pretrained": config.model.pretrained,
            "img_size": config.model.img_size,
            "decoder_channels": config.model.decoder_channels,
            "head_dropout": config.model.head_dropout,
        },
        "training": {
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
            "max_epochs": config.training.max_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "loss": config.training.loss,
            "class_weights": config.training.class_weights,
            "freeze_backbone": config.training.freeze_backbone,
            "freeze_decoder": config.training.freeze_decoder,
            "precision": config.training.precision,
            "accelerator": config.training.accelerator,
            "devices": config.training.devices,
            "log_every_n_steps": config.training.log_every_n_steps,
            "use_wandb": config.training.use_wandb,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
