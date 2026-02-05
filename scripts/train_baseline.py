#!/usr/bin/env python
"""Train baseline Prithvi-EO-2.0-300M model on HLS Burn Scars dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import lightning.pytorch as pl

from src.data.dataset import HLSBurnScarsDataModule
from src.training.trainer import BurnScarSegmentationTask, create_trainer
from src.utils.config import ExperimentConfig, load_config, save_config


def train(config: ExperimentConfig) -> None:
    """Run training with the given configuration.

    Args:
        config: Experiment configuration.
    """
    # Set seed for reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory for reproducibility
    save_config(config, output_dir / "config.yaml")

    print("=" * 60)
    print(f"Experiment: {config.name}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create data module
    print("\nSetting up data module...")
    data_wrapper = HLSBurnScarsDataModule(
        dataset_path=config.data.dataset_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    datamodule = data_wrapper.build()

    # Create model/task
    print("Creating model...")
    task = BurnScarSegmentationTask(
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        img_size=config.model.img_size,
        decoder_channels=config.model.decoder_channels,
        head_dropout=config.model.head_dropout,
        loss=config.training.loss,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        freeze_backbone=config.training.freeze_backbone,
        freeze_decoder=config.training.freeze_decoder,
        class_weights=config.training.class_weights,
    )

    # Count parameters
    total_params = sum(p.numel() for p in task.parameters())
    trainable_params = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        output_dir=config.output_dir,
        experiment_name=config.name,
        project_name=config.project,
        early_stopping_patience=config.training.early_stopping_patience,
        log_every_n_steps=config.training.log_every_n_steps,
        use_wandb=config.training.use_wandb,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model=task, datamodule=datamodule)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    print("\nTraining complete!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Prithvi-EO-2.0-300M on HLS Burn Scars"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.no_wandb:
        config.training.use_wandb = False
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr

    train(config)


if __name__ == "__main__":
    main()
