"""Training infrastructure for burn scar segmentation using TerraTorch."""

from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import terratorch.tasks

from src.data.dataset import BAND_NAMES


# Neck indices for different Prithvi model sizes
NECK_INDICES = {
    "prithvi_eo_v2_300": [5, 11, 17, 23],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
}

BackboneType = Literal["prithvi_eo_v2_300", "prithvi_eo_v2_300_tl"]


class BurnScarSegmentationTask(terratorch.tasks.SemanticSegmentationTask):
    """Semantic segmentation task for burn scar detection.

    Extends TerraTorch's SemanticSegmentationTask with burn-scar-specific defaults.
    """

    def __init__(
        self,
        backbone: BackboneType = "prithvi_eo_v2_300",
        pretrained: bool = True,
        img_size: int = 512,
        decoder_channels: list[int] | None = None,
        head_dropout: float = 0.1,
        loss: str = "ce",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        class_weights: list[float] | None = None,
    ):
        """Initialize burn scar segmentation task.

        Args:
            backbone: Prithvi backbone variant.
            pretrained: Whether to use pretrained weights.
            img_size: Input image size.
            decoder_channels: UNet decoder channels.
            head_dropout: Dropout for segmentation head.
            loss: Loss function ("ce", "focal", "dice").
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            freeze_backbone: Whether to freeze backbone.
            freeze_decoder: Whether to freeze decoder.
            class_weights: Optional class weights for loss.
        """
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        neck_indices = NECK_INDICES[backbone]

        model_args = {
            # Backbone
            "backbone": backbone,
            "backbone_pretrained": pretrained,
            "backbone_num_frames": 1,
            "backbone_img_size": img_size,
            "backbone_bands": BAND_NAMES,
            # Necks
            "necks": [
                {"name": "SelectIndices", "indices": neck_indices},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": decoder_channels,
            # Head
            "head_dropout": head_dropout,
            "num_classes": 2,
        }

        super().__init__(
            model_args=model_args,
            model_factory="EncoderDecoderFactory",
            loss=loss,
            lr=lr,
            ignore_index=-1,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay},
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            plot_on_val=False,
            class_names=["Not burned", "Burn scar"],
            class_weights=class_weights,
        )


def create_trainer(
    max_epochs: int = 100,
    precision: str = "16-mixed",
    accelerator: str = "auto",
    devices: int = 1,
    output_dir: str | Path = "outputs",
    experiment_name: str = "burn-scar-baseline",
    project_name: str = "burn-scar-ssl",
    early_stopping_patience: int = 15,
    log_every_n_steps: int = 10,
    use_wandb: bool = True,
) -> pl.Trainer:
    """Create a PyTorch Lightning trainer with standard callbacks.

    Args:
        max_epochs: Maximum training epochs.
        precision: Training precision ("32", "16-mixed", "bf16-mixed").
        accelerator: Accelerator type ("auto", "gpu", "cpu", "mps").
        devices: Number of devices to use.
        output_dir: Directory for outputs and checkpoints.
        experiment_name: Name for this experiment.
        project_name: W&B project name.
        early_stopping_patience: Patience for early stopping.
        log_every_n_steps: Logging frequency.
        use_wandb: Whether to use W&B logging.

    Returns:
        Configured PyTorch Lightning Trainer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    # Note: TerraTorch logs metrics with "/" prefix (e.g., "val/loss", "val/mIoU")
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch{epoch:02d}-loss{val/loss:.4f}-iou{val/mIoU:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=0.0001,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    if use_wandb:
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            save_dir=str(output_dir),
            log_model=False,  # Don't auto-upload checkpoints
        )
    else:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name=experiment_name,
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    return trainer
