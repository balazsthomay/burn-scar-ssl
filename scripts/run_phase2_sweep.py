#!/usr/bin/env python3
"""Run Phase 2 label efficiency sweep.

Trains all backbone × fraction × seed combinations and saves results
for later analysis.

Usage:
    uv run scripts/run_phase2_sweep.py
    uv run scripts/run_phase2_sweep.py --backbone resnet50 --fraction 0.1
    uv run scripts/run_phase2_sweep.py --dry-run
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from rich.console import Console
from rich.table import Table

from src.data.dataset import HLSBurnScarsDataModule
from src.models.segmentation import SegmentationModel
from src.training.trainer import BurnScarSegmentationTask

console = Console()


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    backbone: str
    fraction: float
    seed: int
    test_iou_burn: float
    test_miou: float
    val_loss: float
    epochs_trained: int
    checkpoint_path: str


class Phase2LightningModule(pl.LightningModule):
    """Lightning module for DINOv3/ResNet training."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SegmentationModel(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=True,
            **model_kwargs,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute metrics
        preds = logits.argmax(dim=1)
        valid_mask = masks != -1

        # Per-class IoU
        ious = []
        for cls in range(2):
            pred_cls = (preds == cls) & valid_mask
            target_cls = (masks == cls) & valid_mask
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
            self.log(f"{stage}/iou_class{cls}", iou, prog_bar=False)

        miou = sum(ious) / len(ious)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/mIoU", miou, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


def run_single_experiment(
    backbone: str,
    fraction: float,
    seed: int,
    config: dict,
    output_dir: Path,
    use_wandb: bool = True,
) -> ExperimentResult:
    """Run a single experiment.

    Args:
        backbone: Backbone name.
        fraction: Label fraction (e.g., 0.1 for 10%).
        seed: Random seed.
        config: Full configuration dict.
        output_dir: Base output directory.
        use_wandb: Whether to use W&B logging.

    Returns:
        ExperimentResult with metrics.
    """
    pl.seed_everything(seed, workers=True)

    # Determine split file
    if fraction < 1.0:
        pct = int(fraction * 100)
        train_split = f"train_{pct}pct.txt"
    else:
        train_split = None  # Use default train.txt

    # Create experiment directory
    exp_name = f"{backbone}_frac{int(fraction*100)}_seed{seed}"
    exp_dir = output_dir / backbone / f"{int(fraction*100)}pct" / f"seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dm = HLSBurnScarsDataModule(
        dataset_path=config["data"]["dataset_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split_file=train_split,
    )
    datamodule = dm.build()

    # Model
    training_cfg = config["training"]

    if backbone == "prithvi_eo_v2_300":
        # Use TerraTorch task for Prithvi
        model_override = config.get("model_overrides", {}).get(backbone, {})
        task = BurnScarSegmentationTask(
            backbone=backbone,
            pretrained=True,
            loss=training_cfg["loss"],
            lr=training_cfg["lr"],
            weight_decay=training_cfg["weight_decay"],
            **model_override,
        )
    else:
        # Use our custom module for DINOv3/ResNet
        model_override = config.get("model_overrides", {}).get(backbone, {})
        task = Phase2LightningModule(
            backbone_name=backbone,
            num_classes=2,
            lr=training_cfg["lr"],
            weight_decay=training_cfg["weight_decay"],
            **model_override,
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=training_cfg["early_stopping_patience"],
        ),
    ]

    # Logger
    if use_wandb:
        logger = WandbLogger(
            project=config["experiment"]["project"],
            name=exp_name,
            save_dir=str(exp_dir),
            tags=[backbone, f"frac{int(fraction*100)}", f"seed{seed}"],
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(exp_dir),
            name="logs",
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        precision=training_cfg["precision"],
        accelerator=training_cfg["accelerator"],
        devices=training_cfg["devices"],
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(exp_dir),
        log_every_n_steps=training_cfg["log_every_n_steps"],
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(task, datamodule)

    # Test
    test_results = trainer.test(task, datamodule, ckpt_path="best")

    # Extract metrics
    test_metrics = test_results[0] if test_results else {}
    best_ckpt = trainer.checkpoint_callback.best_model_path

    # TerraTorch logs as "test/IoU_Burn scar", custom module logs as "test/iou_class1"
    test_iou_burn = test_metrics.get(
        "test/IoU_Burn scar",
        test_metrics.get("test/iou_class1", 0.0),
    )

    result = ExperimentResult(
        backbone=backbone,
        fraction=fraction,
        seed=seed,
        test_iou_burn=test_iou_burn,
        test_miou=test_metrics.get("test/mIoU", 0.0),
        val_loss=float(trainer.checkpoint_callback.best_model_score or 0),
        epochs_trained=trainer.current_epoch,
        checkpoint_path=str(best_ckpt),
    )

    # Save result
    with open(exp_dir / "result.json", "w") as f:
        json.dump(result.__dict__, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 label efficiency sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_sweep.yaml"),
        help="Path to sweep config",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Run only this backbone",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Run only this fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this seed",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine what to run
    backbones = (
        [args.backbone]
        if args.backbone
        else [b["name"] for b in config["backbones"]]
    )
    fractions = (
        [args.fraction]
        if args.fraction
        else config["subsets"]["fractions"]
    )
    seeds = (
        [args.seed]
        if args.seed
        else config["experiment"]["seeds"]
    )

    # Build experiment list
    experiments = [
        (backbone, fraction, seed)
        for backbone in backbones
        for fraction in fractions
        for seed in seeds
    ]

    console.print(f"\n[bold]Phase 2 Label Efficiency Sweep[/bold]")
    console.print(f"Total experiments: {len(experiments)}")
    console.print(f"Backbones: {backbones}")
    console.print(f"Fractions: {fractions}")
    console.print(f"Seeds: {seeds}")

    if args.dry_run:
        table = Table(title="Planned Experiments")
        table.add_column("Backbone")
        table.add_column("Fraction")
        table.add_column("Seed")
        for backbone, fraction, seed in experiments:
            table.add_row(backbone, f"{int(fraction*100)}%", str(seed))
        console.print(table)
        return 0

    # Run experiments
    output_dir = Path(config["experiment"]["output_dir"])
    results = []

    for i, (backbone, fraction, seed) in enumerate(experiments):
        console.print(
            f"\n[bold blue]Experiment {i+1}/{len(experiments)}:[/bold blue] "
            f"{backbone} @ {int(fraction*100)}% labels, seed={seed}"
        )

        try:
            result = run_single_experiment(
                backbone=backbone,
                fraction=fraction,
                seed=seed,
                config=config,
                output_dir=output_dir,
                use_wandb=not args.no_wandb,
            )
            results.append(result)
            console.print(
                f"[green]✓[/green] Test IoU (burn): {result.test_iou_burn:.4f}, "
                f"mIoU: {result.test_miou:.4f}"
            )
        except Exception as e:
            console.print(f"[red]✗[/red] Failed: {e}")

    # Save all results (merge with existing to support per-experiment invocations)
    results_file = output_dir / "all_results.json"
    existing_results = []
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    # Key by (backbone, fraction, seed) to deduplicate
    result_key = lambda r: (r["backbone"], r["fraction"], r["seed"])
    merged = {result_key(r): r for r in existing_results}
    for r in results:
        merged[result_key(r.__dict__)] = r.__dict__

    with open(results_file, "w") as f:
        json.dump(list(merged.values()), f, indent=2)

    console.print(f"\n[green]Completed {len(results)}/{len(experiments)} experiments[/green]")
    console.print(f"Results saved to {output_dir / 'all_results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
