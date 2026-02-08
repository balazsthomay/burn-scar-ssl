#!/usr/bin/env python3
"""Run Phase 4 PEFT (LoRA/DoRA) experiments.

Parameter-efficient fine-tuning of Prithvi-EO-2.0-300M using LoRA/DoRA adapters.
Targets Q+V attention projections in the backbone while keeping decoder fully trainable.

Usage:
    uv run scripts/run_phase4_peft.py
    uv run scripts/run_phase4_peft.py --method dora --rank 8
    uv run scripts/run_phase4_peft.py --label-fraction 0.1 --max-epochs 1
    uv run scripts/run_phase4_peft.py --dry-run
"""

import argparse
import json
import sys
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

from src.data.dataset import HLSBurnScarsDataModule
from src.models.prithvi import build_prithvi_segmentation_model, get_model_info

console = Console()


def _extract_logits(output: torch.Tensor) -> torch.Tensor:
    """Extract raw logits from model output, handling TerraTorch's ModelOutput wrapper."""
    if hasattr(output, "output"):
        return output.output
    return output


def build_peft_config(
    method: str = "lora",
    rank: int = 8,
    alpha: int = 16,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.1,
    replace_qkv: str = "qkv",
    use_dora: bool = False,
) -> dict:
    """Build a TerraTorch-compatible PEFT config dict.

    Args:
        method: PEFT method — "lora" or "dora".
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor.
        target_modules: Which modules to apply adapters to.
            Defaults to ["q_linear", "v_linear"] (after QKV split).
        lora_dropout: Dropout for LoRA layers.
        replace_qkv: Module suffix to split fused QKV projections.
        use_dora: If True, use DoRA (weight-decomposed LoRA).

    Returns:
        Dict compatible with TerraTorch's EncoderDecoderFactory.build_model(peft_config=...).
    """
    if target_modules is None:
        target_modules = ["q_linear", "v_linear"]

    # DoRA is just LoRA with use_dora=True in peft_config_kwargs
    if method == "dora":
        use_dora = True

    return {
        "method": "LORA",
        "replace_qkv": replace_qkv,
        "peft_config_kwargs": {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": target_modules,
            "lora_dropout": lora_dropout,
            "use_dora": use_dora,
        },
    }


class Phase4PEFTModule(pl.LightningModule):
    """Lightning module for LoRA/DoRA fine-tuning of Prithvi segmentation models."""

    def __init__(
        self,
        peft_config: dict,
        backbone: str = "prithvi_eo_v2_300",
        num_classes: int = 2,
        img_size: int = 512,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_prithvi_segmentation_model(
            backbone=backbone,
            pretrained=True,
            num_classes=num_classes,
            img_size=img_size,
            peft_config=peft_config,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images = batch["image"]
        masks = batch["mask"]

        raw_output = self(images)
        logits = _extract_logits(raw_output)
        loss = self.loss_fn(logits, masks)

        # Per-class IoU
        preds = logits.argmax(dim=1)
        valid_mask = masks != -1

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


def run_experiment(
    method: str,
    rank: int,
    alpha: int,
    label_fraction: float,
    seed: int,
    config: dict,
    output_dir: Path,
    use_wandb: bool = True,
    max_epochs: int | None = None,
) -> dict:
    """Run a single PEFT experiment.

    Returns:
        Dict with metrics and param counts.
    """
    pl.seed_everything(seed, workers=True)

    use_dora = method == "dora"
    peft_cfg = build_peft_config(
        method=method,
        rank=rank,
        alpha=alpha,
        target_modules=config["peft"]["target_modules"],
        lora_dropout=config["peft"]["lora_dropout"],
        replace_qkv=config["peft"]["replace_qkv"],
        use_dora=use_dora,
    )

    training_cfg = config["training"]
    epochs = max_epochs if max_epochs is not None else training_cfg["max_epochs"]

    # Data
    if label_fraction < 1.0:
        pct = int(label_fraction * 100)
        train_split = f"train_{pct}pct.txt"
    else:
        train_split = None

    dm = HLSBurnScarsDataModule(
        dataset_path=config["data"]["dataset_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split_file=train_split,
    )
    datamodule = dm.build()

    # Model
    task = Phase4PEFTModule(
        peft_config=peft_cfg,
        backbone=config.get("backbone", "prithvi_eo_v2_300"),
        num_classes=2,
        img_size=config.get("img_size", 512),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )

    # Log param counts
    info = get_model_info(task.model)
    console.print(f"  Total params:     {info['total_parameters_millions']:.2f}M")
    console.print(f"  Trainable params: {info['trainable_parameters_millions']:.2f}M")
    console.print(
        f"  Trainable ratio:  "
        f"{info['trainable_parameters'] / info['total_parameters'] * 100:.1f}%"
    )

    # Experiment directory
    exp_name = f"{method}_r{rank}_a{alpha}_frac{int(label_fraction*100)}_seed{seed}"
    exp_dir = output_dir / method / f"r{rank}" / f"{int(label_fraction*100)}pct" / f"seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

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
            project=config.get("project", "burn-scar-ssl"),
            name=exp_name,
            save_dir=str(exp_dir),
            tags=[method, f"r{rank}", f"frac{int(label_fraction*100)}", f"seed{seed}"],
        )
    else:
        logger = TensorBoardLogger(save_dir=str(exp_dir), name="logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=training_cfg["precision"],
        accelerator=training_cfg["accelerator"],
        devices=training_cfg["devices"],
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(exp_dir),
        log_every_n_steps=training_cfg["log_every_n_steps"],
        enable_progress_bar=False,
    )

    # Train
    trainer.fit(task, datamodule)

    # Test
    test_results = trainer.test(task, datamodule, ckpt_path="best")
    test_metrics = test_results[0] if test_results else {}

    best_ckpt = trainer.checkpoint_callback.best_model_path

    result = {
        "method": method,
        "rank": rank,
        "alpha": alpha,
        "label_fraction": label_fraction,
        "seed": seed,
        "test_iou_burn": test_metrics.get("test/iou_class1", 0.0),
        "test_miou": test_metrics.get("test/mIoU", 0.0),
        "val_loss": float(trainer.checkpoint_callback.best_model_score or 0),
        "epochs_trained": trainer.current_epoch,
        "checkpoint_path": str(best_ckpt),
        "total_params_millions": info["total_parameters_millions"],
        "trainable_params_millions": info["trainable_parameters_millions"],
        "trainable_ratio_pct": info["trainable_parameters"] / info["total_parameters"] * 100,
    }

    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 PEFT experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase4_peft.yaml"),
        help="Path to PEFT config",
    )
    parser.add_argument("--method", type=str, default=None, help="lora or dora")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--label-fraction", type=float, default=None, help="Label fraction")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    method = args.method or config["peft"]["method"]
    rank = args.rank or config["peft"]["rank"]
    alpha = args.alpha or config["peft"]["alpha"]
    label_fraction = args.label_fraction if args.label_fraction is not None else 1.0
    seed = args.seed or config.get("seed", 42)

    console.print(f"\n[bold]Phase 4: PEFT ({method.upper()}) Fine-Tuning[/bold]")
    console.print(f"  Method:    {method}")
    console.print(f"  Rank:      {rank}")
    console.print(f"  Alpha:     {alpha}")
    console.print(f"  Labels:    {int(label_fraction * 100)}%")
    console.print(f"  Seed:      {seed}")

    if args.dry_run:
        peft_cfg = build_peft_config(
            method=method,
            rank=rank,
            alpha=alpha,
            target_modules=config["peft"]["target_modules"],
            lora_dropout=config["peft"]["lora_dropout"],
            replace_qkv=config["peft"]["replace_qkv"],
        )
        console.print(f"\n[bold]PEFT config:[/bold]")
        console.print(json.dumps(peft_cfg, indent=2))
        return 0

    output_dir = Path(config.get("output_dir", "outputs/phase4"))

    result = run_experiment(
        method=method,
        rank=rank,
        alpha=alpha,
        label_fraction=label_fraction,
        seed=seed,
        config=config,
        output_dir=output_dir,
        use_wandb=not args.no_wandb,
        max_epochs=args.max_epochs,
    )

    console.print(f"\n[green]Done.[/green]")
    console.print(f"  Test IoU (burn): {result['test_iou_burn']:.4f}")
    console.print(f"  Test mIoU:       {result['test_miou']:.4f}")
    console.print(f"  Trainable:       {result['trainable_params_millions']:.2f}M / {result['total_params_millions']:.2f}M ({result['trainable_ratio_pct']:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
