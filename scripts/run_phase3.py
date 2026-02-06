#!/usr/bin/env python3
"""Run Phase 3 semi-supervised training (FixMatch + EMA Teacher).

Trains a Prithvi segmentation model using FixMatch with pseudo-labels from
an EMA teacher on unlabeled data, combined with supervised loss on the
labeled subset.

Usage:
    uv run scripts/run_phase3.py
    uv run scripts/run_phase3.py --config configs/phase3_ssl.yaml
    uv run scripts/run_phase3.py --label-fraction 0.25
    uv run scripts/run_phase3.py --no-cutmix       # ablation: disable CutMix
    uv run scripts/run_phase3.py --no-ema           # ablation: student self-trains
    uv run scripts/run_phase3.py --tau 0.90         # ablation: threshold sweep
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from rich.console import Console

from src.data.ssl_datamodule import SemiSupervisedDataModule
from src.training.ssl_task import FixMatchSegmentationTask

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 SSL training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3_ssl.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=None,
        help="Override label fraction (e.g., 0.10, 0.25). Determines split file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override confidence threshold",
    )
    parser.add_argument(
        "--lambda-u",
        type=float,
        default=None,
        help="Override unsupervised loss weight",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="Override EMA decay",
    )
    parser.add_argument(
        "--no-cutmix",
        action="store_true",
        help="Ablation: disable CutMix augmentation",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Ablation: disable EMA teacher (student generates own pseudo-labels)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    seed = args.seed or config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    data_cfg = config["data"]
    model_cfg = config["model"]
    ssl_cfg = config["ssl"]
    train_cfg = config["training"]

    if args.label_fraction is not None:
        pct = int(args.label_fraction * 100)
        data_cfg["labeled_split_file"] = f"train_{pct}pct.txt"

    if args.tau is not None:
        ssl_cfg["tau"] = args.tau
    if args.lambda_u is not None:
        ssl_cfg["lambda_u"] = args.lambda_u
    if args.ema_decay is not None:
        ssl_cfg["ema_decay"] = args.ema_decay
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = args.max_epochs

    # Experiment naming
    label_frac = data_cfg["labeled_split_file"].replace("train_", "").replace("pct.txt", "")
    ablation_tags = []
    if args.no_cutmix:
        ablation_tags.append("no-cutmix")
    if args.no_ema:
        ablation_tags.append("no-ema")
    if args.tau is not None:
        ablation_tags.append(f"tau{args.tau}")

    exp_name = f"phase3_ssl_{label_frac}pct_seed{seed}"
    if ablation_tags:
        exp_name += "_" + "_".join(ablation_tags)

    output_dir = Path(config["experiment"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Phase 3: FixMatch + EMA Teacher[/bold]")
    console.print(f"  Labels: {label_frac}%")
    console.print(f"  Seed: {seed}")
    console.print(f"  tau: {ssl_cfg['tau']}")
    console.print(f"  lambda_u: {ssl_cfg['lambda_u']}")
    console.print(f"  EMA decay: {ssl_cfg['ema_decay']}")
    console.print(f"  Warmup epochs: {ssl_cfg['warmup_epochs']}")
    console.print(f"  CutMix: {not args.no_cutmix}")
    console.print(f"  EMA teacher: {not args.no_ema}")
    console.print(f"  Output: {output_dir}")

    # Data module
    dm = SemiSupervisedDataModule(
        dataset_path=data_cfg["dataset_path"],
        labeled_split_file=data_cfg["labeled_split_file"],
        batch_size_labeled=data_cfg["batch_size_labeled"],
        batch_size_unlabeled=data_cfg["batch_size_unlabeled"],
        num_workers=data_cfg["num_workers"],
    )

    # Task
    task = FixMatchSegmentationTask(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        num_classes=2,
        img_size=model_cfg["img_size"],
        decoder_channels=model_cfg["decoder_channels"],
        head_dropout=model_cfg["head_dropout"],
        lr_backbone=train_cfg["lr_backbone"],
        lr_decoder=train_cfg["lr_decoder"],
        weight_decay=train_cfg["weight_decay"],
        tau=ssl_cfg["tau"],
        lambda_u=ssl_cfg["lambda_u"],
        ema_decay=ssl_cfg["ema_decay"],
        warmup_epochs=ssl_cfg["warmup_epochs"],
    )

    # Ablation: disable CutMix by replacing _cutmix with identity
    if args.no_cutmix:
        task._cutmix = lambda images, labels, mask: (images, labels, mask)

    # Ablation: disable EMA by making teacher = student (shared weights)
    if args.no_ema:
        task.teacher = task.student
        task._update_ema = lambda: None

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=train_cfg["early_stopping_patience"],
        ),
    ]

    # Logger
    use_wandb = train_cfg.get("use_wandb", True) and not args.no_wandb
    if use_wandb:
        tags = [model_cfg["backbone"], f"frac{label_frac}", f"seed{seed}", "phase3"]
        tags.extend(ablation_tags)
        logger = WandbLogger(
            project=config["experiment"]["project"],
            name=exp_name,
            save_dir=str(output_dir),
            tags=tags,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="logs",
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        precision=train_cfg["precision"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=train_cfg["log_every_n_steps"],
        enable_progress_bar=True,
    )

    # Train
    console.print("\n[bold blue]Starting training...[/bold blue]")
    trainer.fit(task, dm)

    # Test
    console.print("\n[bold blue]Running test evaluation...[/bold blue]")
    test_results = trainer.test(task, dm, ckpt_path="best")

    # Save results
    test_metrics = test_results[0] if test_results else {}
    result = {
        "experiment": exp_name,
        "label_fraction": label_frac,
        "seed": seed,
        "ssl_config": ssl_cfg,
        "ablations": {
            "no_cutmix": args.no_cutmix,
            "no_ema": args.no_ema,
        },
        "test_metrics": test_metrics,
        "test_iou_burn": test_metrics.get("test/iou_class1", 0.0),
        "test_miou": test_metrics.get("test/mIoU", 0.0),
        "epochs_trained": trainer.current_epoch,
        "checkpoint_path": str(trainer.checkpoint_callback.best_model_path),
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    console.print(f"\n[green]Done![/green]")
    console.print(f"  Test IoU (burn scar): {result['test_iou_burn']:.4f}")
    console.print(f"  Test mIoU: {result['test_miou']:.4f}")
    console.print(f"  Results saved to {output_dir / 'result.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
