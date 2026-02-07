"""FixMatch + EMA Teacher training task for semi-supervised segmentation.

Implements the FixMatch algorithm with an Exponential Moving Average teacher
and CutMix augmentation for pseudo-label-based semi-supervised learning.

Algorithm per training step:
    1. Labeled batch → student(strong_view) vs true mask → L_supervised
    2. Unlabeled batch → teacher(weak_view) → pseudo-labels (argmax + confidence mask)
    3. CutMix: mix pairs of (unlabeled_strong, pseudo_labels) with random rectangles
    4. Student(unlabeled_strong_mixed) vs pseudo_labels_mixed → L_unsupervised (masked)
    5. L_total = L_sup + lambda_u * L_unsup
    6. EMA update: theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student
"""

import copy

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.prithvi import build_prithvi_segmentation_model


def _extract_logits(output: torch.Tensor) -> torch.Tensor:
    """Extract raw logits from model output, handling TerraTorch's ModelOutput wrapper."""
    if hasattr(output, "output"):
        return output.output
    return output


class FixMatchSegmentationTask(pl.LightningModule):
    """FixMatch + EMA Teacher for semi-supervised semantic segmentation.

    The student model is trained on both labeled data (with true masks) and
    unlabeled data (with pseudo-labels from the EMA teacher). CutMix is applied
    to unlabeled samples for additional regularization.
    """

    def __init__(
        self,
        backbone: str = "prithvi_eo_v2_300",
        pretrained: bool = True,
        num_classes: int = 2,
        img_size: int = 512,
        decoder_channels: list[int] | None = None,
        head_dropout: float = 0.1,
        lr_backbone: float = 1e-5,
        lr_decoder: float = 1e-4,
        weight_decay: float = 0.05,
        tau: float = 0.95,
        lambda_u: float = 1.0,
        ema_decay: float = 0.999,
        warmup_epochs: int = 5,
    ):
        """Initialize FixMatch task.

        Args:
            backbone: Prithvi backbone variant.
            pretrained: Whether to use pretrained weights.
            num_classes: Number of segmentation classes.
            img_size: Input image size.
            decoder_channels: UNet decoder channel dims.
            head_dropout: Dropout for segmentation head.
            lr_backbone: Learning rate for backbone parameters.
            lr_decoder: Learning rate for decoder/head parameters.
            weight_decay: Weight decay for optimizer.
            tau: Confidence threshold for pseudo-label filtering.
            lambda_u: Weight for unsupervised loss.
            ema_decay: EMA decay coefficient for teacher update.
            warmup_epochs: Number of supervised-only warmup epochs (lambda_u=0).
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_decoder = lr_decoder
        self.weight_decay = weight_decay
        self.tau = tau
        self.lambda_u = lambda_u
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs

        # Build student
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        self.student = build_prithvi_segmentation_model(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            decoder_channels=decoder_channels,
            head_dropout=head_dropout,
        )

        # Build teacher as a deep copy of student, no gradients
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the student model."""
        return _extract_logits(self.student(x))

    @torch.no_grad()
    def _generate_pseudo_labels(
        self, images_weak: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo-labels from teacher on weakly-augmented images.

        Args:
            images_weak: Weakly augmented images (B, C, H, W).

        Returns:
            pseudo_labels: Argmax predictions (B, H, W).
            confidence_mask: Boolean mask where max probability >= tau (B, H, W).
        """
        logits = _extract_logits(self.teacher(images_weak))
        probs = F.softmax(logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confidence_mask = max_probs >= self.tau
        return pseudo_labels, confidence_mask

    def _cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CutMix to pairs of (image, pseudo-label, confidence_mask).

        Shuffles the batch and pastes a random rectangle from the shuffled
        sample onto the original. Both labels and confidence masks are
        mixed consistently.

        Args:
            images: Strong-augmented images (B, C, H, W).
            labels: Pseudo-labels (B, H, W).
            mask: Confidence mask (B, H, W).

        Returns:
            Mixed images, labels, and confidence masks.
        """
        B, C, H, W = images.shape
        # Random shuffle indices
        indices = torch.randperm(B, device=images.device)

        # Random rectangle: sample lambda ~ Beta(1, 1) = Uniform(0, 1)
        lam = torch.rand(1, device=images.device).item()
        cut_ratio = (1.0 - lam) ** 0.5  # side length ratio

        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random center
        cy = torch.randint(0, H, (1,), device=images.device).item()
        cx = torch.randint(0, W, (1,), device=images.device).item()

        # Clamp to image bounds
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        # Mix
        mixed_images = images.clone()
        mixed_labels = labels.clone()
        mixed_mask = mask.clone()

        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        mixed_labels[:, y1:y2, x1:x2] = labels[indices, y1:y2, x1:x2]
        mixed_mask[:, y1:y2, x1:x2] = mask[indices, y1:y2, x1:x2]

        return mixed_images, mixed_labels, mixed_mask

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update teacher weights as EMA of student weights."""
        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters(), strict=True
        ):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def _effective_lambda_u(self) -> float:
        """Return lambda_u, respecting warmup (0 during warmup epochs)."""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        return self.lambda_u

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]
        unlabeled_batch = batch["unlabeled"]

        # --- Supervised loss ---
        labeled_strong = labeled_batch["image_strong"]
        true_mask = labeled_batch["mask"]

        student_logits_labeled = _extract_logits(self.student(labeled_strong))
        loss_sup_per_pixel = self.loss_fn(student_logits_labeled, true_mask)
        loss_sup = loss_sup_per_pixel.mean()

        # --- Unsupervised loss ---
        effective_lambda = self._effective_lambda_u()

        if effective_lambda > 0:
            unlabeled_weak = unlabeled_batch["image_weak"]
            unlabeled_strong = unlabeled_batch["image_strong"]

            # Teacher generates pseudo-labels from weak augmentation
            pseudo_labels, confidence_mask = self._generate_pseudo_labels(unlabeled_weak)

            # CutMix
            mixed_strong, mixed_pseudo, mixed_conf = self._cutmix(
                unlabeled_strong, pseudo_labels, confidence_mask
            )

            # Student prediction on mixed strong augmentation
            student_logits_unlabeled = _extract_logits(self.student(mixed_strong))
            loss_unsup_per_pixel = self.loss_fn(student_logits_unlabeled, mixed_pseudo)

            # Mask by confidence
            if mixed_conf.sum() > 0:
                loss_unsup = (loss_unsup_per_pixel * mixed_conf.float()).sum() / mixed_conf.sum()
            else:
                loss_unsup = torch.tensor(0.0, device=self.device)

            pseudo_mask_ratio = confidence_mask.float().mean()
        else:
            loss_unsup = torch.tensor(0.0, device=self.device)
            pseudo_mask_ratio = torch.tensor(0.0, device=self.device)

        loss_total = loss_sup + effective_lambda * loss_unsup

        # EMA update
        self._update_ema()

        # Logging
        self.log("train/loss_sup", loss_sup, prog_bar=True)
        self.log("train/loss_unsup", loss_unsup, prog_bar=True)
        self.log("train/loss_total", loss_total, prog_bar=True)
        self.log("train/pseudo_mask_ratio", pseudo_mask_ratio, prog_bar=False)
        self.log("train/effective_lambda_u", effective_lambda, prog_bar=False)

        return loss_total

    def _eval_step(self, batch, stage: str):
        """Shared evaluation step for val/test."""
        images = batch["image"]
        masks = batch["mask"]

        logits = _extract_logits(self.student(images))
        loss = F.cross_entropy(logits, masks, ignore_index=-1)

        preds = logits.argmax(dim=1)
        valid_mask = masks != -1

        ious = []
        for cls in range(self.num_classes):
            pred_cls = (preds == cls) & valid_mask
            target_cls = (masks == cls) & valid_mask
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
            self.log(f"{stage}/iou_class{cls}", iou, prog_bar=False, sync_dist=True)

        miou = sum(ious) / len(ious)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mIoU", miou, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        # Differential learning rates: backbone gets lower LR
        encoder_params = []
        decoder_params = []

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_backbone},
                {"params": decoder_params, "lr": self.lr_decoder},
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer
