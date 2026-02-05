"""Metrics for burn scar segmentation."""

import torch
from torchmetrics import JaccardIndex, Metric


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -1,
) -> dict[str, float]:
    """Compute Intersection over Union (IoU) for segmentation.

    Args:
        predictions: Predicted class labels (B, H, W).
        targets: Ground truth labels (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        Dictionary with per-class IoU and mean IoU.
    """
    # Create mask for valid pixels
    valid_mask = targets != ignore_index

    # Flatten and filter
    preds_flat = predictions[valid_mask]
    targets_flat = targets[valid_mask]

    iou_per_class = []

    for cls in range(num_classes):
        pred_cls = preds_flat == cls
        target_cls = targets_flat == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(float("nan"))

        iou_per_class.append(iou.item())

    # Compute mean IoU (excluding NaN)
    valid_ious = [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")

    return {
        "iou_not_burned": iou_per_class[0],
        "iou_burn_scar": iou_per_class[1],
        "mean_iou": mean_iou,
    }


class SegmentationMetrics:
    """Wrapper for segmentation metrics using torchmetrics.

    Provides IoU (Jaccard Index) metrics for burn scar segmentation
    with support for ignoring no-data pixels.
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = -1,
        device: str | torch.device = "cpu",
    ):
        """Initialize segmentation metrics.

        Args:
            num_classes: Number of segmentation classes.
            ignore_index: Index to ignore in metric computation.
            device: Device to place metrics on.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Per-class IoU
        self.iou_metric = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ).to(device)

        # Mean IoU
        self.mean_iou_metric = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="macro",
        ).to(device)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with new batch.

        Args:
            predictions: Predicted class labels (B, H, W) or logits (B, C, H, W).
            targets: Ground truth labels (B, H, W).
        """
        # Convert logits to class predictions if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        self.iou_metric.update(predictions, targets)
        self.mean_iou_metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with per-class IoU and mean IoU.
        """
        per_class_iou = self.iou_metric.compute()
        mean_iou = self.mean_iou_metric.compute()

        return {
            "iou_not_burned": per_class_iou[0].item(),
            "iou_burn_scar": per_class_iou[1].item(),
            "mean_iou": mean_iou.item(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.iou_metric.reset()
        self.mean_iou_metric.reset()
