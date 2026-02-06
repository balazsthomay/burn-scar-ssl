"""Tests for FixMatch + EMA Teacher SSL training task."""

import copy

import lightning.pytorch as pl
import pytest
import torch
import torch.nn as nn

from src.training.ssl_task import FixMatchSegmentationTask


# ---------------------------------------------------------------------------
# Helpers: lightweight mock model to avoid loading Prithvi weights in tests
# ---------------------------------------------------------------------------

class _MockEncoder(nn.Module):
    """Minimal encoder that mimics Prithvi's encoder interface."""

    def __init__(self, in_channels: int = 6, hidden: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _MockSegModel(nn.Module):
    """Minimal segmentation model with encoder/decoder split."""

    def __init__(self, in_channels: int = 6, num_classes: int = 2, hidden: int = 16):
        super().__init__()
        self.encoder = _MockEncoder(in_channels, hidden)
        self.decoder = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _MockTrainer:
    """Minimal mock for pl.Trainer to satisfy LightningModule properties."""

    def __init__(self, current_epoch: int = 0):
        self.current_epoch = current_epoch
        self.logger = None
        self.log_dir = "/tmp"
        self.callback_metrics = {}
        self.global_step = 0
        # Satisfy Lightning's _FabricModule checks
        self.strategy = type("S", (), {"root_device": torch.device("cpu")})()
        self.loggers = []


def _make_task(
    tau: float = 0.95,
    lambda_u: float = 1.0,
    ema_decay: float = 0.999,
    warmup_epochs: int = 0,
    num_classes: int = 2,
    current_epoch: int = 0,
) -> FixMatchSegmentationTask:
    """Create a FixMatchSegmentationTask with a mock model (no Prithvi download)."""
    task = FixMatchSegmentationTask.__new__(FixMatchSegmentationTask)
    # Manually init LightningModule
    pl.LightningModule.__init__(task)

    task.num_classes = num_classes
    task.lr_backbone = 1e-5
    task.lr_decoder = 1e-4
    task.weight_decay = 0.05
    task.tau = tau
    task.lambda_u = lambda_u
    task.ema_decay = ema_decay
    task.warmup_epochs = warmup_epochs

    task.student = _MockSegModel(num_classes=num_classes)
    task.teacher = copy.deepcopy(task.student)
    for p in task.teacher.parameters():
        p.requires_grad = False

    task.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    task._hparams_initial = {}
    task._hparams = {}

    # Attach mock trainer so current_epoch / logging works
    task._trainer = _MockTrainer(current_epoch=current_epoch)
    return task


def _make_batch(
    batch_size: int = 4,
    channels: int = 6,
    height: int = 32,
    width: int = 32,
    num_classes: int = 2,
):
    """Create a synthetic combined batch (labeled + unlabeled)."""
    labeled = {
        "image_weak": torch.randn(batch_size, channels, height, width),
        "image_strong": torch.randn(batch_size, channels, height, width),
        "mask": torch.randint(0, num_classes, (batch_size, height, width)),
    }
    unlabeled = {
        "image_weak": torch.randn(batch_size, channels, height, width),
        "image_strong": torch.randn(batch_size, channels, height, width),
    }
    return {"labeled": labeled, "unlabeled": unlabeled}


# ---------------------------------------------------------------------------
# Tests: FixMatch training step
# ---------------------------------------------------------------------------

def _noop_log(self, *args, **kwargs):
    """No-op replacement for self.log in tests without a full Trainer."""
    pass


class TestFixMatchTrainingStep:
    def test_training_step_returns_scalar_loss(self):
        task = _make_task(warmup_epochs=0)
        task.log = _noop_log.__get__(task)
        batch = _make_batch()
        loss = task.training_step(batch, batch_idx=0)
        assert loss.dim() == 0  # scalar
        assert loss.requires_grad

    def test_training_step_during_warmup_has_zero_unsup_loss(self):
        task = _make_task(warmup_epochs=10, current_epoch=0)
        task.log = _noop_log.__get__(task)
        batch = _make_batch()

        loss = task.training_step(batch, batch_idx=0)

        assert task._effective_lambda_u() == 0.0
        assert loss.dim() == 0

    def test_training_step_after_warmup_has_nonzero_lambda(self):
        task = _make_task(warmup_epochs=5, lambda_u=1.0, current_epoch=10)

        assert task._effective_lambda_u() == 1.0

    def test_output_shape_matches_input(self):
        task = _make_task()
        x = torch.randn(2, 6, 32, 32)
        logits = task(x)
        assert logits.shape == (2, 2, 32, 32)


# ---------------------------------------------------------------------------
# Tests: EMA update
# ---------------------------------------------------------------------------

class TestEMAUpdate:
    def test_ema_moves_teacher_toward_student(self):
        task = _make_task(ema_decay=0.9)

        # Record initial teacher params
        initial_teacher = {
            name: p.data.clone()
            for name, p in task.teacher.named_parameters()
        }

        # Perturb student to differ from teacher
        with torch.no_grad():
            for p in task.student.parameters():
                p.add_(torch.randn_like(p))

        # Update EMA
        task._update_ema()

        # Teacher should have moved toward student
        for name, p in task.teacher.named_parameters():
            old = initial_teacher[name]
            student_p = dict(task.student.named_parameters())[name]
            expected = 0.9 * old + 0.1 * student_p.data
            torch.testing.assert_close(p.data, expected)

    def test_ema_with_decay_1_keeps_teacher_fixed(self):
        task = _make_task(ema_decay=1.0)

        initial_teacher = {
            name: p.data.clone()
            for name, p in task.teacher.named_parameters()
        }

        with torch.no_grad():
            for p in task.student.parameters():
                p.add_(torch.randn_like(p) * 10)

        task._update_ema()

        for name, p in task.teacher.named_parameters():
            torch.testing.assert_close(p.data, initial_teacher[name])

    def test_ema_with_decay_0_copies_student(self):
        task = _make_task(ema_decay=0.0)

        with torch.no_grad():
            for p in task.student.parameters():
                p.add_(torch.randn_like(p) * 10)

        task._update_ema()

        for (n_t, p_t), (n_s, p_s) in zip(
            task.teacher.named_parameters(),
            task.student.named_parameters(),
        ):
            torch.testing.assert_close(p_t.data, p_s.data)

    def test_teacher_has_no_grad(self):
        task = _make_task()
        for p in task.teacher.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Tests: Pseudo-label generation & confidence masking
# ---------------------------------------------------------------------------

class TestPseudoLabels:
    def test_pseudo_labels_shape(self):
        task = _make_task()
        images = torch.randn(4, 6, 32, 32)
        pseudo, conf_mask = task._generate_pseudo_labels(images)

        assert pseudo.shape == (4, 32, 32)
        assert conf_mask.shape == (4, 32, 32)
        assert pseudo.dtype == torch.int64 or pseudo.dtype == torch.long
        assert conf_mask.dtype == torch.bool

    def test_high_threshold_filters_most_pixels(self):
        task = _make_task(tau=0.99)

        # Use a model that produces near-uniform predictions (low confidence)
        with torch.no_grad():
            for p in task.teacher.parameters():
                p.fill_(0.0)  # near-uniform logits

        images = torch.randn(4, 6, 32, 32)
        _, conf_mask = task._generate_pseudo_labels(images)

        # With near-uniform logits for 2 classes, max prob ~ 0.5,
        # so tau=0.99 should filter almost everything
        assert conf_mask.float().mean() < 0.5

    def test_low_threshold_keeps_most_pixels(self):
        task = _make_task(tau=0.0)
        images = torch.randn(4, 6, 32, 32)
        _, conf_mask = task._generate_pseudo_labels(images)

        # tau=0 → all pixels kept
        assert conf_mask.all()

    def test_pseudo_labels_are_valid_classes(self):
        task = _make_task(num_classes=2)
        images = torch.randn(4, 6, 32, 32)
        pseudo, _ = task._generate_pseudo_labels(images)
        assert (pseudo >= 0).all()
        assert (pseudo < 2).all()


# ---------------------------------------------------------------------------
# Tests: CutMix
# ---------------------------------------------------------------------------

class TestCutMix:
    def test_cutmix_output_shapes(self):
        task = _make_task()
        images = torch.randn(4, 6, 32, 32)
        labels = torch.randint(0, 2, (4, 32, 32))
        mask = torch.ones(4, 32, 32, dtype=torch.bool)

        mixed_img, mixed_lbl, mixed_mask = task._cutmix(images, labels, mask)

        assert mixed_img.shape == images.shape
        assert mixed_lbl.shape == labels.shape
        assert mixed_mask.shape == mask.shape

    def test_cutmix_modifies_at_least_some_pixels(self):
        """CutMix should paste a rectangle, changing some pixels (probabilistically)."""
        task = _make_task()
        torch.manual_seed(42)

        images = torch.randn(8, 6, 64, 64)
        labels = torch.zeros(8, 64, 64, dtype=torch.long)
        mask = torch.ones(8, 64, 64, dtype=torch.bool)

        # Run multiple times — at least one should modify pixels
        any_changed = False
        for _ in range(10):
            mixed_img, _, _ = task._cutmix(images, labels, mask)
            if not torch.allclose(mixed_img, images):
                any_changed = True
                break

        assert any_changed, "CutMix should modify at least some pixels across multiple runs"

    def test_cutmix_label_consistency_with_image(self):
        """Where image pixels are replaced, labels should also be replaced."""
        task = _make_task()
        torch.manual_seed(0)

        B, C, H, W = 4, 6, 32, 32

        # Each sample has a distinct constant image and label
        images = torch.arange(B).float().view(B, 1, 1, 1).expand(B, C, H, W).clone()
        labels = torch.arange(B).view(B, 1, 1).expand(B, H, W).clone()
        mask = torch.ones(B, H, W, dtype=torch.bool)

        mixed_img, mixed_lbl, _ = task._cutmix(images, labels, mask)

        # For each sample, wherever the image changed from its original
        # constant, the label should also have changed
        for b in range(B):
            original_val = b
            img_changed = (mixed_img[b, 0] != original_val)
            lbl_changed = (mixed_lbl[b] != original_val)
            assert (img_changed == lbl_changed).all(), (
                f"Image and label changes should be spatially consistent for sample {b}"
            )

    def test_cutmix_batch_size_1(self):
        """CutMix should work (degenerate but not crash) with batch_size=1."""
        task = _make_task()
        images = torch.randn(1, 6, 32, 32)
        labels = torch.randint(0, 2, (1, 32, 32))
        mask = torch.ones(1, 32, 32, dtype=torch.bool)

        # Should not raise
        mixed_img, mixed_lbl, mixed_mask = task._cutmix(images, labels, mask)
        assert mixed_img.shape == images.shape


# ---------------------------------------------------------------------------
# Tests: Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_lambda_zero_during_warmup(self):
        for epoch in range(5):
            task = _make_task(warmup_epochs=5, current_epoch=epoch)
            assert task._effective_lambda_u() == 0.0

    def test_warmup_lambda_active_after_warmup(self):
        task = _make_task(warmup_epochs=5, lambda_u=2.0, current_epoch=5)
        assert task._effective_lambda_u() == 2.0

        task._trainer.current_epoch = 100
        assert task._effective_lambda_u() == 2.0

    def test_zero_warmup_means_always_active(self):
        task = _make_task(warmup_epochs=0, lambda_u=1.0, current_epoch=0)
        assert task._effective_lambda_u() == 1.0


# ---------------------------------------------------------------------------
# Tests: Validation / Test step
# ---------------------------------------------------------------------------

class TestEvalStep:
    def test_eval_step_returns_loss(self):
        task = _make_task()
        task.log = _noop_log.__get__(task)
        batch = {
            "image": torch.randn(4, 6, 32, 32),
            "mask": torch.randint(0, 2, (4, 32, 32)),
        }

        loss = task._eval_step(batch, "val")
        assert loss.dim() == 0
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Tests: Optimizer configuration
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_configure_optimizers_returns_adamw(self):
        task = _make_task()
        result = task.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_differential_lr_groups(self):
        task = _make_task()
        optimizer = task.configure_optimizers()

        # Should have 2 param groups
        assert len(optimizer.param_groups) == 2

        lrs = {pg["lr"] for pg in optimizer.param_groups}
        assert 1e-5 in lrs
        assert 1e-4 in lrs
