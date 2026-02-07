"""Tests for Phase 4 PEFT (LoRA/DoRA) model building and training module."""

import pytest
import torch

from src.models.prithvi import build_prithvi_segmentation_model, get_model_info


def _make_peft_config(rank: int = 8, alpha: int = 16, use_dora: bool = False) -> dict:
    """Helper to build a standard LoRA/DoRA peft config."""
    return {
        "method": "LORA",
        "replace_qkv": "qkv",
        "peft_config_kwargs": {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["q_linear", "v_linear"],
            "lora_dropout": 0.1,
            "use_dora": use_dora,
        },
    }


@pytest.mark.slow
class TestPEFTModelBuild:
    """Tests for building Prithvi model with PEFT adapters."""

    def test_build_with_lora(self):
        """Should build model with LoRA config without errors."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=8),
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_build_with_dora(self):
        """Should build model with DoRA (use_dora=True) without errors."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=8, use_dora=True),
        )
        assert model is not None

    def test_build_with_no_peft_config(self):
        """peft_config=None should build a standard model (no adapters)."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=None,
        )
        info = get_model_info(model)
        # All params should be trainable with no PEFT
        assert info["trainable_parameters"] == info["total_parameters"]

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_build_with_different_ranks(self, rank):
        """Should build successfully with different LoRA ranks."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=rank, alpha=rank * 2),
        )
        assert model is not None


@pytest.mark.slow
class TestPEFTParamCounts:
    """Tests for parameter efficiency of PEFT models."""

    @pytest.fixture
    def baseline_info(self):
        """Get param counts for a non-PEFT model."""
        model = build_prithvi_segmentation_model(
            pretrained=False, img_size=224
        )
        return get_model_info(model)

    @pytest.fixture
    def lora_model(self):
        """Build a LoRA model."""
        return build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=8),
        )

    def test_trainable_params_reduced(self, lora_model, baseline_info):
        """LoRA should have far fewer trainable params than full fine-tuning."""
        lora_info = get_model_info(lora_model)
        assert lora_info["trainable_parameters"] < baseline_info["trainable_parameters"]

    def test_trainable_ratio_under_10_pct(self, lora_model):
        """Trainable params should be < 10% of total with LoRA on Q+V only."""
        info = get_model_info(lora_model)
        ratio = info["trainable_parameters"] / info["total_parameters"]
        assert ratio < 0.10, f"Trainable ratio {ratio:.2%} exceeds 10%"

    def test_total_params_similar(self, lora_model, baseline_info):
        """Total param count should be close to baseline (adapters are small)."""
        lora_info = get_model_info(lora_model)
        # LoRA adds a small number of params; total should be within 5% of baseline
        ratio = lora_info["total_parameters"] / baseline_info["total_parameters"]
        assert 0.95 < ratio < 1.05

    def test_higher_rank_more_trainable(self):
        """Higher LoRA rank should result in more trainable parameters."""
        model_r4 = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=4),
        )
        model_r16 = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            peft_config=_make_peft_config(rank=16),
        )
        info_r4 = get_model_info(model_r4)
        info_r16 = get_model_info(model_r16)
        assert info_r16["trainable_parameters"] > info_r4["trainable_parameters"]


@pytest.mark.slow
class TestPEFTForwardPass:
    """Tests for forward pass through PEFT models."""

    @pytest.fixture
    def lora_model(self):
        return build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            num_classes=2,
            peft_config=_make_peft_config(rank=8),
        )

    def test_forward_shape(self, lora_model):
        """Forward pass should produce correct output shape."""
        x = torch.randn(2, 6, 224, 224)
        output = lora_model(x)
        logits = output.output if hasattr(output, "output") else output
        assert logits.shape == (2, 2, 224, 224)

    def test_forward_dtype(self, lora_model):
        """Output should be float32."""
        x = torch.randn(1, 6, 224, 224)
        output = lora_model(x)
        logits = output.output if hasattr(output, "output") else output
        assert logits.dtype == torch.float32

    def test_forward_dora(self):
        """DoRA model should produce correct output shape."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
            num_classes=2,
            peft_config=_make_peft_config(rank=8, use_dora=True),
        )
        x = torch.randn(1, 6, 224, 224)
        output = model(x)
        logits = output.output if hasattr(output, "output") else output
        assert logits.shape == (1, 2, 224, 224)

    def test_backward_pass(self, lora_model):
        """Backward pass should compute gradients for adapter params."""
        x = torch.randn(1, 6, 224, 224)
        output = lora_model(x)
        logits = output.output if hasattr(output, "output") else output
        loss = logits.sum()
        loss.backward()

        # Check that at least some adapter params have gradients
        has_grad = False
        for name, param in lora_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No adapter parameters received gradients"


@pytest.mark.slow
class TestPhase4PEFTModule:
    """Tests for the Phase4PEFTModule Lightning module."""

    @pytest.fixture
    def module(self):
        from scripts.run_phase4_peft import Phase4PEFTModule

        peft_cfg = _make_peft_config(rank=8)
        task = Phase4PEFTModule(
            peft_config=peft_cfg,
            backbone="prithvi_eo_v2_300",
            num_classes=2,
            img_size=224,
            lr=1e-4,
            weight_decay=0.05,
        )
        # Patch log to avoid needing a full Trainer
        task.log = lambda *args, **kwargs: None
        return task

    def test_training_step_returns_scalar_loss(self, module):
        """training_step should return a scalar loss tensor."""
        batch = {
            "image": torch.randn(2, 6, 224, 224),
            "mask": torch.randint(0, 2, (2, 224, 224)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

    def test_validation_step_returns_scalar_loss(self, module):
        """validation_step should return a scalar loss tensor."""
        batch = {
            "image": torch.randn(2, 6, 224, 224),
            "mask": torch.randint(0, 2, (2, 224, 224)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.ndim == 0

    def test_configure_optimizers_returns_adamw(self, module):
        """configure_optimizers should return AdamW."""
        optim = module.configure_optimizers()
        assert isinstance(optim, torch.optim.AdamW)

    def test_configure_optimizers_lr(self, module):
        """Optimizer should use the configured learning rate."""
        optim = module.configure_optimizers()
        assert optim.defaults["lr"] == 1e-4

    def test_configure_optimizers_weight_decay(self, module):
        """Optimizer should use the configured weight decay."""
        optim = module.configure_optimizers()
        assert optim.defaults["weight_decay"] == 0.05

    def test_hparams_saved(self, module):
        """Hyperparameters should be saved."""
        assert module.hparams["lr"] == 1e-4
        assert module.hparams["backbone"] == "prithvi_eo_v2_300"
        assert module.hparams["peft_config"]["method"] == "LORA"


@pytest.mark.slow
class TestBuildPeftConfig:
    """Tests for the build_peft_config helper."""

    def test_lora_config(self):
        from scripts.run_phase4_peft import build_peft_config

        cfg = build_peft_config(method="lora", rank=8, alpha=16)
        assert cfg["method"] == "LORA"
        assert cfg["peft_config_kwargs"]["r"] == 8
        assert cfg["peft_config_kwargs"]["lora_alpha"] == 16
        assert cfg["peft_config_kwargs"]["use_dora"] is False

    def test_dora_config(self):
        from scripts.run_phase4_peft import build_peft_config

        cfg = build_peft_config(method="dora", rank=8, alpha=16)
        assert cfg["method"] == "LORA"  # DoRA uses LORA method with use_dora=True
        assert cfg["peft_config_kwargs"]["use_dora"] is True

    def test_default_target_modules(self):
        from scripts.run_phase4_peft import build_peft_config

        cfg = build_peft_config()
        assert cfg["peft_config_kwargs"]["target_modules"] == ["q_linear", "v_linear"]

    def test_custom_target_modules(self):
        from scripts.run_phase4_peft import build_peft_config

        cfg = build_peft_config(target_modules=["q_linear", "k_linear", "v_linear"])
        assert cfg["peft_config_kwargs"]["target_modules"] == [
            "q_linear",
            "k_linear",
            "v_linear",
        ]

    def test_replace_qkv_key(self):
        from scripts.run_phase4_peft import build_peft_config

        cfg = build_peft_config(replace_qkv="qkv")
        assert cfg["replace_qkv"] == "qkv"
