#!/usr/bin/env bash
# Phase 5: Full deployment pipeline
#
# Orchestrates retraining + ONNX export for both model variants:
#   1. Full fine-tuned Prithvi @ 100% labels (accuracy ceiling)
#   2. LoRA r=16 @ 100% labels (best PEFT result)
#
# Usage:
#   bash scripts/run_phase5_sweep.sh
#   bash scripts/run_phase5_sweep.sh --no-wandb
#   bash scripts/run_phase5_sweep.sh --skip-train

set -euo pipefail

WANDB_FLAG=""
SKIP_TRAIN=false
SEED=42

for arg in "$@"; do
    case $arg in
        --no-wandb) WANDB_FLAG="--no-wandb" ;;
        --skip-train) SKIP_TRAIN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

OUTPUT_DIR="outputs/phase5"
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Phase 5: Deployment Pipeline"
echo "================================================"

# ──────────────────────────────────────────────────
# 1. Retrain full fine-tuned Prithvi @ 100% labels
# ──────────────────────────────────────────────────
FULL_FT_DIR="$OUTPUT_DIR/full_ft"
FULL_FT_CKPT="$FULL_FT_DIR/prithvi_eo_v2_300/100pct/seed${SEED}/checkpoints/best.ckpt"

if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[1/4] Training full fine-tuned Prithvi @ 100% labels..."
    uv run scripts/run_phase2_sweep.py \
        --backbone prithvi_eo_v2_300 \
        --fraction 1.0 \
        --seed "$SEED" \
        $WANDB_FLAG \
        --config configs/phase2_sweep.yaml

    # Find the checkpoint (Phase 2 saves under output_dir/backbone/pct/seed/)
    PHASE2_OUTPUT="outputs/phase2"
    FULL_FT_CKPT_FOUND=$(find "$PHASE2_OUTPUT/prithvi_eo_v2_300/100pct/seed${SEED}/checkpoints" -name "best*.ckpt" | head -1)

    if [ -z "$FULL_FT_CKPT_FOUND" ]; then
        echo "ERROR: Full FT checkpoint not found"
        exit 1
    fi
    FULL_FT_CKPT="$FULL_FT_CKPT_FOUND"
    echo "  Checkpoint: $FULL_FT_CKPT"
fi

# ──────────────────────────────────────────────────
# 2. Retrain LoRA r=16 @ 100% labels
# ──────────────────────────────────────────────────
LORA_DIR="$OUTPUT_DIR/lora_r16"
LORA_CKPT="$LORA_DIR/lora/r16/100pct/seed${SEED}/checkpoints/best.ckpt"

if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[2/4] Training LoRA r=16 @ 100% labels..."
    uv run scripts/run_phase4_peft.py \
        --method lora \
        --rank 16 \
        --alpha 32 \
        --seed "$SEED" \
        $WANDB_FLAG

    # Find the checkpoint
    PHASE4_OUTPUT="outputs/phase4"
    LORA_CKPT_FOUND=$(find "$PHASE4_OUTPUT/lora/r16/100pct/seed${SEED}/checkpoints" -name "best*.ckpt" | head -1)

    if [ -z "$LORA_CKPT_FOUND" ]; then
        echo "ERROR: LoRA checkpoint not found"
        exit 1
    fi
    LORA_CKPT="$LORA_CKPT_FOUND"
    echo "  Checkpoint: $LORA_CKPT"
fi

# ──────────────────────────────────────────────────
# 3. Export + benchmark full fine-tuned model
# ──────────────────────────────────────────────────
echo ""
echo "[3/4] Exporting + benchmarking full fine-tuned Prithvi..."
uv run scripts/run_phase5_deploy.py \
    --checkpoint "$FULL_FT_CKPT" \
    --checkpoint-type terratorch \
    --output-dir "$OUTPUT_DIR/full_ft" \
    $WANDB_FLAG

# ──────────────────────────────────────────────────
# 4. Export + benchmark LoRA model
# ──────────────────────────────────────────────────
echo ""
echo "[4/4] Exporting + benchmarking LoRA r=16..."
uv run scripts/run_phase5_deploy.py \
    --checkpoint "$LORA_CKPT" \
    --checkpoint-type peft \
    --output-dir "$OUTPUT_DIR/lora_r16" \
    $WANDB_FLAG

echo ""
echo "================================================"
echo "Phase 5 complete."
echo "Results: $OUTPUT_DIR/"
echo "================================================"
