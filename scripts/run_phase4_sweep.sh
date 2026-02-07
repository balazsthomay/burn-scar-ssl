#!/usr/bin/env bash
# Phase 4: PEFT (LoRA/DoRA) Sweep
#
# Runs 5 experiments comparing LoRA ranks, DoRA, and label efficiency.
# Estimated ~2.5 hours on RTX 4090.
#
# Usage:
#   bash scripts/run_phase4_sweep.sh
#   bash scripts/run_phase4_sweep.sh --no-wandb

set -euo pipefail

EXTRA_ARGS="${*}"
SEED=42
CONFIG="configs/phase4_peft.yaml"

echo "=========================================="
echo "Phase 4: PEFT Sweep"
echo "=========================================="

# Experiment 1: LoRA r=4, 100% labels
echo ""
echo "[1/5] LoRA r=4, alpha=8, 100% labels"
uv run scripts/run_phase4_peft.py \
    --config "${CONFIG}" \
    --method lora --rank 4 --alpha 8 \
    --label-fraction 1.0 --seed "${SEED}" ${EXTRA_ARGS}

# Experiment 2: LoRA r=8, 100% labels
echo ""
echo "[2/5] LoRA r=8, alpha=16, 100% labels"
uv run scripts/run_phase4_peft.py \
    --config "${CONFIG}" \
    --method lora --rank 8 --alpha 16 \
    --label-fraction 1.0 --seed "${SEED}" ${EXTRA_ARGS}

# Experiment 3: LoRA r=16, 100% labels
echo ""
echo "[3/5] LoRA r=16, alpha=32, 100% labels"
uv run scripts/run_phase4_peft.py \
    --config "${CONFIG}" \
    --method lora --rank 16 --alpha 32 \
    --label-fraction 1.0 --seed "${SEED}" ${EXTRA_ARGS}

# Experiment 4: DoRA r=8, 100% labels
echo ""
echo "[4/5] DoRA r=8, alpha=16, 100% labels"
uv run scripts/run_phase4_peft.py \
    --config "${CONFIG}" \
    --method dora --rank 8 --alpha 16 \
    --label-fraction 1.0 --seed "${SEED}" ${EXTRA_ARGS}

# Experiment 5: LoRA r=8, 10% labels
echo ""
echo "[5/5] LoRA r=8, alpha=16, 10% labels"
uv run scripts/run_phase4_peft.py \
    --config "${CONFIG}" \
    --method lora --rank 8 --alpha 16 \
    --label-fraction 0.1 --seed "${SEED}" ${EXTRA_ARGS}

echo ""
echo "=========================================="
echo "Phase 4 sweep complete."
echo "Results in outputs/phase4/"
echo "=========================================="
