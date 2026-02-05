#!/bin/bash
# Train Prithvi-EO-2.0-300M baseline on HLS Burn Scars
# Usage: ./scripts/train.sh [--epochs N] [--batch-size N] [--lr FLOAT] [--no-wandb]

set -e

# Default config
CONFIG="configs/baseline.yaml"

echo "=============================================="
echo "Burn Scar Segmentation - Baseline Training"
echo "=============================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Run training with all passed arguments
echo "Starting training..."
echo "Config: $CONFIG"
echo "Arguments: $@"
echo ""

uv run python scripts/train_baseline.py --config "$CONFIG" "$@"
