#!/bin/bash
# Evaluate a trained checkpoint on the test set
# Usage: ./scripts/eval.sh <checkpoint_path>

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/eval.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  ./scripts/eval.sh outputs/baseline/checkpoints/last.ckpt"
    exit 1
fi

CHECKPOINT="$1"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=============================================="
echo "Evaluating checkpoint: $CHECKPOINT"
echo "=============================================="

uv run python -c "
import sys
sys.path.insert(0, '.')

import lightning.pytorch as pl
from src.data.dataset import HLSBurnScarsDataModule
from src.training.trainer import BurnScarSegmentationTask

# Load checkpoint
print('Loading checkpoint...')
task = BurnScarSegmentationTask.load_from_checkpoint('$CHECKPOINT')

# Setup data
print('Setting up data...')
data_wrapper = HLSBurnScarsDataModule(
    dataset_path='data/hls_burn_scars',
    batch_size=8,
    num_workers=4,
)
datamodule = data_wrapper.build()

# Create trainer for evaluation
trainer = pl.Trainer(
    accelerator='auto',
    devices=1,
    logger=False,
)

# Run test
print('Running evaluation on test set...')
results = trainer.test(model=task, datamodule=datamodule)

print('')
print('Results:')
for k, v in results[0].items():
    print(f'  {k}: {v:.4f}')
"
