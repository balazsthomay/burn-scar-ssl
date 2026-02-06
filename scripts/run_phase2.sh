#!/bin/bash
# Phase 2: Label Efficiency Experiments
# 9 runs: 3 backbones × 3 fractions × 1 seed
# ~1-2 hours on RTX 4090

set -e

FRACTIONS="0.10 0.50 1.0"
BACKBONES="prithvi_eo_v2_300 dinov3_vitl16_sat resnet50"
SEED=42

echo "=== Phase 2: Label Efficiency Experiments ==="

# Generate subsets if needed
if [ ! -f "data/hls_burn_scars/splits/train_10pct.txt" ]; then
    echo "Generating training subsets..."
    uv run scripts/generate_subsets.py
fi

echo "Backbones: $BACKBONES"
echo "Fractions: $FRACTIONS"
echo ""

for backbone in $BACKBONES; do
    for fraction in $FRACTIONS; do
        echo ">>> $backbone @ $(echo "$fraction * 100" | bc)% labels"
        uv run scripts/run_phase2_sweep.py \
            --backbone "$backbone" \
            --fraction "$fraction" \
            --seed "$SEED"
    done
done

echo "=== Done! Results at outputs/phase2/ and W&B ==="
