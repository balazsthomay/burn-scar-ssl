#!/bin/bash
# Setup script for burn-scar-ssl on CUDA machines
# Tested on: Ubuntu 22.04 with CUDA 12.x

set -e  # Exit on error

echo "=============================================="
echo "Burn Scar SSL - Phase 1 Setup"
echo "=============================================="

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. CUDA may not be available."
else
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "uv version: $(uv --version)"

# Sync dependencies
echo ""
echo "Installing Python dependencies..."
uv sync

# Download dataset
echo ""
echo "Downloading HLS Burn Scars dataset (~2.6GB)..."
uv run python scripts/download_data.py

# Verify installation
echo ""
echo "Verifying TerraTorch installation..."
uv run python -c "
from terratorch.registry import BACKBONE_REGISTRY
models = [m for m in BACKBONE_REGISTRY if 'prithvi' in m.lower()]
print(f'Available Prithvi models: {len(models)}')
print('  - ' + '\n  - '.join(models[:5]))
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To train the baseline model:"
echo "  ./scripts/train.sh"
echo ""
echo "Or with custom settings:"
echo "  ./scripts/train.sh --epochs 50 --batch-size 16"
echo "=============================================="
