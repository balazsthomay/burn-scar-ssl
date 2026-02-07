#!/usr/bin/env bash
# Phase 3: Semi-Supervised Learning Experiments
# Run the main SSL experiment + 2 key ablations on a GPU VM.
#
# Usage:
#   bash scripts/run_phase3_experiments.sh
#   bash scripts/run_phase3_experiments.sh --dry-run
#
# Expected runtime: ~2.5 hours total on RTX 4090
#   Run 1 (main):     ~45 min
#   Run 2 (no-ema):   ~45 min
#   Run 3 (25% labels): ~45 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/outputs/phase3/logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Experiments to run
declare -a NAMES=(
    "main_10pct"
    "ablation_no_ema"
    "main_25pct"
)
declare -a ARGS=(
    ""
    "--no-ema"
    "--label-fraction 0.25"
)
declare -a DESCRIPTIONS=(
    "Main: 10% labels, full FixMatch+EMA+CutMix"
    "Ablation: no EMA teacher (student self-trains)"
    "Main: 25% labels, full FixMatch+EMA+CutMix"
)

NUM_EXPERIMENTS=${#NAMES[@]}

echo "============================================"
echo "Phase 3: Semi-Supervised Learning Experiments"
echo "============================================"
echo "Experiments: $NUM_EXPERIMENTS"
echo ""

for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
    echo "  [$((i + 1))] ${DESCRIPTIONS[$i]}"
    echo "      args: ${ARGS[$i]}"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would run the above experiments. Exiting."
    exit 0
fi

FAILED=0
PASSED=0

for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
    NAME="${NAMES[$i]}"
    EXPERIMENT_ARGS="${ARGS[$i]}"
    DESC="${DESCRIPTIONS[$i]}"
    LOG_FILE="$LOG_DIR/${NAME}.log"

    echo "--------------------------------------------"
    echo "[$((i + 1))/$NUM_EXPERIMENTS] $DESC"
    echo "  Log: $LOG_FILE"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    if uv run scripts/run_phase3.py $EXPERIMENT_ARGS 2>&1 | tee "$LOG_FILE"; then
        PASSED=$((PASSED + 1))
        echo ""
        echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "  FAILED: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo ""
done

# Summary: collect results from result.json files
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "Passed: $PASSED / $NUM_EXPERIMENTS"
if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED"
fi
echo ""

echo "Results:"
echo "--------"
for result_file in "$PROJECT_DIR"/outputs/phase3/*/result.json; do
    if [ -f "$result_file" ]; then
        exp=$(python3 -c "import json; r=json.load(open('$result_file')); print(f\"  {r['experiment']}: IoU(burn)={r['test_iou_burn']:.4f}  mIoU={r['test_miou']:.4f}  epochs={r['epochs_trained']}\")" 2>/dev/null || echo "  (could not parse $result_file)")
        echo "$exp"
    fi
done

echo ""
echo "Phase 2 baseline (10% labels, supervised): IoU(burn)=0.692"
echo "============================================"
