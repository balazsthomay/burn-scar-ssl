#!/usr/bin/env bash
# Download Phase 4 result JSONs from the VM before destroying it.
# Usage: bash scripts/download_phase4_results.sh

set -euo pipefail

VM="root@86.127.245.129"
PORT=22681
REMOTE="/workspace/burn-scar-ssl/outputs/phase4"
LOCAL="outputs"

mkdir -p "${LOCAL}"

scp -P ${PORT} ${VM}:${REMOTE}/lora/r4/100pct/seed42/result.json  ${LOCAL}/phase4_r4_100pct.json
scp -P ${PORT} ${VM}:${REMOTE}/lora/r8/100pct/seed42/result.json  ${LOCAL}/phase4_r8_100pct.json
scp -P ${PORT} ${VM}:${REMOTE}/lora/r16/100pct/seed42/result.json ${LOCAL}/phase4_r16_100pct.json
scp -P ${PORT} ${VM}:${REMOTE}/dora/r8/100pct/seed42/result.json  ${LOCAL}/phase4_dora_r8_100pct.json
scp -P ${PORT} ${VM}:${REMOTE}/lora/r8/10pct/seed42/result.json   ${LOCAL}/phase4_r8_10pct.json

echo "Done. Results in outputs/"
