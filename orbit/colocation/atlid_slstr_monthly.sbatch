#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=nceo_aerosolfire
#SBATCH --job-name=atlid-slstr-month
#SBATCH --array=1-12
#SBATCH --time=04:00:00
#SBATCH --mem=16000
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --qos=short

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
START_YEAR="${START_YEAR:-${TARGET_YEAR:-2024}}"
START_MONTH="${START_MONTH:-08}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

MONTH_KEY=$("${PYTHON_BIN}" - <<'PY' "${START_YEAR}" "${START_MONTH}" "${SLURM_ARRAY_TASK_ID:-1}"
import sys

start_year = int(sys.argv[1])
start_month = int(sys.argv[2])
array_idx = int(sys.argv[3])

offset = array_idx - 1
month = start_month + offset
year = start_year + (month - 1) // 12
month = ((month - 1) % 12) + 1

print(f"{year:04d}-{month:02d}")
PY
)

echo "[$(date --iso-8601=seconds)] Starting ATLID/SLSTR colocation for ${MONTH_KEY}"
${PYTHON_BIN} colocation/atlid_slstr_colocation.py --month "${MONTH_KEY}" --output "colocation/atlid_slstr_matches_${MONTH_KEY}.csv"
echo "[$(date --iso-8601=seconds)] Finished ATLID/SLSTR colocation for ${MONTH_KEY}"
