#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=nceo_aerosolfire
#SBATCH --job-name=atlid-slstr-month
#SBATCH --array=1-12
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --qos=standard

set -euo pipefail

YEAR="${TARGET_YEAR:-2024}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MONTH=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-1}")
MONTH_KEY="${YEAR}-${MONTH}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] Starting ATLID/SLSTR colocation for ${MONTH_KEY}"
${PYTHON_BIN} colocation/atlid_slstr_colocation.py --month "${MONTH_KEY}" --output "colocation/atlid_slstr_matches_${MONTH_KEY}.csv"
echo "[$(date --iso-8601=seconds)] Finished ATLID/SLSTR colocation for ${MONTH_KEY}"
