#!/bin/bash
# Slurm array job: collocate one day of A-CTH frames per task.
#
# Submit with --export to pick the retrieval and month, e.g.
#   sbatch --export=ALL,RETRIEVAL=R11,MONTH=2026-02 scripts/cth_collocate.sbatch
#   sbatch --export=ALL,RETRIEVAL=R10,MONTH=2026-02 scripts/cth_collocate.sbatch
#
# The cth-collocate command resumes cleanly — re-running an array task will
# skip frames whose matches CSV already exists.

#SBATCH --partition=standard
#SBATCH --account=nceo_aerosolfire
#SBATCH --job-name=cth-collocate
#SBATCH --array=1-28
#SBATCH --time=02:00:00
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --qos=short

set -euo pipefail

RETRIEVAL=${RETRIEVAL:-R11}
MONTH=${MONTH:-2026-02}

DAY=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
START="${MONTH}-${DAY}T00:00:00Z"
END=$(date -u -d "${MONTH}-${DAY} +1 day" +%Y-%m-%dT%H:%M:%SZ)

OUT_DIR="validation_data/cth_${MONTH}_${RETRIEVAL}"
mkdir -p "$OUT_DIR" logs

cd /gws/pw/j07/nceo_aerosolfire/rsong/project/cloud_cci

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} retrieval=${RETRIEVAL} month=${MONTH} day=${DAY}"
echo "[$(date)] start=${START} end=${END} out=${OUT_DIR}"

python3 -m validation cth-collocate \
    --start "$START" \
    --end "$END" \
    --retrieval "$RETRIEVAL" \
    --out "$OUT_DIR"

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} done"
