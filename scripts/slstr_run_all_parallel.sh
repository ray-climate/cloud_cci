#!/usr/bin/env bash
# Parallel replacement for slstr_run_all.sh: drives ice COT then water COT using
# N-worker parallel collocation. CTH is already complete. Appends MILESTONE lines
# to the same run_all.log the watcher tails.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/slstr_val/run_all.log
DL=logs/slstr_val/dl_all.log
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
PY="python -m validation"; N="${1:-6}"; W=60
PAR="bash scripts/slstr_collocate_parallel.sh"

say "===== parallel run_all start (N=$N workers, window ${W} min) ====="

# ---------------- Stage 2: ice COT (A-EBD complete) ----------------
# Single-process run already finished days 1-21; parallelise only the undone
# tail (Dec 18 -> Jan 1, margin) so every worker does real matched-frame work.
say "STAGE2p iceCOT: parallel collocation over undone tail (Dec18-Jan1)"
$PAR slstr-collocate validation_data/slstr_cot_2025-12 "$N" "$W" 2025-12-18 2026-01-01
python scripts/slstr_filter_day.py 'validation_data/slstr_cot_2025-12/matches_cot_*.csv' \
    validation_data/slstr_cot_2025-12_day
if ls validation_data/slstr_cot_2025-12_day/matches_cot_*.csv >/dev/null 2>&1; then
  $PY evaluate --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
      --out validation_data/slstr_cot_ice_2025-12.csv \
  && $PY figures --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
      --label "SLSTR cot Dec-2025 (ice, polar, day)" --out figures/slstr_cot_ice_2025-12 \
  && say "MILESTONE ICE_COT_DONE" || say "MILESTONE ICE_COT_FAILED"
else
  say "MILESTONE ICE_COT_EMPTY (no daytime matches)"
fi

# ---------------- Stage 3: water COT (ACM-CAP) ----------------
DEADLINE=$(( $(date +%s) + 8*3600 ))
say "STAGE3p waterCOT: waiting for ACM-CAP download to complete"
until grep -q 'ACM-CAP complete' "$DL" 2>/dev/null \
      || [ "$(ls earthcare_data/ACM_CAP_2B/2025/12 2>/dev/null | wc -l)" -ge 31 ]; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { say "STAGE3p: deadline, proceeding on partial ACM-CAP"; break; }
  sleep 120
done
say "STAGE3p waterCOT: parallel collocation (ACM-CAP $(ls earthcare_data/ACM_CAP_2B/2025/12 2>/dev/null | wc -l)/31)"
$PAR slstr-synergy-collocate validation_data/slstr_synergy_2025-12 "$N" "$W"
python scripts/slstr_filter_day.py 'validation_data/slstr_synergy_2025-12/matches_synergy_*.csv' \
    validation_data/slstr_synergy_2025-12_day
if ls validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv >/dev/null 2>&1; then
  $PY cot-water-evaluate --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
      --out validation_data/slstr_cot_water_2025-12.csv \
  && $PY cot-water-figures --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
      --qc-mode qc_strict --label "SLSTR cot water Dec-2025 (polar, day)" \
      --out figures/slstr_cot_water_2025-12 \
  && say "MILESTONE WATER_COT_DONE" || say "MILESTONE WATER_COT_FAILED"
else
  say "MILESTONE WATER_COT_EMPTY (no daytime matches)"
fi
say "MILESTONE ALL_DONE_PARALLEL"
