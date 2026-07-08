#!/usr/bin/env bash
# Autonomous SLSTR × EarthCARE validation pipeline for Dec 2025.
#
# Runs unattended: waits on the background downloader, then collocates and
# evaluates each variable as its reference data completes. Emits "MILESTONE ..."
# lines to logs/slstr_val/run_all.log so a watcher can pick up each stage.
#
#   Stage 1  CTH   (A-CTH, already downloaded)      -> day+night (thermal)
#   Stage 2  iceCOT(A-EBD)                          -> daytime-only
#   Stage 3  waterCOT(ACM-CAP)                      -> daytime-only
#
# Collocation is resumable (skips existing per-frame CSVs); COT stages restrict
# to illum_orac==1 (ORAC's solar retrieval is night-invalid).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/slstr_val/run_all.log
DL=logs/slstr_val/dl_all.log
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
PY="python -m validation"
START=2025-12-01; END=2026-01-01; W=60

say "===== slstr_run_all start (window ${W} min) ====="

# ---------------- Stage 1: CTH (full month) ----------------
say "STAGE1 CTH: waiting for any in-flight collocation to finish"
while pgrep -f 'slstr-cth-collocate.*slstr_cth_2025-12' >/dev/null; do sleep 60; done
if ! ls validation_data/slstr_cth_2025-12/matches_cth_*.csv >/dev/null 2>&1; then
  say "STAGE1 CTH: launching collocation"
  $PY slstr-cth-collocate --start $START --end $END --max-time-diff-min $W \
      --out validation_data/slstr_cth_2025-12
fi
if ls validation_data/slstr_cth_2025-12/matches_cth_*.csv >/dev/null 2>&1; then
  NF=$(ls validation_data/slstr_cth_2025-12/matches_cth_*.csv | wc -l)
  if $PY cth-evaluate --matches 'validation_data/slstr_cth_2025-12/matches_cth_*.csv' \
        --out validation_data/slstr_cth_2025-12.csv \
     && $PY cth-figures --matches 'validation_data/slstr_cth_2025-12/matches_cth_*.csv' \
        --qc-mode qc_strict --label "SLSTR cth Dec-2025 (polar)" \
        --out figures/slstr_cth_2025-12; then
    say "MILESTONE CTH_DONE frames=$NF"
  else
    say "MILESTONE CTH_FAILED (evaluate/figures error)"
  fi
else
  say "MILESTONE CTH_FAILED (no matches produced)"
fi

# ---------------- Stage 2: ice COT (A-EBD) ----------------
DEADLINE=$(( $(date +%s) + 5*3600 ))
say "STAGE2 iceCOT: waiting for A-EBD download to complete (deadline 5h)"
until grep -q 'A-EBD complete' "$DL" 2>/dev/null \
      || [ "$(ls earthcare_data/ATL_EBD_2A/2025/12 2>/dev/null | wc -l)" -ge 31 ]; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { say "STAGE2: deadline, proceeding on partial A-EBD"; break; }
  sleep 180
done
say "STAGE2 iceCOT: A-EBD days=$(ls earthcare_data/ATL_EBD_2A/2025/12 2>/dev/null | wc -l)/31; collocating"
if $PY slstr-collocate --start $START --end $END --max-time-diff-min $W \
      --out validation_data/slstr_cot_2025-12; then
  python scripts/slstr_filter_day.py 'validation_data/slstr_cot_2025-12/matches_cot_*.csv' \
      validation_data/slstr_cot_2025-12_day
  if ls validation_data/slstr_cot_2025-12_day/matches_cot_*.csv >/dev/null 2>&1; then
    $PY evaluate --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
        --out validation_data/slstr_cot_ice_2025-12.csv
    $PY figures --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
        --label "SLSTR cot Dec-2025 (ice, polar, day)" --out figures/slstr_cot_ice_2025-12
    say "MILESTONE ICE_COT_DONE"
  else
    say "MILESTONE ICE_COT_EMPTY (no daytime matches)"
  fi
else
  say "MILESTONE ICE_COT_FAILED (collocation error)"
fi

# ---------------- Stage 3: water COT (ACM-CAP) ----------------
DEADLINE=$(( $(date +%s) + 10*3600 ))
say "STAGE3 waterCOT: waiting for ACM-CAP download to complete (deadline 10h)"
until grep -q 'ACM-CAP complete' "$DL" 2>/dev/null \
      || [ "$(ls earthcare_data/ACM_CAP_2B/2025/12 2>/dev/null | wc -l)" -ge 31 ]; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { say "STAGE3: deadline, proceeding on partial ACM-CAP"; break; }
  sleep 300
done
if ls earthcare_data/ACM_CAP_2B/2025/12/*/*.h5 >/dev/null 2>&1; then
  say "STAGE3 waterCOT: ACM-CAP days=$(ls earthcare_data/ACM_CAP_2B/2025/12 2>/dev/null | wc -l)/31; collocating"
  if $PY slstr-synergy-collocate --start $START --end $END --max-time-diff-min $W \
        --out validation_data/slstr_synergy_2025-12; then
    python scripts/slstr_filter_day.py 'validation_data/slstr_synergy_2025-12/matches_synergy_*.csv' \
        validation_data/slstr_synergy_2025-12_day
    if ls validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv >/dev/null 2>&1; then
      $PY cot-water-evaluate --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
          --out validation_data/slstr_cot_water_2025-12.csv
      $PY cot-water-figures --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
          --qc-mode qc_strict --label "SLSTR cot water Dec-2025 (polar, day)" \
          --out figures/slstr_cot_water_2025-12
      say "MILESTONE WATER_COT_DONE"
    else
      say "MILESTONE WATER_COT_EMPTY (no daytime matches)"
    fi
  else
    say "MILESTONE WATER_COT_FAILED (collocation error)"
  fi
else
  say "MILESTONE WATER_COT_SKIPPED (no ACM-CAP data yet)"
fi

say "MILESTONE ALL_DONE"
