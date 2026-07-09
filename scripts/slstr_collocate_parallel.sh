#!/usr/bin/env bash
# Parallel SLSTR collocation: split the month into N contiguous date chunks and
# run one collocation worker per chunk. Chunks are disjoint in date => disjoint
# in frames, so workers never write the same CSV (no race). Collocation is
# skip-safe, so this also cleanly RESUMES a partial single-process run.
#
# Usage: slstr_collocate_parallel.sh <subcommand> <out_dir> <n_workers> <max_dt_min>
#   subcommand: slstr-cth-collocate | slstr-synergy-collocate | slstr-collocate
set -u
cd "$(dirname "$0")/.." || exit 1
SUB="${1:?subcommand}"; OUT="${2:?out dir}"; N="${3:-6}"; W="${4:-60}"
RANGE_START="${5:-2025-12-01}"; RANGE_END="${6:-2026-01-01}"
mkdir -p "$OUT" logs/slstr_val

# Contiguous date boundaries splitting [RANGE_START, RANGE_END) into N chunks.
# For an already-partly-done variable, pass the UNDONE sub-range so every worker
# does real matched-frame work (balanced) instead of re-scanning finished days.
mapfile -t BOUNDS < <(python - "$N" "$RANGE_START" "$RANGE_END" <<'PY'
import sys, datetime
n=int(sys.argv[1])
start=datetime.date.fromisoformat(sys.argv[2]); end=datetime.date.fromisoformat(sys.argv[3])
days=(end-start).days
b=[start+datetime.timedelta(days=round(i*days/n)) for i in range(n+1)]
for i in range(n):
    if b[i] < b[i+1]:
        print(f"{b[i].isoformat()} {b[i+1].isoformat()}")
PY
)

pids=(); tags=()
for line in "${BOUNDS[@]}"; do
  s="${line% *}"; e="${line#* }"
  wl="logs/slstr_val/worker_${SUB}_${s}.log"
  python -m validation "$SUB" --start "$s" --end "$e" --max-time-diff-min "$W" --out "$OUT" > "$wl" 2>&1 &
  pids+=($!); tags+=("$s..$e")
  echo "  worker pid $! : $s .. $e ($SUB)"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "  WORKER FAILED: ${tags[$i]}"; fail=1; fi
done
echo "parallel $SUB finished (fail=$fail); CSVs in $OUT: $(ls "$OUT"/*.csv 2>/dev/null | wc -l)"
exit $fail
