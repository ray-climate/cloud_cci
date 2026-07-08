#!/usr/bin/env bash
# Outage-resilient EarthCARE monthly download.
#
# MAAP intermittently returns 504s (even on catalogue-open) and can be down for
# hours. This script loops the whole month in repeated PASSES; already-downloaded
# files are skipped by the client, so each pass is cheap once data lands. It keeps
# passing until a clean pass (no per-day failures) or a wall-clock deadline.
#
# Usage: scripts/download_ec_month.sh <YYYY> <MM> <PRODUCT> [PRODUCT ...]
#   e.g. scripts/download_ec_month.sh 2025 12 A-CTH A-EBD ACM-CAP
set -u

YEAR="${1:?year, e.g. 2025}"; shift
MONTH="${1:?month, e.g. 12}"; shift
PRODUCTS=("$@")
[ "${#PRODUCTS[@]}" -eq 0 ] && { echo "give >=1 product code"; exit 1; }

DEST="earthcare_data"
MAX_PASSES=60                 # generous: survives a long MAAP outage
DEADLINE_HOURS=48
PER_DAY_RETRIES=3

cd "$(dirname "$0")/.." || exit 1
days_in_month=$(python -c "import calendar;print(calendar.monthrange($YEAR,$MONTH)[1])")
deadline=$(( $(date +%s) + DEADLINE_HOURS*3600 ))

for PRODUCT in "${PRODUCTS[@]}"; do
    echo "############ PRODUCT $PRODUCT $YEAR-$MONTH ############"
    for pass in $(seq 1 "$MAX_PASSES"); do
        [ "$(date +%s)" -ge "$deadline" ] && { echo "!! deadline reached"; break; }
        echo "===== $PRODUCT pass $pass/$MAX_PASSES  $(date -u +%H:%M:%SZ) ====="
        fails=0
        for d in $(seq -w 1 "$days_in_month"); do
            start=$(date -u -d "$YEAR-$MONTH-$d" +%Y-%m-%dT00:00:00Z)
            nextday=$(date -u -d "$YEAR-$MONTH-$d +1 day" +%Y-%m-%dT00:00:00Z)
            ok=0
            for attempt in $(seq 1 "$PER_DAY_RETRIES"); do
                if python -m earthcare download --product "$PRODUCT" \
                        --start "$start" --end "$nextday" --dest "$DEST" \
                        >/dev/null 2>&1; then ok=1; break; fi
                sleep $((attempt * 15))
            done
            if [ "$ok" -eq 1 ]; then echo "  $PRODUCT $start ok"
            else echo "  $PRODUCT $start FAIL"; fails=$((fails+1)); fi
        done
        echo "----- $PRODUCT pass $pass done: $fails day-failures -----"
        [ "$fails" -eq 0 ] && { echo "$PRODUCT complete."; break; }
        sleep 120   # let MAAP breathe before the next pass
    done
done
echo "=== downloader finished $(date -u +%H:%M:%SZ) ==="
