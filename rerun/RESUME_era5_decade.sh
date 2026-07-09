#!/bin/bash
# Resume/start any ERA5 decade fixed-PS re-solve. Safe anytime: SKIPS years
# already finished (/Volumes/s/tcpips/data/era5/ps_fixed/era5_ps_fixed_{year}.nc).
# Usage:  ./RESUME_era5_decade.sh <start_year> <end_year>
#   e.g.  ./RESUME_era5_decade.sh 1990 1999
# Requires: SSD mounted at /Volumes/s, the worstsurge conda env active.
set -euo pipefail
START="${1:?usage: RESUME_era5_decade.sh <start_year> <end_year>}"
END="${2:?usage: RESUME_era5_decade.sh <start_year> <end_year>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export HDF5_USE_FILE_LOCKING=FALSE
ln -sfn /Volumes/s/tcpips/data/era5 "$REPO/data/era5"   # point at the SSD data
caffeinate -i -s python3 "$REPO/rerun/mem_guard.py" \
    "$REPO/rerun/era5_ps_local.py" 7168 259200 \
    -- --start-year "$START" --end-year "$END" \
    >> "$REPO/rerun/era5_ps_local.log" 2>&1 &
echo "started ${START}-${END} (pid $!). tail -f $REPO/rerun/era5_ps_local.log"
