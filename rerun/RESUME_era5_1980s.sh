#!/bin/bash
# Resume the ERA5 first-decade (1980-1989) fixed-PS re-solve.
# Safe anytime: SKIPS years already finished
# (/Volumes/s/tcpips/data/era5/ps_fixed/era5_ps_fixed_{year}.nc).
# Requires: SSD mounted at /Volumes/s, the worstsurge conda env active.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export HDF5_USE_FILE_LOCKING=FALSE
ln -sfn /Volumes/s/tcpips/data/era5 "$REPO/data/era5"   # point at the SSD data
caffeinate -i -s python "$REPO/rerun/mem_guard.py" \
    "$REPO/rerun/era5_ps_local.py" 7168 259200 \
    -- --start-year 1980 --end-year 1989 \
    >> "$REPO/rerun/era5_ps_local.log" 2>&1 &
echo "resumed (pid $!). tail -f $REPO/rerun/era5_ps_local.log"
