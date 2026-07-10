#!/bin/bash
# DRAFT — run one ERA5 decade on the VM, GCS-backed + preemption-safe. UNTESTED.
# Usage (on the VM):  ./run_decade_gcs.sh <start_year> <end_year>   e.g.  ./run_decade_gcs.sh 2000 2009
#
# What it does:
#   1. pulls the decade's raw ERA5 inputs from the bucket to local disk
#   2. pulls any already-finished ps_fixed/*.nc so era5_ps_local.py SKIPS them (resume)
#   3. runs the guarded joblib solve, writing per-year checkpoints locally
#   4. a background loop rsyncs finished years up to the bucket every 2 min, so a spot
#      preemption loses at most the in-progress year (re-run this same command to resume)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
START="${1:?usage: run_decade_gcs.sh <start_year> <end_year>}"
END="${2:?usage: run_decade_gcs.sh <start_year> <end_year>}"

export PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$REPO_DIR"
export HDF5_USE_FILE_LOCKING=FALSE   # harmless on ext4; matches the local convention

OUT="$LOCAL_DATA/ps_fixed"
mkdir -p "$OUT" "$LOCAL_DATA/era5"

echo "### pull inputs (raw + pi) for the decade from the bucket ###"
gcloud storage rsync -r "gs://$BUCKET/data/era5" "$LOCAL_DATA/era5"
# Point the repo's data/era5 at the local data. GOTCHA: importing tcpips creates
# data/era5/{raw,...} as REAL dirs (constants.py makedirs at import), and `ln -sfn` into an
# existing real dir does NOT fail — it silently creates a NESTED link inside it, and the job
# then crashes at data-load time. Remove the empty skeleton first (safe: only the empty
# makedirs tree ever lives there), unless it's already the symlink we want.
[ -L "$REPO_DIR/data/era5" ] || rm -rf "$REPO_DIR/data/era5"
ln -sfn "$LOCAL_DATA/era5" "$REPO_DIR/data/era5"

echo "### pull already-finished checkpoints (resume: these years get skipped) ###"
gcloud storage rsync -r "gs://$BUCKET/ps_fixed" "$OUT" || true

echo "### start background sync (finished years -> bucket every 120s) ###"
( while true; do gcloud storage rsync -r "$OUT" "gs://$BUCKET/ps_fixed" >/dev/null 2>&1 || true; sleep 120; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; gcloud storage rsync -r "$OUT" "gs://$BUCKET/ps_fixed" || true' EXIT

echo "### solve $START-$END (n_jobs=$N_JOBS, cap ${CAP_MB}MB) ###"
micromamba run -n "$ENV_NAME" python "$REPO_DIR/rerun/mem_guard.py" \
    "$REPO_DIR/rerun/era5_ps_local.py" "$CAP_MB" "$TIMEOUT_S" \
    -- --start-year "$START" --end-year "$END" --out-dir "$OUT" --n-jobs "$N_JOBS"

echo "### final sync to bucket ###"
kill $SYNC_PID 2>/dev/null || true
gcloud storage rsync -r "$OUT" "gs://$BUCKET/ps_fixed"
echo "DONE $START-$END -> gs://$BUCKET/ps_fixed/"
