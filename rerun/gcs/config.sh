# DRAFT — shared config for the GCS/GCE ERA5 runs. Edit these, then `source` from the other scripts.
# Nothing here is validated; fill in the <PLACEHOLDERS> and sanity-check before use.

# --- Google Cloud project / location ---
# Env-first (VAR="${VAR:-default}"): the VM sources the COMMITTED copy of this file from the
# git clone, so laptop-only edits don't propagate. Either commit your real values (bucket
# names aren't secrets) or override per-invocation: `BUCKET=my-bucket ./run_decade_gcs.sh ...`.
export PROJECT="${PROJECT:-<your-gcp-project-id>}"
export REGION="${REGION:-europe-west2}"   # London (low latency from the UK); pair the zone to it
export ZONE="${ZONE:-europe-west2-c}"
export BUCKET="${BUCKET:-worstsurge-era5}" # gs://$BUCKET — created by provision.sh; globally-unique name

# --- Compute Engine VM ---
export VM="era5-worker"
export MACHINE="c2d-standard-16"       # 16 vCPU AMD Milan, x86_64. c3-standard-16 (Intel) also fine.
export BOOT_GB="100"                    # holds env + one decade of raw ERA5 (~10 GB) + outputs; bump if staging more
export SPOT="true"                      # spot = ~60-70% cheaper, preemptible. Safe: job is per-year checkpointed.

# --- The job ---
export ENV_NAME="worstsurge-era5"
export N_JOBS="14"                      # joblib workers on the VM (leave ~2 vCPU for I/O); 16-vCPU -> 14
export CAP_MB="28000"                   # mem_guard RSS cap; c2d-standard-16 has 64 GB, so generous
export TIMEOUT_S="172800"               # 48 h per decade (plenty; a decade is ~12 h at 16 vCPU)

# --- Layout conventions (local on the VM mirrors the bucket) ---
export REPO_URL="https://github.com/sdat2/worstsurge.git"
export REPO_DIR="$HOME/worstsurge"
export LOCAL_DATA="$HOME/era5_data"     # data/era5 will symlink here on the VM
# Bucket prefixes:
#   gs://$BUCKET/data/era5/{raw,pi_og}/...   <- inputs you upload once (mirrors local data/era5)
#   gs://$BUCKET/ps_fixed/era5_ps_fixed_{year}.nc  <- outputs (the durable product)
