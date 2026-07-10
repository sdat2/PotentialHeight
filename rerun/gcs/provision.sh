#!/bin/bash
# DRAFT — provision the GCS bucket + upload inputs + create the spot VM. UNTESTED.
# Prereqs: `gcloud` CLI installed & authed (`gcloud auth login`; `gcloud config set project <id>`),
# a GCP project with billing enabled, and the local ERA5 decade files present (on the SSD).
# PREREQUISITE: rerun/gcs/ must be COMMITTED AND PUSHED — startup.sh builds the env from the
# fresh clone's rerun/gcs/environment.yml, which only exists on the VM if it's on origin.
# Run steps individually the first time rather than blind — read each command.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

echo "### 1. Create the bucket (one-time) ###"
gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" --location="$REGION" --uniform-bucket-level-access || \
    echo "  (bucket may already exist — continuing)"

echo "### 2. Upload ERA5 inputs for the decade(s) you want to run (one-time per decade) ###"
# Mirror the local data/era5 layout into the bucket. Adjust the source glob to the
# decade(s) you need. These are large (~6.4 GB pressure + 0.56 GB single + PI per decade).
# Example for the 2000s (repeat/adjust for 2010s, 2020-2024):
SRC="/Volumes/s/tcpips/data/era5"
DECADE="2000_2009"
echo "  uploading raw/*${DECADE}* ..."
gcloud storage cp "$SRC/raw/"*${DECADE}*.nc "gs://$BUCKET/data/era5/raw/" || \
    echo "  (nothing matched in raw/ — check paths/decade)"
# pi_og: copy the EXACT file — the glob would also match era5_pi_years*.fix.nc (~1.4 GB of
# CDO-compatibility dead weight the solver never opens; get_era5_pi reads exactly this name).
echo "  uploading pi_og/era5_pi_years${DECADE}.nc ..."
gcloud storage cp "$SRC/pi_og/era5_pi_years${DECADE}.nc" "gs://$BUCKET/data/era5/pi_og/" || \
    echo "  (pi file missing — check paths/decade)"
# Alternative to uploading from home: re-download from Copernicus CDS ON THE VM using
# tcpips/era5.py's download_single_levels(...) (needs a ~/.cdsapirc). Trades upload time for CDS queue time.

echo "### 3. Create the spot worker VM (boots, runs startup.sh) ###"
# plain string, not an array: expanding an empty array under `set -u` kills macOS's
# bash 3.2 (fixed only in bash 4.4), which would break the on-demand (SPOT=false) path.
SPOT_FLAGS=""
[ "$SPOT" = "true" ] && SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
# shellcheck disable=SC2086  # word-splitting of $SPOT_FLAGS is intended
gcloud compute instances create "$VM" \
    --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE" \
    $SPOT_FLAGS \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size="${BOOT_GB}GB" --boot-disk-type=pd-balanced \
    --scopes=storage-rw \
    --metadata-from-file=startup-script=./startup.sh

cat <<EOF

### Done provisioning. Next: ###
  # SSH in (startup.sh will have cloned the repo + built the env, or is still running):
  gcloud compute ssh $VM --zone=$ZONE
  # then on the VM, run a decade (see run_decade_gcs.sh), e.g.:
  #   cd ~/worstsurge/rerun/gcs && ./run_decade_gcs.sh 2000 2009
  #
  # When ALL decades are done, TEAR DOWN so you stop paying:
  gcloud compute instances delete $VM --zone=$ZONE
  # (the outputs live in gs://$BUCKET/ps_fixed/ — the VM is disposable)
EOF
