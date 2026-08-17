#!/bin/bash
# Manning's-n tidal-friction sweep on a GCP VM (adforce/eval/tidal_diagnosis.md):
# {0.022 control, 0.028, 0.035} x tide-only x mid x {Katrina, Ida, Matthew} = 9 runs,
# driven by the hydra eval orchestrator (python -m adforce.eval.launch), which
# generates per-storm tidal fort.15s, rewrites the fort.13 Manning default per cell,
# skips completed runs (spot-safe), extracts gauge_ts.parquet and strips fort.6x/PE*
# after every run (in-container paths -- the hist_sweep host/container strip bug
# does not apply here because the whole sweep runs inside ONE container).
#
# Run ON THE VM host (docker + the adcirc-ws base image present -- see
# run_on_gcp.sh steps 1-2; spot VM is fine, the sweep resumes):
#   BRANCH=refactor/adforce-eval bash mannings_sweep.sh
# Then from the laptop:
#   python -m adforce.eval.harvest remote=<vm>:/root/work/eval study=mannings dry_run=false
#
# Conventions inherited from hist_sweep.sh: the adcirc-ws:lowres prep image
# (adcircpy + IBTrACS baked + aswip) and read-only subtree mounts of the fresh
# branch clone over the image's baked /opt/worstsurge (the baked IBTrACS under
# /opt/worstsurge/data is NOT shadowed -- only code subtrees are mounted).
set -uo pipefail
IMG=adcirc-ws:lowres
BRANCH=${BRANCH:-refactor/adforce-eval}
WS=${WS:-/root/ws}
NP=${NP:-16}   # match the VM's vCPUs (docker -e ADCIRC_NP)
STORMS=${STORMS:-'["Katrina 2005","Ida 2021","Matthew 2016"]'}
IBT_URL="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"

echo "=== fresh clone of $BRANCH for the code mounts ==="
rm -rf "$WS"
git clone --depth 1 -b "$BRANCH" https://github.com/sdat2/worstsurge.git "$WS" || exit 1

# --- prep image (identical to hist_sweep.sh): adcircpy + IBTrACS + aswip ---
if ! docker image inspect $IMG > /dev/null 2>&1; then
  BASE=""
  for b in adcirc-ws:icsfix adcirc-ws:latest adcirc-swan:latest; do
    docker image inspect $b > /dev/null 2>&1 && { BASE=$b; break; }
  done
  [ -z "$BASE" ] && { echo "NO-BASE-IMAGE (build adcirc-ws first: run_on_gcp.sh step 2)"; exit 1; }
  echo "=== building prep image $IMG from $BASE ==="
  docker rm -f mannings-prep 2>/dev/null
  docker run --name mannings-prep $BASE micromamba run -n ws bash -c "
    set -e
    pip install adcircpy > /tmp/pip.log 2>&1
    mkdir -p /opt/worstsurge/data/ibtracs
    python -c \"import urllib.request as u; u.urlretrieve('$IBT_URL', '/opt/worstsurge/data/ibtracs/IBTrACS.since1980.v04r01.nc')\"
    cmake -S /opt/adcirc-src/adcirc -B /tmp/aswip-build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_ASWIP=ON > /tmp/aswip.log 2>&1
    cmake --build /tmp/aswip-build --target aswip -j 8 >> /tmp/aswip.log 2>&1
    cp /tmp/aswip-build/aswip /opt/adcirc/work/aswip
    python -c 'import adcircpy; print(\"adcircpy import ok\")'
    " || { echo PREP-FAILED; docker logs mannings-prep | tail -20; exit 1; }
  docker commit mannings-prep $IMG && docker rm mannings-prep
  echo "committed $IMG"
fi

MNT="-v /root/work:/work \
  -v $WS/adforce/eval:/opt/worstsurge/adforce/eval:ro \
  -v $WS/adforce/training:/opt/worstsurge/adforce/training:ro \
  -v $WS/adforce/fort13.py:/opt/worstsurge/adforce/fort13.py:ro \
  -v $WS/adforce/fort61.py:/opt/worstsurge/adforce/fort61.py:ro"

mkdir -p /root/work/eval

echo "=== dry-run plan ==="
docker run --rm $MNT $IMG micromamba run -n ws \
  python -m adforce.eval.launch study=mannings matrix=mannings_tide \
    "storms=$STORMS" runs_root=/work/eval || exit 1

echo "=== launching (sequential; ~9 tidal runs) ==="
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -e ADCIRC_NP=$NP $MNT \
  $IMG micromamba run -n ws \
  python -m adforce.eval.launch study=mannings matrix=mannings_tide \
    "storms=$STORMS" runs_root=/work/eval dry_run=false \
  2>&1 | tee -a /root/mannings_sweep.log
echo "MANNINGS-SWEEP-DONE disk: $(df -h / | tail -1 | awk '{print $5}')"
