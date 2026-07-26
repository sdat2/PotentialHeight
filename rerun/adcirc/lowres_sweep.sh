#!/bin/bash
# Low-resolution historical validation sweep (mesh-sensitivity test).
# Same GAHM/IBTrACS pipeline as the EC95d (mid) archive; only the mesh
# (fort.14.low + resampled fort.13.low) changes. See lowres_sweep notes in
# rerun/results once scored.
#
# Prep is baked into a committed image (adcirc-ws:lowres) exactly once:
#   * adcircpy pip-installed INTO the ws env (a pip --target + PYTHONPATH
#     approach shadowed the env's xarray with an incompatible copy — never
#     again);
#   * IBTrACS.since1980.v04r01.nc at /opt/worstsurge/data/ibtracs/ (the
#     in-image tcpips constants' fallback data path);
#   * aswip built from the in-image ADCIRC sources into /opt/adcirc/work/.
#
# Expects on the VM: /root/lowres/{training/,fort.13.low,
# extract_gauge_series.py,gauges_both_boxes.csv}
set -uo pipefail
BASE=adcirc-ws:icsfix
IMG=adcirc-ws:lowres
W=/root/work
IBT_URL="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"

MNT="-v /root/work:/work \
  -v /root/lowres/training:/opt/worstsurge/adforce/training:ro \
  -v /root/lowres/fort.13.low:/opt/worstsurge/adforce/setup/fort.13.low:ro"

# --- 1. one-off prep image -------------------------------------------------
if ! docker image inspect $IMG > /dev/null 2>&1; then
  echo "=== building prep image $IMG ==="
  docker rm -f lowres-prep 2>/dev/null
  docker run --name lowres-prep $BASE micromamba run -n ws bash -c "
    set -e
    pip install adcircpy > /tmp/pip.log 2>&1
    mkdir -p /opt/worstsurge/data/ibtracs
    python -c \"import urllib.request as u; u.urlretrieve('$IBT_URL', '/opt/worstsurge/data/ibtracs/IBTrACS.since1980.v04r01.nc')\"
    cmake -S /opt/adcirc-src/adcirc -B /tmp/aswip-build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_ASWIP=ON > /tmp/aswip.log 2>&1
    cmake --build /tmp/aswip-build --target aswip -j 8 >> /tmp/aswip.log 2>&1
    cp /tmp/aswip-build/aswip /opt/adcirc/work/aswip
    tail -2 /tmp/pip.log; ls -la /opt/adcirc/work/aswip
    python -c 'import adcircpy; print(\"adcircpy import ok\")'
    " || { echo PREP-FAILED; docker logs lowres-prep | tail -20; exit 1; }
  docker commit lowres-prep $IMG && docker rm lowres-prep
  echo "committed $IMG"
fi
rm -rf $W/pylibs  # remove the failed --target experiment

# --- 2. the sweep ----------------------------------------------------------
mkdir -p $W/lowres_runs
STORMS="FRANCES_2004 JEANNE_2004 KATRINA_2005 RITA_2005 GUSTAV_2008 IKE_2008 \
ISAAC_2012 MATTHEW_2016 HARVEY_2017 IRMA_2017 NATE_2017 MICHAEL_2018 \
BARRY_2019 LAURA_2020 DELTA_2020 IDA_2021 NICHOLAS_2021 NICOLE_2022 IDALIA_2023"
echo "=== low-res sweep: $(echo $STORMS | wc -w) storms ==="
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE $MNT -e ADCIRC_NP=16 \
  $IMG micromamba run -n ws \
  python -m adforce.generate_training_data --resolution low \
    --recommended-dt 5.0 --runs-parent-name /work/lowres_runs \
    --storms $STORMS > /root/lowres_sweep.log 2>&1
rc=$?
echo "SWEEP-EXIT:$rc"
n=$(grep -c "Successfully completed run" /root/lowres_sweep.log)
echo "completed runs: $n"
if [ "$n" -eq 0 ]; then
  echo "NO-RUNS-COMPLETED (skipping extraction)"
  echo LOWRES-SWEEP-DONE
  exit 1
fi

# --- 3. extraction ---------------------------------------------------------
docker run --rm $MNT \
  -v /root/lowres/extract_gauge_series.py:/opt/extract.py:ro \
  -v /root/lowres/gauges_both_boxes.csv:/work/gauges_both_boxes.csv:ro \
  $IMG micromamba run -n ws python /opt/extract.py \
    --runs-dir /work/lowres_runs --gauges /work/gauges_both_boxes.csv \
    --out /work/lowres_gauge_series.parquet > /root/lowres_extract.log 2>&1
echo "EXTRACT-EXIT:$?"
ls -la $W/lowres_gauge_series.parquet 2>/dev/null
echo LOWRES-SWEEP-DONE
