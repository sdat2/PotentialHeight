#!/bin/bash
# Generalized historical-storm sweep runner (mesh-resolution x forcing-mode
# matrix for the tide-surge-interaction study). Parameterized by env vars:
#   RES=low|mid        mesh resolution (fort.14/.13 pair)
#   MODE=storm|tide|both  forcing (storm-only / tide-only control / storm+tide)
#   SPINUP=0|6         pre-storm spinup days (use 6 for tidal modes)
#   RUNS_DIR           run directory under /work (default /work/${RES}_${MODE}_runs)
#   STORMS             storm list (default: the 19 comp-validation storms)
#   PREP_ONLY=1        build/commit the prep image and exit
#
# The prep image (adcirc-ws:lowres) is committed once per VM from whichever
# validated base image exists there (icsfix / latest / adcirc-swan): adcircpy
# installed into the ws env, IBTrACS baked at the in-image tcpips data path,
# aswip compiled into /opt/adcirc/work. See rerun/adcirc/lowres_sweep.sh for
# the original single-purpose version and its failure history.
set -uo pipefail
IMG=adcirc-ws:lowres
W=/root/work
IBT_URL="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"

RES=${RES:-low}
MODE=${MODE:-storm}
SPINUP=${SPINUP:-0}
RUNS_DIR=${RUNS_DIR:-/work/${RES}_${MODE}_runs}
STORMS=${STORMS:-"FRANCES_2004 JEANNE_2004 KATRINA_2005 RITA_2005 GUSTAV_2008 \
IKE_2008 ISAAC_2012 MATTHEW_2016 HARVEY_2017 IRMA_2017 NATE_2017 MICHAEL_2018 \
BARRY_2019 LAURA_2020 DELTA_2020 IDA_2021 NICHOLAS_2021 NICOLE_2022 IDALIA_2023"}

MNT="-v /root/work:/work \
  -v /root/lowres/training:/opt/worstsurge/adforce/training:ro \
  -v /root/lowres/fort.13.low:/opt/worstsurge/adforce/setup/fort.13.low:ro"

if ! docker image inspect $IMG > /dev/null 2>&1; then
  BASE=""
  for b in adcirc-ws:icsfix adcirc-ws:latest adcirc-swan:latest; do
    docker image inspect $b > /dev/null 2>&1 && { BASE=$b; break; }
  done
  [ -z "$BASE" ] && { echo "NO-BASE-IMAGE"; exit 1; }
  echo "=== building prep image $IMG from $BASE ==="
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
    python -c 'import adcircpy; print(\"adcircpy import ok\")'
    " || { echo PREP-FAILED; docker logs lowres-prep | tail -20; exit 1; }
  docker commit lowres-prep $IMG && docker rm lowres-prep
  echo "committed $IMG"
fi
[ "${PREP_ONLY:-0}" = "1" ] && { echo PREP-ONLY-DONE; exit 0; }

mkdir -p $RUNS_DIR 2>/dev/null || mkdir -p ${W}$(echo $RUNS_DIR | sed s#^/work##)
echo "=== sweep RES=$RES MODE=$MODE SPINUP=$SPINUP -> $RUNS_DIR ==="
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE $MNT -e ADCIRC_NP=16 \
  $IMG micromamba run -n ws \
  python -m adforce.generate_training_data --resolution $RES --mode $MODE \
    --spinup-days $SPINUP --recommended-dt 5.0 --runs-parent-name $RUNS_DIR \
    --storms $STORMS > /root/sweep_${RES}_${MODE}.log 2>&1
rc=$?
n=$(grep -c "Successfully completed run" /root/sweep_${RES}_${MODE}.log)
echo "SWEEP-EXIT:$rc completed:$n"
if [ "$n" -gt 0 ]; then
  docker run --rm $MNT \
    -v /root/lowres/extract_gauge_series.py:/opt/extract.py:ro \
    -v /root/lowres/gauges_both_boxes.csv:/work/gauges_both_boxes.csv:ro \
    $IMG micromamba run -n ws python /opt/extract.py \
      --runs-dir $RUNS_DIR --gauges /work/gauges_both_boxes.csv \
      --out ${RUNS_DIR}_gauge_series.parquet \
      > /root/extract_${RES}_${MODE}.log 2>&1
  echo "EXTRACT-EXIT:$?"
fi
echo "HIST-SWEEP-DONE ${RES}_${MODE}"
