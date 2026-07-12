#!/bin/bash
# Headless SWAN-coupled ADCIRC build: base image + padcswan target, with serial markers.
# Ship via --metadata-from-file=startup-script=... Markers: SWAN-{CLONE,BASE,BUILD}-{OK,FAIL}.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y docker.io git || { echo "SWAN-CLONE-FAIL(apt)"; exit 1; }
git clone --depth 1 https://github.com/sdat2/worstsurge.git /root/worstsurge \
    && echo "SWAN-CLONE-OK" || { echo "SWAN-CLONE-FAIL"; exit 1; }
cd /root/worstsurge/rerun/adcirc
if docker build -t adcirc-ws . > /root/build.log 2>&1; then echo "SWAN-BASE-OK"; else
    echo "SWAN-BASE-FAIL"; tail -15 /root/build.log; exit 1; fi
# extend: build padcswan in the existing source/build tree (flags mirror the base build;
# ADDITIONAL_FLAGS_SWAN passed non-empty to dodge the macros empty-expansion trap)
docker rm -f swb >/dev/null 2>&1 || true
if docker run --name swb adcirc-ws micromamba run -n ws bash -c '
  LL="-ffixed-line-length-132 -ffree-line-length-132"
  cmake -S /opt/adcirc-src/adcirc -B /opt/adcirc-build \
    -DBUILD_PADCSWAN=ON -DADDITIONAL_FLAGS_SWAN="$LL" > /root_cmake.log 2>&1
  cmake --build /opt/adcirc-build -j16 --target padcswan 2>&1 | tail -3
  find /opt/adcirc-build -name padcswan -type f -exec cp {} /opt/adcirc/work/ \;
  test -x /opt/adcirc/work/padcswan && ls -la /opt/adcirc/work/' > /root/swanbuild.log 2>&1
then
    docker commit swb adcirc-swan >/dev/null && echo "SWAN-BUILD-OK"
    tail -3 /root/swanbuild.log
else
    echo "SWAN-BUILD-FAIL"; tail -25 /root/swanbuild.log
fi
echo "SWAN-SEQUENCE-COMPLETE"
