#!/bin/bash
# Headless ADCIRC image test: build the full image + run the smoke test, with
# grep-able markers on the serial console. Shipped to the VM via
# --metadata-from-file=startup-script=... (content travels with the create call,
# so uncommitted edits here are fine). Logs: /root/build.log, /root/smoke.log.
# Markers: ADCIRC-CLONE-{OK,FAIL} ADCIRC-BUILD-{OK,FAIL} ADCIRC-SMOKE-{PASS,FAIL}
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y && apt-get install -y docker.io git || { echo "ADCIRC-CLONE-FAIL (apt)"; exit 1; }

git clone --depth 1 https://github.com/sdat2/worstsurge.git /root/worstsurge \
    && echo "ADCIRC-CLONE-OK" || { echo "ADCIRC-CLONE-FAIL"; exit 1; }

cd /root/worstsurge/rerun/adcirc
if docker build -t adcirc-ws . > /root/build.log 2>&1; then
    echo "ADCIRC-BUILD-OK"
else
    echo "ADCIRC-BUILD-FAIL (tail of /root/build.log follows)"
    tail -20 /root/build.log
    exit 1
fi

mkdir -p /root/work
if docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v /root/work:/work adcirc-ws adcirc-smoke-test 16 > /root/smoke.log 2>&1; then
    echo "ADCIRC-SMOKE-PASS"
    grep -aE "PASS|FAIL|max zeta" /root/smoke.log | tail -15
else
    echo "ADCIRC-SMOKE-FAIL (tail of /root/smoke.log follows)"
    tail -30 /root/smoke.log
fi
echo "ADCIRC-TEST-SEQUENCE-COMPLETE"
