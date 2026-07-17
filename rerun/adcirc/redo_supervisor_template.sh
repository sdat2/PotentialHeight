#!/bin/bash
# Paired MES/EI redo-campaign supervisor (one per VM). Deployed to
# /root/redo_supervisor.sh with __QUEUE__ / __IMG__ substituted, plus a cron
# @reboot entry, so spot preemptions self-heal: on boot it walks its queue,
# skips completed runs (ledger has >= 50 evals), and relaunches interrupted
# ones (adbo auto-resumes from an existing ledger under the same exp_name).
#
# Each run bind-mounts the patched adforce/wrap.py + adbo/exp.py over the
# image's copies (DryObservationPoint vs AdcircRunFailure split — see
# rerun/results/acquisition_mes_vs_ei.md). Disk hygiene is sequential, never
# concurrent: successful evals clean themselves (low_storage in wrap.py), and
# failed-eval leftovers are swept between container attempts while nothing is
# running. Up to 15 container attempts per run, then "GAVE-UP" is logged for
# the off-VM watcher to alert on.
#
# QUEUE items are daf:seed, e.g. "mes:1 ei:2".
QUEUE="__QUEUE__"
IMG=__IMG__
W=/root/work/exp

exec 9>/root/redo.lock
flock -n 9 || {
  echo "supervisor already running, exiting"
  exit 0
}

count() { grep -o '"res"' "$1" 2>/dev/null | wc -l; }

for item in $QUEUE; do
  daf=${item%%:*}
  seed=${item##*:}
  name=redo3d-${daf}-s${seed}
  for attempt in $(seq 1 15); do
    n=$(count $W/$name/experiments.json)
    if [ "$n" -ge 50 ]; then
      echo "[$name] COMPLETE ($n/50)"
      break
    fi
    echo "[$name] attempt $attempt from $n/50 $(date -u)"
    # Sweep leftovers of earlier FAILED evals now, while no container is
    # running, so there is nothing to race (successful evals clean themselves
    # via low_storage in adforce/wrap.py). Skip the newest dir — it holds the
    # most recent failure's forensics. NEVER run cleanup concurrently with
    # the container: a timer-based janitor doing so once deleted the active
    # run's PE dirs mid-startup and produced all-NaN maxele files.
    for d in $(ls -dt $W/$name/exp_0* 2>/dev/null | tail -n +2); do
      rm -f $d/fort.63.nc $d/fort.64.nc $d/fort.73.nc $d/fort.74.nc
      rm -rf $d/PE0*
    done
    docker run --rm --shm-size=8g --cap-add=SYS_PTRACE \
      -v /root/work:/work -e ADCIRC_NP=16 \
      -v /root/wrap_patched.py:/opt/worstsurge/adforce/wrap.py:ro \
      -v /root/exp_patched.py:/opt/worstsurge/adbo/exp.py:ro \
      $IMG micromamba run -n ws \
      python -m adbo.exp_3d --exp_name $name --seed $seed --daf $daf \
      >>/root/$name.log 2>&1
    rc=$?
    echo "[$name] container exit $rc at $(count $W/$name/experiments.json)/50 $(date -u)"
    sleep 10
  done
  n=$(count $W/$name/experiments.json)
  if [ "$n" -lt 50 ]; then
    echo "[$name] GAVE-UP at $n/50 after 15 attempts $(date -u)"
  fi
done
echo "SUPERVISOR-QUEUE-DONE $(date -u)"
