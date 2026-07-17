#!/bin/bash
P=/opt/worstsurge/rerun/gcs/adcirc/profiles
for variant in old fixed; do
  echo "=== materiality_${variant}_2015 ==="
  docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v /root/work:/work adcirc-ws:icsfix micromamba run -n ws \
    python -m adforce.wrap name=materiality_${variant}_2015 \
      tc.profile_name.value=$P/profile_${variant}_2015.json \
      files.exe_path=/opt/adcirc/work files.exp_path=/work/exp files.low_storage=false \
      slurm.modules="" slurm.tasks_per_node=16 slurm.reserved_cpus=0 2>&1 | tail -3
done
echo MATERIALITY-RUNS-DONE
