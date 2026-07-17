#!/bin/bash
( while true; do sleep 600; for d in /root/work/exp/no3d-ei-s42/exp_00*; do rm -f $d/fort.63.nc $d/fort.64.nc $d/fort.73.nc $d/fort.74.nc; rm -rf $d/PE0*; done 2>/dev/null; done ) &
CLEAN=$!
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v /root/work:/work -e ADCIRC_NP=16 adcirc-ws micromamba run -n ws \
  python -m adbo.exp_3d --exp_name no3d-ei-s42 --seed 42 --daf ei > /root/ei.log 2>&1
echo "EI-EXIT:$?"; kill $CLEAN 2>/dev/null
echo EI-DONE
