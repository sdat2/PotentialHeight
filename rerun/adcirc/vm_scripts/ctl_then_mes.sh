#!/bin/bash
# 1. clean tide-decomposition control: storm-only on the identical Aug19/13d/group-3 deck
cd /root/work/exp
rm -rf storm_ctl_g3 && mkdir storm_ctl_g3
cp tide_storm/fort.13 tide_storm/fort.14 tide_storm/fort.22.nc storm_ctl_g3/
docker run --rm -v /root/work:/work adcirc-ws:icsfix bash -c "sed -e \"s/20050825.000000/20050819.000000/\" -e \"s/NWS13GroupForPowell='"'"'2'"'"'/NWS13GroupForPowell='"'"'3'"'"'/\" -e \"24s/^ 7.0 /13.0 /\" /opt/worstsurge/adforce/setup/fort.15.mid.notide > /work/exp/storm_ctl_g3/fort.15; grep -c 20050819 /work/exp/storm_ctl_g3/fort.15"
cd storm_ctl_g3
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v /root/work:/work -w /work/exp/storm_ctl_g3 adcirc-ws:icsfix micromamba run -n ws bash -c "/opt/adcirc/work/adcprep --np 16 --partmesh > adcprep.log 2>&1 && /opt/adcirc/work/adcprep --np 16 --prepall >> adcprep.log 2>&1 && mpirun -np 16 /opt/adcirc/work/padcirc > pad.log 2>&1"
echo "STORMCTL-EXIT:$?"
# 2. MES 3D BO (paper setup, seed 42) with periodic output cleanup so the disk survives 50 runs
( while true; do sleep 600; for d in /root/work/exp/no3d-mes-s42/exp_00*; do rm -f $d/fort.63.nc $d/fort.64.nc $d/fort.73.nc $d/fort.74.nc; rm -rf $d/PE0*; done 2>/dev/null; done ) &
CLEAN=$!
docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v /root/work:/work -e ADCIRC_NP=16 adcirc-ws:icsfix micromamba run -n ws \
  python -m adbo.exp_3d --exp_name no3d-mes-s42 --seed 42 --daf mes > /root/mes.log 2>&1
echo "MES-EXIT:$?"; kill $CLEAN 2>/dev/null
echo CTL-MES-DONE
