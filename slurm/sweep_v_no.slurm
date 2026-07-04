#!/bin/bash --login

#SBATCH --job-name=sweep-v-no
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# 1D intensity sweep along the size-intensity tradeoff curve for New Orleans:
# fixed track (set to the 3D BO optimum below), vmax varying on r(V).
# Phase-2 experiment (a) of the tradeoff-curve scoping.

source /work/n02/n02/sdat2/.bashrc
micromamba activate t1
which python

# --- inputs -----------------------------------------------------------------
# point-timeseries product with the PS inputs (see w22.ps_runs.point_timeseries)
point_nc=w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc
obs_lon=-90.0715
obs_lat=29.9511
# fixed track: SET THESE TO THE 3D BO OPTIMUM for the site/year
angle=0.0
trans_speed=7.71
displacement=0.0
num=15

for year in 2025 2097; do
    curve=w22/data/curves/${year}_new_orleans_r4i1p1f1.nc
    # generate the tradeoff curve from the point env (skips if it exists)
    python -c "
from w22.tradeoff import env_from_point_ds, generate_curve
env = env_from_point_ds('${point_nc}', ${year})
print('env', env)
generate_curve(env, '${year}_new_orleans_r4i1p1f1')
"
    python -m adbo.sweep_vmax --curve_nc=${curve} \
        --exp_name=no-sweep-${year} --num=${num} \
        --angle=${angle} --trans_speed=${trans_speed} \
        --displacement=${displacement} \
        --obs_lon=${obs_lon} --obs_lat=${obs_lat} --resolution=mid
    python -m adbo.plot_tradeoff --mode=sweep \
        --exp_dir=${WORSTSURGE_EXP_DIR:-exp}/no-sweep-${year}
done
