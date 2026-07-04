#!/bin/bash --login

#SBATCH --job-name=bo-4d-no
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# 4D Bayesian optimization (track + on-curve intensity) for New Orleans,
# 2025 vs 2097: each sampled vmax gets its CLE15 profile generated on the
# size-intensity tradeoff curve r(V). Phase-2 experiment (b) of the
# tradeoff-curve scoping. Resumable: rerun after a timeout to continue.

source /work/n02/n02/sdat2/.bashrc
micromamba activate t1
which python

# --- inputs -----------------------------------------------------------------
point_nc=w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc
obs_lon=-90.0715
obs_lat=29.9511
init_steps=35
daf_steps=35

for year in 2025 2097; do
    curve=w22/data/curves/${year}_new_orleans_r4i1p1f1.nc
    python -c "
from w22.tradeoff import env_from_point_ds, generate_curve
env = env_from_point_ds('${point_nc}', ${year})
print('env', env)
generate_curve(env, '${year}_new_orleans_r4i1p1f1')
"
    python -m adbo.exp_4d --curve_nc=${curve} \
        --exp_name=no-4d-${year} \
        --init_steps=${init_steps} --daf_steps=${daf_steps} \
        --obs_lon=${obs_lon} --obs_lat=${obs_lat} --resolution=mid
    python -m adbo.plot_tradeoff --mode=4d \
        --exp_dir=${WORSTSURGE_EXP_DIR:-exp}/no-4d-${year}
done
