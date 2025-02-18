#!/bin/bash --login

# Slurm settings
#SBATCH --job-name=i50b0t45
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:0
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

p25=2025_new_orleans_profile_r4i1p1f1
p97=2097_new_orleans_profile_r4i1p1f1
obs_lon=-90.0715
obs_lat=29.9511
init_steps=50
daf_steps=0

# try to activate micromamba
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba --version
micromamba activate t1

# print out the python version
echo "which python"
which python

for trial in {4..5}; do
    seed=$(( $init_steps * 10000 + $daf_steps * 100 + $trial ))
    echo "trial = $trial, seed = $seed"
    python -m adbo.exp_3d --exp_name=i${init_steps}b${daf_steps}t${trial} --profile_name=$p25 --resolution=mid --obs_lat=$obs_lat --obs_lon=$obs_lon --init_steps=$init_steps --daf_steps=$daf_steps --seed=$seed
done
