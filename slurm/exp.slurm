#!/bin/bash --login

# Slurm settings
#SBATCH --job-name=i10b10
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:0
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

p25=2025_new_orleans_profile_r4i1p1f1
p97=2097_new_orleans_profile_r4i1p1f1
obs_lon=-90.0715
obs_lat=29.9511
init_steps=10
daf_steps=10
seed=1010

# try to activate micromamba
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba --version
micromamba activate t1

# print out the python version
echo "which python"
which python

# define variables
#case_name=$SLURM_JOB_NAME # name for printing
# np=$SLURM_NTASKS  # how many parallel tasks to define
# export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
# export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# run python job
# srun --distribution=block:block --hint=nomultithread python -m tcpips.pangeo

# python -m adbo.exp_2d # --exp_name 2d-ani-new --year 2025 --resolution mid-notide

# 1
# python -m adbo.exp_3d --exp_name=new-orleans-2025 --profile_name=$p25 --resolution=mid --obs_lat=$obs_lat --obs_lon=$obs_lon
python -m adbo.exp_3d --exp_name=i${init_steps}b${daf_steps} --profile_name=$p25 --resolution=mid --obs_lat=$obs_lat --obs_lon=$obs_lon --init_steps=$init_steps --daf_steps=$daf_steps --seed=$seed
