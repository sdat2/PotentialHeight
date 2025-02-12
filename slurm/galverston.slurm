#!/bin/bash --login

#SBATCH --job-name=galverston
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

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
p25=2025_galverston_profile_r4i1p1f1
p97=2097_galverston_profile_r4i1p1f1
obs_lon=-94.7977
obs_lat=29.3013

# 1
# python -m adbo.exp_3d --exp_name=galverston-2025 --profile_name=$p25 --resolution=mid --obs_lat=$obs_lat --obs_lon=$obs_lon
python -m adbo.exp_3d --exp_name=galverston-2097 --profile_name=$p25 --resolution=mid --obs_lat=$obs_lat --obs_lon=$obs_lat