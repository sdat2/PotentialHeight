#!/bin/bash --login

#SBATCH --job-name=data-processing
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:0
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=short

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk


# Load modules
# module load PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1


# try to activate bashrc

# work=/mnt/lustre/a2fs-work1/work/n02/n02/b

source /work/n02/n02/sdat2/.bashrc

micromamba activate t1

# eval "$(conda shell.bash hook)"

echo "which python"
which python

# define variables
case_name=$SLURM_JOB_NAME # name for printing
np=$SLURM_NTASKS  # how many parallel tasks to define
# export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
# export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# run python job
# srun --distribution=block:block --hint=nomultithread python -m tcpips.pangeo

python -m tcpips.regrid

