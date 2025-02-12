#!/bin/bash --login

#SBATCH --job-name=bo1
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

# run python job
# srun --distribution=block:block --hint=nomultithread python -m tcpips.pangeo
p25=2025_new_orleans_profile_r4i1p1f1
p97=2097_new_orleans_profile_r4i1p1f1

# 1
python -m adbo.exp_3d --exp_name=8729840-2097 --profile_name=$p97 --resolution=mid --obs_lat=30.404389 --obs_lon=-87.211194
python -m adbo.exp_3d --exp_name=8735180-2097 --profile_name=$p97 --resolution=mid --obs_lat=30.25 --obs_lon=-88.075
python -m adbo.exp_3d --exp_name=8760922-2097 --profile_name=$p97 --resolution=mid --obs_lat=28.932222 --obs_lon=-89.4075
python -m adbo.exp_3d --exp_name=8761724-2097 --profile_name=$p97 --resolution=mid --obs_lat=29.263 --obs_lon=-89.957


# 2
# python -m adbo.exp_3d --exp_name=8762075-2097 --profile_name=$p97 --resolution=mid --obs_lat=29.114167 --obs_lon=-90.199167
# python -m adbo.exp_3d --exp_name=8762482-2097 --profile_name=$p97 --resolution=mid --obs_lat=29.788611 --obs_lon=-90.420278
# python -m adbo.exp_3d --exp_name=8764044-2097 --profile_name=$p97 --resolution=mid-notide --obs_lat=29.6675 --obs_lon=-91.237611
# python -m adbo.exp_3d --exp_name=8729840-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=30.404389 --obs_lon=-87.211194

# 3
# python -m adbo.exp_3d --exp_name=8735180-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=30.25 --obs_lon=-88.075
# python -m adbo.exp_3d --exp_name=8760922-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=28.932222 --obs_lon=-89.4075
# python -m adbo.exp_3d --exp_name=8761724-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=29.263 --obs_lon=-89.957
# python -m adbo.exp_3d --exp_name=8762075-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=29.114167 --obs_lon=-90.199167
# python -m adbo.exp_3d --exp_name=8762482-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=29.788611 --obs_lon=-90.420278
# python -m adbo.exp_3d --exp_name=8764044-2025 --profile_name=$p25 --resolution=mid-notide --obs_lat=29.6675 --obs_lon=-91.237611


#lon=[30.404389, 30.25, 28.932222, 29.263, 29.114167, 29.788611, 29.6675]
#lat=[-87.211194, -88.075, -89.4075, -89.957, -90.199167,-90.420278, -91.237611]
#stationid=['8729840', '8735180', '8760922', '8761724', '8762075', '8762482', '8764044']