#!/bin/bash --login

#SBATCH --job-name=bo-m32-ei
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n01-biopole
#SBATCH --partition=standard
#SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# try to activate conda
source /work/n01/n01/sithom/.bashrc
which conda
conda --version
conda activate base

# print out the python version
echo "which python"
which python


# run python job
p15=2015_new_orleans_profile_r4i1p1f1
p100=2100_new_orleans_profile_r4i1p1f1

# 1
python -m adbo.exp_3d --exp_name=bo-m32-ei --profile_name=$p15 --kernel=Matern52 --seed=15 --daf=ei