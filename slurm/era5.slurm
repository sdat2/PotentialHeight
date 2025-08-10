#!/bin/bash --login

#SBATCH --job-name=era5
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

python -m tcpips.era5
