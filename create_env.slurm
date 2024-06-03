#!/bin/bash --login

#SBATCH --job-name=create
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=short

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk


# Load modules
module load PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1

# source ~/.bashrc
# ls ~/
source /work/n02/n02/sdat2/.bashrc

which micromamba
micromamba --version

# Create a new conda environment
micromamba create -n t1 -y -f env.yml
micromamba activate t1

# Check the python version
which python
python --version

# Install the local packages
pip install ../sithom


# Run the job
python -m adforce.wrap --exp_name notide --profile_name 2025.json --stationid 3 --resolution mid-notide

python -m adforce.wrap --exp_name notide --profile_name 2097.json --stationid 3 --resolution mid-notide
