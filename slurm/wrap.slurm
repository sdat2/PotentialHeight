#!/bin/bash --login

#SBATCH --job-name=wrap
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
source /work/n02/n02/sdat2/.bashrc
micromamba activate t1

which python
python --version

# Run the job
python -m adforce.wrap angle=45 profile_name=2025 # --exp_name notide-mid-2025 --profile_name 2025.json --resolution mid-notide --animate

# python -m adforce.wrap --exp_name notide-mid-2097 --profile_name 2097.json --resolution mid-notide --animate
