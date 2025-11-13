#!/bin/bash --login

#SBATCH --job-name=create_test_set
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n01-biopole
#SBATCH --partition=standard
#SBATCH --qos=standard

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# Load modules
module load PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1

# try to activate micromamba
source /work/n01/n01/sithom/.bashrc
which conda
conda --version
conda activate base

which python
python --version

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'


# Run the job
python -m adbo.create_test_set


