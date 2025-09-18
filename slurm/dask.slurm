#!/bin/bash --login

#SBATCH --job-name=dask-ps-calc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# Your project code
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

# Your email settings
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# --- KEY CHANGES ---
# The Dask cluster is now defined in the Python script.
# This Slurm script just launches that Python script.

echo "Setting up environment..."
source /work/n02/n02/sdat2/.bashrc
micromamba activate t1
echo "Python environment activated: $(which python)"

export MPLCONFIGDIR="/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/tmp/.matplotlib"
mkdir -p $MPLCONFIGDIR

echo "Starting Dask job controller script..."
# We just run the python script directly. It will submit the other Slurm jobs.
python -u /work/n02/n02/sdat2/adcirc-swan/worstsurge/tcpips/run_dask_calculation.py
echo "Job finished."