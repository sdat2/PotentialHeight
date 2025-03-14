#!/bin/bash --login

#SBATCH --job-name=2d-ani-new
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:0
##%%%%SBATCH --time=00:20:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n01-biopole
#SBATCH --partition=standard
##########%SBATCH --qos=short
#SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# try to activate micromamba
source /work/n01/n01/sithom/.bashrc
which conda
conda --version
conda activate base

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'

# print out the python version
echo "which python"
which python

# run python job
python -m adbo.exp_2d --exp_name 2d-ani-subprocess --resolution mid

# try test version
# python -m adbo.exp_2d --exp_name test-bo-2d-3 --year 2025 --resolution mid-notide --test=True