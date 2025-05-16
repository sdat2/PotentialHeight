#!/bin/bash --login

#SBATCH --job-name=ps-test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
##%%SBATCH --time=24:00:0
#SBATCH --time=00:20:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=short
###%%SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# try to activate micromamba
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba --version
micromamba activate t1

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'

# print out the python version
echo "which python"
which python

# run python job
#python -m w22.potential_size

# pytest ../w22/ps.py
# python -m w22.ps

python -m w22.test --name archer2-slurm
#python -m w22.ps

# Time taken by oct2py for 10 loops is 21.8 s
# Time taken by octave for 10 loops is 7.9 s