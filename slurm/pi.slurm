#!/bin/bash --login

#SBATCH --job-name=pi-calc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk


# Load modules


# try to activate bashrc
source /work/n02/n02/sdat2/.bashrc

micromamba activate t1

echo "which python"
which python

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'

# single bit test:
python -c "from tcpips.pi_driver import pi_cmip6_part; pi_cmip6_part(exp='ssp585', model='CESM2', member='r4i1p1f1')"
