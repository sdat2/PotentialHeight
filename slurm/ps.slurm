#!/bin/bash --login

#SBATCH --job-name=ps-big
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:0
##%%SBATCH --time=00:20:00

# slurm job name as output log file name
#SBATCH --output=logs/%x-%j.out

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n02-bas
#SBATCH --partition=standard
##%%SBATCH --qos=short
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

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'

# print out the python version
echo "which python"
which python

# run python job
# python -m w22.potential_size

# pytest ../cle/ps.py
# python -m w22.ps

# python -m w22.ps


python -c "from w22.ps_runs import trimmed_cmip6_example; trimmed_cmip6_example()"

python -c "from w22.ps_runs import trimmed_cmip6_example; trimmed_cmip6_example(pi_version=3)"
python -c "from w22.ps_runs import trimmed_cmip6_example; trimmed_cmip6_example(pi_version=4)"


python -c "from w22.ps_runs import global_cmip6 as ga; ga()"


python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'new_orleans'); pt(10, 'new_orleans'); pt(11, 'new_orleans')"


python  -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'galverston'); pt(10, 'galverston'); pt(11, 'galverston')"

python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'miami'); pt(10, 'miami'); pt(11, 'miami')"

