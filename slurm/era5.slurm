#!/bin/bash --login

#SBATCH --job-name=era5-80s
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

# module load PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1

# try to activate micromamba
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba --version
micromamba activate t1

# print out the python version
echo "which python"
which python

export MPLCONFIGDIR="/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/tmp/.matplotlib"
mkdir -p $MPLCONFIGDIR

# python -m tcpips.era5
# echo "Starting dask-mpi job on 10 nodes..."
# srun python -u /work/n02/n02/sdat2/adcirc-swan/worstsurge/tcpips/era5_ps_calc.py
# echo "Job finished."
python -c "from tcpips.era5 import calculate_potential_sizes as cps, dask_cluster_wrapper as dcq; dcq(cps, start_year=1980, end_year=1989, dry_run=False)"
# python -c "from tcpips.era5 import calculate_potential_sizes as cps, dask_cluster_wrapper as dcq; dcq(cps, start_year=1990, end_year=1999, dry_run=False)"
# python -c "from tcpips.era5 import calculate_potential_sizes as cps, dask_cluster_wrapper as dcq; dcq(cps, start_year=2000, end_year=2009, dry_run=False)"
# python -c "from tcpips.era5 import calculate_potential_sizes as cps, dask_cluster_wrapper as dcq; dcq(cps, start_year=2010, end_year=2019, dry_run=False)"
# python -c "from tcpips.era5 import calculate_potential_sizes as cps, dask_cluster_wrapper as dcq; dcq(cps, start_year=2020, end_year=2024, dry_run=False)"