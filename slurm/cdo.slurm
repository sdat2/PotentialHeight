#!/bin/bash --login

#SBATCH --job-name=cdo-cesm
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
# module load PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1


# try to activate bashrc
source /work/n02/n02/sdat2/.bashrc

micromamba activate t1

echo "which python"
which python

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'

# single bit test:
# python -c "from tcpips.regrid import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='CESM2', member='r10i1p1f1')"

module load cdo
module load nco

python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='CESM2', member='r10i1p1f1')"
python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='CESM2', member='r10i1p1f1')"
#python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='CESM2', member='r11i1p1f1')"
python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='CESM2', member='r11i1p1f1')"
python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='CESM2', member='r4i1p1f1')"
python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='CESM2', member='r4i1p1f1')"

#python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='UKESM1-0-LL', member='r1i1p1f2')"
# python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='UKESM1-0-LL', member='r2i1p1f2')"
# python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='ocean', model='UKESM1-0-LL', member='r3i1p1f2')"

# python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='UKESM1-0-LL', member='r1i1p1f2')"
# python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='UKESM1-0-LL', member='r2i1p1f2')"
# python -c "from tcpips.regrid_cdo import regrid_cmip6_part; regrid_cmip6_part(exp='ssp585', typ='atmos', model='UKESM1-0-LL', member='r3i1p1f2')"
# # python -m tcpips.regrid_cdo
# ls -lash ../data/cmip6/regridded
# du -h --max-depth=1 ../data/cmip6/regridded
