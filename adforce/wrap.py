"""Wrap.

To set off Slurm I could use (https://stackoverflow.com/questions/61704713/best-practice-submitting-slurm-jobs-via-python):
- Slurmpy https://github.com/brentp/slurmpy
        -- simplest
- Snakemake
@article{molder2021sustainable,
  title={Sustainable data analysis with Snakemake},
  author={M{\"o}lder, Felix and Jablonski, Kim Philipp and Letcher, Brice and Hall, Michael B and Tomkins-Tinch, Christopher H and Sochat, Vanessa and Forster, Jan and Lee, Soohyun and Twardziok, Sven O and Kanitz, Alexander and others},
  journal={F1000Research},
  volume={10},
  year={2021},
  publisher={Faculty of 1000 Ltd}
}
- Joblib (https://joblib.readthedocs.io/en/latest/)
- Fireworks (https://materialsproject.github.io/fireworks/)
@article {CPE:CPE3505,
author = {Jain, Anubhav and Ong, Shyue Ping and Chen, Wei and Medasani, Bharat and Qu, Xiaohui and Kocher, Michael and Brafman, Miriam and Petretto, Guido and Rignanese, Gian-Marco and Hautier, Geoffroy and Gunter, Daniel and Persson, Kristin A.},
title = {FireWorks: a dynamic workflow system designed for high-throughput applications},
journal = {Concurrency and Computation: Practice and Experience},
volume = {27},
number = {17},
issn = {1532-0634},
url = {http://dx.doi.org/10.1002/cpe.3505},
doi = {10.1002/cpe.3505},
pages = {5037--5059},
keywords = {scientific workflows, high-throughput computing, fault-tolerant computing},
year = {2015},
note = {CPE-14-0307.R2},
"""

import os
from typing import Callable, Dict
import numpy as np
import shutil
import time
import xarray as xr
import matplotlib.pyplot as plt
from slurmpy import Slurm
from sithom.time import timeit
from src.constants import NEW_ORLEANS, KATRINA_TIDE_NC
from .fort22 import save_forcing
from .mesh import xr_loader


ROOT: str = "/work/n01/n01/sithom/adcirc-swan/"
OG_PATH: str = "/work/n01/n01/sithom/adcirc-swan/NWS13example"
model_ref_paths: Dict[str, str] = {
    "mid": "/work/n01/n01/sithom/adcirc-swan/NWS13example",
    "high": "/work/n01/n01/sithom/adcirc-swan/kat.nws13.2004",
}
node_dict: Dict[str, int] = {"low": 1, "mid": 1, "high": 8}


@timeit
def setup_new(
    new_path: str,
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -89.4715,
    impact_lat: float = 29.9511,
    impact_time=np.datetime64("2004-08-13T12", "ns"),
    resolution: str = "mid",
) -> None:
    """
    Set up a new ADCIRC folder and add in the forcing data.

    Args:
        new_path (str): New model directory.
    """
    # original path to copy setting files from
    files = [
        "fort.13",
        "fort.14",
        "fort.15",
        "fort.64.nc",
        "fort.73.nc",
        "fort.74.nc",
    ]
    os.makedirs(new_path, exist_ok=True)
    for file in files:
        shutil.copy(
            os.path.join(model_ref_paths[resolution], file),
            os.path.join(new_path, file),
        )

    save_forcing(
        new_path,
        angle=angle,
        trans_speed=trans_speed,
        impact_lon=impact_lon,
        impact_lat=impact_lat,
        impact_time=impact_time,
    )


def is_job_finished(jid: int) -> bool:
    """
    Check if a SLURM job is finished using sacct.

    Args:
        jid (int): Job ID.

    Returns:
        bool: Whether the job is finished.
    """

    args = f"sacct -j {jid} -o state"
    job_states = [x.strip() for x in os.popen(args).read().strip().split("\n")]
    return (
        np.all([x == "COMPLETED" for x in job_states[2:]])
        if len(job_states) > 2
        else False
    )


@timeit
def run_and_wait(
    direc: str, jobname: str = "run", time_limit: float = 3 * 60 * 60, nodes: int = 1
) -> int:
    """
    Run the ADCIRC model and wait for it to finish.

    Args:
        direc (str): Path to ADCIRC run folder.
        jobname (str, optional): Job name. Defaults to "run".
        time_limit (float, optional): Time limit in seconds, before leaving without answer. Defaults to 60*60 (3 hours).

    Returns:
        int: Slurm Job ID.
    """
    s = Slurm(
        jobname,
        {
            "nodes": nodes,
            "account": "n01-SOWISE",
            "partition": "standard",
            "qos": "standard",
            "time": "1:0:0",
            "tasks-per-node": 128,  # number of cpus on archer2 node.
            "cpus-per-task": 1,
            "output": os.path.join(direc, "slurm.out"),
            "error": os.path.join(direc, "slurm.out"),
            "mail-type": "ALL",
            "mail-user": "sdat2@cam.ac.uk",
        },
    )

    jid = s.run(
        f"""
module load PrgEnv-gnu/8.3.3
module load cray-hdf5-parallel/1.12.2.1
module load cray-parallel-netcdf/1.12.3.1

cd {direc}

home_dir=/mnt/lustre/a2fs-work1/work/n01/n01/sithom
source $home_dir/.bashrc

compile_dir=/work/n01/n01/sithom/adcirc-swan/compile_n4

# define variables
case_name=$SLURM_JOB_NAME # name for printing
np=$SLURM_NTASKS # how many parallel tasks to define

export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#...Run the case
echo ""
echo "|---------------------------------------------|"
echo "    TEST CASE: $case_name"
echo ""
echo -n "    Prepping case..."
$compile_dir/adcprep --np $np --partmesh >  adcprep.log
$compile_dir/adcprep --np $np --prepall  >> adcprep.log
if [ $? == 0 ] ; then
    echo "done!"
else
    echo "ERROR!"
    exit 1
fi

echo -n "    Runnning case..."
srun --distribution=block:block --hint=nomultithread $compile_dir/padcirc > padcirc_log.txt
exitstat=$?
echo "Finished"
echo "    ADCIRC Exit Code: $exitstat"
if [ "x$exitstat" != "x0" ] ; then
    echo "    ERROR: ADCIRC did not exit cleanly."
    exit 1
fi
echo ""

"""
    )

    time_total = 0
    is_finished = is_job_finished(jid)
    tinc = 1
    while not is_finished and time_total < time_limit:
        is_finished = is_job_finished(jid)
        time.sleep(tinc)
        time_total += tinc

    if not is_finished:
        print(f"Job {jid} did not finish in time")
    else:
        print(f"Job {jid} finished")
    return jid


def select_point_f(stationid: int, resolution: str = "mid") -> Callable[[str], float]:
    """
    Create a function to select the maximum elevation near a given stationid.

    Args:
        stationid (int): Stationid to select.
        resolution (str, optional): Original path. Defaults to OG_PATH.

    Returns:
        Callable: Function to select the maximum elevation near a given stationid.
    """
    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)
    lon, lat = (
        tide_ds.isel(stationid=stationid).lon.values,
        tide_ds.isel(stationid=stationid).lat.values,
    )
    mele_og = xr_loader(os.path.join(model_ref_paths[resolution], "maxele.63.nc"))
    # read zeta_max point closes to the
    # work out closest point to NEW_ORLEANS
    xs = mele_og.x.values
    ys = mele_og.y.values
    # lon, lat = NEW_ORLEANS.lon, NEW_ORLEANS.lat
    dist = ((xs - lon) ** 2 + (ys - lat) ** 2) ** 0.5
    min_p = np.argmin(dist)  # closest point to stationid

    def select_max(path: str) -> float:
        """
        Select the maximum elevation near a given stationid from time series data.

        Args:
            path (str): Path to ADCIRC run folder.

        Returns:
            float: Maximum elevation near a given stationid.
        """
        mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
        return mele_ds["zeta_max"].values[min_p]

    return select_max


@timeit
def run_wrapped(
    out_path: str = "test-run",
    select_point: str = select_point_f(3, resolution="mid"),
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -89.4715,
    impact_lat: float = 29.9511,
    impact_time=np.datetime64("2004-08-13T12", "ns"),
    resolution: str = "mid",
):
    """
    Run the ADCIRC model and wait for it to finish.

    Args:
        out_path (str, optional): Path to ADCIRC run folder. Defaults to OG_PATH.
        select_point (Callable, optional): Function to select the maximum elevation near a given stationid. Defaults to select_point_f(3).
        angle (float, optional): Angle of the storm. Defaults to 0.
        trans_speed (float, optional): Translation speed of the storm. Defaults to 7.71.
        impact_lon (float, optional): Longitude of the storm impact. Defaults to -89.4715.
        impact_lat (float, optional): Latitude of the storm impact. Defaults to 29.9511.
        impact_time (np.datetime64, optional): Time of the storm impact. Defaults to np.datetime64("2004-08-13T12", "ns").
        resolution (str, optional): Resolution of the ADCIRC model. Defaults to "mid".

    Returns:
        float: Maximum elevation near a given stationid.
    """
    # add new forcing
    setup_new(
        out_path,
        angle=angle,
        trans_speed=trans_speed,
        impact_lon=impact_lon,
        impact_lat=impact_lat,
        impact_time=impact_time,
        resolution=resolution,
    )
    # set off sbatch.
    run_and_wait(out_path, nodes=node_dict[resolution])
    # look at results.
    return select_point(out_path)


@timeit
def read_results(path: str = OG_PATH, stationid: int = 3):
    """
    Read the results of the ADCIRC run.

    Args:
        path (str, optional): Path to ADCIRC run folder. Defaults to OG_PATH.
        stationid (int, optional): Stationid to select. Defaults to 3.

    Returns:
        float: Maximum elevation near a given stationid.
    """
    mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
    # read zeta_max point closes to the
    # work out closest point to NEW_ORLEANS
    xs = mele_ds.x.values
    ys = mele_ds.y.values

    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)
    lon, lat = (
        tide_ds.isel(stationid=stationid).lon.values,
        tide_ds.isel(stationid=stationid).lat.values,
    )

    # lon, lat = NEW_ORLEANS.lon, NEW_ORLEANS.lat
    dist = ((xs - lon) ** 2 + (ys - lat) ** 2) ** 0.5
    min_p = np.argmin(dist)
    # Read the maximum elevation for that point
    return mele_ds["zeta_max"].values[min_p]


if __name__ == "__main__":
    # python -m adforce.wrap
    # python adforce/wrap.py
    # read_angle_exp()
    # run_speed()
    # TODO: add an option to turn the tide off.
    # run_angle_new()
    run_wrapped(
        out_path="/work/n01/n01/sithom/adcirc-swan/kat.nws13.2004.wrap3",
        select_point=select_point_f(3, resolution="high"),
        angle=10,
        resolution="high",
    )
