"""Wrap the ADCIRC call into a single function for BayesOpt etc.

To set off Slurm I could use (https://stackoverflow.com/questions/61704713/best-practice-submitting-slurm-jobs-via-python):
- Slurmpy https://github.com/brentp/slurmpy
"""

import os
from typing import Callable, Dict, Union, List
import numpy as np
import shutil
import time
from datetime import timedelta
import xarray as xr
from slurmpy import Slurm
from sithom.time import timeit
from sithom.place import Point
from cle.constants import DATA_PATH as CLE_DATA_PATH
from .constants import NEW_ORLEANS  # , KATRINA_TIDE_NC
from .fort22 import save_forcing
from .mesh import xr_loader
from .profile import read_profile

# some of these could be moved to a config file.
ROOT: str = "/work/n02/n02/sdat2/adcirc-swan/"
OG_PATH: str = "/work/n02/n02/sdat2/adcirc-swan/examples/NWS13example"
EXE_PATH: str = "/work/n02/n02/sdat2/adcirc-swan/adcirc/work"
MODEL_PATHS_DICT: Dict[str, str] = {
    "low": "/work/n02/n02/sdat2/adcirc-swan/adcirc-testsuite/adcirc/adcirc_katrina-2d-nws13-parallel",
    "mid": "/work/n02/n02/sdat2/adcirc-swan/examples/NWS13example",
    "mid-notide": "/work/n02/n02/sdat2/adcirc-swan/examples/NWS13notide",
    "high": "/work/n02/n02/sdat2/adcirc-swan/examples/kat.nws13.2004",
}
EXP_PATH: str = os.path.join(ROOT, "exp")
NODE_DICT: Dict[str, int] = {"low": 1, "mid": 1, "mid-notide": 1, "high": 8}
QOS_DICT: Dict[str, str] = {
    "low": "short",
    "mid": "short",
    "mid-notide": "short",
    "high": "standard",
}
TIME_DICT: Dict[str, float] = {
    # time in seconds
    "low": 20 * 60,  # wall time of 20 minutes for short qos queue on ARCHER2
    "mid": 20 * 60,
    "mid-notide": 20 * 60,
    "high": 60 * 60,
}
EMAIL_ADDRESS: str = "sdat2@cam.ac.uk"
SLURM_ACCOUNT: str = "n02-bas"
PARTITION: str = "standard"
TASKS_PER_NODE: int = 128
MODULES: str = (
    "PrgEnv-gnu/8.3.3 cray-hdf5-parallel/1.12.2.1 cray-netcdf-hdf5parallel/4.9.0.1"
)
FILES_TO_COPY: List[str] = [
    "fort.13",  # node attributes
    "fort.14",  # mesh
    "fort.15",  # model settings
    # setup files
    # "fort.64.nc",
    # "fort.73.nc",
    # "fort.74.nc",
]
DEFAULT_PROFILE: str = os.path.join(CLE_DATA_PATH, "outputs.json")


def adcirc_run_struct(
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -0.6,
    impact_lat: float = 0,
    impact_time: np.datetime64 = np.datetime64("2004-08-13T12"),
    obs: Point = NEW_ORLEANS,
    resolution: str = "mid",
    profile: Union[str, xr.Dataset] = DEFAULT_PROFILE,
    out_path: str = os.path.join(EXP_PATH, "test-run"),
) -> xr.Dataset:
    """
    Run characteristics to pass between parts of adforce.

    Args:
        angle (float, optional): Angle of the storm. Defaults to 0.
        trans_speed (float, optional): Translation speed of the storm. Defaults to 7.71.
        impact_lon (float, optional): Longitude of the storm impact. Defaults to -0.6.
        impact_lat (float, optional): Latitude of the storm impact. Defaults to 0.
        impact_time (np.datetime64, optional): Time of the storm impact. Defaults to np.datetime64("2004-08-13T12").
        resolution (str, optional): Resolution of the ADCIRC model. Defaults to "mid".
        profile (Union[str, xr.dataset], optional): Profile of the storm. Defaults to DEFAULT_PROFILE.
        out_path (str, optional): Path to ADCIRC run folder. Defaults to os.path.join(EXP_PATH, "test-run").

    TODO: Add in the ability to add a time varying profile.

    Returns:
        xr.Dataset: Experiment structure.
    """
    if resolution not in MODEL_PATHS_DICT:
        raise ValueError(f"Resolution {resolution} not in {MODEL_PATHS_DICT.keys()}")

    if isinstance(profile, str):
        profile = read_profile(profile)

    param = xr.Dataset(
        data_vars={
            "angle": (
                [],
                [angle],
                {"units": "degrees", "description": "Bearing of the storm track"},
            ),
            "trans_speed": (
                [],
                [trans_speed],
                {"units": "m/s", "description": "Translation speed of the storm"},
            ),
            "impact_lon": (
                [],
                [impact_lon],
                {
                    "units": "degrees",
                    "description": "Longitude of the storm impact relative to the observation point",
                },
            ),
            "impact_lat": (
                [],
                [impact_lat],
                {
                    "units": "degrees",
                    "description": "Latitude of the storm impact relative to the observation point",
                },
            ),
            "impact_time": (
                [],
                [impact_time],
                {"units": "ns", "description": "Time of the storm impact"},
            ),
            "obs_lon": (
                [],
                [obs.lon],
                {
                    "description": "Longitude of the observation point",
                    "units": "degrees_east",
                },
            ),
            "obs_lat": (
                [],
                [obs.lat],
                {
                    "description": "Latitude of the observation point",
                    "units": "degrees_north",
                },
            ),
            "resolution": (
                [],
                [resolution],
                {"description": "Resolution of the ADCIRC model"},
            ),
            "out_path": ([], [out_path], {"description": "Path to ADCIRC run folder"}),
            "velocity": (["radius"], profile["VV"], {"units": "m/s"}),
        },
        coords={"radius": profile["rr"]},
        # coords={"time": impact_time,},
    )

    return xr.merge([param, profile])


@timeit
def setup_adcirc_run_folder(
    new_path: str,
    profile_path: str = os.path.join(CLE_DATA_PATH, "outputs.json"),
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
        profile_path (str, optional): Path to the storm profile. Defaults to os.path.join(CLE_DATA_PATH, "outputs.json").
        angle (float, optional): Angle of the storm. Defaults to 0. Degrees from North.
        trans_speed (float, optional): Translation speed of the storm. Defaults to 7.71 m/s.
        impact_lon (float, optional): Longitude of the storm impact. Defaults to -89.4715.
        impact_lat (float, optional): Latitude of the storm impact. Defaults to 29.9511.
        impact_time (np.datetime64, optional): Time of the storm impact. Defaults to np.datetime64("2004-08-13T12", "ns").
        resolution (str, optional): Resolution of the ADCIRC model. Defaults to "mid".
    """
    # original path to copy setting files from
    os.makedirs(new_path, exist_ok=True)
    for file in FILES_TO_COPY:
        shutil.copy(
            os.path.join(MODEL_PATHS_DICT[resolution], file),
            os.path.join(new_path, file),
        )

    save_forcing(
        new_path,
        profile_path=profile_path,
        angle=angle,
        trans_speed=trans_speed,
        impact_lon=impact_lon,
        impact_lat=impact_lat,
        impact_time=impact_time,
    )


def is_slurm_job_finished(jid: int) -> bool:
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
def setoff_slurm_job_and_wait(
    direc: str, jobname: str = "run", time_limit: float = 20 * 60, nodes: int = 1
) -> int:
    """
    Run the ADCIRC model and wait for it to finish.

    Uses slurmpy to set off a slurm job and sacct to check if it has finished.

    Args:
        direc (str): Path to ADCIRC run folder.
        jobname (str, optional): Job name. Defaults to "run".
        time_limit (float, optional): Time limit in seconds, before leaving without answer. Defaults to 60*60 (3 hours).
        nodes (int, optional): Number of nodes to use. Defaults to 1.

    Returns:
        int: Slurm Job ID.
    """
    # time_limit_str = "HH:mm:ss".format(time_limit)
    s = Slurm(
        jobname,
        {
            "nodes": nodes,
            "account": SLURM_ACCOUNT,
            "partition": PARTITION,
            "qos": QOS_DICT["mid"],
            "time": str(timedelta(seconds=time_limit)),
            "tasks-per-node": TASKS_PER_NODE,  # number of cpus on archer2 node.
            "cpus-per-task": 1,
            "output": os.path.join(direc, "slurm.out"),
            "error": os.path.join(direc, "slurm.out"),
            "mail-type": "ALL",
            "mail-user": EMAIL_ADDRESS,
        },
    )

    jid = s.run(
        f"""
module load {MODULES}

cd {direc}

EXE_PATH={EXE_PATH}

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
$EXE_PATH/adcprep --np $np --partmesh >  adcprep.log
$EXE_PATH/adcprep --np $np --prepall  >> adcprep.log
if [ $? == 0 ] ; then
    echo "done!"
else
    echo "ERROR!"
    exit 1
fi

echo -n "    Runnning case..."
srun --distribution=block:block --hint=nomultithread $EXE_PATH/padcirc > padcirc_log.txt
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
    is_finished = is_slurm_job_finished(jid)
    tinc = 1
    # wait twice as long as the time limit given to the slurm job
    # to account for queueing.
    while not is_finished and time_total < time_limit * 2:
        is_finished = is_slurm_job_finished(jid)
        time.sleep(tinc)
        time_total += tinc

    if not is_finished:
        print(f"Job {jid} did not finish in time")
    else:
        print(f"Job {jid} finished")
    return jid


def maxele_observation_func(
    point: Union[Point, int] = NEW_ORLEANS, resolution: str = "mid"
) -> Callable[[str], float]:
    """
    Create a function to select the maximum elevation near a given stationid.

    TODO: Curently now just selects based on the closest point to NEW_ORLEANS.

    Args:
        point (Union[Point, int]): Stationid to select.
        resolution (str, optional): Original path. Defaults to OG_PATH.

    Returns:
        Callable: Function to select the maximum elevation near a given stationid.
    """
    # tide_ds = xr.open_dataset(KATRINA_TIDE_NC)
    # lon, lat = (
    #    tide_ds.isel(stationid=stationid).lon.values,
    #    tide_ds.isel(stationid=stationid).lat.values,
    # )
    if isinstance(point, int):
        min_p = 1000
    else:
        mele_og = xr_loader(os.path.join(MODEL_PATHS_DICT[resolution], "maxele.63.nc"))
        # read zeta_max point closes to the
        # work out closest point to NEW_ORLEANS
        xs = mele_og.x.values
        ys = mele_og.y.values
        distsq = (xs - point.lon) ** 2 + (ys - point.lat) ** 2
        min_p = np.argmin(distsq)  # closest point to stationid
        print("Finding closest point to:\n", point)
        print(
            "closest node to point: {:}, {:.3f}E, {:.3f}N ".format(
                min_p, xs[min_p], ys[min_p]
            )
        )

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
def run_adcirc_idealized_tc(
    out_path: str = "test-run",
    profile_name: str = "2025.json",
    select_point: str = maxele_observation_func(3, resolution="mid"),
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -89.4715,
    impact_lat: float = 29.9511,
    impact_time=np.datetime64("2004-08-13T12", "ns"),
    resolution: str = "mid",
) -> float:
    """
    Run the ADCIRC model and wait for it to finish.

    Args:
        out_path (str, optional): Path to ADCIRC run folder. Defaults to OG_PATH.
        select_point (Callable, optional): Function to select the maximum elevation near a given stationid. Defaults to maxele_observation_func(3).
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
    setup_adcirc_run_folder(
        out_path,
        profile_path=os.path.join(CLE_DATA_PATH, profile_name),
        angle=angle,
        trans_speed=trans_speed,
        impact_lon=impact_lon,
        impact_lat=impact_lat,
        impact_time=impact_time,
        resolution=resolution,
    )
    # set off sbatch.
    setoff_slurm_job_and_wait(out_path, nodes=NODE_DICT[resolution])
    # look at results.
    return select_point(out_path)


if __name__ == "__main__":
    # python -m adforce.wrap
    # python adforce/wrap.py
    # read_angle_exp()
    # run_speed()
    # run_angle_new()

    import argparse

    parser = argparse.ArgumentParser(description="Run ADCIRC model.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="notide-example-2",
        help="Path to ADCIRC run folder.",
    )
    parser.add_argument(
        "--profile_name",
        type=str,
        default="2025.json",
        help="Path to ADCIRC run folder.",
    )
    parser.add_argument(
        "--resolution", type=str, default="mid", help="Resolution of the ADCIRC model."
    )
    parser.add_argument(
        "--stationid",
        type=int,
        default=3,
        help="Stationid to select.",
    )

    args = parser.parse_args()

    res = run_adcirc_idealized_tc(
        out_path=os.path.join(EXP_PATH, args.exp_name),
        profile_name=args.profile_name,
        select_point=maxele_observation_func(
            args.stationid, resolution=args.resolution
        ),
        angle=0,
        resolution=args.resolution,
    )
    print(res)

    # python -m adforce.wrap --exp_name notide-mid --profile_name 2025.json --stationid 3 --resolution mid-notide
