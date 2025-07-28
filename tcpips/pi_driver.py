"""Potential Intensity Driver for CMIP6 Data."""

import os
from typing import Dict
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from sithom.misc import human_readable_size, get_git_revision_hash
from sithom.time import time_stamp
from .constants import (
    CDO_PATH,
    REGRIDDED_PATH,
    PI_PATH,
    PI4_PATH,
    PROJECT_PATH,
)
from .dask import dask_cluster_wrapper
from .files import locker
from .convert import convert
from .pi import calculate_pi


def find_atmos_ocean_pairs(path: str = REGRIDDED_PATH,
                           new_path: str = PI_PATH,
                           ) -> Dict[str, Dict[str, any]]:
    """
    Find the atmospheric and oceanic data pairs that can be combined to calculate potential intensity.

    Args:
        path (str): Path to the regridded data directory. Defaults to REGRIDDED_PATH.
        new_path (str): Path to the directory where potential intensity data will be stored. Defaults to PI_PATH.

    Returns:
        Dict[str, Dict[str, any]]: Dictionary of pairs.
    """

    pairs = {}
    for exp in [
        x
        for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x))
    ]:
        for model in os.listdir(os.path.join(path, exp, "ocean")):
            for member in [
                x.strip(".nc")
                for x in os.listdir(os.path.join(path, exp, "ocean", model))
            ]:
                key = f"{exp}.{model}.{member}"
                pi_lock = os.path.join(new_path, key + ".lock")
                print(key)
                oc_path = os.path.exists(
                    os.path.join(path, exp, "ocean", model, member) + ".nc"
                )
                oc_lock = os.path.exists(
                    os.path.join(path) + f"{exp}.ocean.{model}.{member}.lock"
                )
                at_path = os.path.exists(
                    os.path.join(path, exp, "atmos", model, member) + ".nc"
                )
                at_lock = os.path.exists(
                    os.path.join(path) + f"{exp}.atmos.{model}.{member}.lock"
                )
                if oc_path and at_path and not oc_lock and not at_lock:
                    pairs[f"{exp}.{model}.{member}"] = {
                        "exp": exp,
                        "model": model,
                        "member": member,
                        "locked": os.path.exists(pi_lock),
                    }
                if oc_lock:
                    print(f"Ocean lock file exists for {key}")
                if at_lock:
                    print(f"Atmos lock file exists for {key}")
                if not oc_path:
                    print(f"File missing for {exp}.ocean.{model}.{member}")
                if not at_path:
                    print(f"File missing for {exp}.atmos.{model}.{member}")

    return pairs


def investigate_cmip6_pairs(path: str = REGRIDDED_PATH) -> None:
    """
    Investigate the CMIP6 pairs to see if they are the correct size.
    """

    def hr_file_size(filename: str) -> str:
        st = os.stat(filename)
        return human_readable_size(st.st_size)

    pairs = find_atmos_ocean_pairs()
    for key in pairs:
        print(key)
        for i in ["ocean", "atmos"]:
            print(
                i,
                hr_file_size(
                    os.path.join(
                        path,
                        pairs[key]["exp"],
                        i,
                        pairs[key]["model"],
                        pairs[key]["member"] + ".nc",
                    )
                ),
            )


@locker(PI4_PATH)
def pi_cmip6_part(
    exp: str = "ssp585",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
    time_chunk: int = 1,
    reduce_storage: bool = False,
    fix_temp: bool = True,
) -> None:
    """Potential intensity calculation..."""
    print(f"exp:{exp} model:{model} member:{member}")

    # 1. Open datasets with time coordinates intact
    # 1. Open datasets separately to inspect and align time
    ocean_path = os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc"
    atmos_path = os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc"

    ocean_ds = xr.open_dataset(
        ocean_path,
        chunks={"time": time_chunk},  # decode_times=False
    )

    atmos_ds = xr.open_dataset(
        atmos_path,
        chunks={"time": time_chunk, "plev": -1},  # decode_times=False
    )

    # **This is the key step: Align atmos_ds to ocean_ds's time coordinate**
    print("Aligning atmospheric data to ocean time axis...")
    one_day_tolerance = pd.to_timedelta("1D")  # , unit="D")

    atmos_ds_aligned = atmos_ds.reindex(
        time=ocean_ds.time,
        method="nearest",  # tolerance="1D"
        tolerance=one_day_tolerance,
    )
    atmos_ds_aligned = atmos_ds_aligned.drop_vars("time_bounds", errors="ignore")

    # 2. Now merge the aligned datasets
    ds = xr.merge([ocean_ds, atmos_ds_aligned]).chunk({"time": time_chunk, "plev": -1})

    # --- The rest of your script proceeds as before ---

    ds = convert(ds)
    # If you must drop some variables, do it here, but keep 'time'
    # ds = ds.drop_vars(["time_bounds", "nbnd"], errors='ignore')

    print("Original combined ds:", ds)

    # 2. Perform calculations
    ds = convert(ds)
    pi = calculate_pi(ds, dim="p", fix_temp=fix_temp)
    if reduce_storage:
        ds = ds.drop_vars(["t", "q"], errors="ignore")

    # 3. Merge results (time coordinate is already present)
    ds_out = xr.merge([ds, pi])
    ds_out.attrs["pi_calculated_at_git_hash"] = get_git_revision_hash(
        path=str(PROJECT_PATH)
    )
    ds_out.attrs["pi_calculated_at_time"] = time_stamp()

    # 4. Save to disk
    folder = os.path.join(PI4_PATH, exp, model)
    os.makedirs(folder, exist_ok=True)
    delayed_obj = ds_out.to_zarr(
        os.path.join(folder, member + ".zarr"),
        # format="NETCDF4",
        # engine="h5netcdf",
        # compute=False,
        consolidated=True,
        mode="w",
        compute=False,
    )

    with ProgressBar():
        _ = delayed_obj.compute()  # scheduler="threads")

    print(ds_out)


if __name__ == "__main__":
    # python -m tcpips.pi_driver
    # pi_cmip6_part(exp="ssp585", model="CESM2", member="r4i1p1f1")
    for exp in ["ssp585", "historical"]:
        for model in ["CESM2"]:
            for member in ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"]:
                dask_cluster_wrapper(
                    pi_cmip6_part,
                    exp=exp,
                    model=model,
                    member=member
                )
    # dask_cluster_wrapper(pi_cmip6_part, exp="historical", model="CESM2", member="r4i1p1f1")
    # dask_cluster_wrapper(pi_cmip6_part, exp="ssp585", model="CESM2", member="r10i1p1f1")
    # dask_cluster_wrapper(pi_cmip6_part, exp="ssp585", model="CESM2", member="r11i1p1f1")
    # investigate_cmip6_pairs()
