"""Potential Intensity Driver for CMIP6 Data."""

import os
from typing import Dict
import xarray as xr
from dask.diagnostics import ProgressBar
from sithom.misc import human_readable_size, get_git_revision_hash
from sithom.time import timeit, time_stamp
from .constants import (
    CDO_PATH,
    PI2_PATH,
    REGRIDDED_PATH,
    PI_PATH,
    PI3_PATH,
    PI4_PATH,
    PROJECT_PATH,
)
from .files import locker
from .convert import convert
from .pi import calculate_pi


def find_atmos_ocean_pairs() -> Dict[str, Dict[str, any]]:
    """
    Find the atmospheric and oceanic data pairs that can be combined to calculate potential intensity.

    Returns:
        Dict[str, Dict[str, any]]: Dictionary of pairs.
    """

    pairs = {}
    for exp in [
        x
        for x in os.listdir(REGRIDDED_PATH)
        if os.path.isdir(os.path.join(REGRIDDED_PATH, x))
    ]:
        for model in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean")):
            for member in [
                x.strip(".nc")
                for x in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean", model))
            ]:
                key = f"{exp}.{model}.{member}"
                pi_lock = os.path.join(PI_PATH, key + ".lock")
                print(key)
                oc_path = os.path.exists(
                    os.path.join(REGRIDDED_PATH, exp, "ocean", model, member) + ".nc"
                )
                oc_lock = os.path.exists(
                    os.path.join(REGRIDDED_PATH) + f"{exp}.ocean.{model}.{member}.lock"
                )
                at_path = os.path.exists(
                    os.path.join(REGRIDDED_PATH, exp, "atmos", model, member) + ".nc"
                )
                at_lock = os.path.exists(
                    os.path.join(REGRIDDED_PATH) + f"{exp}.atmos.{model}.{member}.lock"
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


def investigate_cmip6_pairs() -> None:
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
                        REGRIDDED_PATH,
                        pairs[key]["exp"],
                        i,
                        pairs[key]["model"],
                        pairs[key]["member"] + ".nc",
                    )
                ),
            )


@timeit
@locker(PI4_PATH)
def pi_cmip6_part(
    exp: str = "ssp585",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
    time_chunk: int = 1,
    reduce_storage: bool = False,
    fix_temp: bool = True,
) -> None:
    """
    Potential intensity calculation one model ensemble member experiment.

    Args:
        exp (str, optional): Which experiment. Defaults to "ssp585".
        model (str, optional): Which model. Defaults to "CESM2".
        member (str, optional): Which model member. Defaults to "r4i1p1f1".
        time_chunk (int, optional): How big to chunk time. Defaults to 1.
    """
    print(f"exp:{exp} model:{model} member:{member}")

    @timeit
    def open_ds(path: str) -> xr.Dataset:
        """
        Open dataset.

        Args:
            path (str): path to the dataset.

        Returns:
            xr.Dataset: xarray dataset.
        """
        nonlocal time_chunk
        # open netcdf4 file using dask backend
        ds = xr.open_dataset(path, chunks={"time": time_chunk}, decode_times=False)
        ds = ds.drop_vars(
            [
                x
                for x in [
                    "time",
                    "time_bounds",
                    "nbnd",
                ]
                if x in ds
            ]  ## REDUCING time for exp - just taking the first year
        )
        return ds

    ocean_path = os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc"
    ocean_ds = open_ds(ocean_path)
    print("ocean_ds", ocean_ds)
    atmos_ds = open_ds(os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc")
    ocean_time = xr.open_dataset(ocean_path)["time"].values
    print("atmos_ds", atmos_ds)
    # convert units, merge datasets
    ds = convert(xr.merge([ocean_ds, atmos_ds]))
    pi = calculate_pi(ds.compute(), dim="p", fix_temp=fix_temp)
    if reduce_storage:
        del ds["t"]
        del ds["q"]
    ds = xr.merge([ds, pi])
    ds = ds.assign_coords({"time": ("time", ocean_time)})
    ds.attrs["pi_calculated_at_git_hash"] = get_git_revision_hash(
        path=str(PROJECT_PATH)
    )
    ds.attrs["pi_calculated_at_time"] = time_stamp()
    folder = os.path.join(PI4_PATH, exp, model)
    os.makedirs(folder, exist_ok=True)
    delayed_obj = ds.to_netcdf(
        os.path.join(folder, member + ".nc"),
        format="NETCDF4",
        engine="h5netcdf",
        compute=False,
    )
    with ProgressBar():
        _ = delayed_obj.compute()

    print(ds)


if __name__ == "__main__":
    # python -m tcpips.pi_driver
    pi_cmip6_part(exp="ssp585", model="CESM2", member="r4i1p1f1")
    # investigate_cmip6_pairs()
