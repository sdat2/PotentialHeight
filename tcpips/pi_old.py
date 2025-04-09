"""Old processing script for CMIP6."""

from typing import Tuple
import os
import matplotlib.pyplot as plt
import xarray as xr
from sithom.place import Point, BoundingBox
from sithom.plot import (
    plot_defaults,
)  # , axis_formatter
from sithom.time import timeit
from .convert import convert  # , regrid_2d_1degree
from .pi import calculate_pi
from .xr_utils import propagate_attrs, standard_name_to_long_name
from .plot import plot_features
from .constants import (
    FIGURE_PATH,
    GOM,
    DATA_PATH,
)


@timeit
def plot_diffs(times: Tuple[str, str] = ("1850-09-15", "2099-09-15")) -> None:
    """
    Plot the difference maps between two dates.

    Args:
        times (Tuple[str, str], optional): Defaults to ("1850-09-15", "2099-09-15").
    """
    plot_defaults()
    ds_l = [convert(combined_inout_timestep_cmip6(time=time)) for time in times]
    pi_ds_l = [standard_name_to_long_name(calculate_pi(ds, dim="p")) for ds in ds_l]
    diff_ds = ds_l[1] - ds_l[0]
    pi_diff_ds = pi_ds_l[1] - pi_ds_l[0]
    diff_ds = propagate_attrs(ds_l[0], diff_ds)
    pi_diff_ds = propagate_attrs(pi_ds_l[0], pi_diff_ds)
    print(diff_ds)
    print(pi_diff_ds)
    plot_features(
        pi_diff_ds.isel(y=slice(20, -20)),
        [["vmax", "pmin"], ["t0", "otl"]],
        super_titles=["", ""],
    )
    plt.suptitle(f"{times[1]}-{times[0]}")
    plt.savefig(os.path.join(FIGURE_PATH, f"pi-diff-{times[1]}-{times[0]}.png"))
    plt.clf()
    plot_features(
        diff_ds.isel(y=slice(20, -20)).isel(p=0),
        [["sst", "t"], ["msl", "q"]],
        super_titles=["", ""],
    )
    plt.suptitle(f"{times[1]}-{times[0]}")
    plt.savefig(os.path.join(FIGURE_PATH, f"diff-{times[1]}-{times[0]}.png"))
    plt.clf()


def combined_inout_timestep_cmip6(time: str = "2015-01-15") -> xr.Dataset:
    """
    Combined data from the ocean and atmosphere datasets at a given time.

    Args:
        time (str, optional): _description_. Defaults to "2015-01-15".

    Returns:
        xr.Dataset: combined dataset.
    """

    def open(name: str) -> xr.Dataset:
        ds = xr.open_dataset(name, engine="h5netcdf").sel(time=time).isel(time=0)
        return ds.drop_vars([x for x in ["time", "time_bounds", "nbnd"] if x in ds])

    atmos_ds = open(os.path.join(DATA_PATH, "atmos_new_regridded.nc"))
    ocean_ds = open(os.path.join(DATA_PATH, "ocean_regridded.nc"))
    return xr.merge([ocean_ds, atmos_ds])


def processed_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False
) -> xr.Dataset:
    """
    Process the combined data from the ocean and atmosphere datasets at a given time to produce potential intensity.

    Args:
        time (str, optional): Time string. Defaults to "2015-01-15".
        verbose (bool, optional): Whether to print info. Defaults to False.

    Returns:
        xr.Dataset: Processed dataset.
    """
    ds = combined_inout_timestep_cmip6(time=time)
    lats = ds.lat.values
    lons = ds.lon.values
    ds = ds.drop_vars(["lat", "lon"])
    ds = ds.assign_coords({"lat": ("y", lats[:, 0]), "lon": ("x", lons[0, :])})
    if verbose:
        print("ds with 1d coords", ds)
    ds = ds.swap_dims({"y": "lat", "x": "lon"})
    if verbose:
        print(ds)
    return ds


def gom_combined_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False
) -> xr.Dataset:
    """Get the Gulf of Mexico centre data at a given time.

    Args:
        time (str, optional): Defaults to "2015-01-15".
        verbose (bool, optional): Defaults to False.

    Returns:
        xr.Dataset: Gulf of Mexico centre data at a given time.
    """
    ds = processed_inout_timestep_cmip6(time=time, verbose=verbose)
    # select point closest to GOM centre
    ds = ds.sel(lat=GOM[0], lon=GOM[1], method="nearest")
    ds = convert(ds)
    pi = calculate_pi(ds, dim="p")
    if verbose:
        print(pi)
    return xr.merge([ds, pi])


@timeit
def gom_bbox_combined_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False, pad: float = 5
) -> xr.Dataset:
    """Get the Gulf of Mexico bounding box data at a given time.

    Args:
        time (str, optional): Defaults to "2015-01-15".
        verbose (bool, optional): Defaults to False.
        pad (float, optional): Padding around the bounding box. Defaults to 5.

    Returns:
        xr.Dataset: Gulf of Mexico bounding box data at a given time.
    """
    GOM_BBOX: BoundingBox = Point(GOM[1], GOM[0], desc="Gulf of Mexico Centre").bbox(
        pad
    )
    ds = processed_inout_timestep_cmip6(time=time, verbose=verbose)
    ds = ds.sel(lon=slice(*GOM_BBOX.lon), lat=slice(*GOM_BBOX.lat))
    ds = convert(ds)
    pi = calculate_pi(ds, dim="p")
    if verbose:
        print(pi)
    return xr.merge([ds, pi])
