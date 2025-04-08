"""Potential Intensity Calculation script."""

from typing import Callable
import xarray as xr
from tcpyPI import pi
from sithom.time import timeit


TCPYPI_SAMPLE_DATA: str = (
    "../tcpypi/data/sample_data.nc"  # Sample data for the tcpyPI package
)
CKCD: float = 0.9  # Enthalpy exchange coefficient / drag coefficient [dimensionless]
PTOP: float = 50.0  # Top pressure level for the calculation [hPa]


def fix_temp_profile(ds: xr.Dataset, offset_temp_param: float = 1) -> xr.Dataset:
    """
    Fix the temperature profile in the dataset.

    Problem in calculating the potential intensity with incomplete temperature profiles.

    At some places such as the Gulf of Tonkin and the Bay of Bengal, the mean sea level
    pressure in CESM2 in August is low, leading to NaN values in the temperature profile
    at the 1000 hPa level.

    To fix this, we could assume an approximation that the temperature at these low levels
    is equal to sea surface temperature minus one kelvin (a standard parameterization for near surface air temperature).

    TODO: Make this only effect the bottom NaNs of the profile.
    TODO: convert all very high values to NaN (i.e. if t > 300 C, set to NaN).

    Args:
        ds (xr.Dataset): xarray dataset containing the necessary variables "t" and "sst".
        offset_temp_param (float, optional): Temperature offset parameter. Defaults to 1.

    Returns:
        xr.Dataset: xarray dataset with fixed temperature profile.

    Example:
        >>> import numpy as np
        >>> ds = xr.Dataset(data_vars={"sst": (["x", "y"], [[1, 2], [3, 4]], {"units": "degrees_Celsius"}),
        ...                  "t": (["x", "y", "p"], [[[np.nan, 2], [3, 4]], [[5, 6], [np.nan, 8]]], {"units": "degrees_Celsius"}),},
        ...                  coords={"x": [-80, -85], "y": [20, 25], "p": [1000, 850]})
        >>> ds_fixed = fix_temp_profile(ds, offset_temp_param=1)
        >>> np.allclose(ds_fixed.isel(p=0).t.values, [[0, 3], [5, 3]])
        True
    """
    # Fix the temperature profile (doesn't check how )
    sea_level_temp = ds["sst"] - offset_temp_param
    # make all implausibly high values NaN
    # if air temperature > 300 C, set to NaN
    ds["t"] = ds["t"].where(ds["t"] < 300, np.nan)
    # fill in NaN values in the temperature profile with the sea level temperature
    # If air temperature is NaN, set to sea level temperature
    ds["t"] = ds["t"].where(ds["t"].notnull(), sea_level_temp)
    return ds


@timeit
def calculate_pi(ds: xr.Dataset, dim: str = "p", fix_temp=False) -> xr.Dataset:
    """Calculate the potential intensity using the tcpyPI package.

    Data must have been converted to the tcpyPI units by `tcpips.convert'.

    Args:
        ds (xr.Dataset): xarray dataset containing the necessary variables.
        dim (str, optional): Vertical dimension. Defaults to "p" for pressure level.
        fix_temp (bool, optional): Whether to fix the temperature profile. Defaults to True.

    Returns:
        xr.Dataset: xarray dataset containing the calculated variables.
    """
    if fix_temp:
        ds = fix_temp_profile(ds)

    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(
            CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=PTOP, miss_handle=1, V_reduc=1
        ),
        input_core_dims=[
            [],
            [],
            [
                dim,
            ],
            [
                dim,
            ],
            [
                dim,
            ],
        ],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        # dask="allowed", # maybe this could work with dask, but at the moment I get an error
    )

    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    out_ds = xr.Dataset(
        {
            "vmax": vmax,  # maybe change to vp
            "pmin": pmin,
            "ifl": ifl,
            "t0": t0,
            "otl": otl,
        }
    )

    # add names and units to the structure
    out_ds.vmax.attrs["standard_name"], out_ds.vmax.attrs["units"] = (
        "Maximum Potential Intensity",
        "m/s",
    )
    out_ds.pmin.attrs["standard_name"], out_ds.pmin.attrs["units"] = (
        "Minimum Central Pressure",
        "hPa",
    )
    out_ds.ifl.attrs["standard_name"] = "tcpyPI Flag"
    out_ds.t0.attrs["standard_name"], out_ds.t0.attrs["units"] = (
        "Outflow Temperature",
        "K",
    )
    out_ds.otl.attrs["standard_name"], out_ds.otl.attrs["units"] = (
        "Outflow Temperature Level",
        "hPa",
    )

    return standard_name_to_long_name(out_ds)


def standard_name_to_long_name(ds: xr.Dataset) -> xr.Dataset:
    """
    Turn the standard_name attribute into a long_name attribute.

    Args:
        ds (xr.Dataset): dataset with standard_name attributes.

    Returns:
        xr.Dataset: dataset with long_name attributes.
    """
    for var in ds:
        if "standard_name" in ds[var].attrs:
            ds[var].attrs["long_name"] = ds[var].attrs["standard_name"]
    return ds


def propagate_attrs(ds_old: xr.Dataset, ds_new: xr.Dataset) -> xr.Dataset:
    """
    Propagate the standard_name and units attributes from one dataset to another.

    Args:
        ds_old (xr.Dataset): dataset with standard_name and units attributes.
        ds_new (xr.Dataset): dataset with standard_name and units attributes.

    Returns:
        xr.Dataset: dataset with standard_name and units attributes.
    """

    for var in ds_old:
        if var in ds_new:
            if "units" in ds_old[var].attrs:
                ds_new[var].attrs["units"] = ds_old[var].attrs["units"]
            elif "units" not in ds_new[var].attrs:
                ds_new[var].attrs["units"] = ""
            if "long_name" in ds_old[var].attrs:
                ds_new[var].attrs["long_name"] = ds_old[var].attrs["long_name"]
            if "standard_name" in ds_old[var].attrs:
                ds_new[var].attrs["standard_name"] = ds_old[var].attrs["standard_name"]
    return ds_new


def propogate_wrapper(
    func: Callable[[xr.Dataset], xr.Dataset],
) -> Callable[[xr.Dataset], xr.Dataset]:
    """
    Wrapper to propagate the standard_name and units attributes from one dataset to another.

    Args:
        func (Callable[xr.Dataset, xr.Dataset]): function to wrap.

    Returns:
        Callable[xr.Dataset, xr.Dataset]: wrapped function.
    """

    def wrapper(ds_old: xr.Dataset) -> xr.Dataset:
        ds_new = func(ds_old)
        return propagate_attrs(ds_old, ds_new)

    return wrapper
