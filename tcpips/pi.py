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


@timeit
def calculate_pi(ds: xr.Dataset, dim: str = "p") -> xr.Dataset:
    """Calculate the potential intensity using the tcpyPI package.

    Data must have been converted to the tcpyPI units by `tcpips.convert'.

    Args:
        ds (xr.Dataset): xarray dataset containing the necessary variables.
        dim (str, optional): Vertical dimension. Defaults to "p" for pressure level.

    Returns:
        xr.Dataset: xarray dataset containing the calculated variables.
    """
    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=PTOP, miss_handle=1),
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
    )

    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    out_ds = xr.Dataset(
        {
            "vmax": vmax,
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
    func: Callable[[xr.Dataset], xr.Dataset]
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
