"""Some xarray utility functions."""

from typing import Callable
import xarray as xr


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


def propagate_wrapper(
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
