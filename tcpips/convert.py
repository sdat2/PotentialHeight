"""Convert CMIP6 variables to be PI input variables."""

import xarray as xr
from sithom.time import timeit
from tcpips.constants import (
    CONVERSION_NAMES,
    CONVERSION_MULTIPLES,
    CONVERSION_ADDITIONS,
    CONVERSION_UNITS,
)

# CMIP6 equivalent names
# tos: Sea Surface Temperature [degC] [same]
# hus: Specific Humidity [kg/kg] [to g/kg]
# ta: Air Temperature [K] [to degC]
# psl: Sea Level Pressure [Pa] [to hPa]
# calculate PI over the whole data set using the xarray universal function


@timeit
def convert(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert CMIP6 variables to be PI input variables.

    Assume that you can do a linear conversion between the variables:

    y = m*x + c

    m stored in CONVERSION_MULTIPLES, c stored in CONVERSION_ADDITIONS.

    units are stored in CONVERSION_UNITS.

    plev is a special case, it is converted to hPa, assuming it was Pa to start with.

    Args:
        ds (xr.Dataset): CMIP6 dataset.

    Returns:
        xr.Dataset: PI dataset.
    """
    for var in CONVERSION_MULTIPLES:
        if var in ds:
            ds[var] *= CONVERSION_MULTIPLES[var]
    for var in CONVERSION_ADDITIONS:
        if var in ds:
            ds[var] += CONVERSION_ADDITIONS[var]
    for var in CONVERSION_UNITS:
        if var in ds:
            ds[var].attrs["units"] = CONVERSION_UNITS[var]
    ds = ds.rename(CONVERSION_NAMES)
    if "plev" in ds:
        ds = ds.set_coords(["plev"])
        ds["plev"] = ds["plev"] / 100
        ds["plev"].attrs["units"] = "hPa"
        ds = ds.rename({"plev": "p"})
    return ds
