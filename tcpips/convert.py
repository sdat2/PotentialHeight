from typing import Dict
from sithom.time import timeit
import xarray as xr

# CMIP6 equivalent names
# tos: Sea Surface Temperature [degC] [same]
# hus: Specific Humidity [kg/kg] [to g/kg]
# ta: Air Temperature [K] [to degC]
# psl: Sea Level Pressure [Pa] [to hPa]
# calculate PI over the whole data set using the xarray universal function
conversion_names: Dict[str, str] = {"tos": "sst", "hus": "q", "ta": "t", "psl": "msl"}
conversion_multiples: Dict[str, float] = {
    "hus": 1000,
    "psl": 0.01,  # "plev": 0.01
}
conversion_additions: Dict[str, float] = {"ta": -273.15}
conversion_units: Dict[str, str] = {
    "hus": "g/kg",
    "psl": "hPa",
    "tos": "degC",
    "ta": "degC",  # "plev": "hPa"
}


@timeit
def convert(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert CMIP6 variables to be PI input variables

    Args:
        ds (xr.Dataset): CMIP6 dataset.

    Returns:
        xr.Dataset: PI dataset.
    """
    for var in conversion_multiples:
        if var in ds:
            ds[var] *= conversion_multiples[var]
    for var in conversion_additions:
        if var in ds:
            ds[var] += conversion_additions[var]
    for var in conversion_units:
        if var in ds:
            ds[var].attrs["units"] = conversion_units[var]
    ds = ds.rename(conversion_names)
    if "plev" in ds:
        ds = ds.set_coords(["plev"])
        ds["plev"] = ds["plev"] / 100
        ds["plev"].attrs["units"] = "hPa"
        ds = ds.rename({"plev": "p"})
    return ds
