"""Compare the CMIP6 historical period against ERA5.

We might want to load the raw variables, or the derived variables like potential intensity and potential size.

For each ensemble member then might want to calculate the mean bias for each grid point for each variable, but also if this bias has a trend, and perhaps how significant the bias is.

For the ensemble as a whole we might want to compare how the distribution of mean biases and their trends compare, and whether there are significant difference between different families of models.

"""
import os
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from sithom.time import timeit
from .constants import CDO_PATH, PI4_PATH, PS_PATH
from .era5 import get_all_regridded_data

def load_cmip6_data(exp = "historical", model = "CESM2", member = "r4i1p1f1") -> Optional[xr.Dataset]:
    """Load CMIP6 data for a specific experiment, model, and member.

    This function is a placeholder for loading CMIP6 data.
    The actual implementation would depend on the data source and format.

    Args:
        exp (str): Experiment name (default: "historical").
        model (str): Model name (default: "CESM2").
        member (str): Ensemble member (default: "r1i1p1f1").

    Returns:
        xarray.Dataset: Loaded CMIP6 dataset.
    """
    # Placeholder for actual implementation
    print(f"Loading CMIP6 data for {exp}, {model}, {member}...")
    paths = [os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc", os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc", os.path.join(PI4_PATH, exp, model, member) + ".zarr", os.path.join(PS_PATH, exp, model, member) + ".zarr"]
    ds_list = []
    for path in paths:
        if os.path.exists(path):
            if path.endswith('.zarr'):
                ds = xr.open_zarr(path, consolidated=True)
            else:
                ds = xr.open_dataset(path, chunks={"time": 1})  # Adjust chunk size as needed
            ds_list.append(ds)
            print(f"Loaded dataset from {path}")
        else:
            print(f"Dataset not found at {path}")
    one_day_tolerance = pd.to_timedelta("1D")  # , unit="D")
    for i, ds in enumerate(ds_list[1:]):
        ds_list[i+1] = ds.reindex(
        time=ds_list[0].time,
        method="nearest",  # tolerance="1D"
        tolerance=one_day_tolerance,
    )
        ds_list[i+1] = ds_list[i+1].drop_vars("time_bounds", errors="ignore")
    if ds_list:
        combined_ds = xr.merge(ds_list)
        print(f"Combined dataset contains variables: {list(combined_ds.data_vars)}")
        # remove time_bounds, nbnd,
        combined_ds = combined_ds.drop_vars("time_bounds", errors="ignore")
        # combined_ds = combined_ds.drop_vars("nbnd", errors="ignore")
        return combined_ds
    else:
        print("No datasets were loaded.")
        return None




@timeit
def calc_bias(start_year=1980, end_year=2014):
    """Calc mean biases over specified period.

    """
    era5_ds = get_all_regridded_data(start_year=1980, end_year=2024).sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    print("era5", era5_ds)
    print(f"Loaded ERA5 data from {start_year} to {end_year}.")
    print(f"Data variables: {list(era5_ds.data_vars)}")
    cmip6_ds = load_cmip6_data().sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    if cmip6_ds is not None:
        print(f"Loaded CMIP6 data with variables: {list(cmip6_ds.data_vars)}")

    vars_to_compare = set(era5_ds.data_vars).intersection(set(era5_ds.data_vars)).intersection(["sst", "vmax", "pmin", "otl"])
    print(f"Variables to compare: {vars_to_compare}")
    # ok, let's put the time coordinates on the same axis
    cmip6_ds = cmip6_ds.convert_calendar('standard', use_cftime=False)
    cmip6_ds = cmip6_ds.reindex(
        time=era5_ds.time,
        method="nearest",
        tolerance=pd.to_timedelta("1D"),
    )
    print("CMIP6", cmip6_ds)
    cmip6_ds = cmip6_ds.rename({"plev": "pressure_level"})
    cmip6_ds = cmip6_ds.drop_vars("time_bounds", errors="ignore")
    print("Aligned CMIP6 data to ERA5 time axis.")
    bias_ds = cmip6_ds[vars_to_compare] - era5_ds[vars_to_compare]
    print("bias", bias_ds)
    print(f"Bias dataset variables: {list(bias_ds.data_vars)}")
    bias_ds.mean(dim="time", keep_attrs=True).to_netcdf("cmip6_bias.nc", mode="w")





if __name__ == "__main__":
    # python -m tcpips.bias
    # Load the CMIP6 data
    # cmip6_data = load_cmip6_data()
    calc_bias()  # Placeholder for actual implementation

