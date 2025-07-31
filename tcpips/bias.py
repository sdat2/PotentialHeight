"""Compare the CMIP6 historical period against ERA5.

We might want to load the raw variables, or the derived variables like potential intensity and potential size.

For each ensemble member then might want to calculate the mean bias for each grid point for each variable, but also if this bias has a trend, and perhaps how significant the bias is.

For the ensemble as a whole we might want to compare how the distribution of mean biases and their trends compare, and whether there are significant difference between different families of models.

"""
import os
from typing import Optional
import numpy as np
import xarray as xr
from .constants import CDO_PATH, PI4_PATH, PS_PATH
from .era5 import get_all_data

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
    paths = [os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc", os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc", os.path.join(PI4_PATH, exp, model, member) + ".nc", os.path.join(PS_PATH, exp, model, member) + ".nc"]
    ds_list = []
    for path in paths:
        if os.path.exists(path):
            ds = xr.open_dataset(path)
            ds_list.append(ds)
            print(f"Loaded dataset from {path}")
        else:
            print(f"Dataset not found at {path}")
    if ds_list:
        combined_ds = xr.merge(ds_list)
        print(f"Combined dataset contains variables: {list(combined_ds.data_vars)}")
        return combined_ds
    else:
        print("No datasets were loaded.")
        return None



#def load_cmip6_data():
def calc_bias(start_year=1980, end_year=2014):
    """Calc mean biases over specified period.

    """
    era5_ds = get_all_data(start_year=1980, end_year=2024).sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    print(f"Loaded ERA5 data from {start_year} to {end_year}.")
    print(f"Data variables: {list(era5_ds.data_vars)}")
    cmip6_ds = load_cmip6_data().sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    if cmip6_ds is not None:
        print(f"Loaded CMIP6 data with variables: {list(cmip6_ds.data_vars)}")
        # Calculate mean bias for each variable
        bias = {}


if __name__ == "__main__":
    # python -m tcpips.bias
    # Load the CMIP6 data
    # cmip6_data = load_cmip6_data()
    calc_bias()  # Placeholder for actual implementation

