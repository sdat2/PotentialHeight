import os
from typing import List, Union, Literal, Optional
import numpy as np
import pandas as pd
import xarray as xr
import dask
from xmip.preprocessing import combined_preprocessing
import cftime
from intake import open_catalog
from sithom.time import timeit, time_limit, TimeoutException
from sithom.xr import sel, spatial_mean

#  from src.xr_utils import sel, spatial_mean, get_trend, get_clim
from tcpips.reg_new import (
    regrid_2d_to_standard,
    regrid_1d_to_standard,
    regrid_2d,
    regrid_1d,
)

# Configuration and Constants
CMIP6_INTAKE_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
PANGEO_CAT_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/master.yaml"
START_YEAR = "1958"
END_YEAR = "2017"
DEFAULT_FUTURE_SCENARIO = "ssp585"
SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]


# Step 1: Helper Functions (Pure and Stateless)
def standardize_time(
    time: Union[cftime.datetime, np.datetime64],
    calendar: str = "standard",
    standard_day: int = 15,
) -> cftime.datetime:
    """
    Standardizes time to the same day of the month.

    Args:
        time (Union[cftime.datetime, np.datetime64]): Time array.
        calendar (str, optional): Calendar type. Defaults to 'standard'.
        standard_day (int, optional): Day of the month to standardize. Defaults to 15.

    Returns:
        cftime.datetime: Standardized time.
    """
    if isinstance(time, np.datetime64):
        time = pd.to_datetime(time)
    return cftime.datetime(time.year, time.month, standard_day, calendar=calendar)


def preprocess_dataset(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Preprocess a dataset by standardizing time and applying CMIP6 preprocessing.

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): Dataset to preprocess.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Preprocessed dataset.
    """
    ds_copy = ds.copy()
    ds_copy = combined_preprocessing(ds_copy)  # CMIP6-specific preprocessing
    ds_copy = ds_copy.assign_coords(
        time=("time", standardize_time(ds_copy.time.values))
    )
    return ds_copy


# Step 2: Regrid Functions (Pure Functions)
def regrid_ensemble(
    ds: Union[xr.Dataset, xr.DataArray], method: Literal["1d", "2d"] = "1d"
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Regrid a dataset to a 1x1 degree grid.

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): Dataset to regrid.
        method (Literal["1d", "2d"], optional): Regridding method. Defaults to "1d".

    Returns:
        Union[xr.Dataset, xr.DataArray]: Regridded dataset.
    """
    if method == "2d":
        return regrid_2d_to_standard(regrid_2d(ds))
    else:
        return regrid_1d_to_standard(regrid_1d(ds))


# Step 3: Data Retrieval Function (Functional, Stateless)
def get_cmip6_ensemble_data(
    catalog_url: str,
    variable: str,
    institution_id: str,
    experiment_id: str,
    table_id: str = "Amon",
    start_year: str = "1958",
    end_year: str = "2014",
) -> xr.DataArray:
    """
    Retrieve CMIP6 ensemble data for a specific institution and experiment.

    Args:
        catalog_url (str): URL to the intake catalog.
        variable (str): Variable to retrieve (e.g., "ts").
        institution_id (str): Institution ID.
        experiment_id (str): Experiment ID (e.g., "historical").
        table_id (str, optional): Table ID. Defaults to "Amon".
        start_year (str, optional): Start year. Defaults to "1958".
        end_year (str, optional): End year. Defaults to "2014".

    Returns:
        xr.DataArray: Preprocessed ensemble data array.
    """
    cat = open_catalog(catalog_url)["climate"]["cmip6_gcs"]
    query = dict(
        variable_id=[variable],
        experiment_id=[experiment_id],
        table_id=[table_id],
        institution_id=[institution_id],
    )
    subset = cat.search(**query)
    z_kwargs = {"consolidated": True, "decode_times": True}

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        dset_dict = subset.to_dataset_dict(
            zarr_kwargs=z_kwargs, preprocess=preprocess_dataset
        )

    da_list = []
    for ds_key, ds in dset_dict.items():
        da = ds[variable].sel(time=slice(start_year, end_year))
        da_list.append(regrid_ensemble(da))

    return xr.concat(da_list, dim="member")


# Step 4: Object-Oriented Class for Managing High-Level Operations
class EnsembleProcessor:
    """Class to manage the high-level processing of CMIP6 ensemble data."""

    def __init__(
        self,
        var: str = "ts",
        past: str = "historical",
        future: str = DEFAULT_FUTURE_SCENARIO,
    ):
        """
        Initializes the ensemble processor.

        Args:
            var (str, optional): The variable to process. Defaults to "ts".
            past (str, optional): Past scenario. Defaults to "historical".
            future (str, optional): Future scenario. Defaults to "ssp585".
        """
        self.var = var
        self.past = past
        self.future = future
        self.success_list = ["NCAR", "CAMS", "NOAA-GFDL", "NASA-GISS"]  # Example

    def process_ensemble(self) -> None:
        """Process the ensemble data for both past and future scenarios."""
        # Retrieve past data
        past_data = [
            get_cmip6_ensemble_data(PANGEO_CAT_URL, self.var, instit, self.past)
            for instit in self.success_list
        ]
        future_data = [
            get_cmip6_ensemble_data(
                PANGEO_CAT_URL,
                self.var,
                instit,
                self.future,
                start_year="2015",
                end_year="2099",
            )
            for instit in self.success_list
        ]

        # Process each institution's data
        for past_da, future_da in zip(past_data, future_data):
            self._merge_and_save(past_da, future_da)

    def _merge_and_save(self, past_da: xr.DataArray, future_da: xr.DataArray) -> None:
        """Merge past and future data for an ensemble member and save it to disk."""
        merged_da = xr.concat([past_da, future_da], dim="time")
        # Save to NetCDF
        output_path = f"output/{self.var}_{self.past}_{self.future}.nc"
        merged_da.to_netcdf(output_path)
        print(f"Saved merged data to {output_path}")

    def compute_derivatives(self) -> None:
        """Compute derivatives such as trends and climatologies for the ensemble data."""
        da = self.load_ensemble_data()
        trend = get_trend(da)
        climatology = get_clim(da)
        # Save the computed trends and climatologies
        trend.to_netcdf(f"output/{self.var}_trend.nc")
        climatology.to_netcdf(f"output/{self.var}_climatology.nc")

    def load_ensemble_data(self) -> xr.DataArray:
        """Load previously processed ensemble data."""
        return xr.open_dataarray(f"output/{self.var}_{self.past}_{self.future}.nc")


# Step 5: Running the Workflow
if __name__ == "__main__":
    processor = EnsembleProcessor(var="ts", past="historical", future="ssp585")
    processor.process_ensemble()  # Process and save data
    processor.compute_derivatives()  # Compute trends, climatologies, etc.
