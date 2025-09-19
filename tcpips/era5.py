"""
This script downloads monthly ERA5 reanalysis data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

The script uses the `cdsapi` library to interact with the CDS API and download the data in NetCDF format. This requires an API key, which in my case I store at ~/.cdsapirc. You can get your own API key by signing up at the CDS website and following the instructions in the documentation.

The script is designed to download the data to calculate potential intensity (PI) and potential size (PS) for tropical cyclones. Geopotential height is also included so that we could calculate the height of the tropopause.
"""

import os
import uuid
import shutil
import cdsapi
from typing import List, Literal, Optional
import numpy as np
import tempfile
import subprocess
import xarray as xr
import netCDF4 as nc
from matplotlib import pyplot as plt
import statsmodels.api as sm
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from sithom.time import timeit
from sithom.xr import mon_increase
from uncertainties import ufloat, correlated_values, unumpy
from sithom.plot import plot_defaults, label_subplots, get_dim


CARTOPY_INSTALLED = True
try:
    import cartopy
    from cartopy import crs as ccrs
    from cartopy.feature import NaturalEarthFeature
    from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

    CARTOPY_DIR = os.getenv("CARTOPY_DIR")
    if CARTOPY_DIR is not None and CARTOPY_DIR != "":
        print(f"Using Cartopy with CARTOPY_DIR: {CARTOPY_DIR}")
        cartopy.config["data_dir"] = CARTOPY_DIR
        os.makedirs(CARTOPY_DIR, exist_ok=True)  # Ensure the directory exists
    else:
        print("CARTOPY_DIR not set. Using default Cartopy data directory.")
except ImportError:
    print("Cartopy is not installed. Some plotting functions will not work.")
    CARTOPY_INSTALLED = False

from w22.ps import parallelized_ps13_dask
from .constants import (
    ERA5_RAW_PATH,
    ERA5_PI_OG_PATH,
    ERA5_PS_OG_PATH,
    ERA5_REGRIDDED_PATH,
    ERA5_PS_PATH,
    CONFIG_PATH,
    ERA5_PI_PATH,
    ERA5_PRODUCTS_PATH,
    ERA5_FIGURE_PATH,
)
from .pi import calculate_pi
from .rh import relative_humidity_from_dew_point
from .dask_utils import dask_cluster_wrapper

DEFAULT_START_YEAR = 1980
DEFAULT_END_YEAR = 2024
LAT_BOUND = 40  # Degrees North / South for plotting


def plot_var_on_map(
    ax: plt.axis,
    da: xr.DataArray,
    label: str,
    cmap: str,
    shrink: float = 0.8,
    hatch_mask: Optional[xr.DataArray] = None,
    second_hatch_mask: Optional[xr.DataArray] = None,
    **kwargs,
) -> None:
    """Plot a variable on a geographic map.

    Args:
        ax (plt.axis): The axis to plot on.
        da (xr.DataArray): The data array to plot.
        label (str): The label for the colorbar.
        cmap (str): The colormap to use.
        shrink (float): The size of the colorbar.
    """
    if not CARTOPY_INSTALLED:
        return
    # add feature at back of plot

    ax.add_feature(
        NaturalEarthFeature("physical", "land", "110m"),
        edgecolor="black",
        facecolor="green",
        alpha=0.3,
        linewidth=0.5,
    )
    # add in LATITUDE_FORMATTER and LONGITUDE_FORMATTER
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xlocator = plt.MaxNLocator(4)
    gl.ylocator = plt.MaxNLocator(5)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    # only on bottom and left of plot
    gl.top_labels = False
    gl.right_labels = False
    # set the extent of the plot
    ax.set_extent(
        [-180, 180, -LAT_BOUND, LAT_BOUND], crs=ccrs.PlateCarree(central_longitude=180)
    )
    # plot the data
    da = da.where(da > 0, np.nan)
    da.plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cmap=cmap,
        cbar_kwargs={
            "label": "",
            # "label": label,
            "cmap": cmap,
            "shrink": shrink,
        },
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        add_labels=True,
        rasterized=True,
        **kwargs,
    )
    # add hatch mask if provided
    if hatch_mask is not None:
        if not isinstance(hatch_mask, xr.DataArray):
            raise ValueError("hatch_mask must be an xarray DataArray.")
        else:

            plt.rcParams["hatch.linewidth"] = 0.4  # Default is 1.0
            plt.rcParams["hatch.color"] = "gray"  # Default is black
            if np.all(hatch_mask.astype(int).values == 1):
                warning = "All values in the hatch mask are True (1)."
                print(warning)
            else:
                ones = np.sum(hatch_mask.astype(int).values == 1)
                zeros = np.sum(hatch_mask.astype(int).values == 0)
                print(
                    f"{ones/(ones + zeros) * 100:.2f}% of the area for {label} is masked out."
                )

            _ = hatch_mask.astype(int).plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="longitude",
                y="latitude",
                levels=[0, 0.5, 1],  # Define regions for 0s and 1s
                hatches=[None, "xxxx"],  # Apply hatch ONLY to the region of 1s
                colors="none",  # Makes the area transparent
                add_colorbar=False,
            )
    if second_hatch_mask is not None:
        if not isinstance(second_hatch_mask, xr.DataArray):
            raise ValueError("second_hatch_mask must be an xarray DataArray.")
        else:
            plt.rcParams["hatch.linewidth"] = 0.4
            plt.rcParams["hatch.color"] = "green"
            if np.all(second_hatch_mask.astype(int).values == 1):
                warning = "All values in the second hatch mask are True (1)."
                print(warning)
            else:
                ones = np.sum(second_hatch_mask.astype(int).values == 1)
                zeros = np.sum(second_hatch_mask.astype(int).values == 0)
                print(
                    f"{ones/(ones + zeros) * 100:.2f}% of the area for {label} is masked out."
                )
            _ = second_hatch_mask.astype(int).plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="longitude",
                y="latitude",
                levels=[0, 0.5, 1],  # Define regions for 0s and 1s
                hatches=[None, "++++"],  # Apply hatch ONLY to the region of 1s
                colors="none",  # Makes the area transparent
                add_colorbar=False,
            )

    ax.set_title(label)
    plot_defaults()


@timeit
def download_single_levels(
    years: List[str], months: List[str], output_file: str, redo: bool = False
) -> None:
    """
    Downloads monthly-averaged single-level fields in decadal chunks.
      - Sea surface temperature (for PI and PS)
      - Mean sea level pressure (for PI and PS)
      - Relative humidity (PS only)

    Args:
        years (List[str]): List of years to download data for.
        months (List[str]): List of months to download data for.
        output_file (str): The base name for the output files.
        redo (bool): If True, re-download data even if files exist.
    """
    print(f"Called for years: {years}, months: {months}, output_file: {output_file}")

    # If more than 10 years, split into chunks and call recursively
    if len(years) > 10:
        print(
            f"Request for {len(years)} years is too large. Splitting into decadal chunks."
        )
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            # Call the function for each smaller chunk of years
            download_single_levels(years[i:j], months, output_file, redo)
        return

    # Base case: Download a single chunk (10 years or less)
    chunk_file_name = f"{output_file.replace('.nc', '')}_years{years[0]}_{years[-1]}.nc"
    chunk_file_path = os.path.join(ERA5_RAW_PATH, chunk_file_name)

    if os.path.exists(chunk_file_path) and not redo:
        print(f"File {chunk_file_path} already exists. Skipping.")
        return
    else:
        print(f"Downloading chunk: {chunk_file_path} as it does not exist.")

    print(
        f"Downloading single-level data for years {years[0]}-{years[-1]} to {chunk_file_name}"
    )
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "format": "netcdf",
            "variable": [
                "sea_surface_temperature",
                "mean_sea_level_pressure",
                "2m_dewpoint_temperature",
                "2m_temperature",
            ],
            "year": years,
            "month": months,
            "time": "00:00",
        },
        chunk_file_path,
    )
    print(f"Downloaded single-level data to {chunk_file_name}")
    return


@timeit
def download_pressure_levels(
    years: List[str],
    months: List[str],
    output_file: str,
    redo: bool = False,
    stitch_together: bool = True,
) -> None:
    """
    Downloads monthly-averaged pressure-level fields:
      - Temperature (atmospheric profile) (PI and therefore PS)
      - Specific humidity (atmospheric profile) (PI and therefore PS)
      - Geopotential (atmospheric profile) (Neither, only for tropopause height)

    The pressure levels below are specified in hPa.
    Note: Geopotential is provided in m²/s². Divide by ~9.81 to convert to geopotential height (meters).

    The function handles the case where the number of years exceeds 10 by splitting the years into chunks of 10 years and then calling the function recursively, and finally stitching the temporary files together at the end.

    Current download speed from Archer2 login node is around 25 minutes per decade (6.4 GB).

    Args:
        years (List[str]): List of years to download data for.
        months (List[str]): List of months to download data for.
        output_file (str): Name of the output file to save the data.
        redo (bool): If True, download the data again even if the file already exists. Default is False.
        stitch_together (bool): If True, stitch the temporary files together at the end. Default is True.
    """
    if os.path.exists(os.path.join(ERA5_RAW_PATH, output_file)) and not redo:
        print(f"File {output_file} already exists. Use redo=True to download again.")
        return
    print(
        f"Downloading pressure-level data for years: {years} and months: {months}, to {output_file}"
    )
    if len(years) > 10:
        # call recursively to avoid problems with timeout
        # split the years into chunks of 10 years, give each unique name, and then
        # stitch the temporary files together at the end
        file_paths = []
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            file_name = (
                f"{output_file.replace('.nc', '')}_years{years[i]}_{years[j-1]}.nc"
            )
            file_path = os.path.join(ERA5_RAW_PATH, file_name)
            file_paths.append(file_path)

            # check if the temporary sfile already exists (this would indicate the script crashed partway through last time)
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping download.")
            else:
                download_pressure_levels(years[i:j], months, file_name)
        # stitch the files together
        output_path = os.path.join(ERA5_RAW_PATH, output_file)
        # append the files together along the time dimension and save to output_file
        # using xarray
        if stitch_together:
            print(f"Stitching files together: {file_paths} to {output_path}")
            ds = xr.open_mfdataset(
                file_paths,
                combine="nested",
                concat_dim="valid_time",
                parallel=True,
            )
            ds.to_netcdf(output_path)
            # remove the temporary files
            if os.path.exists(output_path):
                for file_path in file_paths:
                    os.remove(file_path)
            print(f"Downloaded pressure-level data to {output_path}")
        return
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "format": "netcdf",
            "variable": ["temperature", "specific_humidity", "geopotential"],
            "pressure_level": [
                "1000",
                "925",
                "850",
                "700",
                "600",
                "500",
                "400",
                "300",
                "250",
                "200",
                "150",
                "100",
                "70",
                "50",
                "30",
                "20",
                "10",
            ],
            "year": years,
            "month": months,
            "time": "00:00",
        },
        os.path.join(ERA5_RAW_PATH, output_file),
    )
    print(f"Downloaded pressure-level data to {output_file}")
    return


def download_era5_data(
    start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR
) -> None:
    """
    Downloads ERA5 data for the specified years and months.
    This function is a wrapper around the download_single_levels
    and download_pressure_levels functions.
    """
    # Specify the desired years and months.
    years = [
        str(year) for year in range(start_year, end_year + 1)
    ]  # Modify or extend this list as needed.
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # Download the single-level (surface) variables.
    download_single_levels(years, months, "era5_single_levels.nc", redo=False)

    # break the single level file into decade chunks
    # and save them as separate files

    # Download the pressure-level variables (including geopotential).
    download_pressure_levels(
        years, months, "era5_pressure_levels.nc", redo=False, stitch_together=False
    )


@timeit
def preprocess_single_level_data(ds: xr.Dataset) -> xr.Dataset:
    """Optimized preprocessing for single-level ERA5 data."""

    # Use assign for cleaner, dask-friendly operations and cast to float32
    updates = {
        "sst": (ds["sst"] - 273.15).astype(np.float32),
        "msl": (ds["msl"] / 100).astype(np.float32),
        "rh": relative_humidity_from_dew_point(ds["d2m"], ds["t2m"]).astype(np.float32),
    }
    ds = ds.assign(**updates)

    # Set attributes for new variables
    ds["sst"].attrs = {"long_name": "Sea Surface Temperature", "units": "Celsius"}
    ds["msl"].attrs = {"long_name": "Mean Sea Level Pressure", "units": "hPa"}
    ds["rh"].attrs = {"long_name": "Relative Humidity", "units": "fraction"}

    # Drop original variables at the end
    ds = ds.drop_vars(["d2m", "t2m"])

    # Rename time dimension if it exists
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


@timeit
def preprocess_pressure_level_data(ds: xr.Dataset) -> xr.Dataset:
    """Optimized preprocessing for pressure-level ERA5 data.

    Converts temperature to Celsius and specific humidity to g/kg."""

    updates = {
        "t": (ds["t"] - 273.15).astype(np.float32),
        "q": (ds["q"] * 1000).astype(np.float32),
    }
    ds = ds.assign(**updates)

    ds["t"].attrs = {"long_name": "Temperature", "units": "Celsius"}
    ds["q"].attrs = {"long_name": "Specific Humidity", "units": "g/kg"}

    # Drop geopotential
    if "z" in ds:
        ds = ds.drop_vars("z")

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


@timeit
def era5_pi_decade(single_level_path: str, pressure_level_path: str) -> None:
    """Optimized calculation of Potential Intensity for a decade.

    Args:
        single_level_path (str): Path to the single-level data file.
        pressure_level_path (str): Path to the pressure-level data file.
    """

    print(f"Calculating PI for {single_level_path} and {pressure_level_path}")
    cluster = LocalCluster()  # n_workers=10, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    # 1. Define a SINGLE, EXPLICIT chunking specification.
    # This ensures both datasets are chunked identically along shared dimensions.
    # Adjust values based on your available RAM.
    chunk_spec = {"time": 12, "latitude": 120, "longitude": 120}

    # 2. Open both datasets using the SAME chunks.
    single_ds = xr.open_dataset(single_level_path, chunks=chunk_spec)

    # For the pressure data, add the non-shared dimension chunking.
    pressure_chunk_spec = chunk_spec.copy()
    pressure_chunk_spec["pressure_level"] = -1  # Keep full vertical column in one chunk
    pressure_ds = xr.open_dataset(pressure_level_path, chunks=pressure_chunk_spec)

    # 3. Preprocess the data. This remains lazy.
    single_ds = preprocess_single_level_data(single_ds)
    pressure_ds = preprocess_pressure_level_data(pressure_ds)

    # 4. Merge. Because chunks are aligned, this is now a fast, metadata-only operation.
    combined_ds = xr.merge([single_ds, pressure_ds])

    # 5. Calculate Potential Intensity (this remains lazy).
    pi_ds = calculate_pi(combined_ds, dim="pressure_level")

    # 6. Save to NetCDF with an efficient engine and compression.
    # This is the step that triggers the computation.
    years_str = single_level_path.split("years")[1].replace(".nc", "")
    output_path = os.path.join(ERA5_PI_OG_PATH, f"era5_pi_years{years_str}.nc")

    # Define encoding for efficient writing
    encoding = {var: {"zlib": True, "complevel": 5} for var in pi_ds.variables}

    print(f"Saving results to {output_path}...")

    with ProgressBar():
        pi_ds.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
    print("...Done.")

    client.close()


def assign_coordinate_attributes(file_path: str):
    """Assigns CF-compliant 'coordinates' attributes to data variables.

    This function opens a NetCDF file in append mode and inspects each variable.
    For any variable that has both 'latitude' and 'longitude' as dimensions
    (and is not a coordinate variable itself), it sets the 'coordinates'
    attribute to 'longitude latitude'. This ensures compliance with CF
    conventions and improves interoperability with analysis tools.

    Args:
        file_path (str): The path to the NetCDF file to be modified.

    Doctests:
    >>> # Setup a dummy NetCDF file for the test
    >>> file_name = 'test_coord.nc'
    >>> with nc.Dataset(file_name, 'w') as ds:
    ...     _ =  ds.createDimension('latitude', 10)
    ...     _ = ds.createDimension('longitude', 20)
    ...     _ = ds.createDimension('time', 2)
    ...     _ = ds.createVariable('latitude', 'f4', ('latitude',))
    ...     _ = ds.createVariable('longitude', 'f4', ('longitude',))
    ...     _ = ds.createVariable('time', 'i4', ('time',))
    ...     _ = ds.createVariable('vmax', 'f8', ('time', 'latitude', 'longitude'))
    ...     _ = ds.createVariable('pmin', 'f8', ('time', 'latitude', 'longitude'))
    >>> # Run the function
    >>> assign_coordinate_attributes(file_name)
    Processing variable: vmax... Added 'coordinates' attribute.
    Processing variable: pmin... Added 'coordinates' attribute.
    >>> # Verify the attributes were set
    >>> with nc.Dataset(file_name, 'r') as ds:
    ...     print(ds.variables['vmax'].coordinates)
    ...     print(ds.variables['pmin'].coordinates)
    longitude latitude
    longitude latitude
    >>> # Clean up the test file
    >>> os.remove(file_name)
    """
    # Define the spatial coordinate variables as named in your file
    lon_name = "longitude"
    lat_name = "latitude"

    try:
        with nc.Dataset(file_path, "a") as ds:
            # Get the names of all coordinate variables
            coord_vars = list(ds.dimensions.keys())

            # Iterate over all variables in the file
            for var_name in ds.variables:
                # Skip coordinate variables themselves

                var_obj = ds.variables[var_name]
                # Check if the variable has the required dimensions
                if lon_name in var_obj.dimensions and lat_name in var_obj.dimensions:
                    print(f"Processing variable: {var_name}... ", end="")
                    # Assign the coordinates attribute
                    var_obj.setncattr("coordinates", f"{lon_name} {lat_name}")
                    print("Added 'coordinates' attribute.")

            ds[lon_name].setncattr("standard_name", "longitude")
            ds[lat_name].setncattr("standard_name", "latitude")
            ds[lon_name].setncattr("axis", "X")
            ds[lat_name].setncattr("axis", "Y")
            ds[lon_name].units = "degrees_east"
            ds[lat_name].units = "degrees_north"

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def fix_era5_pi_decade(output_path: str) -> None:
    """Currently the ERA5 PI data is saved in a format that cannot be read by cdo."""
    print(output_path)
    print("initial ncdump")
    os.system(f"ncdump -h '{output_path}'")
    ds = xr.open_dataset(
        output_path, engine="h5netcdf"
    )  # Load the dataset to ensure it's processed
    print(ds)

    coords_to_remove = ["number", "expver"]

    # Use reset_coords with drop=True to completely remove them from the dataset
    # This is the cleanest way to solve the CDO compatibility issue.
    ds_fixed = ds.reset_coords(coords_to_remove, drop=True)
    print(f"  - Removed coordinates: {coords_to_remove}")
    print(ds_fixed)

    new_output_path = output_path.replace(".nc", ".fix.nc")
    ds_fixed.to_netcdf(new_output_path, engine="netcdf4", format="NETCDF4")
    print()
    os.system(f"ncdump -h '{new_output_path}'")
    print("Now running netcdf4 to change coords.")
    # os.system(f"ncatted -O -a coordinates,.,d,, '{new_output_path}'")
    # print("Final ncdump after ncatted")
    assign_coordinate_attributes(new_output_path)
    os.system(f"ncdump -h '{new_output_path}'")


def era5_pi(years: List[str]) -> None:
    """
    Calculate potential intensity (PI) using the downloaded ERA5 data for a range of years. Go decade by decade.
    """
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        single_level_path = os.path.join(
            ERA5_RAW_PATH, f"era5_single_levels_years{years[i]}_{years[j-1]}.nc"
        )
        pressure_level_path = os.path.join(
            ERA5_RAW_PATH, f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc"
        )
        era5_pi_decade(single_level_path, pressure_level_path)


def select_seasonal_hemispheric_data(
    ds: xr.Dataset,
    months_to_average: int = 1,
    lon: str = "longitude",
    lat: str = "latitude",
) -> xr.Dataset:
    """
    Selects seasonal hemispheric data, with an option to average over peak months.

    Args:
        ds (xr.Dataset): An xarray Dataset with 'time', 'latitude', and
            'longitude' coordinates. 'time' must be a datetime-like coordinate.
        months_to_average (int): The number of months to average.
            - 1: Selects August for NH and February for SH.
            - 3: Averages over Aug-Sep-Oct for NH and Jan-Feb-Mar for SH.

    Returns:
        xr.Dataset: A new Dataset with an integer 'year' dimension, containing
        the combined seasonal and hemispheric data.

    >>> # --- Test Setup ---
    >>> # Create a 2-year dataset where the data value is the month number.
    >>> import pandas as pd
    >>> time = pd.date_range('2020-01-01', '2021-12-31', freq='MS')
    >>> lats = [-20, 20]  # One SH, one NH point
    >>> lons = [100]
    >>> month_data = time.month.values.reshape(-1, 1, 1) * np.ones((len(time), len(lats), len(lons)))
    >>> ds_test = xr.Dataset(
    ...     {'some_variable': (['time', 'latitude', 'longitude'], month_data)},
    ...     coords={'time': time, 'latitude': lats, 'longitude': lons}
    ... )

    >>> # --- Test Case 1: 3-Month Average ---
    >>> result_3m = select_seasonal_hemispheric_data(ds_test, months_to_average=3)
    >>> # For NH (lat=20), average of Jul(7), Aug(8), Sep(9) should be 8.0
    >>> assert int(result_3m.sel(year=2021, latitude=20).some_variable) == 8
    >>> # For SH (lat=-20), average of Jan(1), Feb(2), Mar(3) should be 2.0
    >>> assert int(result_3m.sel(year=2020, latitude=-20).some_variable) == 2
    >>> # Check that the dimensions are correct and 'time' is replaced by 'year'.
    >>> assert sorted(result_3m.dims) == ['latitude', 'longitude', 'year']
    >>> assert result_3m.year.values.tolist() == [2020, 2021]

    >>> # --- Test Case 2: 1-Month Selection (Original behavior) ---
    >>> result_1m = select_seasonal_hemispheric_data(ds_test, months_to_average=1)
    >>> # For NH (lat=20), the value for August (month 8) should be 8.
    >>> assert int(result_1m.sel(year=2021, latitude=20).some_variable) == 8
    >>> # For SH (lat=-20), the value for February (month 2) should be 2.
    >>> assert int(result_1m.sel(year=2020, latitude=-20).some_variable) == 2
    """
    if months_to_average == 1:
        # Original logic: Select a single peak month
        nh_data = ds.where((ds.time.dt.month == 8) & (ds[lat] >= 0), drop=True)
        sh_data = ds.where((ds.time.dt.month == 2) & (ds[lat] < 0), drop=True)

        nh_data = (
            nh_data.assign_coords(year=nh_data.time.dt.year)
            .swap_dims({"time": "year"})
            .drop_vars("time")
        )
        sh_data = (
            sh_data.assign_coords(year=sh_data.time.dt.year)
            .swap_dims({"time": "year"})
            .drop_vars("time")
        )

        return nh_data.combine_first(sh_data)

    elif months_to_average == 3:
        # New logic: Average over 3-month peak season
        # Northern Hemisphere: July-August-September (JAS)
        nh_data = ds.where(ds.time.dt.month.isin([7, 8, 9]) & (ds[lat] >= 0), drop=True)
        nh_annual = nh_data.groupby(nh_data.time.dt.year).mean("time")

        # Southern Hemisphere: January-February-February (JFM)
        sh_data = ds.where(ds.time.dt.month.isin([1, 2, 3]) & (ds[lat] < 0), drop=True)
        sh_annual = sh_data.groupby(sh_data.time.dt.year).mean("time")

        # Merge the two hemispheric, annually-averaged datasets
        return nh_annual.combine_first(sh_annual)

    else:
        raise ValueError("`months_to_average` must be 1 or 3.")


def get_era5_coordinates(
    start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR
) -> xr.Dataset:
    """
    Get the coordinates of the ERA5 data.

    Should have coordinates longitude, latitude, valid_time.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.

    Returns:
        xr.Dataset: The xarray dataset containing the coordinates.
    """
    sfp = os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc")
    if os.path.exists(sfp):
        # open the single level file
        return xr.open_dataset(sfp)[["longitude", "latitude", "valid_time"]]
    else:
        # open the pressure level files for all of the decades using xr.open_mfdataset
        file_paths = []
        years = [str(year) for year in range(start_year, end_year + 1)]
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            file_paths += [
                os.path.join(
                    ERA5_RAW_PATH,
                    f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc",
                )
            ]
        print(f"Opening pressure level files: {file_paths}")
        ds = xr.open_mfdataset(file_paths, chunks={"valid_time": 1}, engine="netcdf4")
        return ds[["longitude", "latitude", "valid_time"]]


# --- data access functions ---
# TODO: add an option to chunk for trends (times all one chunk) or chunk for spatial averages.

CHUNK_IN = {
    "space": {"longitude": 30, "latitude": 30, "valid_time": -1},
    "time": {"valid_time": 1},
}


def get_era5_combined(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    chunk_in: Literal["space", "time"] = "space",
) -> xr.Dataset:
    """
    Get the raw ERA5 data for potential intensity and size as a lazily loaded xarray dataset.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.
        chunk_in (Literal["space", "time"]): The chunking strategy to use.

    Returns:
        xr.Dataset: The xarray dataset containing the combined single-level and pressure-level data.
    """
    # open the single level file
    if os.path.exists(os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc")):
        single_ds = preprocess_single_level_data(
            xr.open_dataset(
                os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc"),
                chunks=CHUNK_IN[chunk_in],
                engine="netcdf4",
            )
        ).sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    else:
        fp = []
        years = [str(year) for year in range(start_year, end_year + 1)]
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            fp += [
                os.path.join(
                    ERA5_RAW_PATH, f"era5_single_levels_years{years[i]}_{years[j-1]}.nc"
                )
            ]
        single_ds = preprocess_single_level_data(
            xr.open_mfdataset(fp, chunks=CHUNK_IN[chunk_in], engine="netcdf4")
        )
    # open the pressure level files for all of the decades using xr.open_mfdataset
    file_paths = []
    years = [str(year) for year in range(start_year, end_year + 1)]
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        file_paths += [
            os.path.join(
                ERA5_RAW_PATH, f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc"
            )
        ]
    print(f"Opening pressure level files: {file_paths}")
    pressure_ds = preprocess_pressure_level_data(
        xr.open_mfdataset(file_paths, chunks=CHUNK_IN[chunk_in], engine="netcdf4")
    )
    # merge the datasets
    return xr.merge([single_ds, pressure_ds])


def get_era5_pi(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    chunk_in: Literal["space", "time"] = "space",
) -> xr.Dataset:
    """Load the potential intensity (PI) data calculated from ERA5 data.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.
        chunk_in (Literal["space", "time"]): The chunking strategy to use.

    Returns:
        xr.Dataset: The xarray dataset containing the PI data.
    """
    file_paths = []
    years = [str(year) for year in range(start_year, end_year + 1)]
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        file_paths += [
            os.path.join(ERA5_PI_OG_PATH, f"era5_pi_years{years[i]}_{years[j-1]}.nc")
        ]

    print(f"Opening PI files: {file_paths}")
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Please run era5_pi() first.")
            return
        else:
            print(f"Found file: {file_path}")
            # print(xr.open_dataset(file_path, engine="h5netcdf"))
    return xr.open_mfdataset(
        file_paths,
        combine="nested",
        concat_dim="time",
        chunks=CHUNK_IN[chunk_in],
        engine="h5netcdf",
    )


@timeit
def get_all_data(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    chunk_in: Literal["space", "time"] = "space",
) -> xr.Dataset:
    """Get both the derived data from PI calculation, and the processed era5 data.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.
        chunk_in (Literal["space", "time"]): The chunking strategy to use.

    Returns:
        xr.Dataset: The xarray dataset containing both the ERA5 data and the PI data.
    """
    era5_ds = get_era5_combined(
        start_year=start_year, end_year=end_year, chunk_in=chunk_in
    )
    if "valid_time" in era5_ds.dims:
        era5_ds = era5_ds.rename({"valid_time": "time"})
    # print(era5_ds.sst.values.sel(longitude))
    print(era5_ds)
    era5_pi = get_era5_pi(start_year=start_year, end_year=end_year, chunk_in=chunk_in)
    print(era5_pi)
    return xr.merge([era5_ds, era5_pi], compat="override", join="override")


def trend_with_neweywest_full(
    # x: np.darray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Calculates the linear trend (slope, intercept) and its p-value
    using OLS with Newey-West standard errors.

    Args:
        # x (np.ndarray): A 1D array representing the time variable (e.g., years).
        y (np.ndarray): A 1D array representing the time series.

    Returns:
        tuple[float, float, float]: A tuple with (slope, intercept, p-value).

    Doctests:
    >>> import numpy as np
    >>> # Test with a simple increasing trend
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> slope, intercept, p_value = trend_with_neweywest_full(y)
    >>> round(slope, 2)
    1.0
    >>> round(intercept, 2)
    1.0
    >>> round(p_value, 4) < 0.05
    True
    """
    # print("Calculating trend with Newey-West standard errors...")
    if np.all(np.isnan(y)):
        return np.nan, np.nan, np.nan

    x = np.arange(len(y))
    x_const = sm.add_constant(x)

    model = sm.OLS(y, x_const, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    # model.params is [intercept, slope]
    # model.pvalues is [p_value_intercept, p_value_slope]
    intercept = model.params[0]
    slope = model.params[1]
    p_value = model.pvalues[1]

    return slope, intercept, p_value


# @dask_cluster_wrapper
def era5_pi_trends(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 1,
) -> None:
    """
    Let's find the linear trends in the potential intensity

    Args:
        start_year (int, optional): The start year for the analysis. Defaults to DEFAULT_START
        end_year (int, optional): The end year for the analysis. Defaults to DEFAULT_END_YEAR.
        months (int, optional): The number of months to average over. Defaults to 1.
            - 1: Selects August for NH and February for SH.
            - 3: Averages over Aug-Sep-Oct for NH and Jan-Feb-Mar for SH.
    """
    # load all the decades of data for potential intensity
    pi_vars = ["vmax", "t0", "otl", "sst", "msl", "rh"]  # [x for x in ds.variables]

    ds = get_all_data(start_year=start_year, end_year=end_year, chunk_in="space")[
        pi_vars
    ]
    if ds is None:
        print("No ERA5 PI data found. Please run era5_pi() first.")
        return

    # print(ds)
    # sst_ds = preprocess_single_level_data(
    #     get_era5_combined(start_year=start_year, end_year=end_year)
    # )["sst"]
    # print("sst_ds", sst_ds)
    # # ds["sst"] = sst_ds["sst"]
    # # lets just select the augusts in the northern hemisphere
    # ds["sst"] = sst_ds
    ds = select_seasonal_hemispheric_data(mon_increase(ds), months_to_average=months)
    # ds = mon_increase(ds.sel(time=ds.time.dt.month == 8))  # .sel(latitude=slice(0, 30))
    print("Calculating trends in potential intensity...")
    print(ds)
    # calculate the linear trends in potential intensity
    # let's replace the time axis with the year
    new_year = ds.year - ds.year.min()
    assert len(new_year) > 20, "Not enough time points to calculate trends."
    ds = ds.assign_coords(year=new_year)
    ds = ds.chunk({"year": -1})  # all time in one chunk
    print("new rechunked ds", ds)

    print("New time coordinates:", ds.year.values)
    print("Calculating trends in potential intensity...")

    # pi_vars = ["vmax", "t0", "otl", "sst"]
    trend_ds = ds[pi_vars].polyfit(dim="year", deg=1, cov=True)
    # let's go through and work out the hatch mask for each variable

    for var in pi_vars:
        cov_n = var + "_polyfit_covariance"
        trend_n = var + "_polyfit_coefficients"
        if var + "_polyfit_covariance" in trend_ds:
            frac_error = np.abs(
                np.sqrt(
                    trend_ds[cov_n].isel(cov_i=0, cov_j=0)
                )  # degree=1 is in the first position
                / trend_ds[trend_n].sel(degree=1)
            )
            trend_ds[var + "_hatch_mask"] = frac_error >= 1.0
            if np.all(trend_ds[var + "_hatch_mask"].astype(int).values == 1):
                print(f"Warning: All values in the hatch mask for {var} are True (1).")
        # second hatch mask from running significance test on the rise
        results = xr.apply_ufunc(
            trend_with_neweywest_full,
            ds[var],
            input_core_dims=[["year"]],
            # Add a third empty list for the new intercept output
            output_core_dims=[[], [], []],
            exclude_dims=set(("year",)),
            vectorize=True,
            dask="parallelized",
            # rechunk=True,
            dask_gufunc_kwargs={"allow_rechunk": True},
            # allow_rechunk=True,
            output_dtypes=[float, float, float],
        )

        slope, intercept, p_value = results

        # Create the final Dataset
        new_trend_ds = xr.Dataset(
            {"slope": slope, "intercept": intercept, "p_value": p_value}
        )
        trend_ds[var + "_p_value"] = new_trend_ds["p_value"]
        trend_ds[var + "_hatch_mask_p"] = new_trend_ds["p_value"] >= 0.05
        trend_ds[var + "_slope_nw"] = new_trend_ds["slope"]
        trend_ds[var + "_intercept_nw"] = new_trend_ds["intercept"]

    print("Saving to zarr...")
    temp_path = tempfile.mkdtemp(
        suffix=".zarr", dir="."
    )  # Write temp file in current dir
    out_path = os.path.join(
        ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.zarr"
    )

    try:
        with ProgressBar():
            trend_ds.to_zarr(temp_path, mode="w", consolidated=True)

        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        os.rename(temp_path, out_path)
        print(f"Successfully saved to {out_path}")

    except Exception as e:
        print(f"Failed to save data. Error: {e}")
        shutil.rmtree(temp_path)  # Clean up on failure
        raise


def find_tropical_m(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 1,
) -> None:
    """I want to find what the trends are for the tropical cyclone potential intensity and size over the pseudo observational period.

    I defined m as the ratio of the change in the outflow temperature to the change in surface temperature.
    T_s = T_s0 + delta_t
    T_o = T_o0 + m * delta_t
    where T_s0 is the initial sea surface temperature, T_o0 is the initial outflow temperature, and delta_t is the change in temperature.

    Args:
        start_year (int, optional): The start year for the analysis. Defaults to DEFAULT_START
        end_year (int, optional): The end year for the analysis. Defaults to DEFAULT_END_YEAR.
        months (int, optional): The number of months to average over. Defaults to 1.

    Returns:
        None: This function saves the results to a file and does not return anything.
    """
    trend_ds = mon_increase(
        xr.open_dataset(
            os.path.join(
                ERA5_PRODUCTS_PATH,
                f"era5_pi_trends_{start_year}_{end_year}_{months}m.nc",
            ),
            engine="h5netcdf",
        )
    ).sel(latitude=slice(-40, 40))

    trend_ds["m"] = (
        ["latitude", "longitude"],
        trend_ds["t0_polyfit_coefficients"].sel(degree=1).values
        / trend_ds["sst_polyfit_coefficients"].sel(degree=1).values,
        {"long_name": "Temp Change Ratio", "units": "dimensionless"},
    )
    print(trend_ds)

    plot_defaults()
    fig, axs = plt.subplots(
        3,
        1,
        sharex=True,
        sharey=True,
        figsize=get_dim(ratio=1.0),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    plot_var_on_map(
        axs[0],
        trend_ds["m"],
        label="$\Delta T_o / \Delta T_s$",
        cmap="cmo.balance",
        vmin=-2,
        vmax=2,
        hatch_mask=trend_ds["t0_hatch_mask"],  # or trend_ds["sst_hatch_mask"],
    )
    plot_var_on_map(
        axs[1],
        trend_ds["t0_polyfit_coefficients"].sel(degree=1) * 10,
        label="$T_o$ trend [K decade$^{-1}$]",
        cmap="cmo.balance",
        vmin=-0.5,
        vmax=0.5,
        hatch_mask=trend_ds["t0_hatch_mask"],
    )
    plot_var_on_map(
        axs[2],
        trend_ds["sst_polyfit_coefficients"].sel(degree=1) * 10,
        label="$T_s$ trend [K/decade]",
        cmap="cmo.balance",
        vmin=-0.5,
        vmax=0.5,
        hatch_mask=trend_ds["sst_hatch_mask"],
    )
    label_subplots(axs)  # , override="outside")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            ERA5_FIGURE_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.pdf"
        ),
        dpi=300,
    )

    print(trend_ds["sst_polyfit_coefficients"].sel(degree=0).mean())
    print(trend_ds["sst_polyfit_coefficients"].sel(degree=1).mean())
    print(trend_ds["t0_polyfit_coefficients"].sel(degree=0).mean())
    print(trend_ds["t0_polyfit_coefficients"].sel(degree=1).mean())
    print(trend_ds["m"].mean())
    print(trend_ds["m"].median())


def plot_trend_maps(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 1,
) -> None:
    """Plot trend in vmax, t0 and otl in another 3 panel subplot with no cbar label but
    labels in subplot title instead.

    Args:
        start_year (int, optional): The start year for the analysis. Defaults to DEFAULT_START
        end_year (int, optional): The end year for the analysis. Defaults to DEFAULT_END_YEAR.
        months (int, optional): The number of months to average over. Defaults to 3.
    """
    trend_ds = xr.open_dataset(
        os.path.join(
            ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.zarr"
        ),
    )
    print(trend_ds)
    trend_ds = mon_increase(trend_ds).sel(latitude=slice(-40, 40))
    vars = ["sst", "t0", "otl", "vmax"]  # "msl", "rh",
    vlim = [(-0.5, 0.5), (-0.5, 0.5), (-1, 1), (-5, 5)]  # (-1, 1), (-0.1, 0.1),
    labels = [
        # "$P_s$ trend [hPa decade$^{-1}$]",
        # r"$\mathcal{H}$ trend [% decade$^{-1}$]",
        "$T_s$ trend [K decade$^{-1}$]",
        "$T_o$ trend [K decade$^{-1}$]",
        "$z_{\mathrm{o}}$ trend [hPa decade$^{-1}$]",
        "$V_p$ trend [m s$^{-1}$ decade$^{-1}$]",
    ]
    trends = [trend_ds[var + "_polyfit_coefficients"].sel(degree=1) for var in vars]
    hatch_masks = [trend_ds[var + "_hatch_mask"] for var in vars]

    plot_defaults()
    fig, axs = plt.subplots(
        len(vars),
        1,
        sharex=True,
        sharey=True,
        figsize=get_dim(ratio=1.0),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )

    for i, var in enumerate(vars):
        print(f"Plotting {var}...")
        plot_var_on_map(
            axs[i],
            trends[i] * 10,  # convert to per decade
            label=labels[i],
            cmap="cmo.balance",
            hatch_mask=hatch_masks[i],
            second_hatch_mask=trend_ds[var + "_hatch_mask_p"],
            vmin=vlim[i][0],
            vmax=vlim[i][1],
        )

    label_subplots(axs)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            ERA5_FIGURE_PATH, f"era5_vmax_trends_{start_year}_{end_year}_{months}m.pdf"
        ),
        dpi=300,
    )


def plot_trend_lineplots(
    lon: float,
    lat: float,
    label: str = "new_orleans",
    ds: Optional[xr.Dataset] = None,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 1,
) -> None:
    """Plot the linear trend and variables for a specific point
    in the ERA5 dataset.

    Args:
        lon (float): Longitude of the point to plot.
        lat (float): Latitude of the point to plot.
        label (str, optional): Label for the plot. Defaults to "new_orleans".
        ds (Optional[xr.Dataset], optional): If provided, use this dataset instead of loading it.
        start_year (int, optional): Start year for the data. Defaults to 1980.
        end_year (int, optional): End year for the data. Defaults to 2020
        months (int, optional): Number of months to average over. Defaults to 1.
    """
    plot_defaults()
    trend_ds = xr.open_dataset(
        os.path.join(
            ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.zarr"
        ),
    )
    if ds is None:
        era5_ds = get_all_data(start_year=start_year, end_year=end_year)
    else:
        era5_ds = ds
    era5_ds = mon_increase(era5_ds).sel(latitude=lat, longitude=lon, method="nearest")
    trend_ds = mon_increase(trend_ds).sel(latitude=lat, longitude=lon, method="nearest")
    print("era5_ds", era5_ds.sst.values)

    print("era5_ds", era5_ds)
    print("trend_ds", trend_ds)
    if lat > 0:
        # get August data (in the Northern Hemisphere)
        era5_ds = era5_ds.sel(time=era5_ds.time.dt.month == 8)
    else:
        # get February data (in the Southern Hemisphere)
        era5_ds = era5_ds.sel(time=era5_ds.time.dt.month == 3)
    # get the time in years
    era5_ds = era5_ds.assign_coords(
        year=era5_ds.time.dt.year - era5_ds.time.dt.year.min()
    )
    print("era5_ds", era5_ds)
    del era5_ds["time"]
    # plot the variables

    vars = ["sst", "t0", "otl", "vmax"]  # "msl", "rh",
    labels = [
        # r"$P_s$ [hPa]",
        # r"$\mathcal{H}$ [%]",
        r"$T_s$ [$^{\circ}$C]",
        r"$T_o$ [K]",
        r"$z_{\mathrm{o}}$ [hPa]",
        r"$V_p$ [m s$^{-1}$]",
    ]
    units = [r"$^{\circ}$C", "K", "hPa", r"m s$^{-1}$"]  # "hPa", "%",
    colors = [
        # "tab:purple",
        # "tab:brown",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    fig, axs = plt.subplots(
        len(vars),
        1,
        sharex=True,
        figsize=get_dim(ratio=0.8),
    )

    def format_p_latex(value: float, sig_figs: int = 2) -> str:
        """Formats a float into a LaTeX scientific notation string.

        Args:
            value: The floating-point number to format.
            sig_figs: The number of significant figures to display.

        Returns:
            A LaTeX-formatted string for the p-value.

        Doctests:
        >>> format_p_latex(0.00012345)
        '$p=1.2 \\\\times 10^{-4}$'
        >>> format_p_latex(12345.0)
        '$p=1.2 \\\\times 10^{4}$'
        >>> format_p_latex(0.051)
        '$p=5.1 \\\\times 10^{-2}$'
        """
        precision = sig_figs - 1
        sci_notation_str = f"{value:.{precision}e}"
        mantissa, exponent_str = sci_notation_str.split("e")
        exponent = int(exponent_str)  # Cleans up leading zeros and '+' sign

        # Don't use scientific notation for exponents of 0 or -1
        if exponent in [0, -1]:
            return f"$p={value:.{sig_figs}f}$"

        return f"$p={mantissa} \\times 10^{{{exponent}}}$"

    def plot_trend(ax: plt.axis, var: str, color: str, label: str, unit: str) -> None:
        """Plot the trend for a specific variable."""
        # ys = era5_ds[var].values
        x = era5_ds["year"].values
        slope = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=1).values
        intercept = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=0).values
        trend = slope * era5_ds["year"].values + intercept
        cov_matrix_unscaled = trend_ds[f"{var}_polyfit_covariance"].values
        coeffs = np.array([slope, intercept])
        cov_matrix = cov_matrix_unscaled

        print(f"{var} Fit Coefficients (slope, intercept): {coeffs}")
        print(f"{var} Covariance Matrix:\n{cov_matrix}\n")

        # 3. Use correlated_values to create ufloat objects
        # This function takes the nominal values and the full covariance matrix.
        u_coeffs = correlated_values(coeffs, cov_matrix)
        slope, intercept = u_coeffs

        print(f"{var} Correlated Slope (m):       {slope}")
        print(f"{var} Correlated Intercept (c):   {intercept}")
        print("-" * 30)

        ax.plot(
            era5_ds["year"].values + start_year,
            trend,
            label=f"{label} Trend",
            color=color,
            linestyle="--",
        )
        # new_ys = unumpy.polyval(np.array(u_coeffs), x)
        perc_inc = slope / intercept * 100  # percentage increase

        new_ys = u_coeffs[0] * x + u_coeffs[1]
        ax.fill_between(
            x + start_year,
            [y.n - y.s for y in new_ys],
            [y.n + y.s for y in new_ys],
            # new_ys.n - new_ys.s,
            color=color,
            alpha=0.2,
        )
        ax.set_title(
            f"{label} Trend: {slope.n*10:.2f} ± {slope.s*10:.2f}  {unit} decade{r'$^{-1}$'}, {format_p_latex(trend_ds[f'{var}_p_value'].values)}",  # , {perc_inc.n*10:.1f} ± {perc_inc.s*10:.1f} % decade{r'$^{-1}$'}",
            fontsize=10,
            color=color,
        )

    for i, ax in enumerate(axs):
        ax.plot(
            era5_ds["year"].values + start_year,
            era5_ds[vars[i]].values,
            label=labels[i],
            color=colors[i],
        )
        plot_trend(ax, vars[i], colors[i], labels[i], units[i])
        ax.set_ylabel(labels[i])

    axs[-1].set_xlabel("Year")
    plt.xlim(0 + start_year, end_year)
    label_subplots(axs)

    plt.savefig(
        os.path.join(
            ERA5_FIGURE_PATH, f"era5_pi_{label}_{start_year}_{end_year}_{months}m.pdf"
        ),
        dpi=300,
    )


def rechunk_file(
    file_name: str,
    new_chunks: dict = {
        "year": -1,  # Keep all years in one chunk
        "latitude": 321,  # Keep all latitudes in one chunk
        "longitude": 720,  # Split longitude into 2 chunks (1440/720=2)
    },
) -> None:
    """Rechunk a zarr file to have larger chunks for latitude and longitude.

    Args:
        file_name (str): The path to the zarr file to rechunk.
        new_chunks (dict, optional): The new chunk sizes. Defaults to {"year": -1, "latitude": 321, "longitude": 720}.
    """
    ds = xr.open_zarr(file_name, chunks="auto")
    print("Original chunks:", ds.chunks)
    ds = ds.chunk(new_chunks)
    print("New chunks:", ds.chunks)
    temp_path = file_name + "." + str(uuid.uuid4().hex) + ".tmp"
    print(f"Rechunking and saving to temporary file {temp_path}...")
    try:
        with ProgressBar():
            ds.to_zarr(temp_path, mode="w", consolidated=True)

        if os.path.exists(file_name):
            shutil.rmtree(file_name)

        os.rename(temp_path, file_name + ".new")
        print(f"Successfully rechunked and saved to {file_name}")

    except Exception as e:
        print(f"Failed to rechunk data. Error: {e}")
        shutil.rmtree(temp_path)  # Clean up on failure
        raise


def calculate_potential_sizes(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    dry_run: bool = False,
) -> None:
    """Calculate the potential sizes of tropical cyclones from ERA5 data.

    This is going to be a very expensive function to run.

    Args:
        start_year (int, optional): The start year for the analysis. Defaults to DEFAULT_START
        end_year (int, optional): The end year for the analysis. Defaults to DEFAULT_END_YEAR.
        dry_run (bool, optional): If True, do not actually run the calculation. Defaults to True.

    Returns:
        None: This function saves the results to a zarr file.
    """
    ds = get_all_data(start_year=start_year, end_year=end_year)
    if ds is None:
        print("No ERA5 data found. Please run era5_pi() first.")
        return

    # just select -40 to 40 latitude, August for NH and February for SH
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds = mon_increase(select_seasonal_hemispheric_data(ds, months_to_average=1)).sel(
        latitude=slice(-40, 40)
    )
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # currently I'm assuming that the data does not need to be converted
    # delete the unnecessary volume variables
    del ds["t"]
    del ds["q"]
    del ds["pressure_level"]
    # ds = convert(ds)
    ds = ds.rename({"vmax": "vmax_3"})
    ds["vmax_1"] = (ds.vmax_3.dims, np.ones(ds.vmax_3.shape) * 33 / 0.8)
    ds = ds.chunk({"lat": 5, "lon": 5, "year": 1})

    print("input ds", ds)
    output_file = os.path.join(
        ERA5_PS_OG_PATH, f"era5_potential_sizes_{start_year}_{end_year}.zarr"
    )
    tmp_file = output_file + "." + str(uuid.uuid4().hex) + ".tmp"
    if not dry_run:
        ds = parallelized_ps13_dask(
            ds  # .chunk({"time": 64, "lat": 480, "lon": 480})
        )  # .chunk(
        # {"time": 64, "lat": 480, "lon": 480}
        # )  # chunk the data for saving
        print("output ds", ds)

        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        ds.to_zarr(
            tmp_file,
            mode="w",
            consolidated=True,
        )
        print(f"Successfully saved to temporary file {tmp_file}")
        if os.path.exists(
            output_file
        ):  # get rid of old file (but only if the new one worked)
            print(f"Removing old file {output_file}...")
            shutil.rmtree(output_file)
            print(f"Removed old file {output_file}.")
        print(f"Renaming {tmp_file} to {output_file}...")
        os.rename(tmp_file, output_file)
        print(f"Successfully saved to {output_file}")

        rechunk_file(
            output_file,
            new_chunks={
                "year": -1,  # Keep all years in one chunk
                "latitude": 321,  # Keep all latitudes in one chunk
                "longitude": 720,  # Split longitude into 2 chunks (1440/720=2
            },
        )
    else:
        print("Dry run, not actually calculating potential sizes.")
        print(f"Would have saved to {output_file}")


@timeit
def call_cdo(input_path: str, output_path: str) -> None:
    """Call CDO to regrid the input netCDF file to a half-degree grid using multiple CPU cores.

    Includes ncdump calls for debugging.

    Args:
        input_path (str): Path to the input netCDF file.
        output_path (str): Path to the output netCDF file.
    """
    print(f"Regridding {input_path} to {output_path} using CDO...")

    # --- Define the number of threads to use ---
    # You can set this to a specific number or use os.cpu_count() to use all available cores.
    # On an M1 Pro with 10 cores, using 8 or 10 is a good starting point.
    num_threads = os.cpu_count() or 8  # Fallback to 8 if cpu_count() returns None

    # Temporary file for the intermediate step
    tmp_path = output_path + ".tmp"

    # --- Debug: Show header of the original input file ---
    print(f"\n--- Header for original input file: {input_path} ---")
    subprocess.run(["ncdump", "-h", input_path])
    print("--- End Header ---")

    # --- Step 1: Use ncks to remove unnecessary variables ---
    # This step helps to reduce the file size before regridding.
    ncks_options = [
        "-C",
        "-O",
        "-4",
    ]  # -C: exclude coordinate variables, -O: overwrite, -4: netCDF-4
    ncks_vars_to_exclude = (
        "lon_verticies,lat_bounds,lon_bounds,lat_verticies,expver,ifl"
    )
    ncks_cmd = (
        ["ncks"]
        + ncks_options
        + [
            "-x",
            "-v",
            ncks_vars_to_exclude,
            "--thr",
            str(num_threads),
            input_path,
            tmp_path,
        ]
    )

    print(f"\nRunning ncks to create temporary file: {' '.join(ncks_cmd)}")
    # Using subprocess.run is generally safer and more flexible than os.system
    ncks_result = subprocess.run(ncks_cmd, capture_output=True, text=True)

    if ncks_result.returncode != 0:
        print(f"Error: ncks failed for {input_path}")
        print(f"ncks stderr: {ncks_result.stderr}")
        return

    print(f"Temporary file created: {tmp_path}")

    # --- Debug: Show header of the temporary file ---
    if os.path.exists(tmp_path):
        print(f"\n--- Header for temporary file: {tmp_path} ---")
        subprocess.run(["ncdump", "-h", tmp_path])
        print("--- End Header ---")

    # --- Step 2: Run the parallelized cdo command ---
    grid_file = os.path.join(CONFIG_PATH, "halfdeg.txt")

    # Added the -P flag to specify the number of parallel threads.
    cdo_cmd = [
        "cdo",
        "-P",
        str(num_threads),  # Enable parallel processing with specified threads
        "-f",
        "nc4",  # Set output format to netCDF-4
        "-s",  # Silent mode
        f"remapbil,{grid_file}",  # The regridding operation
        tmp_path,  # Input file
        output_path,  # Output file
    ]

    print(f"\nRunning parallel CDO command: {' '.join(cdo_cmd)}")
    cdo_result = subprocess.run(cdo_cmd, capture_output=True, text=True)

    if cdo_result.returncode != 0:
        print(f"Error: CDO command failed for {input_path}")
        print(f"CDO stderr: {cdo_result.stderr}")
    else:
        print("CDO command completed successfully.")

    # --- Step 3: Clean up the temporary file ---
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"\nRemoved temporary file: {tmp_path}")
    except OSError as e:
        print(f"Error removing temporary file: {e}")

    # --- Final check ---
    if not os.path.exists(output_path):
        print(f"Error: Final output file was not created: {output_path}")
    else:
        print(f"Successfully created: {output_path}")


def regrid_all(years: List[str]) -> None:
    """Regrid all the ERA5 data for the given years.
    Args:
        years (List[str]): List of years to regrid.
    """
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))

        single_level_path = os.path.join(
            ERA5_RAW_PATH, f"era5_single_levels_years{years[i]}_{years[j-1]}.nc"
        )
        output_single_level_path = os.path.join(
            ERA5_REGRIDDED_PATH,
            f"era5_single_levels_years{years[i]}_{years[j-1]}.nc",
        )
        if os.path.exists(single_level_path) and not os.path.exists(
            output_single_level_path
        ):
            print(f"Regridding single level data for years {years[i]}-{years[j-1]}")
            call_cdo(single_level_path, output_single_level_path)

        pressure_level_path = os.path.join(
            ERA5_RAW_PATH, f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc"
        )
        output_pressure_level_path = os.path.join(
            ERA5_REGRIDDED_PATH,
            f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc",
        )
        if os.path.exists(pressure_level_path) and not os.path.exists(
            output_pressure_level_path
        ):
            print(f"Regridding pressure level data for years {years[i]}-{years[j-1]}")
            call_cdo(pressure_level_path, output_pressure_level_path)
        pi_path = os.path.join(
            ERA5_PI_OG_PATH, f"era5_pi_years{years[i]}_{years[j-1]}.nc"
        )
        fix_era5_pi_decade(pi_path)
        pi_path = pi_path.replace(".nc", ".fix.nc")
        output_pi_path = os.path.join(
            ERA5_PI_PATH, f"era5_pi_years{years[i]}_{years[j-1]}.nc"
        )
        if os.path.exists(pi_path) and not os.path.exists(output_pi_path):
            print(f"Regridding PI data for years {years[i]}-{years[j-1]}")
            call_cdo(pi_path, output_pi_path)
        ps_path = os.path.join(
            ERA5_PS_OG_PATH, f"era5_potential_sizes_{years[i]}_{years[j-1]}.zarr"
        )
        output_ps_path = os.path.join(
            ERA5_PS_PATH, f"era5_potential_sizes_{years[i]}_{years[j-1]}.zarr"
        )
        if os.path.exists(ps_path) and not os.path.exists(output_ps_path):
            print(f"Regridding potential sizes data for years {years[i]}-{years[j-1]}")
            call_cdo(ps_path, output_ps_path)


def get_all_regridded_data(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    chunk_in: Literal["space", "time"] = "space",
) -> xr.Dataset:
    """Get all regridded ERA5 data for the specified years.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.
        chunk_in (Literal["space", "time"]): The chunking strategy to use.

    Returns:
        xr.Dataset: The xarray dataset containing the regridded ERA5 data.
    """
    single_ds = preprocess_single_level_data(
        xr.open_mfdataset(
            os.path.join(
                ERA5_REGRIDDED_PATH, "era5_single*.nc"
            ),  # "era5_single_levels_years2*.nc"),
            combine="by_coords",
            chunks=CHUNK_IN[chunk_in],
            engine="netcdf4",
        )
    )
    pressure_ds = preprocess_pressure_level_data(
        xr.open_mfdataset(
            os.path.join(
                ERA5_REGRIDDED_PATH, "era5_pressure*.nc"
            ),  # _levels_years2*.nc"),
            combine="by_coords",
            chunks=CHUNK_IN[chunk_in],
            engine="netcdf4",
        )
    )
    pi_ds = xr.open_mfdataset(
        os.path.join(ERA5_PI_PATH, "era5_pi_years*.nc"),
        combine="by_coords",
        chunks=CHUNK_IN[chunk_in],
        engine="netcdf4",
    )

    return xr.merge(
        [single_ds, pressure_ds, pi_ds],
        compat="override",
        join="override",
    ).sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def plot_trends():
    from w22.constants import OFFSET_D

    new_point_d = {
        var: (
            OFFSET_D[var]["point"].lon + OFFSET_D[var]["lon_offset"],
            OFFSET_D[var]["point"].lat + OFFSET_D[var]["lat_offset"],
        )
        for var in OFFSET_D
    }

    for point in new_point_d:
        print(f"Plotting trend lineplots for {point} at {new_point_d[point]}")
        plot_trend_lineplots(
            (new_point_d[point][0] + 360) % 360,  # ensure in 0-360 range
            new_point_d[point][1],
            label=point,
            start_year=1980,
        )


if __name__ == "__main__":
    # hatch_mask.astype(int)
    # python -m tcpips.era5 &> era5_pi_2.log
    # download_era5_data(start_year=1940)
    # era5_pi(
    #     [str(year) for year in range(1980, 2025)]  # 2025)]
    # )  # Modify or extend this list as needed.)
    # era5_pi(
    #      [str(year) for year in range(1980, 2025)]  # 2025)]
    # )  # Modify or extend this list as needed.)
    # problem: the era5 pressure level data is too big to save in one file
    # so I have split it into chunks of 10 years.
    # This means that future scripts also need to be able to handle this.
    # era5_pi(
    #     [str(year) for year in range(1980, 2025)]
    # )  # Modify or extend this list as needed.
    # era5_pi_trends(start_year=1940, months=1)
    # dask_cluster_wrapper(
    #     era5_pi_trends, start_year=1980, end_year=2024, months=1
    # )
    # dask_cluster_wrapper(
    #     era5_pi_trends, start_year=1980, end_year=2024, months=1
    # )
    # dask_cluster_wrapper(
    #     era5_pi_trends, start_year=1940, end_year=2024, months=1
    # # )
    # from w22.constants import OFFSET_D

    # era5_pi_trends(start_year=1980, months=1)
    # find_tropical_m(start_year=1980, months=1)
    # find_tropical_m(start_year=1940, months=1)

    # plot_trend_maps(start_year=1980, months=1)
    # plot_trend_maps(start_year=1940, months=1)
    # plot_trend_lineplots(
    #     360 - 90.0,
    #     29.0,
    #     label="new_orleans",
    #     start_year=1980,
    # )  # New Orleans, USA (30N, 90W)
    # plot_trend_lineplots(
    #     360 - 80.0,
    #     25.0,
    #     label="miami",
    #     start_year=1980,
    # )  # Miami, USA (25N, 80W)
    # plot_trend_lineplots(
    #     360 - 90.0,
    #     27.0,
    #     label="new_orleans",
    #     start_year=1940,
    # )  # New Orleans, USA (30N, 90W)
    # plot_trend_lineplots(
    #     360 - 80.0,
    #     25.0,
    #     label="miami",
    #     start_year=1940,
    # )  # Miami, USA (25N, 80W)

    # ds = get_all_data()
    # plot_trend_lineplots(
    #     360 - 90.0, 29.0, label="new_orleans", ds=ds
    # )  # New Orleans, USA (30N, 90W)
    # plot_trend_lineplots(360 - 80.0, 25.0, label="miami", ds=ds)  # Miami, USA (25N, 80W)
    # plot_trend_lineplots(360 - 90.0, 27.0, label="new_orleans")  # New Orleans, USA (30N, 90W)
    # time to emergence - signal to noise  - variable record with nonstationarity - how long until new properties are 2 sigma. Perhaps you need to assume standard deviation constant.
    # Ed Hawkins?
    #
    # dask_cluster_wrapper(
    #     calculate_potential_sizes,
    #     start_year=DEFAULT_START_YEAR,
    #     end_year=DEFAULT_START_YEAR + 9,
    #     dry_run=True,
    # )  # This will take a long time to run.
    # dask_cluster_wrapper(
    #     calculate_potential_sizes,
    #     start_year=DEFAULT_START_YEAR + 10,
    #     end_year=DEFAULT_START_YEAR + 19,
    #     dry_run=True,
    # )
    # dask_cluster_wrapper(
    #     calculate_potential_sizes,
    #     start_year=DEFAULT_START_YEAR + 20,
    #     end_year=DEFAULT_START_YEAR + 29,
    #     dry_run=True,
    # )
    # dask_cluster_wrapper(
    #     calculate_potential_sizes,
    #     start_year=DEFAULT_START_YEAR + 30,
    #     end_year=DEFAULT_START_YEAR + 39,
    #     dry_run=True,
    # )
    # dask_cluster_wrapper(
    #     calculate_potential_sizes,
    #     start_year=DEFAULT_START_YEAR + 40,
    #     end_year=DEFAULT_END_YEAR,
    #     dry_run=True,
    # )
    # calculate_potential_sizes(
    #     start_year=DEFAULT_START_YEAR, end_year=DEFAULT_START_YEAR + 9
    # )  # This will take a long time to run.
    # years = [str(year) for year in range(1940, DEFAULT_END_YEAR + 1)]
    # regrid_all(years)
    rechunk_file(
        os.path.join(ERA5_PS_OG_PATH, f"era5_potential_sizes_2020_2024.zarr"),
        new_chunks={
            "year": -1,
            "latitude": 321,  # Keep all latitudes in one chunk
            "longitude": 720,  # Split longitude into 2 chunks (1440/720=2)
        },
    )
