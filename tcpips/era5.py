"""
This script downloads monthly ERA5 reanalysis data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

The script uses the `cdsapi` library to interact with the CDS API and download the data in NetCDF format. This requires an API key, which in my case I store at ~/.cdsapirc. You can get your own API key by signing up at the CDS website and following the instructions in the documentation.

The script is designed to download the data to calculate potential intensity (PI) and potential size (PS) for tropical cyclones. Geopotential height is also included so that we could calculate the height of the tropopause.
"""

import os
import cdsapi
import time
from typing import List, Union, Tuple, Literal, Optional, Callable
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from sithom.time import timeit
from sithom.xr import mon_increase
from w22.ps import parallelized_ps_dask
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


from .constants import (
    ERA5_RAW_PATH,
    ERA5_PI_OG_PATH,
    ERA5_PRODUCTS_PATH,
    ERA5_FIGURE_PATH,
)
from .pi import calculate_pi
from .rh import relative_humidity_from_dew_point

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
            # plot the hatch mask
            # hatch_mask.plot(
            #     ax=ax,
            #     x="longitude",
            #     y="latitude",
            #     add_colorbar=False,
            #     transform=ccrs.PlateCarree(),
            #     rasterized=True,
            #     hatch=".",
            #     alpha=0.3,
            #     color="none",
            #     edgecolor="gray",
            #     # z_order=1,
            # )
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
    """Optimized preprocessing for pressure-level ERA5 data."""

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


def calculate_trend(
    da: xr.DataArray,
    output: Literal["slope", "rise"] = "rise",
    t_var: str = "T",
    make_hatch_mask: bool = False,
    keep_ds: bool = False,
    uncertainty: bool = False,
) -> Union[float, ufloat, xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Returns either the linear trend rise, or the linear trend slope,
    possibly with the array to hatch out where the trend is not significant.

    Uses `xr.polyfit` order 1 to do everything.

    Args:
        da (xr.DataArray): the timeseries.
        output (Literal[, optional): What to return. Defaults to "rise".
        t_var (str, optional): The time variable name. Defaults to "T".
            Could be changed to another variable that you want to fit along.
        make_hatch_mask (bool, optional): Whether or not to also return a DataArray
            of boolean values to indicate where is not significant.
            Defaults to False. Will only work if you're passing in an xarray object.
        uncertainty (bool, optional): Whether to return a ufloat object
            if doing linear regression on a single timeseries. Defaults to false.

    Returns:
        Union[float, ufloat, xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
            The rise/slope over the time period, possibly with the hatch array if
            that opition is selected for a grid.
    """

    def length_time(da):
        return (da.coords[t_var][-1] - da.coords[t_var][0]).values

    def get_float(inp: Union[np.ndarray, list, float]):
        try:
            if hasattr(inp, "__iter__"):
                inp = inp[0]
        # pylint: disable=bare-except
        except:
            print(type(inp))
        return float(inp)

    if "X" in da.dims or "Y" in da.dims or "member" in da.dims or keep_ds:

        fit_da = da.polyfit(t_var, 1, cov=make_hatch_mask)

        slope = fit_da.polyfit_coefficients.isel(degree=1).drop("degree")

        if make_hatch_mask:
            frac_error = np.abs(
                np.sqrt(fit_da.polyfit_covariance.isel(cov_i=0, cov_j=0))
                / slope  # degree=1 is in the first position
            )
            hatch_mask = frac_error >= 1.0

        slope = fit_da.polyfit_coefficients.sel(degree=1).drop("degree")
    else:
        if uncertainty:
            # print("uncertainty running")
            fit_da = da.polyfit(t_var, 1, cov=True)
            error = np.sqrt(fit_da.polyfit_covariance.isel(cov_i=0, cov_j=0)).values
            error = get_float(error)
            slope = fit_da.polyfit_coefficients.values
            slope = get_float(slope)
            slope = ufloat(slope, error)
        else:
            slope = da.polyfit(t_var, 1).polyfit_coefficients.values
            slope = get_float(slope)

    if output == "rise":

        run = length_time(da)
        rise = slope * run

        if isinstance(rise, xr.DataArray):
            for pr_name in ["units", "long_name"]:
                if pr_name in da.attrs:
                    rise.attrs[pr_name] = da.attrs[pr_name]
            rise = rise.rename("rise")

        # print("run", run, "slope", slope, "rise = slope * run", rise)

        if make_hatch_mask and not isinstance(rise, float, ufloat):
            return rise, hatch_mask
        else:
            return rise

    elif output == "slope":
        if make_hatch_mask and not isinstance(slope, float):
            return slope, hatch_mask
        else:
            return slope


def select_seasonal_hemispheric_data(
    ds: xr.Dataset, months_to_average: int = 1
) -> xr.Dataset:
    """
    Selects seasonal hemispheric data, with an option to average over peak months.

    Args:
        ds (xr.Dataset): An xarray Dataset with 'time', 'latitude', and
            'longitude' coordinates. 'time' must be a datetime-like coordinate.
        months_to_average (int): The number of months to average.
            - 1: Selects August for NH and March for SH.
            - 3: Averages over Aug-Sep-Oct for NH and Jan-Feb-Mar for SH.

    Returns:
        xr.Dataset: A new Dataset with an integer 'year' dimension, containing
        the combined seasonal and hemispheric data.

    >>> # --- Test Setup ---
    >>> # Create a 2-year dataset where the data value is the month number.
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
    >>> # For NH (lat=20), average of Aug(8), Sep(9), Oct(10) should be 9.0
    >>> assert int(result_3m.sel(year=2021, latitude=20).some_variable) == 9
    >>> # For SH (lat=-20), average of Jan(1), Feb(2), Mar(3) should be 2.0
    >>> assert int(result_3m.sel(year=2020, latitude=-20).some_variable) == 2
    >>> # Check that the dimensions are correct and 'time' is replaced by 'year'.
    >>> assert sorted(result_3m.dims) == ['latitude', 'longitude', 'year']
    >>> assert result_3m.year.values.tolist() == [2020, 2021]

    >>> # --- Test Case 2: 1-Month Selection (Original behavior) ---
    >>> result_1m = select_seasonal_hemispheric_data(ds_test, months_to_average=1)
    >>> # For NH (lat=20), the value for August (month 8) should be 8.
    >>> assert int(result_1m.sel(year=2021, latitude=20).some_variable) == 8
    >>> # For SH (lat=-20), the value for March (month 3) should be 3.
    >>> assert int(result_1m.sel(year=2020, latitude=-20).some_variable) == 3
    """
    if months_to_average == 1:
        # Original logic: Select a single peak month
        nh_data = ds.where((ds.time.dt.month == 8) & (ds.latitude >= 0), drop=True)
        sh_data = ds.where((ds.time.dt.month == 3) & (ds.latitude < 0), drop=True)

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
        # Northern Hemisphere: August-September-October (ASO)
        nh_data = ds.where(
            ds.time.dt.month.isin([8, 9, 10]) & (ds.latitude >= 0), drop=True
        )
        nh_annual = nh_data.groupby(nh_data.time.dt.year).mean("time")

        # Southern Hemisphere: January-February-March (JFM)
        sh_data = ds.where(
            ds.time.dt.month.isin([1, 2, 3]) & (ds.latitude < 0), drop=True
        )
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
    """
    # open the single level file
    if os.path.exists(os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc")):
        single_ds = xr.open_dataset(
            os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc"),
            chunks=CHUNK_IN[chunk_in],
            engine="netcdf4",
        )
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
    """Load the potential intensity (PI) data calculated from ERA5 data."""
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
    """Get both the derived data from PI calculation, and the processed era5 data."""
    era5_ds = get_era5_combined(
        start_year=start_year, end_year=end_year, chunk_in=chunk_in
    )
    if "valid_time" in era5_ds.dims:
        era5_dsds = era5_dsds.rename({"valid_time": "time"})
    # print(era5_ds.sst.values.sel(longitude))
    print(era5_ds)
    era5_pi = get_era5_pi(start_year=start_year, end_year=end_year, chunk_in=chunk_in)
    print(era5_pi)
    return xr.merge([era5_ds, era5_pi], compat="override", join="override")


# @dask_cluster_wrapper
def era5_pi_trends(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 1,
) -> None:
    """
    Let's find the linear trends in the potential intensity
    """
    # load all the decades of data for potential intensity
    ds = get_all_data(start_year=start_year, end_year=end_year, chunk_in="space")
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
    print("New time coordinates:", ds.year.values)
    print("Calculating trends in potential intensity...")
    # pi_vars = ["vmax", "t0", "otl", "sst"]
    pi_vars = [x for x in ds.variables]
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
    # print(trend_ds)
    trend_ds.to_netcdf(
        os.path.join(
            ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.nc"
        ),
        engine="h5netcdf",
        encoding={var: {"zlib": True, "complevel": 5} for var in trend_ds.variables},
    )


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
        figsize=get_dim(ratio=0.8),
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


def plot_vmax_trends(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    months: int = 3,
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
            ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.nc"
        ),
        engine="h5netcdf",
    )
    trend_ds = mon_increase(trend_ds).sel(latitude=slice(-40, 40))

    vmax = trend_ds["vmax_polyfit_coefficients"].sel(degree=1)
    t0 = trend_ds["t0_polyfit_coefficients"].sel(degree=1)
    otl = trend_ds["otl_polyfit_coefficients"].sel(degree=1)

    plot_defaults()
    fig, axs = plt.subplots(
        3,
        1,
        sharex=True,
        sharey=True,
        figsize=get_dim(ratio=0.8),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    plot_var_on_map(
        axs[0],
        vmax * 10,  # convert to m/s/decade
        label="$V_p$ trend [m s$^{-1}$ decade$^{-1}$]",
        cmap="cmo.balance",
        hatch_mask=trend_ds["vmax_hatch_mask"],
        vmin=-10,
        vmax=10,
    )
    plot_var_on_map(
        axs[1],
        t0 * 10,  # convert to K/decade
        label="$T_o$ trend [K decade$^{-1}$]",
        cmap="cmo.balance",
        hatch_mask=trend_ds["t0_hatch_mask"],
        vmin=-0.5,
        vmax=0.5,
    )
    plot_var_on_map(
        axs[2],
        otl * 10,  # convert to m/decade
        label="$z_{\mathrm{o}}$ trend [hPa decade$^{-1}$]",
        cmap="cmo.balance",
        hatch_mask=trend_ds["otl_hatch_mask"],
        vmin=-1,
        vmax=1,
    )

    # vmax.plot(ax=axs[0], cmap="cmo.balance", cbar_kwargs={"label": ""})
    # axs[0].set_title("Vmax Trend [m/s/decade]")
    # axs[0].set_ylabel(r"Latitude [$^{\circ}$N]")
    # axs[0].set_xlabel("")

    # t0.plot(ax=axs[1], cmap="cmo.balance", cbar_kwargs={"label": ""})
    # axs[1].set_title("T0 Trend [K/decade]")
    # axs[1].set_ylabel(r"Latitude [$^{\circ}$N]")
    # axs[1].set_xlabel("")

    # otl.plot(ax=axs[2], cmap="cmo.balance", cbar_kwargs={"label": ""})
    # axs[2].set_title("OTL Trend [m/decade]")
    # axs[2].set_ylabel(r"Latitude [$^{\circ}$N]")
    # axs[2].set_xlabel(r"Longitude [$^{\circ}$E]")

    label_subplots(axs)  # , override="outside")
    plt.tight_layout()
    plt.savefig(os.path.join(ERA5_FIGURE_PATH, "era5_vmax_trends_3m.pdf"), dpi=300)

    print("vmax trend mean:", vmax.mean())
    print("t0 trend mean:", t0.mean())
    print("otl trend mean:", otl.mean())


def plot_lineplots(
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
            ERA5_PRODUCTS_PATH, f"era5_pi_trends_{start_year}_{end_year}_{months}m.nc"
        ),
        engine="h5netcdf",
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
        # get March data (in the Southern Hemisphere)
        era5_ds = era5_ds.sel(time=era5_ds.time.dt.month == 3)
    # get the time in years
    era5_ds = era5_ds.assign_coords(
        year=era5_ds.time.dt.year - era5_ds.time.dt.year.min()
    )
    print("era5_ds", era5_ds)
    del era5_ds["time"]
    # plot the variables

    fig, axs = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=get_dim(ratio=0.8),
    )

    def plot_trend(ax: plt.axis, var: str, color: str, label: str, unit: str) -> None:
        """Plot the trend for a specific variable."""
        # ys = era5_ds[var].values
        x = era5_ds["year"].values
        slope = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=1).values
        intercept = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=0).values
        trend = slope * era5_ds["year"].values + intercept
        cov_matrix_unscaled = trend_ds[f"{var}_polyfit_covariance"].values
        coeffs = np.array([slope, intercept])
        # residuals = ys - np.polyval(coeffs, x)
        # n_dof = len(x) - len(coeffs)  # Number of degrees of freedom
        # residual_variance = np.sum(residuals**2) / n_dof
        # cov_matrix = cov_matrix_unscaled * residual_variance
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
            # 0.3,
            # 0.95,
            f"{label} Trend: {slope.n*10:.2f} ± {slope.s*10:.2f}  {unit} decade{r'$^{-1}$'}",
            # transform=ax.transAxes,
            fontsize=10,
            # verticalalignment="top",
            color=color,
        )

    print("sst", era5_ds["sst"].values.tolist())
    print("t0", era5_ds["t0"].values.tolist())
    print("otl", era5_ds["otl"].values.tolist())
    axs[0].plot(
        era5_ds["year"].values + start_year,
        era5_ds["sst"].values - 273.15,
        label=r"$T_s$ [$^{\circ}$C]",
        color="tab:blue",
    )
    plot_trend(axs[0], "sst", "tab:blue", r"$T_s$ [$^{\circ}$C]", r"$^{\circ}$C")
    axs[1].plot(
        era5_ds["year"].values + start_year,
        era5_ds["t0"].values,
        label=r"$T_o$ [K]",
        color="tab:orange",
    )
    plot_trend(axs[1], "t0", "tab:orange", r"$T_o$ [K]", "K")
    axs[2].plot(
        era5_ds["year"].values + start_year,
        era5_ds["otl"].values,
        label=r"$z_o$ [m]",
        color="tab:green",
    )
    plot_trend(axs[2], "otl", "tab:green", r"$z_o$ [hPa]", "hPa")
    axs[3].plot(
        era5_ds["year"].values + start_year,
        era5_ds["vmax"].values,
        label=r"$V_p$ [m s$^{-1}$]",
        color="tab:red",
    )
    plot_trend(axs[3], "vmax", "tab:red", r"$V_p$ [m s$^{-1}$]", r"m s$^{-1}$")
    axs[0].set_ylabel(r"$T_s$ [$^{\circ}$C]")
    axs[1].set_ylabel(r"$T_o$ [K]")
    axs[2].set_ylabel(r"$z_o$ [hPa]")
    axs[3].set_ylabel(r"$V_p$ [m s$^{-1}$]")
    axs[3].set_xlabel("Year")
    # era5_ds["sst"].plot(ax=axs[0], x="year", label="SST [C]", color="tab:blue")
    # era5_pi["t0"].plot(ax=axs[1], x="year", label="T0 [K]", color="tab:orange")
    plt.xlim(0 + start_year, end_year)
    label_subplots(axs)  # , override="outside")
    # era5_ds["otl"].plot(ax=axs[2], label="OTL [
    plt.savefig(
        os.path.join(
            ERA5_FIGURE_PATH, f"era5_pi_{label}_{start_year}_{end_year}_{months}m.pdf"
        ),
        dpi=300,
    )


def calculate_potential_sizes(
    start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR
) -> None:
    """Calculate the potential sizes of tropical cyclones from ERA5 data.

    This is going to be a very expensive function to run."""
    ds = get_all_data(start_year=start_year, end_year=end_year)
    if ds is None:
        print("No ERA5 data found. Please run era5_pi() first.")
        return
    # First potential size corresponding to the potential intensity velocity -- needs vmax, t0, sst, t2

    # Then calculate the potential size corresponding to the lower limit of category 1 which is 33 m/s --


if __name__ == "__main__":
    # hatch_mask.astype(int)
    # python -m tcpips.era5 &> era5_pi_2.log
    # download_era5_data(start_year=1940)
    # era5_pi(
    #     [str(year) for year in range(1940, 1950)]  # 2025)]
    # )  # Modify or extend this list as needed.)
    # problem: the era5 pressure level data is too big to save in one file
    # so I have split it into chunks of 10 years.
    # This means that future scripts also need to be able to handle this.
    # era5_pi(
    #     [str(year) for year in range(1980, 2025)]
    # )  # Modify or extend this list as needed.
    era5_pi_trends(start_year=1940, months=1)
    era5_pi_trends(start_year=1980, months=1)
    # find_tropical_m(start_year=1980, months=1)
    # find_tropical_m(start_year=1940, months=1)

    # plot_vmax_trends(start_year=1980, months=1)
    # plot_vmax_trends(start_year=1940, months=1)
    # plot_lineplots(
    #     360 - 90.0,
    #     29.0,
    #     label="new_orleans",
    #     start_year=1980,
    # )  # New Orleans, USA (30N, 90W)
    # plot_lineplots(
    #     360 - 80.0,
    #     25.0,
    #     label="miami",
    #     start_year=1980,
    # )  # Miami, USA (25N, 80W)
    # plot_lineplots(
    #     360 - 90.0,
    #     27.0,
    #     label="new_orleans",
    #     start_year=1940,
    # )  # New Orleans, USA (30N, 90W)
    # plot_lineplots(
    #     360 - 80.0,
    #     25.0,
    #     label="miami",
    #     start_year=1940,
    # )  # Miami, USA (25N, 80W)

    # ds = get_all_data()
    # plot_lineplots(
    #     360 - 90.0, 29.0, label="new_orleans", ds=ds
    # )  # New Orleans, USA (30N, 90W)
    # plot_lineplots(360 - 80.0, 25.0, label="miami", ds=ds)  # Miami, USA (25N, 80W)
    # plot_lineplots(360 - 90.0, 27.0, label="new_orleans")  # New Orleans, USA (30N, 90W)
    # time to emergence - signal to noise  - variable record with nonstationarity - how long until new properties are 2 sigma. Perhaps you need to assume standard deviation constant.
    # Ed Hawkins?
