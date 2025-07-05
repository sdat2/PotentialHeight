"""
This script downloads monthly ERA5 reanalysis data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

The script uses the `cdsapi` library to interact with the CDS API and download the data in NetCDF format. This requires an API key, which in my case I store at ~/.cdsapirc. You can get your own API key by signing up at the CDS website and following the instructions in the documentation.

The script is designed to download the data to calculate potential intensity (PI) and potential size (PS) for tropical cyclones. Geopotential height is also included so that we could calculate the height of the tropopause.
"""

import os
import cdsapi
from typing import List, Union, Tuple, Literal, Optional
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from sithom.time import timeit
from sithom.xr import mon_increase
from w22.ps import parallelized_ps_dask
from uncertainties import ufloat, correlated_values, unumpy
from matplotlib import pyplot as plt
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
    ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree(central_longitude=180))
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
    if hatch_mask is not None and False:
        # ensure hatch_mask is boolean
        if not isinstance(hatch_mask, xr.DataArray):
            raise ValueError("hatch_mask must be an xarray DataArray.")
        # plot the hatch mask
        hatch_mask.plot(
            ax=ax,
            x="longitude",
            y="latitude",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            hatch="xx",
            color="none",
            edgecolor="black",
        )
    ax.set_title(label)


@timeit
def download_single_levels(
    years: List[str], months: List[str], output_file: str, redo: bool = False
) -> None:
    """
    Downloads monthly-averaged single-level fields:
      - Sea surface temperature (for PI and PS)
      - Mean sea level pressure (for PI and PS)
      - Relative humidity (PS only)

    Args:
        years (List[str]): List of years to download data for.
        months (List[str]): List of months to download data for.
        output_file (str): Name of the output file to save the data.
        redo (bool): If True, download the data again even if the file already exists. Default is False.
    """
    if os.path.exists(os.path.join(ERA5_RAW_PATH, output_file)) and not redo:
        print(f"File {output_file} already exists. Use redo=True to download again.")
        return
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
        os.path.join(ERA5_RAW_PATH, output_file),
    )
    print(f"Downloaded single-level data to {output_file}")
    return


def break_single_levels(years: List[str], output_file: str) -> None:
    """
    Breaks the single-level data into decade chunks and saves them as separate files.

    Args:
        years (List[str]): List of years to download data for.
        output_file (str): Name of the output file to save the data.
    """
    # break the single level file into decade chunks
    # and save them as separate files
    prefix = output_file.replace(".nc", "")
    ds = xr.open_dataset(os.path.join(ERA5_RAW_PATH, output_file))
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        file_name = f"{prefix}_years{years[i]}_{years[j-1]}.nc"
        file_path = os.path.join(ERA5_RAW_PATH, file_name)
        out_ds = ds.sel(valid_time=slice(f"{years[i]}-01-01", f"{years[j-1]}-12-31"))
        out_ds.to_netcdf(file_path)
        print(f"Saved {file_path}")


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


def download_era5_data() -> None:
    """
    Downloads ERA5 data for the specified years and months.
    This function is a wrapper around the download_single_levels
    and download_pressure_levels functions.
    """
    # Specify the desired years and months.
    years = [
        str(year) for year in range(1980, 2025)
    ]  # Modify or extend this list as needed.
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # Download the single-level (surface) variables.
    download_single_levels(years, months, "era5_single_levels.nc", redo=True)

    # break the single level file into decade chunks
    # and save them as separate files
    break_single_levels(years, "era5_single_levels.nc")

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
    cluster = LocalCluster(n_workers=10, threads_per_worker=1)
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


def get_era5_coordinates() -> xr.Dataset:
    """
    Get the coordinates of the ERA5 data.
    """
    sfp = os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc")
    if os.path.exists(sfp):
        # open the single level file
        return xr.open_dataset(sfp)[["longitude", "latitude", "valid_time"]]
    else:
        # open the pressure level files for all of the decades using xr.open_mfdataset
        file_paths = []
        years = [str(year) for year in range(1980, 2025)]
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


def get_era5_combined() -> xr.Dataset:
    """
    Get the raw ERA5 data for potential intensity and size as a lazily loaded xarray dataset.
    """
    # open the single level file
    if os.path.exists(os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc")):
        single_ds = xr.open_dataset(
            os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc"),
            chunks={"valid_time": 1},
            engine="netcdf4",
        )
    else:
        fp = []
        years = [str(year) for year in range(1980, 2025)]
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            fp += [
                os.path.join(
                    ERA5_RAW_PATH, f"era5_single_levels_years{years[i]}_{years[j-1]}.nc"
                )
            ]
        single_ds = xr.open_mfdataset(fp, chunks={"valid_time": 1}, engine="netcdf4")
    # open the pressure level files for all of the decades using xr.open_mfdataset
    file_paths = []
    years = [str(year) for year in range(1980, 2025)]
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        file_paths += [
            os.path.join(
                ERA5_RAW_PATH, f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc"
            )
        ]
    print(f"Opening pressure level files: {file_paths}")
    pressure_ds = xr.open_mfdataset(
        file_paths, chunks={"valid_time": 1}, engine="netcdf4"
    )
    # merge the datasets
    return xr.merge([single_ds, pressure_ds])


def get_trend(
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


def select_seasonal_hemispheric_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Selects August data from the Northern Hemisphere and March data
    from the Southern Hemisphere, combining them into a single dataset
    with an integer 'year' coordinate.

    Args:
        ds: An xarray Dataset with 'time', 'latitude', and 'longitude'
            coordinates. 'time' must be a datetime-like coordinate.

    Returns:
        A new xarray Dataset with an integer 'year' dimension, containing
        the combined seasonal and hemispheric data.

    >>> # 1. Set up a sample dataset for the doctest
    >>> time = pd.date_range('2000-01-01', '2002-12-31', freq='MS')
    >>> latitudes = [-30, -10, 10, 30]
    >>> longitudes = [100, 120]
    >>> data = np.arange(len(time) * len(latitudes) * len(longitudes)).reshape(len(time), len(latitudes), len(longitudes))
    >>> ds_test = xr.Dataset(
    ...     {'some_variable': (['time', 'latitude', 'longitude'], data)},
    ...     coords={'time': time, 'latitude': latitudes, 'longitude': longitudes}
    ... )
    >>> # 2. Run the function
    >>> result = select_seasonal_hemispheric_data(ds_test)
    >>> # 3. Check the result
    >>> print(result)
    <xarray.Dataset> Size: 264B
    Dimensions:        (year: 3, latitude: 4, longitude: 2)
    Coordinates:
      * latitude       (latitude) int64 32B -30 -10 10 30
      * longitude      (longitude) int64 16B 100 120
      * year           (year) int64 24B 2000 2001 2002
    Data variables:
        some_variable  (year, latitude, longitude) float64 192B 16.0 17.0 ... 255.0
    >>> # 1. Set up a sample dataset for the doctest.
    >>> # The data value for each point is simply its month number (1-12).
    >>> time = pd.date_range('2020-01-01', '2021-12-31', freq='MS')
    >>> lats = [-20, -10, 10, 20] # Southern and Northern latitudes
    >>> lons = [100, 120]
    >>> month_data = time.month.values.reshape(-1, 1, 1) * np.ones((len(time), len(lats), len(lons)))
    >>> ds_test = xr.Dataset(
    ...     {'some_variable': (['time', 'latitude', 'longitude'], month_data)},
    ...     coords={'time': time, 'latitude': lats, 'longitude': lons}
    ... )
    >>>
    >>> # 2. Run the function
    >>> result = select_seasonal_hemispheric_data(ds_test)
    >>>
    >>> # 3. Check the results
    >>> # For any Northern Hemisphere point, the value for any year should be 8 (from August).
    >>> print(int(result.sel(year=2021, latitude=10, longitude=120).some_variable))
    8
    >>> # For any Southern Hemisphere point, the value for any year should be 3 (from March).
    >>> print(int(result.sel(year=2020, latitude=-20, longitude=100).some_variable))
    3
    >>> # Check that the dimensions are correct and 'time' has been replaced by 'year'.
    >>> print(sorted(result.dims))
    ['latitude', 'longitude', 'year']
    """
    # 1. Isolate August data for the Northern Hemisphere
    nh_data = ds.where((ds.time.dt.month == 8) & (ds.latitude >= 0), drop=True)

    # 2. Isolate March data for the Southern Hemisphere
    sh_data = ds.where((ds.time.dt.month == 3) & (ds.latitude < 0), drop=True)

    # 3. Align both datasets to a common 'year' coordinate
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

    # 4. Combine the two datasets.
    # combine_first fills NaN values in nh_data with corresponding values from sh_data.
    # This correctly merges the two hemispheres.
    return nh_data.combine_first(sh_data)

    # # ds = mon_increase(ds)
    # # Define a boolean mask for Augusts in the Northern Hemisphere (latitude >= 0)
    # is_nh_august = (ds["time"].dt.month == 8) & (ds["latitude"] >= 0)

    # # Define a boolean mask for Marches in the Southern Hemisphere (latitude < 0)
    # is_sh_march = (ds["time"].dt.month == 3) & (ds["latitude"] < 0)

    # # Use .where() with the combined boolean mask.
    # # For any given time, data is kept only for the latitudes matching the condition.
    # # For a March timestamp, NH data becomes NaN; for an August timestamp, SH data becomes NaN.
    # # drop=True removes timestamps where neither condition is met.
    # combined = ds.where(is_sh_march | is_nh_august, drop=True)

    # # Group the data by year. For each year, a lat/lon point will have at most one
    # # valid data point (either from March or August).
    # # .first() selects the first non-NaN value in each group along the 'time' dimension,
    # # effectively collapsing 'time' into 'year'.
    # result = combined.groupby(combined.time.dt.year).first("time")

    # return result


def get_era5_pi() -> xr.Dataset:
    """Load the potential intensity (PI) data calculated from ERA5 data."""
    file_paths = []
    years = [str(year) for year in range(1980, 2025)]
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
        chunks={"time": 1},
        engine="h5netcdf",
    )


def era5_pi_trends() -> xr.Dataset:
    """
    Let's find the linear trends in the potential intensity
    """
    # load all the decades of data for potential intensity
    ds = get_era5_pi()
    if ds is None:
        print("No ERA5 PI data found. Please run era5_pi() first.")
        return

    print(ds)
    sst_ds = preprocess_single_level_data(get_era5_combined())["sst"]
    print("sst_ds", sst_ds)
    # ds["sst"] = sst_ds["sst"]
    # lets just select the augusts in the northern hemisphere
    ds["sst"] = sst_ds
    ds = select_seasonal_hemispheric_data(mon_increase(ds))
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
    pi_vars = ["vmax", "t0", "otl", "sst"]
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
    # print(trend_ds)
    trend_ds.to_netcdf(
        os.path.join(ERA5_PRODUCTS_PATH, "era5_pi_trends.nc"),
        engine="h5netcdf",
        encoding={var: {"zlib": True, "complevel": 5} for var in trend_ds.variables},
    )


def find_tropical_m():
    """I want to find what the trends are for the tropical cyclone potential intensity and size over the pseudo observational period.

    I defined m as the ratio of the change in the outflow temperature to the change in surface temperature.
    T_s = T_s0 + delta_t
    T_o = T_o0 + m * delta_t
    where T_s0 is the initial sea surface temperature, T_o0 is the initial outflow temperature, and delta_t is the change in temperature.
    """
    trend_ds = mon_increase(
        xr.open_dataset(
            os.path.join(ERA5_PRODUCTS_PATH, "era5_pi_trends.nc"),
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
        label="Change in Outflow Temp / Change in SST",
        cmap="viridis",
        hatch_mask=trend_ds["t0_hatch_mask"],
    )
    plot_var_on_map(
        axs[1],
        trend_ds["t0_polyfit_coefficients"].sel(degree=1) * 10,
        label="Outflow Temperature Trend [K/decade]",
        cmap="cmo.balance",
        hatch_mask=trend_ds["t0_hatch_mask"],
    )
    plot_var_on_map(
        axs[2],
        trend_ds["sst_polyfit_coefficients"].sel(degree=1) * 10,
        label="SST Trend [K/decade]",
        cmap="cmo.balance",
        hatch_mask=trend_ds["sst_hatch_mask"],
    )
    label_subplots(axs)  # , override="outside")
    plt.tight_layout()
    plt.savefig(os.path.join(ERA5_FIGURE_PATH, "era5_pi_trends.pdf"), dpi=300)

    print(trend_ds["sst_polyfit_coefficients"].sel(degree=0).mean())
    print(trend_ds["sst_polyfit_coefficients"].sel(degree=1).mean())
    print(trend_ds["t0_polyfit_coefficients"].sel(degree=0).mean())
    print(trend_ds["t0_polyfit_coefficients"].sel(degree=1).mean())
    print(trend_ds["m"].mean())
    print(trend_ds["m"].median())


def plot_vmax_trends():
    """Plot trend in vmax, t0 and otl in another 3 panel subplot with no cbar label but
    labels in subplot title instead."""
    trend_ds = xr.open_dataset(
        os.path.join(ERA5_PRODUCTS_PATH, "era5_pi_trends.nc"),
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
        label="$V_p$ Trend [m s$^{-1}$ decade$^{-1}$]",
        cmap="viridis",
        hatch_mask=trend_ds["vmax_hatch_mask"],
    )
    plot_var_on_map(
        axs[1],
        t0 * 10,  # convert to K/decade
        label="$T_o$ Trend [K decade$^{-1}$]",
        cmap="viridis",
        hatch_mask=trend_ds["t0_hatch_mask"],
    )
    plot_var_on_map(
        axs[2],
        otl * 10,  # convert to m/decade
        label="$z_{\mathrm{lnb}}$ Trend [hpa decade$^{-1}$]",
        cmap="viridis",
        hatch_mask=trend_ds["otl_hatch_mask"],
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
    plt.savefig(os.path.join(ERA5_FIGURE_PATH, "era5_vmax_trends.pdf"), dpi=300)

    print("vmax trend mean:", vmax.mean())
    print("t0 trend mean:", t0.mean())
    print("otl trend mean:", otl.mean())


@timeit
def get_all_data():
    era5_ds = get_era5_combined().rename({"valid_time": "time"})
    # print(era5_ds.sst.values.sel(longitude))
    print(era5_ds)
    era5_pi = get_era5_pi()
    print(era5_pi)
    return xr.merge([era5_ds, era5_pi], compat="override", join="override")


def plot_lineplots(
    lon: float, lat: float, label: str = "new_orleans", ds: Optional[xr.Dataset] = None
) -> None:
    """Plot the linear trend and variables for a specific point
    in the ERA5 dataset.

    Args:
        lon (float): Longitude of the point to plot.
        lat (float): Latitude of the point to plot.
        label (str, optional): Label for the plot. Defaults to "new_orleans".
        ds (Optional[xr.Dataset], optional): If provided, use this dataset instead of loading it.
    """
    plot_defaults()
    trend_ds = xr.open_dataset(
        os.path.join(ERA5_PRODUCTS_PATH, "era5_pi_trends.nc"),
        engine="h5netcdf",
    )
    if ds is None:
        era5_ds = get_all_data()
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

    def plot_trend(ax, var: str, color: str, label: str):
        ys = era5_ds[var].values
        x = era5_ds["year"].values
        slope = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=1).values
        intercept = trend_ds[f"{var}_polyfit_coefficients"].sel(degree=0).values
        trend = slope * era5_ds["year"].values + intercept
        cov_matrix_unscaled = trend_ds[f"{var}_polyfit_covariance"].values
        coeffs = np.array([slope, intercept])
        residuals = ys - np.polyval(coeffs, x)
        n_dof = len(x) - len(coeffs)  # Number of degrees of freedom
        residual_variance = np.sum(residuals**2) / n_dof
        cov_matrix = cov_matrix_unscaled * residual_variance

        print(f"{var} Fit Coefficients (slope, intercept): {coeffs}")
        print(f"{var} Covariance Matrix:\n{cov_matrix}\n")

        # 3. Use correlated_values to create ufloat objects
        # This function takes the nominal values and the full covariance matrix.
        u_coeffs = correlated_values(coeffs, cov_matrix)
        slope, intercept = u_coeffs

        print(f"{var} Correlated Slope (m):     {slope}")
        print(f"{var} Correlated Intercept (c): {intercept}")
        print("-" * 30)

        ax.plot(
            era5_ds["year"].values + 1980,
            trend,
            label=f"{label} Trend",
            color=color,
            linestyle="--",
        )
        # new_ys = unumpy.polyval(np.array(u_coeffs), x)

        new_ys = u_coeffs[0] * x + u_coeffs[1]
        ax.fill_between(
            x + 1980,
            [y.n - y.s for y in new_ys],
            [y.n + y.s for y in new_ys],
            # new_ys.n - new_ys.s,
            color=color,
            alpha=0.2,
        )
        ax.text(
            0.3,
            0.95,
            f"{label} Trend: {slope.n:.2f} ± {slope.s:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color=color,
        )

    print(era5_ds["sst"].values)
    print(era5_ds["t0"].values)
    print(era5_ds["otl"].values)
    axs[0].plot(
        era5_ds["year"].values + 1980,
        era5_ds["sst"].values - 273.15,
        label=r"$T_s$ [$^{\circ}$C]",
        color="tab:blue",
    )
    plot_trend(axs[0], "sst", "tab:blue", r"$T_s$ [$^{\circ}$C]")
    axs[1].plot(
        era5_ds["year"].values + 1980,
        era5_ds["t0"].values,
        label=r"$T_o$ [K]",
        color="tab:orange",
    )
    plot_trend(axs[1], "t0", "tab:orange", r"$T_o$ [K]")
    axs[2].plot(
        era5_ds["year"].values + 1980,
        era5_ds["otl"].values,
        label=r"$z_o$ [m]",
        color="tab:green",
    )
    plot_trend(axs[2], "otl", "tab:green", r"$z_o$ [hPa]")
    axs[3].plot(
        era5_ds["year"].values + 1980,
        era5_ds["vmax"].values,
        label=r"$V_p$ [m s$^{-1}$]",
        color="tab:red",
    )
    plot_trend(axs[3], "vmax", "tab:red", r"$V_p$ [m s$^{-1}$]")
    axs[0].set_ylabel(r"$T_s$ [$^{\circ}$C]")
    axs[1].set_ylabel(r"$T_o$ [K]")
    axs[2].set_ylabel(r"$z_o$ [hPa]")
    axs[3].set_ylabel(r"$V_p$ [m s$^{-1}$]")
    axs[3].set_xlabel("Year")
    # era5_ds["sst"].plot(ax=axs[0], x="year", label="SST [C]", color="tab:blue")
    # era5_pi["t0"].plot(ax=axs[1], x="year", label="T0 [K]", color="tab:orange")
    plt.xlim(0 + 1980, 44 + 1980)  # limit the x-axis to 45 years
    label_subplots(axs)  # , override="outside")
    # era5_ds["otl"].plot(ax=axs[2], label="OTL [
    plt.savefig(
        os.path.join(ERA5_FIGURE_PATH, f"era5_pi_{label}.pdf"),
        dpi=300,
    )


if __name__ == "__main__":
    # python -m tcpips.era5
    # python -m tcpips.era5 &> era5_pi_2.log
    # download_era5_data()
    # era5_pi(
    #    [str(year) for year in range(1980, #2025)]
    # )  # Modify or extend this list as needed.)
    # problem: the era5 pressure level data is too big to save in one file
    # so I have split it into chunks of 10 years.
    # This means that future scripts also need to be able to handle this.
    # era5_pi(
    #     [str(year) for year in range(1980, 2025)]
    # )  # Modify or extend this list as needed.
    # era5_pi_trends()
    find_tropical_m()
    plot_vmax_trends()
    plot_lineplots(360 - 90.0, 29.0, label="new_orleans")  # New Orleans, USA (30N, 90W)
    # plot_lineplots(360 - 90.0, 27.0, label="new_orleans")  # New Orleans, USA (30N, 90W)
