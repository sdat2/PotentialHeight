"""
This script downloads monthly ERA5 reanalysis data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

The script uses the `cdsapi` library to interact with the CDS API and download the data in NetCDF format. This requires an API key, which in my case I store at ~/.cdsapirc. You can get your own API key by signing up at the CDS website and following the instructions in the documentation.

The script is designed to download the data to calculate potential intensity (PI) and potential size (PS) for tropical cyclones. Geopotential height is also included so that we could calculate the height of the tropopause.
"""

import os
import cdsapi
from typing import List
import xarray as xr
from sithom.time import timeit
from .constants import ERA5_RAW_PATH, ERA5_PI_OG_PATH, ERA5_PI_PATH
from .pi import calculate_pi
from .rh import relative_humidity_from_dew_point


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
    # if len(years) <= 10, download the data normally using cdsapi

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


def preprocess_single_level_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess the single-level data from ERA5.
    This includes converting units and renaming variables.

    Args:
        ds (xr.Dataset): xarray dataset containing the single-level data.

    Returns:
        xr.Dataset: Preprocessed xarray dataset.
    """
    assert ds["d2m"].units == "K", "Sea surface temperature is not in Kelvin"
    assert ds["t2m"].units == "K", "Mean sea level pressure is not in Pa"

    ds["rh"] = relative_humidity_from_dew_point(ds["d2m"], ds["t2m"])
    ds["rh"].attrs = {
        "long_name": "Relative Humidity",
        "units": "fraction",
        "description": "Relative humidity at 2m calculated from dew point temperature and air temperature",
    }
    assert ds["sst"].units == "K", "Sea surface temperature is not in Kelvin"
    ds["sst"] = ds["sst"] - 273.15
    ds["sst"].attrs = {
        "long_name": "Sea Surface Temperature",
        "units": "Celsius",
        "description": "Sea surface temperature",
    }
    assert ds["msl"].units == "Pa", "Mean sea level pressure is not in Pa"
    ds["msl"] = ds["msl"] / 100
    ds["msl"].attrs = {
        "long_name": "Mean Sea Level Pressure",
        "units": "hPa",
        "description": "Mean sea level pressure",
    }
    del ds["d2m"]
    del ds["t2m"]
    ds = ds.rename({"valid_time": "time"})
    return ds


def preprocess_pressure_level_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess the pressure-level data from ERA5.
    This includes converting units and renaming variables.
    Args:
        ds (xr.Dataset): xarray dataset containing the pressure-level data.
    Returns:
        xr.Dataset: Preprocessed xarray dataset.
    """
    assert ds["t"].units == "K", "Temperature is not in Kelvin"

    ds["t"] = ds["t"] - 273.15
    ds["t"].attrs = {
        "long_name": "Temperature",
        "units": "Celsius",
        "description": "Temperature at pressure levels",
    }
    assert ds["q"].units == "kg kg**-1", "Specific humidity is not in kg/kg"
    ds["q"] = ds["q"] * 1000
    ds["q"].attrs = {
        "long_name": "Specific Humidity",
        "units": "g/kg",
        "description": "Specific humidity at pressure levels",
    }
    del ds["z"]
    ds.rename({"valid_time": "time"})
    return ds


def era5_pi_decade(single_level_path: str, pressure_level_path: str) -> None:
    """
    Calculate potential intensity (PI) using the downloaded ERA5 data for a decade.
    This function is a placeholder and should be implemented with the actual calculation logic.

    Args:
        single_level_path (str): Path to the single-level data file.
        pressure_level_path (str): Path to the pressure-level data file.

    Returns:
        None
    """
    years_str = single_level_path.split("years")[1].replace(".nc", "")
    single_ds = preprocess_single_level_data(xr.open_dataset(single_level_path))
    pressure_ds = preprocess_pressure_level_data(xr.open_dataset(pressure_level_path))
    # let's assume they are on the same grid
    # and that the time dimension is the same
    combined_ds = xr.merge([single_ds, pressure_ds])
    # calculate potential intensity
    pi_ds = calculate_pi(combined_ds, dim="pressure_level")
    # save the potential intensity dataset
    # pi_ds.to_netcdf(os.path.join(ERA5_PI_OG_PATH, "era5_pi.nc"))
    combined_ds.to_netcdf(os.path.join(ERA5_PI_OG_PATH, f"era5_pi_years{years_str}.nc"))


def era5_pi(years: List[str]) -> None:
    """
    Calculate potential intensity (PI) using the downloaded ERA5 data.
    This function is a placeholder and should be implemented with the actual calculation logic.
    """
    # Placeholder for potential intensity calculation logic
    # First convert the data
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
    This function is a placeholder and should be implemented with the actual calculation logic.
    """
    return xr.open_dataset(os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc"))[
        ["longitude", "latitude", "valid_time"]
    ]


def get_era5_combined() -> xr.Dataset:
    """
    Get the raw ERA5 data for potential intensity and size as a lazily loaded xarray dataset.
    """
    # open the single level file
    single_ds = xr.open_dataset(
        os.path.join(ERA5_RAW_PATH, "era5_single_levels.nc"), chunks={"time": 1}
    )
    # open the pressure level files for all of the decades using xr.open_mfdataset
    file_paths = []
    years = [str(year) for year in range(1980, 2025)]
    for i in range(0, len(years), 10):
        j = min(i + 10, len(years))
        file_paths += os.path.join(
            ERA5_RAW_PATH, f"era5_pressure_levels_years{years[i]}_{years[j-1]}.nc"
        )
    pressure_ds = xr.open_dataset(file_paths, chunks={"time": 1})
    # merge the datasets
    return xr.merge([single_ds, pressure_ds])


if __name__ == "__main__":
    # python -m tcpips.era5
    download_era5_data()
    # era5_pi(
    #    [str(year) for year in range(1980, #2025)]
    # )  # Modify or extend this list as needed.)
    # problem: the era5 pressure level data is too big to save in one file
    # so I have split it into chunks of 10 years.
    # This means that future scripts also need to be able to handle this.
