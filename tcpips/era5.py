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
from .constants import ERA5_PATH


@timeit
def download_single_levels(
    years: List[str], months: List[str], output_file: str
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
    """
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "format": "netcdf",
            "variable": [
                "sea_surface_temperature",
                "mean_sea_level_pressure",
                "relative_humidity",
            ],
            "year": years,
            "month": months,
            "time": "00:00",
        },
        os.path.join(ERA5_PATH, output_file),
    )
    print(f"Downloaded single-level data to {output_file}")
    return


@timeit
def download_pressure_levels(
    years: List[str], months: List[str], output_file: str
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
    """
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
            file_path = os.path.join(ERA5_PATH, file_name)
            file_paths.append(file_path)

            # check if the temporary sfile already exists (this would indicate the script crashed partway through last time)
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping download.")
            else:
                download_pressure_levels(years[i:j], months, file_name)
        # stitch the files together
        output_path = os.path.join(ERA5_PATH, output_file)
        # append the files together along the time dimension and save to output_file
        # using xarray
        print(f"Stitching files together: {file_paths} to {output_path}")
        ds = xr.open_mfdataset(
            file_paths,
            combine="by_coords",
            concat_dim="time",
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
        os.path.join(ERA5_PATH, output_file),
    )
    print(f"Downloaded pressure-level data to {output_file}")
    return


if __name__ == "__main__":
    # Specify the desired years and months.
    # python -m tcpips.era5
    years = [
        str(year) for year in range(1980, 2025)
    ]  # Modify or extend this list as needed.
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # Download the single-level (surface) variables.
    # download_single_levels(years, months, "era5_single_levels.nc")

    # Download the pressure-level variables (including geopotential).
    download_pressure_levels(years, months, "era5_pressure_levels.nc")
