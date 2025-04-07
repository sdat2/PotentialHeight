"""
This script downloads ERA5 reanalysis data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
"""

import os
import cdsapi
from typing import List
import xarray as xr
from sithom.time import timeit
from .constants import ERA5_PATH


@timeit
def download_single_levels(years: List[str], months: List[str], output_file: str):
    """
    Downloads monthly-averaged single-level fields:
      - Sea surface temperature
      - Mean sea level pressure
    """
    c = cdsapi.Client()
    c.retrieve(
        # "reanalysis-era5-single-levels-monthly-means"
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "format": "netcdf",
            "variable": ["sea_surface_temperature", "mean_sea_level_pressure"],
            "year": years,
            "month": months,
            "time": "00:00",
        },
        os.path.join(ERA5_PATH, output_file),
    )  # .download()
    print(f"Downloaded single-level data to {output_file}")


@timeit
def download_pressure_levels(
    years: List[str], months: List[str], output_file: str
) -> None:
    """
    Downloads monthly-averaged pressure-level fields:
      - Temperature (atmospheric profile)
      - Specific humidity (atmospheric profile)
      - Geopotential (atmospheric profile)

    The pressure levels below are specified in hPa.
    Note: Geopotential is provided in m²/s². Divide by ~9.81 to convert to geopotential height (meters).

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
        file_names = []
        for i in range(0, len(years), 10):
            j = min(i + 10, len(years))
            file_name = f"{output_file}_years{years[i]}_{years[j-1]}.nc"
            download_pressure_levels(years[i:j], months, file_name)
            file_names.append(file_name)
        # stitch the files together
        file_names = [os.path.join(ERA5_PATH, file_name) for file_name in file_names]
        output_file = os.path.join(ERA5_PATH, output_file)
        # append the files together along the time dimension and save to output_file
        # using xarray
        ds = xr.open_mfdataset(
            file_names,
            combine="by_coords",
            concat_dim="time",
            parallel=True,
        )
        ds.to_netcdf(output_file)
        # remove the temporary files
        if os.path.exists(output_file):
            for file_name in file_names:
                os.remove(file_name)
        print(f"Downloaded pressure-level data to {output_file}")
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
        os.path.join(ERA5_PATH, output_file),
    )
    print(f"Downloaded pressure-level data to {output_file}")
    return


if __name__ == "__main__":
    # Specify the desired years and months.
    # python -m tcpips.era5
    years = [
        str(year) for year in range(1980, 2024)
    ]  # Modify or extend this list as needed.
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # Download the single-level (surface) variables.
    # download_single_levels(years, months, "era5_single_levels.nc")

    # Download the pressure-level variables (including geopotential).
    download_pressure_levels(years, months, "era5_pressure_levels.nc")
