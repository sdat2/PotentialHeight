#!/usr/bin/env python3
import os
import cdsapi
from .constants import ERA5_PATH


def download_single_levels(years, months, output_file):
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


def download_pressure_levels(years, months, output_file):
    """
    Downloads monthly-averaged pressure-level fields:
      - Temperature (atmospheric profile)
      - Specific humidity (atmospheric profile)
      - Geopotential (atmospheric profile)

    The pressure levels below are specified in hPa.
    Note: Geopotential is provided in m²/s². Divide by ~9.81 to convert to geopotential height (meters).
    """
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


if __name__ == "__main__":
    # Specify the desired years and months.
    # python -m tcpips.era5
    years = [
        str(year) for year in range(1980, 2024)
    ]  # Modify or extend this list as needed.
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # Download the single-level (surface) variables.
    download_single_levels(years, months, "era5_single_levels.nc")

    # Download the pressure-level variables (including geopotential).
    download_pressure_levels(years, months, "era5_pressure_levels.nc")
