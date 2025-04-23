"""Relative Humidity Calculation Module.

This module provides functions to calculate relative humidity based on
dew point temperature and air temperature. It includes a function to
calculate the saturation pressure of water vapor at a given temperature.

https://earthscience.stackexchange.com/a/24167
"""

import numpy as np


def saturation_pressure(temperature: float) -> float:
    """
    Calculate the saturation pressure of water vapor at a given temperature.

    Args:
        temperature (float): Temperature in degrees Celsius.

    Returns:
        float: Saturation pressure in hPa.
    """
    # Constants for the Magnus-Tetens approximation
    a = 6.1078  # hPa
    b = 17.1
    c = 235  # degrees Celsius

    return a * np.exp(b * (temperature / (c + temperature)))


def relative_humidity_from_dew_point(
    dew_point_temp: float, temperature: float
) -> float:
    """
    Calculate the relative humidity given the dew point temperature and air temperature.

    Args:
        dew_point_temp (float): Dew point temperature in degrees Kelvin.
        temperature (float): Air temperature in degrees Kelvin.

    Returns:
        float: Relative humidity as fraction (0 to 1).
    """
    # Calculate saturation pressure at dew point
    dew_point_saturation_pressure = saturation_pressure(dew_point_temp - 273.15)

    # Calculate saturation pressure at air temperature
    air_saturation_pressure = saturation_pressure(temperature - 273.15)

    # Calculate relative humidity
    relative_humidity = dew_point_saturation_pressure / air_saturation_pressure

    return relative_humidity
