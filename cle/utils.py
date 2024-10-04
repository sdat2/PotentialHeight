"""
Utilities for idealized tropical cyclone calculations.
"""

import numpy as np
from .constants import TEMP_0K


def coriolis_parameter_from_lat(lat: np.ndarray) -> np.ndarray:
    """
    Calculate the Coriolis parameter from latitude.

    Args:
        lat (np.ndarray): Latitude [degrees].

    Returns:
        np.ndarray: Coriolis parameter [s-1].

    Example::
        >>> cp_out = coriolis_parameter_from_lat(30)
        >>> cp_manual = 2 * 2 * np.pi / 24 / 60 / 60 * 1/2  ## 2 * omega * sin(30deg)
        >>> np.isclose(cp_out, cp_manual, rtol=1e-3, atol=1e-6)
        True
        >>> np.isclose(coriolis_parameter_from_lat(0), 0, rtol=1e-3, atol=1e-6)
        True
    """

    return 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))


def pressure_from_wind(
    rr: np.ndarray,  # [m]
    vv: np.ndarray,  # [m/s]
    p0: float = 1015 * 100,  # Pa
    rho0: float = 1.15,  # kg m-3
    fcor: float = 5e-5,  # s-2
) -> np.ndarray:  # [Pa]
    """
    Use coriolis force and pressure gradient force to find physical
    pressure profile to correspond to the velocity profile.

    TODO: decrease air density in response to decreased pressure (will make central pressure lower).

    Args:
        rr (np.ndarray): radii array [m].
        vv (np.ndarray): velocity array [m/s]
        p0 (float): ambient pressure [Pa].
        rho0 (float): Air density at ambient pressure [kg/m3].
        fcor (float): Coriolis force [s-1].

    Returns:
        np.ndarray: Pressure array [Pa].

    Example::
        >>> rr = np.array([0, 1, 2, 3, 4, 5])
        >>> vv = np.array([0, 0, 0, 0, 0, 0])
        >>> p = pressure_from_wind(rr, vv)
        >>> np.allclose(p, np.array([101500, 101500, 101500, 101500, 101500, 101500]))
    """
    p = np.zeros(rr.shape)  # [Pa]
    # rr ascending
    assert np.all(rr == np.sort(rr))  # check if rr is sorted
    p[-1] = p0  # set the last value to the background pressure
    for j in range(len(rr) - 1):
        i = -j - 2
        # Assume Coriolis force and pressure-gradient balance centripetal force.
        # delta P = - rho * ( v^2/r + fcor * vv[i] ) * delta r
        p[i] = p[i + 1] - rho0 * (
            vv[i] ** 2 / (rr[i + 1] / 2 + rr[i] / 2) + fcor * vv[i]
        ) * (rr[i + 1] - rr[i])
        # centripetal pushes out, pressure pushes inward, coriolis pushes inward
    return p  # pressure profile [Pa]


def buck_sat_vap_pressure(
    temp: float,
) -> float:  # temp in K -> saturation vapour pressure in Pa
    """
    Arden buck_sat_vap_pressure equation.

    https://en.wikipedia.org/wiki/Arden_buck_sat_vap_pressure_equation

    Args:
        temp (float): temperature in Kelvin.

    Returns:
        float: saturation vapour pressure in Pa.

    """
    # https://en.wikipedia.org/wiki/Arden_buck_sat_vap_pressure_equation
    temp: float = temp - TEMP_0K  # convert from degK to degC
    return 0.61121 * np.exp((18.678 - temp / 234.5) * (temp / (257.14 + temp))) * 1000


def carnot_factor(temp_hot: float, temp_cold: float) -> float:
    """
    Calculate carnot factor.

    Args:
        temp_hot (float): Temperature of hot reservoir [K].
        temp_cold (float): Temperature of cold reservoir [K].

    Returns:
        float: Carnot factor [dimensionless].
    """
    return (temp_hot - temp_cold) / temp_hot


def absolute_angular_momentum(v: float, r: float, f: float) -> float:
    """
    Calculate absolute angular momentum.

    Args:
        v (float): Azimuthal wind speed [m/s].
        r (float): Radius from storm centre [m].
        f (float): Coriolis parameter [s-1].

    Returns:
        float: Absolute angular momentum [m2/s].
    """
    return v * r + 0.5 * f * r**2  # [m2/s]
