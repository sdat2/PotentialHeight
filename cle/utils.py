"""
Utilities for idealized tropical cyclone calculations.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
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
    p0: float = 1015 * 100,  # [Pa]
    rho0: float = 1.15,  # [kg m-3]
    fcor: float = 5e-5,  # [s-1]
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
        >>> vv = np.array([0] * 6)
        >>> p = pressure_from_wind(rr, vv)
        >>> np.allclose(p, np.array([101500] * 6), rtol=1e-3, atol=1e-6) # zero velocity -> no change.
        True
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


def pressure_from_wind_new(
    rr: np.ndarray,  # [m]
    vv: np.ndarray,  # [m s-1]
    p0: float = 1015 * 100,  # [Pa]
    rho0: float = 1.15,  # [kg m-3]
    fcor: float = 5e-5,  # [s-1]
) -> np.ndarray:  # [Pa]
    """
    Pressure from wind new.

    Assume cyclogeostrophic balance

    \frac{dp}{dr}=\rho(r)\left(\frac{v(r)^2}{r} +fv(r)\right)

    If ideal gas following isothermal expansion then rho1/p1 = rho2/p2.

    So rho(r) = rho0/p0*p(r)

    \frac{dp}{dr}=p(r) \frac{rho0}{p0}\left(\frac{v(r)^2}{r} +fv(r)\right)

    \frac{dlog(p)}{dr} = \frac{rho0}{p0}\left(\frac{v(r)^2}{r} +fv(r)\right)

    p(r) = \int^{\inf}_{r}\frac{rho0}{p0}\left(\frac{v(r)^2}{r} +fv(r)\right) dr


    Args:
        rr (np.ndarray): radii. Ascending order.
        vv (np.ndarray): velocities. From center to edge.
        p0 (float): background pressure. Defaults to 1015_00 Pa.
        rho0 (float): background density. Defaults to 1.15 kg m-3.
        fcor (float): coriolis force. Defaults to 5e-5 [s-1].

    Returns:
        np.ndarray: pressure profile [Pa]

    Example:
        >>> r0 = 1_000_000
        >>> rr = np.linspace(0, r0, 100)
        >>> vmax = 50
        >>> rmax = 30_000
        >>> vv = vmax * (rr / rmax) * np.exp(1 - rr / rmax)
        >>> pp1 = pressure_from_wind_new(rr, vv)
        >>> pp2 = pressure_from_wind(rr, vv)
        >>> assert pp1[0] < pp2[0]
        >>> rr = np.array([0, 1, 2, 3, 4, 5])
        >>> vv = np.array([0] * 6)
        >>> p = pressure_from_wind_new(rr, vv)
        >>> np.allclose(p, np.array([1015_00] * 6), rtol=1e-3, atol=1e-6)
        True
    """
    if isinstance(rr, list):
        rr = np.array(rr)
    if isinstance(vv, list):
        vv = np.array(vv)
    assert np.all(rr == np.sort(rr))  # check if rr is sorted
    integrand = (vv**2 / (rr + 1e-6) + fcor * vv) * rho0 / p0  # adding small delta
    integral = cumulative_trapezoid(integrand[::-1], rr[::-1], initial=0)
    return p0 * np.exp(integral[::-1])


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

    Example::
        >>> absolute_angular_momentum(0, 0, 0)
        0.0
        >>> absolute_angular_momentum(1, 1, 1)
        1.5
    """
    return v * r + 0.5 * f * r**2  # [m2/s]
