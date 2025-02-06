"""
Utilities for idealized tropical cyclone calculations.
"""

from typing import Union
import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
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

    This assumes cyclogeostrophic balance and constant density.

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
        >>> r0 = 1_000_000
        >>> ds = 1/1_000 # decay scale [m-1]
        >>> rho0 = 1.0 # [kg m-3]
        >>> p0 = 1000_00.0 # [Pa]
        >>> rr = np.linspace(0, r0, 1_000_000)
        >>> a0 = 1.0 # [m0.5 s-1]
        >>> vv = a0 * np.exp(-ds * rr) * (rr)**0.5
        >>> pest = pressure_from_wind(rr, vv, p0=p0, rho0=rho0, fcor=0.0)
        >>> pcalc = p0 - rho0 * a0**2 / (2*ds) * (np.exp(-2*ds*rr) - np.exp(-2*ds*r0))
        >>> if not np.allclose(pest, pcalc, rtol=1e-2, atol=1e-2):
        ...    print("pest", pest[:10], pest[::-1][:10][::-1])
        ...    print("pcalc", pcalc[:10], pcalc[::-1][:10][::-1])
    """
    p = np.zeros(rr.shape)  # [Pa]
    # rr ascending
    assert np.all(rr == np.sort(rr))  # check if rr is sorted
    p[-1] = p0  # set the last value to the background pressure
    for j in range(len(rr) - 1):
        i = -j - 2
        # Assume Coriolis force and pressure-gradient balance centripetal force.
        # delta P = - rho * ( vv[i]^2/r + fcor * vv[i] ) * delta r
        p[i] = p[i + 1] - rho0 * (
            vv[i] ** 2 / (rr[i + 1] / 2 + rr[i] / 2) + fcor * vv[i]
        ) * (rr[i + 1] - rr[i])
        # centripetal pushes out, pressure pushes inward, coriolis pushes inward
    return p  # pressure profile [Pa]


def pressure_from_wind_new(
    rr: np.ndarray,  # [m]
    vv: np.ndarray,  # [m s-1]
    p0: float = 1015 * 100.0,  # [Pa]
    rho0: float = 1.15,  # [kg m-3]
    fcor: float = 5e-5,  # [s-1]
) -> np.ndarray:  # [Pa]
    """
    Pressure from wind, assuming cyclogeostrophic balance and constant temperature.

    We assume cyclogeostrophic balance, with pressure gradient balancing the circular acceleration and coriolis force.

    \frac{dp}{dr}=\rho(r)\left(\frac{v(r)^2}{r} +fv(r)\right)

    where p is pressure, r is radius, \rho is density, f is coriolis parameter and v is velocity.
    If ideal gas following isothermal expansion then rho1/p1 = rho2/p2, rho(r) = rho0/p0*p(r)

    Where rho0 and p0 are the background density and pressure respectively.
    Substituting in we find that,

    \frac{dp}{dr}=p(r) \frac{rho0}{p0}\left(\frac{v(r)^2}{r} +fv(r)\right)

    and then bringing p(r) to the left hand side yields

    \frac{1}{p}\frac{dp}{dr}=\frac{dlog(p)}{dr} = \frac{rho0}{p0}\left(\frac{v(r)^2}{r} +fv(r)\right),

    allowing us to express p(r) by integrating from infinity to r,

    p(r) = p0*\exp\left(-\frac{rho0}{p0}\int^{\inf}_{r}\left(\frac{v(r)^2}{r} +fv(r)\right)dr\right).

    For an exact solution, let fcor=0, v=a0*exp(-\lambda r)*r**0.5

    Then we look:

    p(r) = p0*\exp\left(-\frac{rho0}{p0}\int^{\inf}_{r}\left(a0**2*exp(-2\lambda r)\right)\right)
        = p0*\exp\left(\frac{a0**2 * rho0}{2* p0 *\lambda}\left[*exp(-2\lambda r)\right]^{\inf}_{r}\right)
        = p0*\exp\left(-\frac{a0**2 * rho0}{2 * p0 * \lambda}*a0**2*exp(- 2 \lambda r)\right)

    Args:
        rr (np.ndarray): radii. Ascending order [m].
        vv (np.ndarray): velocities. From center to edge [m s-1].
        p0 (float, optional): background pressure. Defaults to 1015_00 [Pa].
        rho0 (float, optional): background density. Defaults to 1.15 [kg m-3].
        fcor (float, optional): coriolis force. Defaults to 5e-5 [s-1].

    Returns:
        np.ndarray: pressure profile [Pa].

    Example:
        >>> r0 = 1_000_000
        >>> rr = np.linspace(0, r0, 1_000_000)
        >>> vmax = 50
        >>> rmax = 30_000
        >>> vv = vmax * (rr / rmax) * np.exp(1 - rr / rmax)
        >>> pp1 = pressure_from_wind_new(rr, vv)
        >>> pp2 = pressure_from_wind(rr, vv)
        >>> assert pp1[0] > pp2[0]
        >>> rr = np.array([0, 1, 2, 3, 4, 5])
        >>> vv = np.array([0] * 6)
        >>> p = pressure_from_wind_new(rr, vv)
        >>> np.allclose(p, np.array([1015_00] * 6), rtol=1e-3, atol=1e-6)
        True
        >>> ds = 1/1_000 # decay scale [m-1]
        >>> rho0 = 1.0 # [kg m-3]
        >>> rr = np.linspace(0, r0, 1_000_000)
        >>> p0 = 1000_00.0 # [Pa]
        >>> a0 = 1.0 # [m0.5 s-1]
        >>> vv = a0 * np.exp(-ds * rr) * (rr)**0.5
        >>> pest = pressure_from_wind_new(rr, vv, p0=p0, rho0=rho0, fcor=0.0)
        >>> integral = -(a0**2*rho0)/(2*p0*ds) * (np.exp(-2*ds*rr) - np.exp(-2*ds*r0))
        >>> assert np.all(integral <= 0)
        >>> pcalc = p0 * np.exp(integral)
        >>> if not np.allclose(pest, pcalc, rtol=1e-2, atol=1e-2):
        ...    print("pest", pest[:10], pest[::-1][:10])
        ...    print("pcalc", pcalc[:10], pcalc[::-1][:10])
    """
    # all real inputs
    assert isinstance(rho0, Union[float, int])
    assert isinstance(p0, Union[float, int])
    assert isinstance(fcor, Union[float, int])
    assert rho0 < 2 and rho0 > 0.5  # definitely between 0.5 and 2 kg m-3
    assert p0 > 900_00 and p0 < 1100_00  # definitely between 900 hPa and 1100 hPa
    assert (
        fcor >= 0
    )  # assuming cyclonic winds, just take absolute value if you are in southern hemisphere
    if isinstance(rr, list):
        rr = np.array(rr)
    if isinstance(vv, list):
        vv = np.array(vv)
    assert np.all(rr == np.sort(rr))  # check if rr is sorted
    assert not (np.min(rr) < 0)  # no negative radii
    integrand = (
        (vv**2 / (rr + 1e-6) + fcor * vv) * rho0 / p0
    )  # adding small delta 1e-6 to avoid singularity
    integral = cumulative_trapezoid(integrand[::-1], rr[::-1], initial=0)[::-1]
    assert np.all(integral <= 0)
    p = p0 * np.exp(integral)
    assert p[0] <= p[-1]
    assert np.all(p <= p0)
    assert np.all(p >= 0)
    return p


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
    assert temp > 150 and temp < 350
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
    assert 250 < temp_hot and 320 > temp_hot
    assert 150 < temp_cold and 250 > temp_cold
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
    assert v >= 0
    assert r >= 0
    return v * r + 0.5 * f * r**2  # [m2/s]
