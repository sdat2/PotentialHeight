"""
Utilities for idealized tropical cyclone calculations.
"""

from typing import Union, Literal
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from .constants import TEMP_0K, F_COR_DEFAULT, RHO_AIR_DEFAULT, BACKGROUND_PRESSURE


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
    p0: float = BACKGROUND_PRESSURE,  # [Pa]
    rho0: float = RHO_AIR_DEFAULT,  # [kg m-3]
    fcor: float = F_COR_DEFAULT,  # [s-1]
    assumption: Literal[
        "isopycnal", "isothermal"
    ] = "isopycnal",  # choose isopycnal or isothermal assumption for density
) -> np.ndarray:  # [Pa]
    """
    Pressure from wind, assuming cyclogeostrophic balance and either constant temperature or pressure.

    We assume cyclogeostrophic balance, with pressure gradient balancing the circular acceleration and coriolis force.

    \\frac{dp}{dr}=\\rho(r)\\left(\\frac{v(r)^2}{r} +fv(r)\\right)

    where p is pressure, r is radius, \\rho is density, f is coriolis parameter and v is velocity.
    If ideal gas following isothermal expansion then rho1/p1 = rho2/p2, rho(r) = rho0/p0*p(r)

    Where rho0 and p0 are the background density and pressure respectively.
    Substituting in we find that,

    \\frac{dp}{dr}=p(r) \\frac{rho0}{p0}\\left(\\frac{v(r)^2}{r} +fv(r)\\right)

    and then bringing p(r) to the left hand side yields

    \\frac{1}{p}\\frac{dp}{dr}=\\frac{dlog(p)}{dr} = \\frac{rho0}{p0}\\left(\\frac{v(r)^2}{r} +fv(r)\\right),

    allowing us to express p(r) by integrating from infinity to r,

    p(r) = p0*\\exp\\left(-\\frac{rho0}{p0}\\int^{\\inf}_{r}\\left(\\frac{v(r)^2}{r} +fv(r)\\right)dr\\right).

    For an exact solution, let fcor=0, v=a0*exp(-\\lambda r)*r**0.5

    Then we see,

    p(r) = p0*\\exp\\left(-\\frac{rho0}{p0}\\int^{\\inf}_{r}\\left(a0**2*exp(-2\\lambda r)\\right)\\right)
        = p0*\\exp\\left(\\frac{a0**2 * rho0}{2* p0 *\\lambda}\\left[*exp(-2\\lambda r)\\right]^{\\inf}_{r}\\right)
        = p0*\\exp\\left(-\\frac{a0**2 * rho0}{2 * p0 * \\lambda}*a0**2*exp(- 2 \\lambda r)\\right).

    Args:
        rr (np.ndarray): radii array [m].
        vv (np.ndarray): velocity array [m/s]
        p0 (float): ambient pressure [Pa].
        rho0 (float, optional): Air density at ambient pressure [kg/m3].
        fcor (float, optional): Coriolis force [s-1].
        assumption (Literal["isopycnal", "isothermal"], optional)

    Returns:
        np.ndarray: Pressure array [Pa].

    Example::
        >>> rr = np.array([0, 1, 2, 3, 4, 5])
        >>> vv = np.array([0] * 6)
        >>> p = pressure_from_wind(rr, vv, assumption="isopycnal")
        >>> np.allclose(p, np.array([101500] * 6), rtol=1e-3, atol=1e-6) # zero velocity -> no change.
        True
        >>> p = pressure_from_wind(rr, vv, assumption="isothermal")
        >>> np.allclose(p, np.array([101500] * 6), rtol=1e-3, atol=1e-6) # zero velocity -> no change.
        True
        >>> r0 = 1_000_000
        >>> ds = 1/1_000 # decay scale [m-1]
        >>> rho0 = 1.0 # [kg m-3]
        >>> p0 = 1000_00.0 # [Pa]
        >>> rr = np.linspace(0, r0, 1_000_000)
        >>> a0 = 1.0 # [m0.5 s-1]
        >>> vv = a0 * np.exp(-ds * rr) * (rr)**0.5
        >>> pest = pressure_from_wind(rr, vv, p0=p0, rho0=rho0, fcor=0.0, assumption="isopycnal")
        >>> pcalc = p0 - rho0 * a0**2 / (2*ds) * (np.exp(-2*ds*rr) - np.exp(-2*ds*r0))
        >>> if not np.allclose(pest, pcalc, rtol=1e-4, atol=1e-6):
        ...    print("pest", pest[:10], pest[::-1][:10][::-1])
        ...    print("pcalc", pcalc[:10], pcalc[::-1][:10][::-1])
        >>> pest = pressure_from_wind(rr, vv, p0=p0, rho0=rho0, fcor=0.0, assumption="isothermal")
        >>> integral = -(a0**2*rho0)/(2*p0*ds) * (np.exp(-2*ds*rr) - np.exp(-2*ds*r0))
        >>> assert np.all(integral <= 0)
        >>> pcalc = p0 * np.exp(integral)
        >>> if not np.allclose(pest, pcalc, rtol=1e-3, atol=1e-6):
        ...    print("pest", pest[:10], pest[::-1][:10][::-1])
        ...    print("pcalc", pcalc[:10], pcalc[::-1][:10][::-1])
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
    if assumption == "isopycnal":
        integrand = (
            vv**2 / (rr + 1e-6) + fcor * vv
        ) * rho0  # adding small delta 1e-6 to avoid singularity
        integral = cumulative_trapezoid(integrand[::-1], rr[::-1], initial=0)[::-1]
        p = p0 + integral
    elif assumption == "isothermal":
        integrand = (
            (vv**2 / (rr + 1e-6) + fcor * vv) * rho0 / p0
        )  # adding small delta 1e-6 to avoid singularity
        integral = cumulative_trapezoid(integrand[::-1], rr[::-1], initial=0)[::-1]
        assert np.all(integral <= 0)
        p = p0 * np.exp(integral)
    else:
        assert False
    assert p[0] <= p[-1]
    assert np.all(p <= p0)
    assert np.all(p >= 0)
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
    assert temp > 150 and temp < 350
    # https://en.wikipedia.org/wiki/Arden_buck_sat_vap_pressure_equation
    temp: float = temp - TEMP_0K  # convert from degK to degC
    return 0.61121 * np.exp((18.678 - temp / 234.5) * (temp / (257.14 + temp))) * 1000


def carnot_efficiency(temp_hot: float, temp_cold: float) -> float:
    """
    Calculate Carnot efficiency.

    Args:
        temp_hot (float): Temperature of hot reservoir [K].
        temp_cold (float): Temperature of cold reservoir [K].

    Returns:
        float: Carnot efficiency [dimensionless].
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


def qair2rh(
    qair: Union[xr.DataArray, float],
    temp: Union[xr.DataArray, float],
    press: Union[xr.DataArray, float] = BACKGROUND_PRESSURE,
) -> Union[xr.DataArray, float]:
    """
    Convert specific humidity to relative humidity.

    Inspired by: https://earthscience.stackexchange.com/a/2385

    Args:
        qair (Union[xr.DataArray, float]): Specific humidity [dimensionless].
        temp (Union[xr.DataArray, float]): Temperature [K].
        press (Union[xr.DataArray, float]): Pressure [hPa].

    Returns:
        Union[xr.DataArray, float]: Relative humidity [dimensionless].

    Example::
        >>> print(f'{qair2rh(0.01, 300, 1013.25):.3f}')
        0.458
    """
    temp_c = temp - TEMP_0K
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e = qair * press / (0.378 * qair + 0.622)
    rh = e / es
    rh = np.clip(rh, 0, 1)
    if isinstance(rh, xr.DataArray):
        rh.rename("rh")
        rh.attrs["long_name"] = "Relative Humdity"
        rh.attrs["short_name"] = "Relative Humdity"
        rh.attrs["units"] = "dimensionless"
    return rh
