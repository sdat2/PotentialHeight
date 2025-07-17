"""
Utilities for idealized tropical cyclone calculations.
"""

from typing import Union, Literal, List, Tuple
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from sithom.time import timeit
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
        >>> qair = xr.DataArray([0.001, 0.002, 0.003], dims=["x"], attrs={"units": "dimensionless"})
        >>> temp = xr.DataArray([300, 300, 300], dims=["x"], attrs={"units": "K"})
        >>> press = xr.DataArray([1013.25, 1013.25, 1013.25], dims=["x"], attrs={"units": "hPa"})
        >>> rh = qair2rh(qair, temp, press) #>>> np.allclose(rh, np.array([0.0458*i for i in range(1, 4)]), rtol=1e-3, atol=1e-6)
        >>> assert rh.attrs["units"] == "dimensionless"
        >>> assert rh.attrs["long_name"] == "Relative Humdity"
    """

    if isinstance(qair, xr.DataArray):
        if qair.attrs["units"] in ["dimensionless", "kg/kg"]:
            qair = qair
        elif qair.attrs["units"] in ["g/kg"]:
            qair = qair / 1000
        else:
            raise ValueError("qair units not recognized")
    # convert to temperature to degrees
    if isinstance(temp, xr.DataArray):
        if temp.attrs["units"] in ["K", "degK"]:
            temp = temp - TEMP_0K
        elif temp.attrs["units"] in ["degC", "C", "c", "degres_celsius"]:
            temp = temp
        else:
            raise ValueError("temp units not recognized")
    else:
        temp = temp - TEMP_0K

    if isinstance(press, xr.DataArray):
        if press.attrs["units"] in ["Pa"]:
            press = press / 100
        elif press.attrs["units"] in ["hPa", "mb", "mbar"]:
            press = press
        else:
            raise ValueError("press units not recognized")
    # saturation vapour pressure assuming temp in degC
    # and using buck equation
    es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    # vapour pressure from specific humidity
    e = qair * press / (0.378 * qair + 0.622)
    # relative humidity as the ratio of vapour pressure to saturation vapour pressure
    rh = e / es
    # clip between 0 and 1
    rh = np.clip(rh, 0, 1)
    if isinstance(rh, xr.DataArray):
        rh.rename("rh")
        rh.attrs["long_name"] = "Relative Humdity"
        rh.attrs["short_name"] = "Relative Humdity"
        rh.attrs["units"] = "dimensionless"
        rh.attrs["description"] = "Relative humidity of air (between 0 and 1)"
    return rh


def qtp2rh(qa: xr.DataArray, ta: xr.DataArray, msl: xr.DataArray) -> xr.DataArray:
    """
    Generate surface relative humidity from the specific humidity and temperature volumes, and pressure. The pressure level coordinate is "p" in Pa.

    Args:
        qa (xr.DataArray): Specific humidity [dimensionless] at many pressure levels.
        ta (xr.DataArray): Air temperature [K] at the same pressure levels.
        msl (xr.DataArray): The mean sea level pressure [Pa].

    Returns:
        xr.DataArray: Relative humidity [dimensionless] at the surface.

    Example::
        >>> pressure_levels = np.array([1000, 900])
        >>> qa = xr.DataArray(np.array([0.01, 0.02]), dims=["p"], coords={"p": ("p", pressure_levels, {"units": "hPa"})}, attrs={"units": "dimensionless"})
        >>> ta = xr.DataArray(np.array([300, 300]), dims=["p"], coords={"p": ("p", pressure_levels, {"units": "hPa"})}, attrs={"units": "K"})
        >>> msl = xr.DataArray(np.array([1000]), attrs={"units": "hPa"})
        >>> rh = qtp2rh(qa, ta, msl)
        >>> assert rh.attrs["units"] == "dimensionless"
        >>> assert rh.attrs["long_name"] == "Relative Humdity"
    """
    # select the closest pressure level to msl pressure from the q and t volumes
    qa = qa.sel(p=msl, method="nearest")
    # make layers identical
    ta = ta.sel(p=qa.p)
    # convert specific humidity to relative humidity
    rh = qair2rh(qa, ta, ta.p)
    return rh


@timeit
def curveintersect(
    x1: List[float], y1: List[float], x2: List[float], y2: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Find intersection points of two curves (x1, y1) and (x2, y2).

    Parameters:
    x1, y1 : array-like, coordinates of the first curve
    x2, y2 : array-like, coordinates of the second curve

    Returns:
    Tuple[List[float], List[float]]: intersections x and y coordinates.
    """
    assert len(x1) == len(y1)
    assert len(x2) == len(y2)
    print("intersection sizes", len(x1), len(x2), len(x1) * len(x2))

    def line_intersect(p1, p2, q1, q2):
        """Check if line segments p1p2 and q1q2 intersect."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    def segment_intersection(p1, p2, q1, q2):
        """Find intersection point of line segments p1p2 and q1q2 if they intersect."""

        def line(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0] * p2[1] - p2[0] * p1[1]
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            return None

        L1 = line(p1, p2)
        L2 = line(q1, q2)
        return intersection(L1, L2)

    intersections_x = []
    intersections_y = []
    for i in range(len(x1) - 1):
        for j in range(len(x2) - 1):
            p1, p2 = (x1[i], y1[i]), (x1[i + 1], y1[i + 1])
            q1, q2 = (x2[j], y2[j]), (x2[j + 1], y2[j + 1])
            if line_intersect(p1, p2, q1, q2):
                intersect_point = segment_intersection(p1, p2, q1, q2)
                if intersect_point:
                    intersections_x.append(intersect_point[0])
                    intersections_y.append(intersect_point[1])

    return intersections_x, intersections_y


if __name__ == "__main__":
    # Example usage
    x1 = np.random.rand(50)
    y1 = np.random.rand(50)
    x2 = np.random.rand(50)
    y2 = np.random.rand(50)

    import matplotlib.pyplot as plt

    plt.plot(x1, y1, color="orange")
    plt.plot(x2, y2, color="blue")

    x, y = curveintersect(x1, y1, x2, y2)
    print(x, len(x))
    plt.plot(x, y, "o", color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test/intersect.pdf")
