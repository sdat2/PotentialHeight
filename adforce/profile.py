"""adforce/profile.py: Profile module for adforce."""
import numpy as np
import xarray as xr
from sithom.io import read_json


def pressures_profile(  # add pressure profile to wind profile
    rr: np.ndarray,  # radii [m]
    vv: np.ndarray,  # wind speeds [m/s]
    fcor: float = 5e-5,  # fcor
    p0: float = 1015 * 100,  # [Pa]
    rho0: float = 1.15,  # [kg m-3]
) -> np.ndarray:
    """
    Add pressure profile to wind profile.

    Args:
        rr (np.ndarray): Radii [m].
        vv (np.ndarray): Wind speeds [m/s].
        fcor (float, optional): Coriolis parameter. Defaults to 5e-5.
        p0 (float, optional): Background pressure. Defaults to 1015 * 100.
        rho0 (float, optional): Background density. Defaults to 1.15.

    Returns:
        np.ndarray: Pressures [hPa].
    """

    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    # could speed up but is very quick anyway
    p = np.zeros(rr.shape)  # [Pa]
    # rr ascending
    assert np.all(rr == np.sort(rr))
    p[-1] = p0
    for j in range(len(rr) - 1):
        i = -j - 2
        # Assume Coriolis force and pressure-gradient balance centripetal force.
        p[i] = p[i + 1] - rho0 * (
            vv[i] ** 2 / (rr[i + 1] / 2 + rr[i] / 2) + fcor * vv[i]
        ) * (rr[i + 1] - rr[i])
        # centripetal pushes out, pressure pushes inward, coriolis pushes inward

    return  p / 100  # pressures in hPa


def read_profile(profile_path: str) -> dict:
    """Read a TC profile from a JSON file."""
    chavas_profile = read_json(profile_path)
    rr = np.array(chavas_profile["rr"], dtype="float32")
    vv = np.array(chavas_profile["VV"], dtype="float32")
    chavas_profile = pressures_profile(rr, vv)
    radii = np.array(chavas_profile["rr"], dtype="float32")
    windspeeds = np.array(chavas_profile["VV"], dtype="float32")
    pressures = np.array(chavas_profile["p"], dtype="float32")
    return xr.Dataset(data_vars={"windspeeds": (["radii"], windspeeds, {"units": "m/s"}),
                                 "pressures": (["radii"], pressures, {"units": "hPa"})},
                        coords={"radii": (["radii"], radii, {"units": "m"})})
