"""adforce/profile.py: Profile module for adforce."""

import numpy as np
import xarray as xr
from sithom.io import read_json
from cle.utils import pressure_from_wind


def pressures_profile(  # add pressure profile to wind profile
    rr: np.ndarray,  # radii [m]
    vv: np.ndarray,  # wind speeds [m/s]
    fcor: float = 5e-5,  # fcor [s-1] = 2 * omega * sin(lat)
    p0: float = 1015 * 100,  # [Pa]
    rho0: float = 1.15,  # [kg m-3]
) -> np.ndarray:
    """
    Add pressure profile to wind profile.

    Args:
        rr (np.ndarray): Radii [m].
        vv (np.ndarray): Wind speeds [m/s].
        fcor (float, optional): Coriolis parameter [s-1]. Defaults to 5e-5.
        p0 (float, optional): Background pressure [Pa]. Defaults to 1015 * 100.
        rho0 (float, optional): Background density [kg m-3]. Defaults to 1.15.

    Returns:
        np.ndarray: Pressures [hPa].
    """
    pressure_from_wind(rr, vv, p0, rho0, fcor, assumption="isothermal") / 100


def read_profile(profile_path: str) -> xr.Dataset:
    """Read a TC profile from a JSON file.

    Args:
        profile_path (str): Path to the JSON file.

    Returns:
        xr.Dataset: Dataset containing the wind speeds and pressures
            of the TC profile. The dataset has the following data variables:
            - windspeeds: Wind speeds [m/s].
            - pressures: Pressures [hPa].
            The dataset has the following coordinate:
            - radii: Radii [m].

    Raises:
        ValueError: If the profile does not contain 'rr' and 'VV' keys.

    Examples:
        >>> import os
        >>> from cle.constants import DATA_PATH
        >>> profile_ds = read_profile(os.path.join(DATA_PATH, "2025.json"))
        >>> assert "windspeeds" in profile_ds.data_vars
        >>> assert "pressures" in profile_ds.data_vars
        >>> assert "radii" in profile_ds.coords
    """
    chavas_profile = read_json(profile_path)

    if "rr" not in chavas_profile or "VV" not in chavas_profile:
        raise ValueError("The profile must contain 'rr' and 'VV' keys.")

    radii = np.array(chavas_profile["rr"], dtype="float32")
    windspeeds = np.array(chavas_profile["VV"], dtype="float32")

    if "p" not in chavas_profile:
        pressures = pressures_profile(radii, windspeeds)
    else:
        pressures = np.array(chavas_profile["p"], dtype="float32")

    return xr.Dataset(
        data_vars={
            "windspeeds": (["radii"], windspeeds, {"units": "m/s"}),
            "pressures": (["radii"], pressures, {"units": "hPa"}),
        },
        coords={"radii": (["radii"], radii, {"units": "m"})},
    )
