"""Potential Intensity Calculation script."""

from typing import Callable
import numpy as np
import xarray as xr
from tcpyPI import pi
from sithom.time import timeit, time_stamp
from sithom.misc import get_git_revision_hash
from .constants import PROJECT_PATH
from .xr_utils import standard_name_to_long_name


CKCD: float = 0.9  # Enthalpy exchange coefficient / drag coefficient [dimensionless]
PTOP: float = 50.0  # Top pressure level for the calculation [hPa]
KAPPA: float = 0.286  # R/cp for dry air [dimensionless]


def fix_temp_profile(
    ds: xr.Dataset,
    method: str = "lapse_rate",
    max_temp_c: float = 100.0,
) -> xr.Dataset:
    """
    Fills missing values in a vertical temperature profile using a principled method.

    Methods:
    1.  'interpolate': Fills internal NaNs using linear interpolation. Cannot fill
        NaNs at the top or bottom of the profile.
    2.  'lapse_rate': Fills NaNs at the bottom of the profile by extrapolating
        downwards from the lowest valid data point using a dry adiabatic lapse rate.
        This is ideal for fixing missing surface-level pressure data.

    Args:
        ds (xr.Dataset): Xarray dataset containing temperature 't' with a
                         vertical pressure coordinate 'p'.
        method (str, optional): The method to use: 'interpolate' or 'lapse_rate'.
                                Defaults to "lapse_rate".
        max_temp_c (float, optional): Plausibility check. Temperatures above this
                                      value (in Celsius) are set to NaN. Defaults to 100.0.

    Returns:
        xr.Dataset: Dataset with the temperature profile 't' filled.

    Example:
        >>> p_coords = [1000, 925, 850]
        >>> t_data = [[[np.nan, 290, 285], [305, 298, 292]], # Profile 1 has NaN at bottom
        ...           [[np.nan, np.nan, 280], [302, 296, 290]]] # Profile 2 has two NaNs at bottom
        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "t": (("x", "y", "p"), t_data, {"units": "K"}),
        ...     },
        ...     coords={
        ...         "p": (("p",), p_coords, {"units": "hPa"}),
        ...         "x": (("x",), [1, 2]), "y": (("y",), [1, 2])
        ...     },
        ... )
        >>> ds_fixed = fix_temp_profile(ds, method='lapse_rate')
        >>> # For profile 1: T_ref=290K at p_ref=925hPa. T(1000) = 290 * (1000/925)**0.286
        >>> expected_t1 = 290 * (1000/925)**KAPPA
        >>> # For profile 2: T_ref=280K at p_ref=850hPa. T(1000) = 280 * (1000/850)**0.286 etc.
        >>> expected_t2_925 = 280 * (925/850)**KAPPA
        >>> expected_t2_1000 = 280 * (1000/850)**KAPPA
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[0, 0, 0], expected_t1, decimal=2)
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[1, 0, 1], expected_t2_925, decimal=2)
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[1, 0, 0], expected_t2_1000, decimal=2)
    """
    ds["t"] = ds["t"].where(ds["t"] < (max_temp_c + 273.15), np.nan)

    if ds.p.attrs.get("units", "hPa").lower() in ["pa", "pascal"]:
        if ds.p.max() > 2000:  # Heuristic check if already in Pa
            p_hpa = ds.p / 100.0
    else:  # Assume hPa if not specified or specified otherwise
        p_hpa = ds.p

    if method == "interpolate":
        # Note: This will not fill NaNs at the boundaries of the dimension
        ds["t"] = ds["t"].interpolate_na(dim="p", method="linear")

    elif method == "lapse_rate":
        # Extrapolate downwards (from low pressure to high pressure)
        # bfill finds the next valid observation along the dimension
        t_ref = ds["t"].bfill(dim="p")
        # Create a p_ref array that matches the t_ref values
        p_ref = p_hpa.where(ds["t"].notnull()).bfill(dim="p")

        # Where t was originally NaN, calculate the new temperature
        # using the lapse rate formula from the reference level.
        extrapolated_t = t_ref * (p_hpa / p_ref) ** KAPPA
        ds["t"] = ds["t"].fillna(extrapolated_t)

    else:
        raise ValueError(f"Unknown method: {method}")

    return ds


@timeit
def calculate_pi(
    ds: xr.Dataset, dim: str = "p", fix_temp=False, V_reduc=1.0
) -> xr.Dataset:
    """Calculate the potential intensity using the tcpyPI package.

    Data must have been converted to the tcpyPI units by `tcpips.convert'.

    Args:
        ds (xr.Dataset): xarray dataset containing the necessary variables.
        dim (str, optional): Vertical dimension. Defaults to "p" for pressure level.
        fix_temp (bool, optional): Whether to fix the temperature profile. Defaults to True.
        V_reduc (float, optional): Reduction factor for the wind speed. Defaults to 1.0.


    Returns:
        xr.Dataset: xarray dataset containing the calculated variables.

    Example:
        >>> ds = xr.Dataset(data_vars={"sst": (["x", "y"], [[20, 30], [30, 32]],
        ...                                    {"units": "degrees_Celsius"}),
        ...                            "msl": (["x", "y"], [[1000, 1005], [1010, 1015]],
        ...                                    {"units": "hPa"}),
        ...                            "t": (["x", "y", "p"],
        ...                                  [[[np.nan, 23], [30, 21]], [[30, 21], [np.nan, 23]]],
        ...                                  {"units": "degrees_Celsius"}),
        ...                            "q": (["x", "y", "p"],
        ...                                  [[[10, 20], [30, 40]], [[50, 60], [70, 80]]],
        ...                                  {"units": "g/kg"})},
        ...                 coords={"x": (["x"], [-80, -85], {"units": "degrees_East"}),
        ...                         "y": (["y"], [20, 25], {"units": "degrees_North"}),
        ...                         "p": (["p"], [1000, 850], {"units": "hPa"})})
        >>> pi_ds = calculate_pi(ds, fix_temp=True) # doctest: +SKIP
    """
    if fix_temp:
        print("Fixing temperature profile...")
        print("Before fixing:", ds)
        ds = fix_temp_profile(ds)
        print("After fixing:", ds)

    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(
            CKCD=CKCD,
            ascent_flag=0,
            diss_flag=1,
            ptop=PTOP,
            miss_handle=1,
            V_reduc=V_reduc,
        ),
        input_core_dims=[
            [],
            [],
            [
                dim,
            ],
            [
                dim,
            ],
            [
                dim,
            ],
        ],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        # dask="allowed", # maybe this could work with dask, but at the moment I get an error
        dask="parallelized",
    )

    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    out_ds = xr.Dataset(
        {
            "vmax": vmax,  # maybe change to vp
            "pmin": pmin,
            "ifl": ifl,
            "t0": t0,
            "otl": otl,
        }
    )

    # add names and units to the structure
    out_ds.vmax.attrs["standard_name"], out_ds.vmax.attrs["units"] = (
        "Potential Intensity",
        "m/s",
    )
    out_ds.pmin.attrs["standard_name"], out_ds.pmin.attrs["units"] = (
        "Minimum Central Pressure",
        "hPa",
    )
    out_ds.ifl.attrs["standard_name"] = "tcpyPI Flag"
    out_ds.t0.attrs["standard_name"], out_ds.t0.attrs["units"] = (
        "Outflow Temperature",
        "K",
    )
    out_ds.otl.attrs["standard_name"], out_ds.otl.attrs["units"] = (
        "Outflow Temperature Level",
        "hPa",
    )
    out_ds.attrs["V_reduc"] = V_reduc
    out_ds.attrs["CKCD"] = CKCD
    out_ds.attrs["ptop"] = PTOP
    ds.attrs["pi_calculated_at_git_hash"] = get_git_revision_hash(
        path=str(PROJECT_PATH)
    )
    ds.attrs["pi_calculated_at_time"] = time_stamp()
    print("Calculated potential intensity:", out_ds)

    return standard_name_to_long_name(out_ds)


def simple_sensitivity(delta_t: float = 1,
                       k: float = 0.07,
                       m: float = 1,
                       avg_to_tropical_ocean: float = 0.8
                       ) -> None:
    """
    Simple sensitivity analysis for the change in potential intensity
    based on the change in temperature and humidity.

    Args:
        delta_t (float): Change in average global surface temperature in Kelvin.
        k (float): Scaling factor for humidity change based on CC (default 0.07).
        m (float): Scaling factor for the change in outflow temperature (default 1).
        avg_to_tropical_ocean (float): Average temperature change
           for tropical ocean compared to global temperature (default 0.8).
    """
    delta_t = (
        delta_t * avg_to_tropical_ocean
    )  # scale to tropical ocean average temperature change
    # let's start by doing a simple sensitivty for potential intensity
    T_s0 = 300  # K
    T_00 = 200  # K
    # approx as Bister 1998 and use 7% scaling for humidity CC per degree to scale enthalpy change
    # T_s = T_s0 + delta_t  # K
    # T_0 = T_00 + m* delta_t  # K

    vp1_div_vp0 = np.sqrt(
        (T_s0 - T_00 + (1 - m) * delta_t)
        / (T_00 + m * delta_t)
        * T_00
        / (T_s0 - T_00)
        * (1 + k * delta_t)
    )
    print(
        f"vp1/vp0 = {vp1_div_vp0:.3f} for delta_t = {delta_t} K, T_s0 = {T_s0} K, T_00 = {T_00} K, m = {m}, k = {k}"
    )
    print(f"fractional change = {(vp1_div_vp0 - 1)* 100:.3f}%")


# vp1/vp0 = 1.042 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = -1.7, k = 0.07
# fractional change = 4.221%
# vp1/vp0 = 1.038 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = -1, k = 0.07
# fractional change = 3.788%
# vp1/vp0 = 1.026 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1, k = 0.07
# fractional change = 2.557%
# vp1/vp0 = 1.024 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1.2, k = 0.07
# fractional change = 2.434%
# vp1/vp0 = 1.022 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1.5, k = 0.07
# fractional change = 2.250%


if __name__ == "__main__":
    # python -m tcpips.pi
    simple_sensitivity(1, k=0.07, m=-1.7)
    simple_sensitivity(1, k=0.07, m=-1)
    simple_sensitivity(1)
    simple_sensitivity(1, k=0.07, m=1.2)
    simple_sensitivity(1, k=0.07, m=1.5)
