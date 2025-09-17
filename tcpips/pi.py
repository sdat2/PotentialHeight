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
P0_HPA: float = 1000.0  # Reference pressure for potential temperature [hPa]


def reconstruct_profile_well_mixed(
    ds: xr.Dataset,
    temp_var: str = "t",
    hum_var: str = "q",
    p_coord: str = "p",
    max_temp_c: float = 100.0,
) -> xr.Dataset:
    """
    Reconstructs incomplete atmospheric profiles assuming a well-mixed boundary layer.

    This function fills missing values (NaNs) at the bottom of vertical temperature
    and humidity profiles. It operates on the principle that the planetary
    boundary layer is well-mixed, meaning potential temperature (theta) and
    specific humidity (q) are constant with height in this layer.

    The method proceeds as follows:
    1. For each vertical profile, it identifies the lowest valid data point (the
       reference level).
    2. It assumes that potential temperature and specific humidity are constant
       from this reference level down to the surface (higher pressures).
    3. It extrapolates these constant values downward to fill any NaNs.
    4. The constant potential temperature is then converted back to in-situ
       temperature at each pressure level.

    This approach is physically principled for climatological studies of the
    tropical atmosphere and ensures consistent treatment of both temperature and
    humidity, which is critical for accurate Potential Intensity (PI) calculations.

    Args:
        ds (xr.Dataset): Xarray dataset containing atmospheric profiles.
                         Must include temperature, humidity, and a pressure coordinate.
        temp_var (str): The name of the temperature variable in the dataset (e.g., 't', 'ta').
                        Units must be Kelvin.
        hum_var (str): The name of the specific humidity variable (e.g., 'q', 'hus').
                       Units should be kg/kg or dimensionless.
        p_coord (str): The name of the vertical pressure coordinate (e.g., 'p', 'plev').
                       Units must be hPa or Pa.
        max_temp_c (float): A plausibility check. Temperatures above this value (in Celsius)
                            are treated as missing data. Defaults to 100.0.

    Returns:
        xr.Dataset: A new dataset with the temperature and humidity profiles filled.
    """
    # --- 1. Input Validation and Preparation ---
    # Create a copy to avoid modifying the original dataset in place.
    ds_out = ds.copy(deep=True)

    # Perform a plausibility check on temperature, masking unrealistic values.
    ds_out[temp_var] = ds_out[temp_var].where(ds_out[temp_var] < (max_temp_c + 273.15))

    # Ensure pressure coordinate is in hPa for the calculation.
    p_hpa = ds_out[p_coord]
    if p_hpa.attrs.get("units", "hPa").lower() in ["pa", "pascal"]:
        # Check if values are already large (suggesting Pa) and convert.
        if p_hpa.max() > 2000:
            p_hpa = p_hpa / 100.0
            p_hpa.attrs["units"] = "hPa"

    # --- 2. Identify Reference Level and Values ---
    # The 'bfill' (backward fill) operation propagates the first valid value
    # from the bottom of the profile (low pressure) upwards (to high pressure),
    # effectively finding the reference value for each column.
    # This is applied to temperature, humidity, and the pressure at that level.

    # Find the reference temperature (T_ref) for each profile.
    t_ref = ds_out[temp_var].bfill(dim=p_coord)

    # Find the reference specific humidity (q_ref) for each profile.
    q_ref = ds_out[hum_var].bfill(dim=p_coord)

    # Find the reference pressure (p_ref) corresponding to the T_ref and q_ref values.
    # This is done by masking the pressure coordinate where temperature is valid,
    # then backward filling.
    p_ref = p_hpa.where(ds_out[temp_var].notnull()).bfill(dim=p_coord)

    # --- 3. Extrapolate Using Well-Mixed Assumption ---
    # Calculate the potential temperature (theta_ref) at the reference level.
    theta_ref = (t_ref) * (P0_HPA / p_ref) ** KAPPA

    # Extrapolate constant potential temperature downward.
    # For any level p_hpa, the extrapolated temperature T_new is calculated from theta_ref.
    extrapolated_t = theta_ref * (p_hpa / P0_HPA) ** KAPPA

    # The extrapolated humidity is simply the constant reference humidity.
    extrapolated_q = q_ref

    # --- 4. Fill Missing Values in the Dataset ---
    # Use the original NaN mask to fill only the cells that were initially missing.
    ds_out[temp_var] = ds_out[temp_var].fillna(extrapolated_t)
    ds_out[hum_var] = ds_out[hum_var].fillna(extrapolated_q)

    return ds_out


def fix_profile(
    ds: xr.Dataset,
    method: str = "lapse_rate",
    max_temp_c: float = 100.0,
) -> xr.Dataset:
    """
    Fills missing values in a vertical temperature and specific humdity profile using a principled method.

    Methods:
    1.  'interpolate': Fills internal NaNs using linear interpolation. Cannot fill
        NaNs at the top or bottom of the profile.
    2.  'lapse_rate': Fills NaNs at the bottom of the profile by extrapolating
        downwards from the lowest valid data point using a dry adiabatic lapse rate.
        This is ideal for fixing missing surface-level pressure data.
    3.  'well_mixed': Fills NaNs at the bottom of the profile by assuming a well-mixed
        boundary layer. This method fills both temperature and specific humidity
        profiles consistently and is physically principled for tropical atmospheres.

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
        >>> ds_fixed = fix_profile(ds, method='lapse_rate')
        >>> # For profile 1: T_ref=290K at p_ref=925hPa. T(1000) = 290 * (1000/925)**0.286
        >>> expected_t1 = 290 * (1000/925)**KAPPA
        >>> # For profile 2: T_ref=280K at p_ref=850hPa. T(1000) = 280 * (1000/850)**0.286 etc.
        >>> expected_t2_925 = 280 * (925/850)**KAPPA
        >>> expected_t2_1000 = 280 * (1000/850)**KAPPA
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[0, 0, 0], expected_t1, decimal=2)
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[1, 0, 1], expected_t2_925, decimal=2)
        >>> np.testing.assert_almost_equal(ds_fixed.t.values[1, 0, 0], expected_t2_1000, decimal=2)
    """
    ds = ds.copy(deep=True)  # Avoid modifying the original dataset in place.

    if ds.p.attrs.get("units", "hPa").lower() in ["pa", "pascal"]:
        if ds.p.max() > 2000:  # Heuristic check if already in Pa
            p_hpa = ds.p / 100.0
    else:  # Assume hPa if not specified or specified otherwise
        p_hpa = ds.p

    # Check if temperature is in Celsius and convert to Kelvin if needed
    if "units" not in ds.t.attrs or ds.t.attrs["units"].lower() in [
        "celsius",
        "degrees_celsius",
        "degc",
    ]:
        in_celsius = True
        ds["t"] = ds["t"] + 273.15  # Convert to Kelvin
    else:
        in_celsius = False

    ds["t"] = ds["t"].where(ds["t"] < (max_temp_c + 273.15), np.nan)

    ds["t"] = ds["t"].where(ds["t"] > 0.0, np.nan)  # Remove non-physical temps

    # print("t", ds.t.min().values, ds.t.max().values)

    if method == "interpolate":
        # 1. Keep a reference to the original p coordinate
        original_p_coord = ds.coords["p"]

        # 2. Sort the dataset by 'p' to allow for interpolation
        ds_sorted = ds.sortby("p")

        # 3. Perform the interpolation on the sorted dataset
        t_interpolated = ds_sorted["t"].interpolate_na(dim="p", method="linear")
        ds_sorted["t"] = t_interpolated

        # 4. Also perform this for specific humidity.
        print("Interpolating specific humidity 'q' as well.")
        q_interpolated = ds_sorted["q"].interpolate_na(dim="p")
        ds_sorted["q"] = q_interpolated

        # 4. Reindex the dataset back to the original coordinate order
        ds = ds_sorted.reindex(p=original_p_coord)

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

    elif method == "well_mixed":
        ds = reconstruct_profile_well_mixed(ds, temp_var="t", hum_var="q", p_coord="p")

    else:
        raise ValueError(f"Unknown method: {method}")

    if in_celsius:
        ds["t"] = ds["t"] - 273.15  # Convert back to Celsius if needed

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
        ds = fix_profile(ds)
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
            miss_handle=0,  # 1,
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
    # print("Calculated potential intensity:", out_ds)

    return standard_name_to_long_name(out_ds)
