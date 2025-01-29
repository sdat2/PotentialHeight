import os
import numpy as np
import xarray as xr
from sithom.io import read_json
from sithom.time import timeit
from chavas15.intersect import curveintersect
from .constants import (
    TEMP_0K,
    DATA_PATH,
    W_COOL_DEFAULT,
    RHO_AIR_DEFAULT,
    SUPERGRADIENT_FACTOR,
)
from .potential_size import run_cle15, wang_diff, wang_consts
from .utils import (
    coriolis_parameter_from_lat,
    buck_sat_vap_pressure,
    pressure_from_wind,
)
from .solve import bisection


@timeit
def point_solution(
    ds: xr.Dataset,
    supergradient_factor: float = SUPERGRADIENT_FACTOR,
    include_profile: bool = False,
) -> xr.Dataset:
    """
    Find the solution for a given point in the grid.

    Args:
        ds (xr.Dataset): Dataset with the input values.
        supergradient_factor (float, optional): Supergradient. Defaults to 1.2.
        include_profile: (bool, optional)

    Returns:
        xr.Dataset: Find the solution dataset.

    Example:
        >>> in_ds = xr.Dataset(data_vars={
        ...     "msl": 1016.7, # hpa
        ...     "vmax": 49.5, # m/s, potential intensity
        ...     "sst": 28, # degC
        ...     "t0": 200, # degK
        ...     },
        ...     coords={"lat":28})
        >>> out_ds = point_solution(in_ds)
    """
    r0s = np.linspace(200, 5000, num=30) * 1000  # [m] try different outer radii
    pcs = []
    pcw = []
    rmaxs = []
    near_surface_air_temperature = (
        ds["sst"].values + TEMP_0K - 1
    )  # Celsius Kelvin, subtract 1K for parameterization for near surface
    outflow_temperature = ds["t0"].values
    coriolis_parameter = coriolis_parameter_from_lat(ds["lat"].values)

    for r0 in r0s:
        pm_cle, rmax_cle, _, pc = run_cle15(
            plot=True,
            inputs={
                "r0": r0,
                "Vmax": ds["vmax"].values,
                "w_cool": W_COOL_DEFAULT,
                "fcor": coriolis_parameter,
                "p0": float(ds["msl"].values),
            },
        )

        ys = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=ds["vmax"].values * supergradient_factor,
                    coriolis_parameter=coriolis_parameter,
                    pressure_dry_at_inflow=ds["msl"].values * 100
                    - buck_sat_vap_pressure(near_surface_air_temperature),
                    near_surface_air_temperature=near_surface_air_temperature,
                    outflow_temperature=outflow_temperature,
                )
            ),
            0.9,
            1.2,
            1e-6,
        )
        # convert solution to pressure
        pm_car = (
            ds["msl"].values * 100 - buck_sat_vap_pressure(near_surface_air_temperature)
        ) / ys + buck_sat_vap_pressure(near_surface_air_temperature)

        pcs.append(pm_cle)
        pcw.append(pm_car)
        rmaxs.append(rmax_cle)
        print("r0, rmax_cle, pm_cle, pm_car", r0, rmax_cle, pm_cle, pm_car)
    pcs = np.array(pcs)
    pcw = np.array(pcw)
    rmaxs = np.array(rmaxs)
    intersect = curveintersect(r0s, pcs, r0s, pcw)

    pm_cle, rmax_cle, vmax, pc = run_cle15(
        inputs={"r0": intersect[0][0], "Vmax": ds["vmax"].values}, plot=False
    )
    # read the solution
    ds["r0"] = intersect[0][0]
    ds["r0"].attrs = {"units": "m", "long_name": "Outer radius of tropical cyclone"}
    ds["pm"] = intersect[1][0]
    ds["pm"].attrs = {"units": "Pa", "long_name": "Pressure at maximum winds"}
    ds["pc"] = pc
    ds["pc"].attrs = {"units": "Pa", "long_name": "Central pressure for CLE15 profile"}
    ds["rmax"] = rmax_cle
    ds["rmax"].attrs = {"units": "m", "long_name": "Radius of maximum winds"}
    if include_profile:
        # TODO: This optimal profile is reading the wrong data.
        # PV = NRT; Rho = NM/V; N/V = P/RT; assume isothermal -> Rho(P) = PM/RT, Rho1 = Rho2*P1/P2
        out = read_json(os.path.join(DATA_PATH, "outputs.json"))
        ds["radii"] = ("r", out["rr"], {"units": "m", "long_name": "Radius"})
        ds["velocities"] = (
            "r",
            out["VV"],
            {"units": "m s-1", "long_name": "Azimuthal Velocity"},
        )
        ds["pressures"] = (
            "r",
            pressure_from_wind(
                np.array(out["rr"]),  # [m]
                np.array(out["VV"]),  # [m/s]
                ds["msl"].values * 100,  # [Pa]
                RHO_AIR_DEFAULT,  # [kg m-3]
                coriolis_parameter,  # [rad s-1]
            )
            / 100,
            {"units": "mbar", "long_name": "Surface Pressure"},
        )
    return ds
