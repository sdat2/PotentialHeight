"""Functions to calculate potential size from xarray inputs in parallel."""

import os
from joblib import Parallel, delayed
import numpy as np
import xarray as xr
from sithom.io import read_json
from sithom.time import timeit
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

    # TODO: Implement bisection instead
    # TODO: Speed up?
    # TODO: Get parallization working well
    # TODO: Implement into potential intensity step?

    Args:
        ds (xr.Dataset): Dataset with the input values.
        supergradient_factor (float, optional): Supergradient. Defaults to 1.2.
        include_profile: (bool, optional)

    Returns:
        xr.Dataset: Find the solution dataset.

    Example:
        >>> in_ds = xr.Dataset(data_vars={
        ...     "msl": 1016.7, # mbar or hPa
        ...     "vmax": 49.5, # m/s, potential intensity
        ...     "sst": 28, # degC
        ...     "t0": 200, # degK
        ...     },
        ...     coords={"lat":28})
        >>> out_ds = point_solution(in_ds)
        >>> out_ds
    """
    near_surface_air_temperature = (
        ds["sst"].values + TEMP_0K - 1
    )  # Celsius Kelvin, subtract 1K for parameterization for near surface
    outflow_temperature = ds["t0"].values
    vmax = ds["vmax"].values
    coriolis_parameter = abs(coriolis_parameter_from_lat(ds["lat"].values))
    # assert coriolis_parameter > 0
    print("coriolis_parameter", coriolis_parameter)
    print("ds['lat'].values", ds["lat"].values)

    def try_for_r0(r0: float):
        pm_cle, rmax_cle, _, _ = run_cle15(
            plot=True,
            inputs={
                "r0": r0,
                "Vmax": vmax,
                "w_cool": W_COOL_DEFAULT,
                "fcor": coriolis_parameter,
                "p0": float(ds["msl"].values),  # in hPa / mbar
            },
        )

        ys = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=vmax * supergradient_factor,
                    coriolis_parameter=coriolis_parameter,
                    pressure_dry_at_inflow=ds["msl"].values
                    * 100  # 100 to convert from hPa to Pa
                    - buck_sat_vap_pressure(near_surface_air_temperature),
                    near_surface_air_temperature=near_surface_air_temperature,
                    outflow_temperature=outflow_temperature,
                )
            ),
            0.3,
            1.2,
            1e-6,  # threshold
        )
        # convert solution to pressure
        pm_car = (  # 100 to convert from hPa to Pa
            ds["msl"].values * 100 - buck_sat_vap_pressure(near_surface_air_temperature)
        ) / ys + buck_sat_vap_pressure(near_surface_air_temperature)

        return pm_cle - pm_car
        # print("r0, rmax_cle, pm_cle, pm_car", r0, rmax_cle, pm_cle, pm_car)

    r0 = bisection(
        try_for_r0, 200 * 1000, 5000 * 1000, 1e-3
    )  # find potential size \(r_a\) between 200 and 5000 km with 1e-3 m absolute tolerance

    pm_cle, rmax_cle, vmax, pc = run_cle15(
        inputs={"r0": r0, "Vmax": ds["vmax"].values}, plot=False
    )
    # read the solution
    ds["r0"] = r0
    ds["r0"].attrs = {
        "units": "m",
        "long_name": "Potential size of tropical cyclone, $r_a$",
    }
    ds["pm"] = pm_cle
    ds["pm"].attrs = {"units": "Pa", "long_name": "Pressure at maximum winds, $p_m$"}
    ds["pc"] = pc
    ds["pc"].attrs = {"units": "Pa", "long_name": "Central pressure for CLE15 profile"}
    ds["rmax"] = rmax_cle
    ds["rmax"].attrs = {
        "units": "m",
        "long_name": "Radius of maximum winds, $r_{\mathrm{max}}$",
    }
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
                ds["msl"].values * 100,  # [Pa] 100 to convert from hPa to Pa
                RHO_AIR_DEFAULT,  # [kg m-3]
                coriolis_parameter,  # [rad s-1]
            )
            / 100,
            {"units": "mbar", "long_name": "Surface Pressure"},
        )
    return ds


@timeit
def loop_through_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply point solution to all of the points in the dataset, using joblib to paralelize.

    Args:
        ds (xr.Dataset): contains msl, vmax, sst, t0 and lat.

    Returns:
        xr.Dataset: additionally contains r0, pm, pc, and rmax.
    """
    # the dataset might have a series of dimensions: time, lat, lon, member, etc.
    # it must have variables msl, vmax, sst, and t0, and coordinate lat for each point
    # we want to loop through these (in parallel) and apply the point_solution function
    # to each point, so that we end up with a new dataset with the same dimensions
    # that now has additional variables r0, pm, pc, rmax
    # we should call point_solution(ds_part, include_profile=False)
    # Step 1: Stack all spatial dimensions into a single dimension
    # ds.dims
    dims = list(ds.dims)

    def ps_skip(ids: xr.Dataset) -> xr.Dataset:
        try:
            assert not np.isnan(ids.vmax.values)  # did not converge
            assert not np.isnan(ids.sst.values)  # is not sea
            return point_solution(ids, include_profile=False)
        except Exception as e:
            print("Exception:", e)
            ids["pm"] = np.nan
            ids["pc"] = np.nan
            ids["r0"] = np.nan
            ids["rmax"] = np.nan
            return ids

    if len(dims) == 0:
        assert False
    elif len(dims) == 1:

        def process_point(index: int) -> xr.Dataset:
            """Apply point_solution to a single data point."""
            # return point_solution(ds_stacked.isel(stacked_dim=index), include_profile=False)
            return ps_skip(ds.isel({dims[0]: index}))

        print(f"About to conduct {ds.sizes[dims[0]]} jobs in parallel")
        results = Parallel(n_jobs=10)(
            delayed(process_point)(i) for i in range(ds.sizes[dims[0]])
        )
        return xr.concat(results, dim=dims[0])
    else:
        ds_stacked = ds.stack(stacked_dim=dims)

        def process_point(index: int) -> xr.Dataset:
            """Apply point_solution to a single data point."""
            # return point_solution(ds_stacked.isel(stacked_dim=index), include_profile=False)
            return ps_skip(ds_stacked.isel(stacked_dim=index))

        print(f"About to conduct {ds_stacked.sizes['stacked_dim']} jobs in parallel")
        # Step 2: Parallelize computation across all stacked points
        results = Parallel(n_jobs=10)(
            delayed(process_point)(i) for i in range(ds_stacked.sizes["stacked_dim"])
        )

        # Step 3: Reconstruct the original dataset dimensions
        return (
            xr.concat(results, dim="stacked_dim")
            .set_index(stacked_dim=dims)  # Restore multi-index before unstacking
            .unstack("stacked_dim")
        )


def convert_2d_coords_to_1d(ds: xr.Dataset) -> xr.Dataset:
    """
    Converts 2D coordinate variables (lon, lat) into 1D coordinates safely.

    Assumes:
    - `lon(y, x)` only varies along `x`, so we take the first row to get `lon(x)`.
    - `lat(y, x)` only varies along `y`, so we take the first column to get `lat(y)`.

    Args:
        ds (xr.Dataset): Input dataset with 2D coordinates.

    Returns:
        xr.Dataset: Dataset with 1D `lon(x)` and `lat(y)`, replacing the old ones.
    """
    # Extract 1D latitude and longitude
    lon_1d = ds["lon"].isel(y=0)  # Take the first row (constant across y)
    lat_1d = ds["lat"].isel(x=0)  # Take the first column (constant across x)

    # Drop the original 2D coordinates first
    ds = ds.drop_vars(["lon", "lat"])

    # Assign new 1D coordinates with the same names
    ds = ds.assign_coords(lon=("x", lon_1d.values), lat=("y", lat_1d.values))

    ds = ds.set_coords(["lon", "lat"])

    return ds


if __name__ == "__main__":
    # python -m cle.ps
    in_ds = xr.Dataset(
        data_vars={
            "msl": 1016.7,  # mbar or hPa
            "vmax": 49.5,  # m/s, potential intensity
            "sst": 28,  # degC
            "t0": 200,  # degK
        },
        coords={"lat": 28},  # degNorth
    )
    out_ds = point_solution(in_ds)
    # print(out_ds)
    # out_ds = loop_through_dimensions(in_ds)
    # print(out_ds)
    in_ds = xr.Dataset(
        data_vars={
            "msl": ("y", [1016.7, 1016.7]),  # mbar or hPa
            "vmax": ("y", [49.5, 49.5]),  # m/s, potential intensity
            "sst": ("y", [28, 28]),  # degC
            "t0": ("y", [200, 200]),  # degK
        },
        coords={"lat": ("y", [28, 29])},  # degNorth
    )
    out_ds = loop_through_dimensions(in_ds)
    print(out_ds)

    in_ds = xr.Dataset(
        data_vars={
            "msl": (("y", "x"), [[1016.7, 1016.7], [1016.7, 1016.7]]),  # mbar or hPa
            "vmax": (
                ("y", "x"),
                [[50, 51], [49, 49.5]],
            ),  # m/s, potential intensity
            "sst": (("y", "x"), [[29, 30], [28, 28]]),  # degC
            "t0": (("y", "x"), [[200, 200], [200, 200]]),  # degK
        },
        coords={"lat": (("y", "x"), [[30, 30], [45, 45]])},  # degNorth
    )
    print(in_ds)
    out_ds = loop_through_dimensions(in_ds)
    print(out_ds)
    ex_data_path = "/work/n02/n02/sdat2/adcirc-swan/worstsurge/data/cmip6/pi/ssp585/CESM2/r4i1p1f1.nc"
    in_ds = convert_2d_coords_to_1d(
        xr.open_dataset(ex_data_path)[["sst", "msl", "vmax", "t0"]]
    ).isel(
        time=7, y=slice(160, 260), x=slice(160, 300)
    )  # .sel(
    #    lon=slice(-100, -80), lat=slice(25, 35)
    # )  # get necessry inputs, get relevant box
    out_ds = loop_through_dimensions(in_ds)
    print(out_ds)
    # from tcpips.constants import DATA_PATH

    out_ds.to_netcdf(os.path.join(DATA_PATH, "example_potential_size_output.nc"))
    print(in_ds)
