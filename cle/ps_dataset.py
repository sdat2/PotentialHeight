"""
Create dataset for potential size / intensity.
"""

import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from chavas15.intersect import curveintersect
from tcpips.pi_old import gom_combined_inout_timestep_cmip6
from .constants import TEMP_0K, DATA_PATH, FIGURE_PATH, W_COOL_DEFAULT
from .potential_size import (
    run_cle15,
    wang_diff,
    wang_consts,
    vary_r0_c15,
    vary_r0_w22,
    find_solution_rmaxv,
)
from .utils import coriolis_parameter_from_lat, buck_sat_vap_pressure
from .solve import bisection


def find_solution_ds(
    ds: xr.Dataset, plot: bool = False, supergradient_factor: float = 1.2
) -> xr.Dataset:
    """
    Find the solution for a given dataset.

    Args:
        ds (xr.Dataset): Dataset with the input values.
        plot (bool, optional): Whether to plot. Defaults to False.
        supergradient_factor (float, optional): Supergradient. Defaults to 1.2.

    Returns:
        xr.Dataset: Find the solution dataset.
    """
    r0s = np.linspace(200, 5000, num=30) * 1000  # [m] try different outer radii
    pcs = []
    pcw = []
    rmaxs = []
    near_surface_air_temperature = ds["sst"].values + TEMP_0K - 1  # Kelvin
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
                    maximum_wind_speed=vmax * supergradient_factor,
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

    if plot:
        plt.plot(r0s / 1000, pcs / 100, "k", label="CLE15")
        plt.plot(r0s / 1000, pcw / 100, "r", label="W22")
        print("intersect", intersect)
        # plt.plot(intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution")
        plt.xlabel("Radius, $r_a$, [km]")
        plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
        if len(intersect) > 0:
            plt.plot(
                intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution"
            )
        plt.legend()
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pc_rmaxadj.pdf"))
        plt.clf()
        plt.plot(r0s / 1000, rmaxs / 1000, "k")
        plt.xlabel("Radius, $r_a$, [km]")
        plt.ylabel("Radius of maximum winds, $r_m$, [km]")
        plt.savefig(os.path.join(FIGURE_PATH, "r0_rmax.pdf"))
        plt.clf()
        # run the model with the solution

    pm_cle, rmax_cle, vmax, pc = run_cle15(
        inputs={"r0": intersect[0][0], "Vmax": ds["vmax"].values}, plot=True
    )
    # read the solution

    out = read_json(os.path.join(DATA_PATH, "outputs.json"))

    ds["r0"] = intersect[0][0]
    ds["r0"].attrs = {"units": "m", "long_name": "Outer radius of tropical cyclone"}
    ds["pm"] = intersect[1][0]
    ds["pm"].attrs = {"units": "Pa", "long_name": "Pressure at maximum winds"}
    ds["pc"] = pc
    ds["pc"].attrs = {"units": "Pa", "long_name": "Central pressure"}
    ds["rmax"] = rmax_cle
    ds["rmax"].attrs = {"units": "m", "long_name": "Radius of maximum winds"}
    ds["radii"] = ("r", out["rr"], {"units": "m", "long_name": "Radius"})
    ds["velocities"] = (
        "r",
        out["VV"],
        {"units": "m s-1", "long_name": "Azimuthal Velocity"},
    )

    pm_ds = xr.Dataset(
        data_vars={
            "pm_cle": ("r0s", pcs, {"units": "Pa"}),
            "pm_car": ("r0s", pcw, {"units": "Pa"}),
            "rmaxs": ("r0s", rmaxs, {"units": "m"}),
        },
        coords={"r0s": r0s},
    )

    return xr.merge([ds, pm_ds])


@timeit
def find_solution() -> None:
    """
    Find a single solution and plot the results.
    """

    # without rmax adjustment
    r0s = np.linspace(200, 5000, num=30) * 1000  # [m]

    pcs = vary_r0_c15(r0s)
    pcw = vary_r0_w22(r0s)

    plt.plot(r0s / 1000, pcs / 100, "k", label="CLE15")
    plt.plot(r0s / 1000, pcw / 100, "r", label="W22")
    intersect = curveintersect(r0s, pcs, r0s, pcw)
    plt.plot(intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution")
    plt.xlabel("Radius, $r_a$, [km]")
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.legend()
    plt.savefig(os.path.join(FIGURE_PATH, "r0_pc_joint.pdf"))
    plt.clf()
    run_cle15(inputs={"r0": intersect[0][0]}, plot=True)


def gom_timestep(time: str = "1850-09-15", plot: bool = False) -> np.ndarray:
    """
    Find the solution at the Gulf of Mexico point for a given time.

    Args:
        time (str, optional): Time. Defaults to "1850-09-15".
        plot (bool, optional): Whether to plot. Defaults to False.

    Returns:
        np.ndarray: Solution.
    """
    # find_solution()
    # find_solution_rmaxv()
    ds = gom_combined_inout_timestep_cmip6(time=time, verbose=True)
    print(ds)
    soln = find_solution_rmaxv(
        vmax_pi=ds["vmax"].values,
        outflow_temperature=ds["t0"].values,
        near_surface_air_temperature=ds["sst"].values + TEMP_0K - 1,
        coriolis_parameter=coriolis_parameter_from_lat(ds["lat"].values),
        background_pressure=ds["msl"].values * 100,
        plot=plot,
    )
    print(soln)
    return soln


@timeit
def calc_solns_for_times(
    num: int = 50, ds_name=os.path.join(DATA_PATH, "gom_solns.nc")
) -> None:
    """
    Calculate a timeseries of solutions for the GOM midpoint.

    Args:
        num (int, optional): How many timesteps to process. Defaults to 50.
        ds_name ([type], optional): Name of the dataset. Defaults to os.path.join(DATA_PATH, "gom_solns.nc").
    """
    print("NUM", num)
    solns = []
    # times = [1850, 1900, 1950, 2000, 2050, 2099]
    times = [int(x) for x in np.linspace(1850, 2099, num=num)]

    for time in [str(t) + "-08-15" for t in times]:
        solns += [gom_timestep(time=time, plot=False)]

    solns = np.array(solns)
    print("solns", solns)

    ds = xr.Dataset(
        data_vars={
            "r0": ("year", solns[:, 0]),
            "vmax": ("year", solns[:, 1]),
            "pm": ("year", solns[:, 2]),
        },
        coords={"year": times},
        attrs={"description": "GOM midpoints solutions"},
    )
    ds.to_netcdf(ds_name)

    print("NUM", num)


@timeit
def ds_solns(
    num: int = 10,
    verbose: bool = False,
    ds_name: str = os.path.join(DATA_PATH, "gom_soln_timeseries_.nc"),
) -> None:
    """
    Record all the details of the GOM solution for a range of times.

    Args:
        num (int, optional): Number of years to calculate for. Defaults to 50.
        verbose (bool, optional): Verbose. Defaults to False.
        ds_name (str, optional): Name of the dataset. Defaults to os.path.join(DATA_PATH, "gom_soln_timeseries.nc").

    """
    times = [int(x) for x in np.linspace(1850, 2099, num=num)]
    ds_list = []

    # Calculate for monthly average centered on August 15th each year.
    dates = [str(t) + "-08-15" for t in times]
    for i, date in enumerate(dates):
        print(i, date)
        ds = find_solution_ds(
            gom_combined_inout_timestep_cmip6(time=date, verbose=True), plot=True
        )
        ds = ds.expand_dims("time", axis=-1)
        ds.coords["time"] = ("time", [times[i]])
        ds = ds.rename({"r": str(f"r{i+1}")})  # radii can be different lengths
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim="time")
    if verbose:
        print(ds)
    ds.to_netcdf(ds_name)


if __name__ == "__main__":
    # python -m cle.create_ds
    ds_solns(
        num=200,
        verbose=True,
        ds_name=os.path.join(DATA_PATH, "gom_soln_timeseries_new.nc"),
    )
