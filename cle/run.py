"""Run the CLE15 model with json files."""
from typing import Callable, Tuple, Optional, Dict, List
import os
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from matplotlib import pyplot as plt
from sithom.io import read_json, write_json
from sithom.plot import plot_defaults, pairplot, label_subplots
from sithom.time import timeit
from chavas15.intersect import curveintersect
from tcpips.pi import get_gom, get_gom_bbox


plot_defaults()

BACKGROUND_PRESSURE = 1015 * 100  # [Pa]
TEMP_0K = 273.15  # [K]
DEFAULT_SURF_TEMP = 299  # [K]


@timeit
def run_cle15(
    execute: bool = True, plot: bool = False, inputs: Optional[Dict[str, any]] = None
) -> Tuple[float, float, float, float]:  # pm, rmax, vmax, pc
    """
    Run the CLE15 model.

    Args:
        execute (bool, optional): Execute the model. Defaults to True.
        plot (bool, optional): Plot the output. Defaults to False.
        inputs (Optional[Dict[str, any]], optional): Input parameters. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: pm, rmax, vmax, pc
    """
    ins = read_json("inputs.json")
    if inputs is not None:
        for key in inputs:
            if key in ins:
                ins[key] = inputs[key]

    write_json(ins, "inputs.json")

    # Storm parameters

    # run octave file r0_pm.m
    if execute:
        # disabling gui leads to one order of magnitude speedup
        # also the pop-up window makes me feel sick due to the screen moving about.
        os.system("octave --no-gui --no-gui-libs r0_pm.m")

    # read in the output from r0_pm.m
    ou = read_json("outputs.json")

    if plot:
        # print(ou)
        # plot the output
        rr = np.array(ou["rr"]) / 1000
        rmerge = ou["rmerge"] / 1000
        vv = np.array(ou["VV"])
        plt.plot(rr[rr < rmerge], vv[rr < rmerge], "g", label="ER11 inner profile")
        plt.plot(rr[rr > rmerge], vv[rr > rmerge], "orange", label="E04 outer profile")
        plt.plot(
            ou["rmax"] / 1000, ins["Vmax"], "b.", label="$r_{\mathrm{max}}$ (output)"
        )
        plt.plot(
            ou["rmerge"] / 1000,
            ou["Vmerge"],
            "kx",
            label="$r_{\mathrm{merge}}$ (input)",
        )
        plt.plot(ins["r0"] / 1000, 0, "r.", label="$r_a$ (input)")

        def _f(x: any) -> float:
            if isinstance(x, float):
                return x
            else:
                return np.nan

        plt.ylim(
            [0, np.nanmax([_f(v) for v in vv]) * 1.10]
        )  # np.nanmax(out["VV"]) * 1.05])
        plt.legend()
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Rotating wind speed, $V$, [m s$^{-1}$]")
        plt.title("CLE15 Wind Profile")
        plt.savefig("r0_pm.pdf", format="pdf")
        plt.clf()

    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    p0 = ins["p0"] * 100  # [Pa]
    rho0 = 1.15  # [kg m-3]
    rr = np.array(ou["rr"])
    vv = np.array(ou["VV"])
    p = np.zeros(rr.shape)
    # rr ascending
    assert np.all(rr == np.sort(rr))
    p[-1] = p0
    for j in range(len(rr) - 1):
        i = -j - 2
        # Assume Coriolis force and pressure-gradient balance centripetal force.
        p[i] = p[i + 1] - rho0 * (
            vv[i] ** 2 / (rr[i + 1] / 2 + rr[i] / 2) + ins["fcor"] * vv[i]
        ) * (rr[i + 1] - rr[i])
        # centripetal pushes out, pressure pushes inward, coriolis pushes inward

    if plot:
        plt.plot(rr / 1000, p / 100, "k")
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Pressure, $p$, [hPa]")
        plt.title("CLE15 Pressure Profile")
        plt.ylim([np.min(p) / 100, np.max(p) * 1.0005 / 100])
        plt.xlim([0, rr[-1] / 1000])
        plt.savefig("r0_pmp.pdf", format="pdf")
        plt.clf()

    # plot the pressure profile

    return (
        interp1d(rr, p)(
            ou["rmax"]
        ),  # find the pressure at the maximum wind speed radius [Pa]
        ou["rmax"],  # rmax radius [m]
        ins["Vmax"],  # maximum wind speed [m/s]
        p[0],
    )  # p[0]  # central pressure [Pa]


def wang_diff(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:  # y = exp(a*y + b*log(y)*y + c)
        return y - np.exp(a * y + b * np.log(y) * y + c)

    return f


@timeit
def bisection(f: Callable, left: float, right: float, tol: float) -> float:
    """
    Bisection method.

    https://en.wikipedia.org/wiki/Root-finding_algorithms#Bisection_method

    Args:
        f (Callable): Function to find root of.
        left (float): Left boundary.
        right (float): Right boundary.
        tol (float): tolerance for convergence.

    Returns:
        float: x such that |f(x)| < tol.
    """
    fleft = f(left)
    fright = f(right)
    if fleft * fright > 0:
        print("Error: f(left) and f(right) must have opposite signs.")
        return np.nan

    while fleft * fright < 0 and right - left > tol:
        mid = (left + right) / 2
        fmid = f(mid)
        if fleft * fmid < 0:
            right = mid
            fright = fmid
        else:
            left = mid
            fleft = fmid
    return (left + right) / 2


def buck(temp: float) -> float:  # temp in K -> saturation vapour pressure in Pa
    """
    Arden Buck equation.

    https://en.wikipedia.org/wiki/Arden_Buck_equation

    Args:
        temp (float): temperature in Kelvin.

    Returns:
        float: saturation vapour pressure in Pa.
    """
    # https://en.wikipedia.org/wiki/Arden_Buck_equation
    temp = temp - TEMP_0K
    return 0.61121 * np.exp((18.678 - temp / 234.5) * (temp / (257.14 + temp))) * 1000


def carnot(temp_hot: float, temp_cold: float) -> float:
    """
    Calculate carnot factor.

    Args:
        temp_hot (float): Temperature of hot reservoir [K].
        temp_cold (float): Temperature of cold reservoir [K].

    Returns:
        float: Carnot factor [dimensionless].
    """
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
    """
    return v * r + 0.5 * f * r**2


def wang_consts(
    near_surface_air_temperature: float = 299,  # K
    outflow_temperature: float = 200,  # K
    latent_heat_of_vaporization: float = 2.27e6,  # J/kg
    gas_constant_for_water_vapor: float = 461,  # J/kg/K
    gas_constant: float = 287,  # J/kg/K
    beta_lift_parameterization: float = 1.25,  # dimensionless
    efficiency_relative_to_carnot: float = 0.5,
    pressure_dry_at_inflow: float = 985 * 100,
    coriolis_parameter: float = 5e-5,
    maximum_wind_speed: float = 83,
    radius_of_inflow: float = 2193 * 1000,
    radius_of_max_wind: float = 64 * 1000,
) -> Tuple[float, float, float]:
    # a, b, c
    absolute_angular_momentum_at_vmax = absolute_angular_momentum(
        maximum_wind_speed, radius_of_max_wind, coriolis_parameter
    )
    carnot_efficiency = carnot(near_surface_air_temperature, outflow_temperature)
    near_surface_saturation_vapour_presure = buck(near_surface_air_temperature)

    return (
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            * (
                efficiency_relative_to_carnot
                * carnot_efficiency
                * latent_heat_of_vaporization
                / gas_constant_for_water_vapor
                / near_surface_air_temperature
                - 1
            )
            / (
                (
                    beta_lift_parameterization
                    - efficiency_relative_to_carnot * carnot_efficiency
                )
            )
        ),
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            / (
                beta_lift_parameterization
                - efficiency_relative_to_carnot * carnot_efficiency
            )
        ),
        (
            beta_lift_parameterization
            * (
                0.5 * maximum_wind_speed**2
                - 0.25 * coriolis_parameter**2 * radius_of_inflow**2
                + 0.5 * coriolis_parameter * absolute_angular_momentum_at_vmax
            )
            / (
                (
                    beta_lift_parameterization
                    - efficiency_relative_to_carnot * carnot_efficiency
                )
                * near_surface_air_temperature
                * gas_constant
            )
        ),
    )


def vary_r0_c15(r0s: np.ndarray) -> np.ndarray:
    pcs = np.array([run_cle15(plot=False, inputs={"r0": r0})[0] for r0 in r0s])
    plt.plot(r0s / 1000, pcs / 100, "k")
    plt.xlabel("Radius, $r_a$, [km]")
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.savefig("r0_pc.pdf")
    plt.clf()
    return pcs


def vary_r0_w22(r0s: np.ndarray) -> np.ndarray:
    ys = np.array(
        [
            bisection(wang_diff(*wang_consts(radius_of_inflow=r0)), 0.3, 1.2, 1e-6)
            for r0 in r0s
        ]
    )
    pms = (BACKGROUND_PRESSURE - buck(DEFAULT_SURF_TEMP)) / ys + buck(DEFAULT_SURF_TEMP)
    plt.plot(r0s / 1000, np.array(pms) / 100, "r")
    plt.xlabel("Radius, $r_a$, [km]")
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.savefig("r0_pc_wang.pdf")
    plt.clf()
    return pms


def find_solution_rmaxv(
    vmax_pi: float = 86,  # m/s
    coriolis_parameter: float = 5e-5,  # s-1
    background_pressure: float = BACKGROUND_PRESSURE,  # Pa
    near_surface_air_temperature: float = DEFAULT_SURF_TEMP,  # K
    w_cool: float = 0.002,
    outflow_temperature: float = 200,  # K
    supergradient_factor: float = 1.2,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r0s = np.linspace(200, 5000, num=30) * 1000
    pcs = []
    pcw = []
    rmaxs = []
    for r0 in r0s:
        pm_cle, rmax_cle, vmax, pc = run_cle15(
            plot=False,
            inputs={
                "r0": r0,
                "Vmax": vmax_pi,
                "w_cool": w_cool,
                "fcor": coriolis_parameter,
                "p0": background_pressure / 100,
            },
        )

        ys = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=vmax * supergradient_factor,
                    coriolis_parameter=coriolis_parameter,
                    pressure_dry_at_inflow=background_pressure
                    - buck(near_surface_air_temperature),
                    near_surface_air_temperature=near_surface_air_temperature,
                    outflow_temperature=outflow_temperature,
                )
            ),
            0.9,
            1.2,
            1e-6,
        )
        # convert solution to pressure
        pm_car = (background_pressure - buck(near_surface_air_temperature)) / ys + buck(
            near_surface_air_temperature
        )

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
        plt.savefig("r0_pc_rmaxadj.pdf")
        plt.clf()
        plt.plot(r0s / 1000, rmaxs / 1000, "k")
        plt.xlabel("Radius, $r_a$, [km]")
        plt.ylabel("Radius of maximum winds, $r_m$, [km]")
        plt.savefig("r0_rmax.pdf")
        plt.clf()
        run_cle15(inputs={"r0": intersect[0][0], "Vmax": vmax_pi}, plot=True)
    return intersect[0][0], vmax_pi, intersect[1][0]


def find_solution_ds(
    ds: xr.Dataset, plot: bool = False, supergradient_factor=1.2
) -> xr.Dataset:
    r0s = np.linspace(200, 5000, num=30) * 1000  # [m] try different outer radii
    pcs = []
    pcw = []
    rmaxs = []
    near_surface_air_temperature = ds["sst"].values + TEMP_0K - 1
    outflow_temperature = ds["t0"].values
    coriolis_parameter = 2 * 7.2921e-5 * np.sin(np.deg2rad(ds["lat"].values))

    for r0 in r0s:
        pm_cle, rmax_cle, vmax, pc = run_cle15(
            plot=True,
            inputs={
                "r0": r0,
                "Vmax": ds["vmax"].values,
                "w_cool": 0.002,
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
                    - buck(near_surface_air_temperature),
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
            ds["msl"].values * 100 - buck(near_surface_air_temperature)
        ) / ys + buck(near_surface_air_temperature)

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
        plt.savefig("r0_pc_rmaxadj.pdf")
        plt.clf()
        plt.plot(r0s / 1000, rmaxs / 1000, "k")
        plt.xlabel("Radius, $r_a$, [km]")
        plt.ylabel("Radius of maximum winds, $r_m$, [km]")
        plt.savefig("r0_rmax.pdf")
        plt.clf()
        # run the model with the solution

    pm_cle, rmax_cle, vmax, pc = run_cle15(
        inputs={"r0": intersect[0][0], "Vmax": ds["vmax"].values}, plot=True
    )
    # read the solution

    out = read_json("outputs.json")

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
def find_solution():
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
    plt.savefig("r0_pc_joint.pdf")
    plt.clf()
    run_cle15(inputs={"r0": intersect[0][0]}, plot=True)


def gom_time(time: str = "1850-09-15", plot: bool = False) -> np.ndarray:
    # find_solution()
    # find_solution_rmaxv()
    ds = get_gom(time=time, verbose=True)
    print(ds)
    soln = find_solution_rmaxv(
        vmax_pi=ds["vmax"].values,
        outflow_temperature=ds["t0"].values,
        near_surface_air_temperature=ds["sst"].values + TEMP_0K - 1,
        coriolis_parameter=2 * 7.2921e-5 * np.sin(np.deg2rad(ds["lat"].values)),
        background_pressure=ds["msl"].values * 100,
        plot=plot,
    )
    print(soln)
    return soln


def plot_and_calc_gom() -> None:
    solns = []
    # times = [1850, 1900, 1950, 2000, 2050, 2099]
    times = [int(x) for x in range(1850, 2100, 20)]

    for time in [str(t) + "-08-15" for t in times]:
        solns += [gom_time(time=time, plot=False)]

    solns = np.array(solns)
    print("solns", solns)
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(times, solns[:, 0] / 1000, "k")
    axs[1].plot(times, solns[:, 1], "k")
    axs[2].plot(times, solns[:, 2] / 100, "k")
    plt.xlabel("Year")
    axs[0].set_ylabel("Radius of outer winds, $r_a$, [km]")
    axs[1].set_ylabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    axs[2].set_ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.savefig("rmax_time.pdf")


@timeit
def calc_solns_for_times(num: int = 50) -> None:
    """
    Calculate a timeseries of solutions for the GOM midpoint.

    Args:
        num (int, optional): How many timesteps to process. Defaults to 50.
    """
    print("NUM", num)
    solns = []
    # times = [1850, 1900, 1950, 2000, 2050, 2099]
    times = [int(x) for x in np.linspace(1850, 2099, num=num)]

    for time in [str(t) + "-08-15" for t in times]:
        solns += [gom_time(time=time, plot=False)]

    solns = np.array(solns)
    print("solns", solns)

    ds = xr.Dataset(
        data_vars={
            "r0": ("year", solns[:, 0]),
            "vmax": ("year", solns[:, 1]),
            "pm": ("year", solns[:, 2]),
        },
        coords={"year": times},
    )
    ds.to_netcdf("gom_solns.nc")

    print("NUM", num)


@timeit
def plot_gom_solns() -> None:
    ds = xr.open_dataset("gom_solns.nc")
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(ds["year"], ds["r0"] / 1000, "k")
    axs[1].plot(ds["year"], ds["vmax"], "k")
    axs[2].plot(ds["year"], ds["pm"] / 100, "k")
    plt.xlabel("Year")
    axs[0].set_ylabel("Radius of outer winds, $r_a$, [km]")
    axs[1].set_ylabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    axs[2].set_ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.savefig("rmax_time.pdf")


@timeit
def ds_solns(
    num: int = 10, verbose: bool = False, ds_name: str = "gom_soln_new.nc"
) -> None:
    """
    Record all the details of the GOM solution for a range of times.

    Args:
        num (int, optional): Number of years to calculate for. Defaults to 50.
        verbose (bool, optional): Verbose. Defaults to False.
    """
    times = [int(x) for x in np.linspace(1850, 2099, num=num)]
    ds_list = []

    # Calculate for monthly average centered on August 15th each year.
    dates = [str(t) + "-08-15" for t in times]
    for i, date in enumerate(dates):
        print(i, date)
        ds = find_solution_ds(get_gom(time=date, verbose=True), plot=True)
        ds = ds.expand_dims("time", axis=-1)
        ds.coords["time"] = ("time", [times[i]])
        ds = ds.rename({"r": str(f"r{i+1}")})  # radii can be different lengths
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim="time")
    if verbose:
        print(ds)
    ds.to_netcdf(ds_name)


def plot_from_ds(ds_name: str = "gom_soln_new.nc") -> None:
    """
    Plot the relationships between the variables.
    """
    ds = xr.open_dataset(ds_name)
    folder = "sup"
    print("ds", ds)
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(ds["time"], ds["r0"] / 1000, "k")
    axs[1].plot(ds["time"], ds["vmax"], "k")
    axs[2].plot(ds["time"], ds["pm"] / 100, "k")
    plt.xlabel("Year")
    axs[0].set_ylabel("Radius of outer winds, $r_a$, [km]")
    axs[1].set_ylabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    axs[2].set_ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.savefig(folder + "/rmax_time_new.pdf")
    plt.clf()

    im = plt.scatter(
        ds["vmax"].values, ds["r0"].values / 1000, c=ds["time"], marker="x"
    )
    plt.colorbar(im, label="Year", shrink=0.5)
    plt.xlabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    plt.ylabel("Radius of outer winds, $r_a$, [km]")
    plt.savefig("rmax_vmax.pdf")
    plt.clf()
    ds["year"] = ("time", ds["time"].values)
    vars: List[str] = ["r0", "vmax", "pm", "sst", "msl", "t0", "year"]  # , "time"]

    pairplot(ds[vars].to_dataframe()[vars])
    plt.savefig(folder + "/pairplot.pdf")
    plt.clf()

    vars = ["t", "q", "vmax", "r0", "pm", "t0", "year"]
    pairplot(ds.isel(p=0)[vars].to_dataframe()[vars])
    plt.savefig(folder + "/pairplot2.pdf")
    plt.clf()

    # do a line plot of sst, vmax, and r0
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(ds["time"], ds["sst"], "k")
    axs[1].plot(ds["time"], ds["vmax"], "k")
    axs[2].plot(ds["time"], ds["r0"] / 1000, "k")
    plt.xlabel("Year")
    axs[0].set_ylabel("Sea surface temperature, $T_s$, [$^\circ$C]")
    axs[1].set_ylabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    axs[2].set_ylabel("Radius of outer winds, $r_a$, [km]")
    plt.savefig(folder + "/sst_vmax_rmax.pdf")

    # do a line plot of carnot factor, vmax, and r0
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    ds["carnot"] = ("time", 1 / carnot(ds["sst"].values + TEMP_0K, ds["t0"].values))
    axs[0].plot(ds["time"], ds["carnot"], "k")
    axs[1].plot(ds["time"], ds["vmax"], "k")
    axs[2].plot(ds["time"], ds["r0"] / 1000, "k")
    plt.xlabel("Year")
    axs[0].set_ylabel(
        r"Carnot factor, $\eta_c$, $\frac{T_h}{T_h- T_c}$ [dimensionless]"
    )
    axs[1].set_ylabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    axs[2].set_ylabel("Radius of outer winds, $r_a$, [km]")
    plt.savefig(folder + "/carnot_vmax_rmax.pdf")

    # pairplot of carnot factor, vmax, and r0
    vars = ["carnot", "vmax", "r0", "year"]
    pairplot(ds[vars].to_dataframe()[vars])
    plt.savefig(folder + "/pairplot3.pdf")
    plt.clf()


def plot_soln_curves(ds_name: str = "gom_soln_new.nc") -> None:
    """
    Plot the solution curves for different.

    Args:
        ds_name (str, optional): Defaults to "gom_soln_new.nc".
    """
    plot_defaults()
    ds = xr.open_dataset(ds_name)
    folder = "sup"
    print("ds", ds)
    for time in range(len(ds.time.values)):
        dst = ds.isel(time=time)
        plt.plot(
            dst["r0s"] / 1000,
            dst["pm_cle"] / 100,
            "b",
            label="CLE15 Dynamics",
            alpha=0.5,
            linewidth=0.5,
        )
        plt.plot(
            dst["r0s"] / 1000,
            dst["pm_car"] / 100,
            "r",
            alpha=0.5,
            label="W22 Thermodynamics",
            linewidth=0.5,
        )

        if time == 0:
            plt.legend()

    im = plt.scatter(
        ds["r0"] / 1000,
        ds["pm"] / 100,
        c=ds["time"],
        marker="x",
        label="Solution",
        linewidth=0.5,
        zorder=100,
    )
    plt.colorbar(im, label="Year", shrink=0.5)

    plt.xlabel("Outer Radius of Tropical Cyclone, $r_a$, [km]")
    plt.xlim([1000, 3000])
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.ylim([875, 1000])
    plt.savefig(folder + "/r0_solns.pdf")
    plt.clf()

    im = plt.scatter(ds.vmax, ds.r0 / 1000, c=ds.pm / 100, marker="x", linewidth=0.5)
    plt.colorbar(im, label="Pressure at maximum winds, $p_m$, [hPa]", shrink=0.5)
    plt.xlabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    plt.ylabel("Outer Radius of Tropical Cyclone, $r_a$, [km]")
    plt.savefig(folder + "/r0_vmax_pm.pdf")
    plt.clf()

    im = plt.scatter(ds.vmax, ds.r0 / 1000, c=ds.time, marker="x", linewidth=0.5)
    plt.colorbar(im, label="Year", shrink=0.5)
    plt.xlabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    plt.ylabel("Outer Radius of Tropical Cyclone, $r_a$, [km]")
    plt.savefig(folder + "/r0_vmax_time.pdf")
    plt.clf()

    im = plt.scatter(ds.vmax, ds.pm / 100, c=ds.r0, marker="x", linewidth=0.5)
    plt.colorbar(im, label="Outer Radius of Tropical Cyclone, $r_a$, [km]", shrink=0.5)
    plt.xlabel("Maximum wind speed, $V_{\mathrm{max}}$, [m s$^{-1}$]")
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    # add in pearson correlation coefficient
    from scipy.stats import pearsonr

    r = pearsonr(ds.vmax, ds.pm / 100)[0]
    plt.title(r"$\rho$ " + f"= {r:.2f}")
    plt.savefig(folder + "/pm_vmax_r0.pdf")
    plt.clf()


def plot_profiles(ds_name: str = "gom_soln_2.nc") -> None:
    """
    Plot the azimuthal wind profiles for different times.

    Args:
        ds_name (str, optional): _description_. Defaults to "gom_soln_2.nc".
    """
    plot_defaults()
    ds = xr.open_dataset(ds_name)
    folder = "sup"
    print("ds", ds)
    for time in range(len(ds.time.values)):
        dst = ds.isel(time=time)
        plt.plot(
            dst["radii"] / 1000,
            dst["velocities"],
            alpha=0.5,
            linewidth=0.5,
            # label=f"{[1850, 2099][time]}",
        )

    plt.legend()
    plt.xlabel("Radius, $r$, [km]")
    plt.ylabel("Wind speed, $V$, [m s$^{-1}$]")
    plt.savefig(folder + "/profiles.pdf")
    plt.clf()


def plot_gom_bbox() -> None:
    """Try and calculate the solution for the GOM bbox."""
    plot_defaults()

    ds = get_gom_bbox(time="2015-01-15", pad=10)
    folder = "sup"
    print(ds)
    ds.vmax.plot()
    plt.xlabel(r"Longitude, $\lambda$, [$^\circ$]")
    plt.ylabel(r"Latitude, $\phi$, [$^\circ$]")
    plt.savefig(folder + "/gom_bbox_pi.pdf")
    plt.clf()
    ds.sst.plot()
    plt.xlabel(r"Longitude, $\lambda$, [$^\circ$]")
    plt.ylabel(r"Latitude, $\phi$, [$^\circ$]")
    plt.savefig(folder + "/gom_bbox_sst.pdf")
    plt.clf()

    new_var = [
        "rmax",
        "pm",
        "pc",
        "r0",
    ]

    ds_list = []

    for i in range(len(ds.lat.values)):
        ds_list_lon = []
        for j in range(len(ds.lon.values)):
            print(i, j)  # zoom through grid.
            dsp = ds.isel(lat=i, lon=j)

            def add_nan() -> None:
                nonlocal ds_list_lon
                nonlocal dsp
                for var in new_var:
                    dsp[var] = np.nan
                ds_list_lon.append(dsp)

            def add_rmax() -> None:
                nonlocal ds_list_lon
                nonlocal dsp
                ds2 = find_solution_ds(dsp, plot=False)[new_var + [var for var in dsp]]
                for var in new_var:
                    dsp[var] = ds2[var]
                print("dsp", dsp)
                print([var for var in dsp])
                del dsp["radii"]
                del dsp["velocities"]
                ds_list_lon.append(dsp)

            # print(dsp)
            if True:
                # ok, let's just try it for one example
                # it is a big problem
                # we should check if vmax is in a reasonable range and non nan before running the calculation.
                if np.isnan(dsp.vmax.values):
                    print("nan vmax")
                    add_nan()
                elif dsp.vmax.values > 100:
                    print("vmax too high")
                    add_nan()
                elif dsp.vmax.values < 20:
                    print("vmax too low")
                    add_nan()
                else:
                    add_rmax()
        # if i == 5:
        ds_lon = xr.concat(ds_list_lon, dim="lon")
        ds_list.append(ds_lon)

    ds = xr.concat(ds_list, dim="lat")
    print("ds_list", ds_list)
    ds.to_netcdf("gom_soln_bbox.nc")


def plot_gom_bbox_soln() -> None:
    plot_defaults()
    ds = xr.open_dataset("gom_soln_bbox.nc")
    folder = "sup"
    print("ds", ds)
    ds["lon"].attrs = {"units": "$^{\circ}E$", "long_name": "Longitude"}
    ds["lat"].attrs = {"units": "$^{\circ}N$", "long_name": "Latitude"}

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axs = axs.T
    ds["sst"].where(~np.isnan(ds["t0"])).plot(
        ax=axs[0, 0],
        cbar_kwargs={
            "label": "Sea surface temperature, $T_s$ [$^\circ$C]",
        },
    )
    ds["t0"].plot(
        ax=axs[1, 0], cbar_kwargs={"label": "Outflow temperature, $T_0$, [K]"}
    )
    ds["msl"].where(~np.isnan(ds["t0"])).plot(
        ax=axs[2, 0], cbar_kwargs={"label": "Mean sea level pressure, $P_0$, [hPa]"}
    )
    ds["r0"].plot(ax=axs[0, 1], cbar_kwargs={"label": "Potential size, $r_a$, [km]"})
    ds["vmax"].plot(
        ax=axs[1, 1],
        cbar_kwargs={"label": "Potential intensity, $V_{\mathrm{max}}$, [m s$^{-1}$]"},
    )
    ds["pm"].plot(
        ax=axs[2, 1], cbar_kwargs={"label": "Pressure at maximum winds, $P_m$, [hPa]"}
    )
    for i in range(3):
        for j in range(2):
            if j != 1:
                axs[i, j].set_xlabel("")
            if i != 0:
                axs[i, j].set_ylabel("")
    label_subplots(axs, override="outside")
    # axs[1].plot(ds["lat"], ds["vmax"], "k")
    # axs[2].plot(ds["lat"], ds["pc"] / 100, "k")
    plt.savefig(folder + "/gom_bbox_r0_pm_rmax.pdf")

    vars: List[str] = ["t", "q", "vmax", "r0", "pc", "t0"]
    pairplot(ds.isel(p=0)[vars].to_dataframe()[vars])
    plt.savefig(folder + "/gom_bbox_pairplot2.pdf")
    plt.clf()


if __name__ == "__main__":
    # python run.py
    # find_solution()
    # find_solution_rmaxv()
    # calc_solns_for_times(num=50)
    # plot_gom_solns()
    # ds_solns(num=2, verbose=True, ds_name="gom_soln_2.nc")
    # plot_from_ds()  # ds_name="gom_soln_2.nc")
    # plot_soln_curves()
    plot_gom_bbox()
    plot_gom_bbox_soln()
