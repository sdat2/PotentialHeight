"""Run the CLE15 model with json files."""
from typing import Callable, Tuple, Optional, Dict
import os
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sithom.io import read_json, write_json
from sithom.plot import plot_defaults
from sithom.time import timeit
from chavas15.intersect import curveintersect

plot_defaults()

BACKGROUND_PRESSURE = 1015 * 100  # [Pa]
TEMP_0K = 273.15  # [K]


@timeit
def run_cle15(
    execute: bool = True, plot: bool = False, inputs: Optional[Dict[str, any]] = None
) -> Tuple[float, float, float]:  # pm, rmax, vmax
    ins = read_json("inputs.json")
    if inputs is not None:
        for key in inputs:
            if key in ins:
                ins[key] = inputs[key]

    write_json(ins, "inputs.json")

    # Storm parameters

    # run octave file r0_pm.m
    if execute:
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
    p0 = BACKGROUND_PRESSURE  # [Pa]
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
        interp1d(rr, p)(ou["rmax"]),
        ou["rmax"],
        ins["Vmax"],
    )  # p[0]  # central pressure [Pa]


def wang_diff(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:
        return y - np.exp(a * y + b * np.log(y) * y + c)

    return f


@timeit
def bisection(f: Callable, left: float, right: float, tol: float) -> float:
    # https://en.wikipedia.org/wiki/Root-finding_algorithms#Bisection_method
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
    # https://en.wikipedia.org/wiki/Arden_Buck_equation
    temp = temp - TEMP_0K
    return 0.61121 * np.exp((18.678 - temp / 234.5) * (temp / (257.14 + temp))) * 1000


def carnot(temp_hot: float, temp_cold: float) -> float:
    return (temp_hot - temp_cold) / temp_hot


def absolute_angular_momentum(v: float, r: float, f: float) -> float:
    return v * r + 0.5 * f * r**2


def wang_consts(
    near_surface_air_temperature=299,  # K
    outflow_temperature=200,  # K
    latent_heat_of_vaporization=2.27e6,  # J/kg
    gas_constant_for_water_vapor=461,  # J/kg/K
    gas_constant=287,  # J/kg/K
    beta_lift_parameterization=1.25,  # dimensionless
    efficiency_relative_to_carnot=0.5,
    pressure_dry_at_inflow=985 * 100,
    coriolis_parameter=5e-5,
    maximum_wind_speed=83,
    radius_of_inflow=2193 * 1000,
    radius_of_max_wind=64 * 1000,
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


DEFAULT_SURF_TEMP = 299  # K


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


def find_solution_rmaxv() -> Tuple[np.ndarray, np.ndarray]:
    r0s = np.linspace(200, 5000, num=30) * 1000
    pcs = []
    pcw = []
    rmaxs = []
    for r0 in r0s:
        pm_cle, rmax_cle, vmax = run_cle15(plot=True, inputs={"r0": r0})

        ys = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=vmax,
                )
            ),
            0.9,
            1.2,
            1e-6,
        )
        pm_car = (BACKGROUND_PRESSURE - buck(DEFAULT_SURF_TEMP)) / ys + buck(
            DEFAULT_SURF_TEMP
        )

        pcs.append(pm_cle)
        pcw.append(pm_car)
        rmaxs.append(rmax_cle)
        print("r0, rmax_cle, pm_cle, pm_car", r0, rmax_cle, pm_cle, pm_car)
    pcs = np.array(pcs)
    pcw = np.array(pcw)
    rmaxs = np.array(rmaxs)

    plt.plot(r0s / 1000, pcs / 100, "k", label="CLE15")
    plt.plot(r0s / 1000, pcw / 100, "r", label="W22")
    intersect = curveintersect(r0s, pcs, r0s, pcw)
    print("intersect", intersect)
    # plt.plot(intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution")
    plt.xlabel("Radius, $r_a$, [km]")
    plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
    plt.plot(intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution")
    plt.legend()
    plt.savefig("r0_pc_rmaxadj.pdf")
    plt.clf()
    plt.plot(r0s / 1000, rmaxs / 1000, "k")
    plt.xlabel("Radius, $r_a$, [km]")
    plt.ylabel("Radius of maximum winds, $r_m$, [km]")
    plt.savefig("r0_rmax.pdf")
    plt.clf()
    run_cle15(inputs={"r0": intersect[0][0], "Vmax": 86}, plot=True)
    return pcs, pcw


def check_w22():
    wd = wang_diff(
        *wang_consts(
            radius_of_max_wind=64 * 1000,
            radius_of_inflow=2193 * 1000,
        )
    )


@timeit
def find_solution():
    r0s = np.linspace(200, 5000, num=30) * 1000

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


if __name__ == "__main__":
    # find_solution()
    find_solution_rmaxv()
