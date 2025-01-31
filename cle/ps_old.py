import os
import numpy as np
import matplotlib.pyplot as plt
from chavas15.intersect import curveintersect
from .potential_size import run_cle15, _run_cle15_octave, wang_consts, wang_diff
from .solve import bisection
from .utils import buck_sat_vap_pressure, pressure_from_wind
from .constants import (
    F_COR_DEFAULT,
    BACKGROUND_PRESSURE,
    DEFAULT_SURF_TEMP,
    FIGURE_PATH,
    W_COOL_DEFAULT,
    SUPERGRADIENT_FACTOR,
    OUTFLOW_TEMPERATURE_DEFAULT,
)


def vary_r0_c15(r0s: np.ndarray) -> np.ndarray:
    """
    Vary r0 for CLE15.

    Args:
        r0s (np.ndarray): r0s.

    Returns:
        np.ndarray: pcs.
    """
    return np.array([run_cle15(plot=False, inputs={"r0": r0})[0] for r0 in r0s])


def vary_r0_w22(r0s: np.ndarray) -> np.ndarray:
    """
    Vary r0 for W22.

    Args:
        r0s (np.ndarray): r0s.

    Returns:
        np.ndarray: pms.
    """
    ys = np.array(
        [
            bisection(wang_diff(*wang_consts(radius_of_inflow=r0)), 0.3, 1.2, 1e-6)
            for r0 in r0s
        ]
    )
    pms = (
        BACKGROUND_PRESSURE - buck_sat_vap_pressure(DEFAULT_SURF_TEMP)
    ) / ys + buck_sat_vap_pressure(DEFAULT_SURF_TEMP)
    return pms


def profile_from_vals(
    rmax: float, vmax: float, r0: float, fcor=F_COR_DEFAULT, p0=BACKGROUND_PRESSURE
):
    """
    Profile from values.

    Args:
        rmax (float): rmax [m]
        vmax (float): vmax [m/s]
        r0 (float): r0 [m]
        fcor (float, optional): fcor. Defaults to 7.836084948051749e-05.
        p0 (float, optional): p0. Defaults to 1015_00.

    Returns:
        dict: Dictionary of values using pressure wind relationship.
    """
    # rr = np.linspace(0, r0, num=1000)
    ou = _run_cle15_octave(
        {"r0": r0, "Vmax": vmax, "rmax": rmax, "fcor": fcor, "p0": p0 / 100}
    )
    for key in ou:
        if isinstance(ou[key], np.ndarray):
            ou[key] = ou[key].flatten()
    ou["p"] = pressure_from_wind(ou["rr"], ou["VV"], fcor=fcor, p0=p0)
    return ou


def find_solution_rmaxv(
    vmax_pi: float = 86,  # m/s
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
    background_pressure: float = BACKGROUND_PRESSURE,  # Pa
    near_surface_air_temperature: float = DEFAULT_SURF_TEMP,  # K
    w_cool: float = W_COOL_DEFAULT,  # K m s-1
    outflow_temperature: float = OUTFLOW_TEMPERATURE_DEFAULT,  # K
    gamma_supergradient_factor: float = SUPERGRADIENT_FACTOR,  # dimensionless
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the solution for rmax.

    Args:
        vmax_pi (float, optional): Maximum wind speed. Defaults to 86 m/s.
        coriolis_parameter (float, optional): Coriolis parameter. Defaults to 5e-5 s-1.
        background_pressure (float, optional): Background pressure. Defaults to 1015 hPa.
        near_surface_air_temperature (float, optional): Near surface air temperature. Defaults to 299 K.
        w_cool (float, optional): Cooling rate. Defaults to 0.002. 2e-3
        outflow_temperature (float, optional): Outflow temperature. Defaults to 200 K.
        gamma_supergradient_factor (float, optional): Supergradient factor. Defaults to 1.2.
        plot (bool, optional): Plot the output. Defaults to False.

    Returns:
        Tuple[float, float, float]: rmax, vmax_pi, pm.
    """
    r0s = np.linspace(200, 5000, num=30) * 1000
    pcs = []
    pcw = []
    rmaxs = []
    for r0 in r0s:
        pm_cle15_dyn, rmax_cle, _, _ = run_cle15(
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
                    maximum_wind_speed=vmax_pi * gamma_supergradient_factor,
                    coriolis_parameter=coriolis_parameter,
                    pressure_dry_at_inflow=background_pressure
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
        pm_w22_car = (
            background_pressure - buck_sat_vap_pressure(near_surface_air_temperature)
        ) / ys + buck_sat_vap_pressure(near_surface_air_temperature)

        pcs.append(pm_cle15_dyn)
        pcw.append(pm_w22_car)
        rmaxs.append(rmax_cle)
        # print(
        #    "r0, rmax_cle, pm_cle15_dyn, pm_w22_car",
        #    r0,
        #    rmax_cle,
        #    pm_cle15_dyn,
        #    pm_w22_car,
        # )
    pcs = np.array(pcs)
    pcw = np.array(pcw)
    rmaxs = np.array(rmaxs)
    # find intersection between r0s and pcs and pcw
    intersect = curveintersect(r0s, pcs, r0s, pcw)

    if plot:
        plt.plot(r0s / 1000, pcs / 100, "k", label="CLE15")
        plt.plot(r0s / 1000, pcw / 100, "r", label="W22")
        print("intersect", intersect)
        # plt.plot(intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution")
        plt.xlabel("Radius of outer winds, $r_0$, [km]")
        plt.ylabel("Pressure at maximum winds, $p_m$, [hPa]")
        if len(intersect) > 0:
            plt.plot(
                intersect[0][0] / 1000, intersect[1][0] / 100, "bx", label="Solution"
            )
        plt.legend()
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pc_rmaxadj.pdf"))
        plt.clf()
        plt.plot(r0s / 1000, rmaxs / 1000, "k")
        plt.xlabel("Outer radius, $r_9$, [km]")
        plt.ylabel("Radius of maximum winds, $r_m$, [km]")
        plt.savefig(os.path.join(FIGURE_PATH, "r0_rmax.pdf"))
        plt.clf()
        run_cle15(inputs={"r0": intersect[0][0], "Vmax": vmax_pi}, plot=True)
    # why do we put vmax_pi back out? We put it in in the first place.
    return intersect[0][0], vmax_pi, intersect[1][0]  # rmax, vmax, pm
