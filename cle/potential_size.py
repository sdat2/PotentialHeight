"""Run the CLE15 model with json files or octopy."""

from typing import Callable, Tuple, Optional, Dict
import os, shutil
import numpy as np
import time
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# from oct2py import Oct2Py, get_log
from sithom.io import read_json, write_json
from sithom.plot import plot_defaults
from sithom.time import timeit
from chavas15.intersect import curveintersect
from .constants import (
    BACKGROUND_PRESSURE,
    DEFAULT_SURF_TEMP,
    DATA_PATH,
    FIGURE_PATH,
    SRC_PATH,
    F_COR_DEFAULT,
    W_COOL_DEFAULT,
    RHO_AIR_DEFAULT,
    SUPERGRADIENT_FACTOR,
)
from .utils import (
    pressure_from_wind,
    absolute_angular_momentum,
    carnot_factor,
    buck_sat_vap_pressure,
)
from .solve import bisection

plot_defaults()

os.environ["OCTAVE_CLI_OPTIONS"] = str(
    "--no-gui --no-gui-libs"  # --jit-compiler"  # disable gui to improve performance jit-compiler broke code
)
if shutil.which("octave") is not None:
    os.environ["OCTAVE_EXECUTABLE"] = shutil.which("octave")
"""
try:
    OC = Oct2Py(logger=get_log())
    OC.eval(f"addpath(genpath('{os.path.join(SRC_PATH, 'mcle')}'))")
except Exception as e:
    # allow the code to be tested without octave
    print("Octopy Initialization Exception", e)
    # assert False
    OC = None
"""
OC = None


@timeit
def _run_cle15_oct2py(
    **kwargs,
) -> dict:  # Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Run the CLE15 model using oct2py.

    Returns:
        dict: dict(rr=rr, VV=VV, rmax=rmax, rmerge=rmerge, Vmerge=Vmerge)
    """
    in_dict = read_json(os.path.join(DATA_PATH, "inputs.json"))
    in_dict.update(kwargs)
    # print(in_dict)
    # OC.eval("path")
    rr, VV, rmax, rmerge, Vmerge = OC.feval(
        "ER11E04_nondim_r0input",
        in_dict["Vmax"],
        in_dict["r0"],
        in_dict["fcor"],
        in_dict["Cdvary"],
        in_dict["Cd"],
        in_dict["w_cool"],
        in_dict["CkCdvary"],
        in_dict["CkCd"],
        in_dict["eye_adj"],
        in_dict["alpha_eye"],
        nout=5,
    )
    return dict(rr=rr, VV=VV, rmax=rmax, rmerge=rmerge, Vmerge=Vmerge)


def _inputs_to_name(inputs: dict) -> str:
    """Create a unique naming string based on the input parameters
    (could be hashed to shorten).

    Args:
        inputs (dict): input dict

    Returns:
        str: unique(ish) name
    """
    name = ""
    for key in sorted(inputs.keys()):  # for consistent order
        name += key + f"_{inputs[key]:.7e}"  # reasonable precision
    return name


def _run_cle15_octave(inputs: dict, execute: bool) -> dict:
    """
    Run the CLE15 model using octave.

    Args:
        inputs (dict): Input parameters.
        execute (bool): Whether to execute the model.

    Returns:
        dict: dict(rr=rr, VV=VV, rmax=rmax, rmerge=rmerge, Vmerge=Vmerge)
    """

    ins = read_json(os.path.join(DATA_PATH, "inputs.json"))
    if inputs is not None:
        for key in inputs:
            if key in ins:
                ins[key] = inputs[key]

    # Storm parameters
    name = _inputs_to_name(ins)

    write_json(ins, os.path.join(DATA_PATH, "tmp", name + "-inputs.json"))

    # run octave file r0_pm.m
    if execute:
        # disabling gui leads to one order of magnitude speedup
        # also the pop-up window makes me feel sick due to the screen moving about.
        os.system(
            f"octave --no-gui --no-gui-libs {os.path.join(SRC_PATH, 'mcle', 'r0_pm.m')} {name}"
        )

    # read in the output from r0_pm.m
    return read_json(os.path.join(DATA_PATH, "tmp", name + "-outputs.json"))


def run_cle15(
    execute: bool = True,
    plot: bool = False,
    inputs: Optional[Dict[str, any]] = None,
    oct2py: bool = False,
) -> Tuple[float, float, float, float]:  # pm, rmax, vmax, pc
    """
    Run the CLE15 model.

    Args:
        execute (bool, optional): Execute the model. Defaults to True.
        plot (bool, optional): Plot the output. Defaults to False.
        inputs (Optional[Dict[str, any]], optional): Input parameters. Defaults to None.
        oct2py (bool, optional): Use oct2py. Defaults to False, as oct2py seems to be slower than direct octave on ARCHER2 compute nodes.

    Returns:
        Tuple[float, float, float, float]: pm [Pa], rmax [m], vmax [m/s], pc [Pa]
    """

    if oct2py and OC is not None:  # should be faster if graphical element disabled
        ou = _run_cle15_oct2py(inputs)
    else:
        ou = _run_cle15_octave(inputs, execute)
    # read default values from the inputs.json file
    ins = read_json(os.path.join(DATA_PATH, "inputs.json"))

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
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pm.pdf"), format="pdf")
        plt.clf()

    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    p0 = ins["p0"] * 100  # [Pa]
    rho0 = RHO_AIR_DEFAULT  # [kg m-3]
    rr = np.array(ou["rr"])  # [m]
    vv = np.array(ou["VV"])  # [m/s]
    p = pressure_from_wind(rr, vv, p0=p0, rho0=rho0, fcor=ins["fcor"])  # [Pa]
    ou["p"] = p.tolist()

    if plot:
        plt.plot(rr / 1000, p / 100, "k")
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Pressure, $p$, [hPa]")
        plt.title("CLE15 Pressure Profile")
        plt.ylim([np.min(p) / 100, np.max(p) * 1.0005 / 100])
        plt.xlim([0, rr[-1] / 1000])
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pmp.pdf"), format="pdf")
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


def wang_diff(
    a: float = 0.062, b: float = 0.031, c: float = 0.008
) -> Callable[[float], float]:
    """
    Wang difference function.

    Args:
        a (float, optional): a. Defaults to 0.062.
        b (float, optional): b. Defaults to 0.031.
        c (float, optional): c. Defaults to 0.008.

    Returns:
        Callable[[float], float]: Function to find root of.
    """

    def f(y: float) -> float:  # y = exp(a*y + b*log(y)*y + c)
        return y - np.exp(a * y + b * np.log(y) * y + c)

    return f


def wang_consts(
    near_surface_air_temperature: float = 299,  # K
    outflow_temperature: float = 200,  # K
    latent_heat_of_vaporization: float = 2.27e6,  # J/kg
    gas_constant_for_water_vapor: float = 461,  # J/kg/K
    gas_constant: float = 287,  # J/kg/K
    beta_lift_parameterization: float = 1.25,  # dimensionless
    efficiency_relative_to_carnot: float = 0.5,  # dimensionless
    pressure_dry_at_inflow: float = 985 * 100,  # Pa
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
    maximum_wind_speed: float = 83,  # m/s
    radius_of_inflow: float = 2193 * 1000,  # m
    radius_of_max_wind: float = 64 * 1000,  # m
) -> Tuple[float, float, float]:  # a, b, c
    """
    Wang carnot engine model parameters.

    Args:
        near_surface_air_temperature (float, optional): Defaults to 299 [K].
        outflow_temperature (float, optional): Defaults to 200 [K].
        latent_heat_of_vaporization (float, optional): Defaults to 2.27e6 [J/kg].
        gas_constant_for_water_vapor (float, optional): Defaults to 461 [J/kg/K].
        gas_constant (float, optional): Defaults to 287 [J/kg/K].
        beta_lift_parameterization (float, optional): Defaults to 1.25 [dimesionless].
        efficiency_relative_to_carnot (float, optional): Defaults to 0.5 [dimensionless].
        pressure_dry_at_inflow (float, optional): Defaults to 985 * 100 [Pa].
        coriolis_parameter (float, optional): Defaults to 5e-5 [s-1].
        maximum_wind_speed (float, optional): Defaults to 83 [m/s].
        radius_of_inflow (float, optional): Defaults to 2193 * 1000 [m].
        radius_of_max_wind (float, optional): Defaults to 64 * 1000 [m].

    Returns:
        Tuple[float, float, float]: a, b, c
    """
    # a, b, c
    absolute_angular_momentum_at_vmax = absolute_angular_momentum(
        maximum_wind_speed, radius_of_max_wind, coriolis_parameter
    )
    carnot_efficiency = carnot_factor(near_surface_air_temperature, outflow_temperature)
    near_surface_saturation_vapour_presure = buck_sat_vap_pressure(
        near_surface_air_temperature
    )

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


def find_solution_rmaxv(
    vmax_pi: float = 86,  # m/s
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
    background_pressure: float = BACKGROUND_PRESSURE,  # Pa
    near_surface_air_temperature: float = DEFAULT_SURF_TEMP,  # K
    w_cool: float = W_COOL_DEFAULT,  # K m s-1
    outflow_temperature: float = 200,  # K
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
    rmax: float, vmax: float, r0: float, fcor=F_COR_DEFAULT, p0=1016 * 100
):
    """
    Profile from values.

    Args:
        rmax (float): rmax [m]
        vmax (float): vmax [m/s]
        r0 (float): r0 [m]
        fcor (float, optional): fcor. Defaults to 7.836084948051749e-05.
        p0 (float, optional): p0. Defaults to 1016 * 100.

    Returns:
        dict: Dictionary of values using pressure wind relationship.
    """
    # rr = np.linspace(0, r0, num=1000)
    ou = _run_cle15_oct2py(
        **{"r0": r0, "Vmax": vmax, "rmax": rmax, "fcor": fcor, "p0": p0 / 100}
    )
    for key in ou:
        if isinstance(ou[key], np.ndarray):
            ou[key] = ou[key].flatten()
    ou["p"] = pressure_from_wind(ou["rr"], ou["VV"], fcor=fcor, p0=p0)
    return ou


if __name__ == "__main__":
    # python -m cle.potential_size
    # find_solution()
    # find_solution_rmaxv()
    # calc_solns_for_times(num=50)
    # plot_gom_solns()
    # ds_solns(num=2, verbose=True, ds_name="gom_soln_2.nc")
    # plot_from_ds()  # ds_name="gom_soln_2.nc")
    # plot_soln_curves()
    # plot_gom_bbox()
    # ds_solns(num=50, verbose=True, ds_name="data/gom_soln_new.nc")
    # find_solution_rmaxv()

    # from timeit import timeit

    # tick = time.perf_counter()

    # for _ in range(10):
    #     _run_cle15_oct2py()  # oct2py not working on ARCHER2

    # tock = time.perf_counter()
    # oct2py_time = tock - tick

    tick = time.perf_counter()

    vmaxs = np.linspace(80, 90, num=10)

    for vmax in vmaxs:
        _run_cle15_octave({"Vmax": vmax}, True)
    tock = time.perf_counter()
    octave_time = tock - tick

    # print(f"Time taken by oct2py for 10 loops is {oct2py_time:.1f} s")  # 21.4 s
    print(f"Time taken by octave for 10 loops is {octave_time:.1f} s")  # 50.9 s
