"""Run the CLE15 model with json files or octopy."""

from typing import Callable, Tuple, Optional, Dict
import os
import subprocess
import numpy as np
import time
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sithom.io import read_json, write_json
from sithom.plot import plot_defaults
from .constants import (
    BACKGROUND_PRESSURE,
    DATA_PATH,
    FIGURE_PATH,
    F_COR_DEFAULT,
    W_COOL_DEFAULT,
    RHO_AIR_DEFAULT,
    LATENT_HEAT_OF_VAPORIZATION,
    GAS_CONSTANT_FOR_WATER_VAPOR,
    GAS_CONSTANT,
    BETA_LIFT_PARAMETERIZATION_DEFAULT,
    EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,
    PRESSURE_DRY_AT_INFLOW_DEFAULT,
    MAX_WIND_SPEED_DEFAULT,
    RADIUS_OF_MAX_WIND_DEFAULT,
    RADIUS_OF_INFLOW_DEFAULT,
    NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,
    SRC_PATH,
    OUTFLOW_TEMPERATURE_DEFAULT,
)
from .utils import (
    pressure_from_wind,
    absolute_angular_momentum,
    carnot_factor,
    buck_sat_vap_pressure,
)

plot_defaults()


def delete_tmp():
    """Delete temporary folder."""
    import os, shutil

    folder = os.path.join(DATA_PATH, "tmp")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def _inputs_to_name(inputs: dict) -> str:
    """Create a unique naming string based on the input parameters
    (now hashed to shorten).

    There is a small probability of a hash collision during runs.

    Args:
        inputs (dict): input dict

    Returns:
        str: unique(ish) name
    """
    name = ""
    for key in sorted(inputs.keys()):  # for consistent order
        name += key + f"_{inputs[key]:.15e}"  # v. high precision
    return str(abs(hash(name)))


def process_inputs(inputs: dict) -> dict:
    ins = read_json(os.path.join(DATA_PATH, "inputs.json"))
    ins["w_cool"] = W_COOL_DEFAULT
    ins["p0"] = BACKGROUND_PRESSURE
    # ins["CkCd"]

    if inputs is not None:
        for key in inputs:
            if key in ins:
                ins[key] = inputs[key]
    return ins


def _run_cle15_octave(inputs: dict) -> dict:
    """
    Run the CLE15 model using octave.

    Args:
        inputs (dict): Input parameters.
        execute (bool): Whether to execute the model.

    Returns:
        dict: dict(rr=rr, VV=VV, rmax=rmax, rmerge=rmerge, Vmerge=Vmerge)
    """

    ins = process_inputs(inputs)

    print("inputs", inputs)

    # Storm parameters
    name = _inputs_to_name(ins)

    # write input file for octave to read
    write_json(ins, os.path.join(DATA_PATH, "tmp", name + "-inputs.json"))

    # check file has been written
    assert os.path.isfile(os.path.join(DATA_PATH, "tmp", name + "-inputs.json"))

    # time.sleep(0.1)  # wait for filesystem to sync for 0.1 s

    # run octave file r0_pm.m
    # disabling gui leads to one order of magnitude speedup
    # also the pop-up window makes me feel sick due to the screen moving about.
    # import subprocess

    # os.system(
    #    f"octave --no-gui --no-gui-libs {os.path.join(SRC_PATH, 'mcle', 'r0_pm.m')} {name}"
    # )
    # os.path.join()
    try:
        result = subprocess.run(
            [
                "octave",
                "--no-gui",
                "--no-gui-libs",
                "r0_pm.m",
                f"{name}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.join(SRC_PATH, "mcle"),
            # "/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/adcirc-swan/worstsurge/cle/mcle",
        )
        # print(result.stderr.decode())
        # print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Octave exited with code", e.returncode)
        print("Standard error:", e.stderr.decode())
        print("Standard output:", e.stdout.decode())

    # subprocess.call(
    #     (
    #         "micromamba activate t1",
    #         f"octave --no-gui --no-gui-libs {os.path.join(SRC_PATH, 'mcle', 'r0_pm.m')} {name}",
    #     )
    # )

    # read in the output from r0_pm.m
    # time.sleep(0.5)  # sleep another 0.5s for file system to catch up with itself
    return read_json(os.path.join(DATA_PATH, "tmp", name + "-outputs.json"))


def run_cle15(
    plot: bool = False,
    inputs: Optional[Dict[str, any]] = None,
) -> Tuple[float, float, float]:  # pm, rmax, vmax, pc
    """
    Run the CLE15 model.

    Args:
        plot (bool, optional): Plot the output. Defaults to False.
        inputs (Optional[Dict[str, any]], optional): Input parameters. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: pm [Pa], rmax [m], pc [Pa]
    """
    ins = process_inputs(inputs)  # find old data.
    ou = _run_cle15_octave(inputs)

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
    p0 = ins["p0"] * 100  # [Pa] [originally in hPa]
    rho0 = RHO_AIR_DEFAULT  # [kg m-3]
    ou["VV"][-1] = 0  # get rid of None at end of output.
    assert None not in ou["VV"]
    rr = np.array(ou["rr"], dtype="float32")  # [m]
    vv = np.array(ou["VV"], dtype="float32")  # [m/s]
    print("rr", rr[:10], rr[-10:])
    print("vv", vv[:10], vv[-10:])
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
    near_surface_air_temperature: float = NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,  # K
    outflow_temperature: float = OUTFLOW_TEMPERATURE_DEFAULT,  # K
    latent_heat_of_vaporization: float = LATENT_HEAT_OF_VAPORIZATION,  # J/kg
    gas_constant_for_water_vapor: float = GAS_CONSTANT_FOR_WATER_VAPOR,  # J/kg/K
    gas_constant: float = GAS_CONSTANT,  # J/kg/K
    beta_lift_parameterization: float = BETA_LIFT_PARAMETERIZATION_DEFAULT,  # dimensionless
    efficiency_relative_to_carnot: float = EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,  # dimensionless
    pressure_dry_at_inflow: float = PRESSURE_DRY_AT_INFLOW_DEFAULT,  # Pa
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
    maximum_wind_speed: float = MAX_WIND_SPEED_DEFAULT,  # m/s
    radius_of_inflow: float = RADIUS_OF_INFLOW_DEFAULT,  # m
    radius_of_max_wind: float = RADIUS_OF_MAX_WIND_DEFAULT,  # m
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


if __name__ == "__main__":
    # python -m cle.potential_size
    # delete_tmp()
    tick = time.perf_counter()
    vmaxs = np.linspace(80, 90, num=10)
    for vmax in vmaxs:
        ou = _run_cle15_octave({"Vmax": vmax})
        print(vmax, "Vmerge", ou["Vmerge"], "rmax", ou["rmax"])
    tock = time.perf_counter()
    octave_time = tock - tick
    print(f"Time taken by octave for 10 loops is {octave_time:.1f} s")  # 50.9 s
