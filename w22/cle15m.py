"""CLE15 model for matlab/octave version."""

from typing import Tuple, Optional, Dict, Union
import os
import numpy as np
import time
import shutil
import subprocess
import socket
import uuid
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sithom.io import read_json, write_json
from sithom.plot import plot_defaults
from .constants import (
    SRC_PATH,
    DATA_PATH,
    TMPS_PATH,
    FIGURE_PATH,
    BACKGROUND_PRESSURE,
    W_COOL_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    RHO_AIR_DEFAULT,
)
from .utils import pressure_from_wind

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


def get_unique_folder() -> str:
    """Create a unique folder for this run.
    The folder is created in the TMPS_PATH directory.
    The folder name is based on the hostname, process ID, and a UUID.
    This ensures that the folder name is unique and does not collide with other
    folders created by other processes or runs.

    Returns:
        str: The path to the unique folder.
    """
    # create unique folder for this run.
    pid = os.getpid()
    hostname = socket.gethostname()
    unique_id = uuid.uuid4()
    temp_dir = os.path.join(TMPS_PATH, f"job_{hostname}_{pid}_{unique_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _inputs_to_name(inputs: dict, hash_name=True) -> str:
    """Create a unique naming string based on the input parameters
    (now hashed to shorten).

    There is a small probability of a hash collision during runs.

    Args:
        inputs (dict): input dict
        hash_name (bool, optional): whether to hash the name. Defaults to True.

    Returns:
        str: unique(ish) name
    """
    name = ""
    for key in sorted(inputs.keys()):  # for consistent order
        val = inputs[key]
        if isinstance(val, Union[float, int]):
            if val < 0:
                name += key + f"_n{abs(val):.15e}"
            else:
                name += key + f"_{val:.15e}"
        elif isinstance(val, str):
            name += key + "_" + val

    if hash_name:  # shortens name, but small probability of collision
        return str(abs(hash(name)))
    else:
        return name


def process_inputs(inputs: dict) -> dict:
    """Process the input parameters for the CLE15 model.

    Args:
        inputs (dict): Input parameters.

    Returns:
        dict: Processed input parameters.
    """
    # load default inputs
    ins = read_json(os.path.join(DATA_PATH, "inputs.json"))
    ins["w_cool"] = W_COOL_DEFAULT
    ins["p0"] = BACKGROUND_PRESSURE / 100  # in hPa instead
    ins["CkCd"] = CK_CD_DEFAULT
    ins["Cd"] = CD_DEFAULT

    if "p0" in inputs:
        assert inputs["p0"] > 900 and inputs["p0"] < 1100  # between 900 and 1100 hPa
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

    Returns:
        dict: dict(rr=rr, VV=VV, rmax=rmax, rmerge=rmerge, Vmerge=Vmerge)
    """

    ins = process_inputs(inputs)

    # print("inputs", inputs)

    # Storm parameters
    name = _inputs_to_name(ins)

    data_folder = get_unique_folder()

    # write input file for octave to read
    write_json(ins, os.path.join(data_folder, name + "-inputs.json"))

    # check file has been written
    assert os.path.isfile(os.path.join(data_folder, name + "-inputs.json"))

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
                f"{data_folder}",
                f"{name}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.join(SRC_PATH, "mcle"),
        )
    except subprocess.CalledProcessError as e:
        print("Octave exited with code", e.returncode)
        print("Standard error:", e.stderr.decode())
        print("Standard output:", e.stdout.decode())

    # read in the output from r0_pm.m
    # time.sleep(0.5)  # sleep another 0.5s for file system to catch up with itself
    output = read_json(os.path.join(data_folder, name + "-outputs.json"))

    # delete the temporary folder 'data_folder'
    shutil.rmtree(data_folder)
    return output


def run_cle15(
    plot: bool = False,
    inputs: Optional[Dict[str, any]] = None,
    rho0=RHO_AIR_DEFAULT,  # [kg m-3]
    pressure_assumption: str = "isopycnal",
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
    ou["VV"][-1] = 0  # get rid of None at end of output.
    assert None not in ou["VV"]
    rr = np.array(ou["rr"], dtype="float32")  # [m]
    vv = np.array(ou["VV"], dtype="float32")  # [m/s]
    # print("rr", rr[:10], rr[-10:])
    # print("vv", vv[:10], vv[-10:])
    p = pressure_from_wind(
        rr,
        vv,
        p0=p0,
        rho0=rho0,
        fcor=ins["fcor"],
        assumption=pressure_assumption,
    )  # [Pa]
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
        float(
            interp1d(rr, p)(ou["rmax"])
        ),  # find the pressure at the maximum wind speed radius [Pa]
        ou["rmax"],  # rmax radius [m]
        p[0],
    )  # p[0]  # central pressure [Pa]


def profile_from_stats(vmax: float, fcor: float, r0: float, p0: float) -> dict:
    ins = process_inputs({"Vmax": vmax, "fcor": fcor, "r0": r0, "p0": p0})
    out = _run_cle15_octave(ins)
    out["VV"][-1] = 0
    out["p"] = pressure_from_wind(out["rr"], out["VV"], p0=p0 * 100, fcor=fcor) / 100
    return out


if __name__ == "__main__":
    # python -m w22.potential_size
    # delete_tmp()
    tick = time.perf_counter()
    vmaxs = np.linspace(80, 90, num=10)
    for vmax in vmaxs:
        ou = _run_cle15_octave({"Vmax": vmax})
        print(vmax, "Vmerge", ou["Vmerge"], "rmax", ou["rmax"])
    tock = time.perf_counter()
    octave_time = tock - tick
    print(f"Time taken by octave for 10 loops is {octave_time:.1f} s")  # 50.9 s
