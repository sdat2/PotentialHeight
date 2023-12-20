"""Run the CLE15 model with json files."""
import os
import numpy as np
from matplotlib import pyplot as plt
from sithom.io import read_json
from sithom.plot import plot_defaults
from sithom.time import timeit

plot_defaults()


@timeit
def run_cle15(execute: bool = True, plot: bool = False) -> float:
    ins = read_json("inputs.json")
    print(ins)

    # Storm parameters

    # run octave file r0_pm.m
    if execute:
        os.system("octave r0_pm.m")

    # read in the output from r0_pm.m
    ou = read_json("outputs.json")

    if plot:
        print(ou)
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
        plt.ylim([0, 55])  # np.nanmax(out["VV"]) * 1.05])
        plt.legend()
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Rotating wind speed, $V$, [m s$^{-1}$]")
        plt.title("CLE15 Wind Profile")
        plt.savefig("r0_pm.pdf", format="pdf")
        plt.clf()

    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    p0 = 1005 * 100  # [Pa]
    rho0 = 1.15  # [kg m-3]
    rmerge = ou["rmerge"]
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
            vv[i] ** 2
            / (rr[i + 1] / 2 + rr[i] / 2)
            # + fcor * vv[i]
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

    return p[0]  # central pressure [Pa]


if __name__ == "__main__":
    run_cle15(plot=True)
