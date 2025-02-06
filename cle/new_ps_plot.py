"""Plot the new potential size calculation results for PIPS chapter/paper"""

import os
import xarray as xr
import matplotlib.pyplot as plt
from sithom.plot import feature_grid, label_subplots, plot_defaults
from .constants import DATA_PATH, FIGURE_PATH


def plot_panels() -> None:
    plot_defaults()

    # initial calculation with Ck/Cd = 1 for C15, gamma =1.2
    example_ds = xr.open_dataset(
        os.path.join(DATA_PATH, "example_potential_size_output_small.nc")
    )

    var = [["sst", "vmax"], ["msl", "rmax"], ["t0", "r0"]]
    units = [[r"$^{\circ}$C", r"m s$^{-1}$"], ["hPa", "m"], ["K", "m"]]
    names = [
        ["Sea surface temp., $T_s$", "Potential intensity, $V_p$"],
        ["Sea level pressure, $p_0$", r"Radius max winds, $r_{\mathrm{max}}$"],
        ["Outflow temperature, $T_0$", "Potential size, $r_a$"],
    ]
    cbar_lims = [[None, None], [None, None], [None, None]]
    super_titles = ["Inputs", "Outputs"]

    xy = [
        ("lon", "Longitude", r"$^{\circ}$E"),
        ("lat", "Latitude", r"$^{\circ}$N"),
    ]

    # pc, pm

    _, axs = feature_grid(
        example_ds, var, units, names, cbar_lims, super_titles, figsize=(6, 6), xy=xy
    )

    label_subplots(axs)

    plt.savefig(os.path.join(FIGURE_PATH, "new_ps_calculation_output_gom.pdf"))


if __name__ == "__main__":
    # python -m cle.new_ps_plot
    plot_panels()
