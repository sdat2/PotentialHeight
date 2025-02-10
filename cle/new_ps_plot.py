"""Plot the new potential size calculation results for PIPS chapter/paper"""

import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
from sithom.plot import feature_grid, label_subplots, plot_defaults, get_dim, pairplot
from sithom.curve import fit
from .constants import DATA_PATH, FIGURE_PATH


def plot_panels() -> None:
    plot_defaults()

    # initial calculation with Ck/Cd = 1 for C15, gamma =1.2
    example_ds = xr.open_dataset(
        os.path.join(DATA_PATH, "example_potential_size_output_small_2.nc")
    )
    # to km
    del example_ds["time"]  # was annoying to have overly precise time
    example_ds["r0"][:] /= 1000
    example_ds["rmax"][:] /= 1000
    var = [["sst", "vmax"], ["msl", "rmax"], ["t0", "r0"]]
    units = [[r"$^{\circ}$C", r"m s$^{-1}$"], ["hPa", "km"], ["K", "km"]]
    names = [
        ["Sea surface temp., $T_s$", "Potential intensity, $V_p$"],
        ["Sea level pressure, $p_0$", r"Radius max winds, $r_{\mathrm{max}}$"],
        ["Outflow temperature, $T_0$", "Potential size, $r_a$"],
    ]
    cbar_lims = [
        [(28, 33, "cmo.thermal"), None],
        [(1010, 1020, "cmo.dense"), None],
        [(200, 210, "cmo.thermal"), None],
    ]
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


def safe_grad(xt, yt):
    # get rid of nan values
    xt, yt = xt[~np.isnan(xt)], yt[~np.isnan(xt)]
    xt, yt = xt[~np.isnan(yt)], yt[~np.isnan(yt)]
    # normalize the data between 0 and 10
    xrange = np.max(xt) - np.min(xt)
    yrange = np.max(yt) - np.min(yt)
    xt = (xt - np.min(xt)) / xrange * 10
    yt = (yt - np.min(yt)) / yrange * 10
    # fit the data with linear fit using OLS
    param, _ = fit(xt, yt)  # defaults to y=mx+c fit
    return param[0] * yrange / xrange


def safe_corr(xt, yt):
    corr = ma.corrcoef(ma.masked_invalid(xt), ma.masked_invalid(yt))
    return corr[0, 1]


def _float_to_latex(x, precision=2):
    """
    Convert a float x to a LaTeX-formatted string with the given number of significant figures.

    Args:
        x (float): The number to format.
        precision (int): Number of significant figures (default is 2).

    Returns:
        str: A string like "2.2\\times10^{-6}" or "3.1" (if no exponent is needed).
    """
    # Handle the special case of zero.
    if x == 0:
        return "0"

    # Format the number using general format which automatically uses scientific notation when needed.
    s = f"{x:.{precision}g}"

    # If scientific notation is used, s will contain an 'e'
    if "e" in s:
        mantissa, exp = s.split("e")
        # Convert the exponent string to an integer (this removes any extra zeros)
        exp = int(exp)
        # Choose the multiplication symbol.
        mult = "\\times"
        return f"{mantissa}{mult}10^{{{exp}}}"
    else:
        # If no exponent is needed, just return the number inside math mode.
        return f"{s}"


def _m_to_text(m):
    if m.s in (np.nan, np.inf, -np.inf):
        if m.s not in (np.nan, np.inf, -np.inf):
            return "$m={:}$".format(_float_to_latex(m))
        else:
            return "NaN"
    else:
        if m.n > 1000 or m.n < 0.1:
            return "$m={:.1eL}$".format(m)
        else:
            return "$m={:.2L}$".format(m)


def timeseries_plot():
    # plot CESM2 ensemble members for ssp585 near New Orleans
    plot_defaults()
    members = [4, 10, 11]
    colors = ["purple", "green", "orange"]
    file_names = [
        os.path.join(DATA_PATH, f"new_orleans_august_ssp585_r{member}i1p1f1.nc")
        for member in members
    ]
    ds_l = [xr.open_dataset(file_name) for file_name in file_names]
    _, axs = plt.subplots(4, 1, sharex=True, figsize=get_dim(ratio=1.5))
    vars = ["sst", "vmax", "rmax", "r0"]
    var_labels = [
        "Sea surface temp., $T_s$",
        "Potential intensity, $V_p$",
        r"Radius max winds, $r_{\mathrm{max}}$",
        "Potential size, $r_a$",
    ]
    units = ["$^{\circ}$ C", "m s$^{-1}$", "km", "km"]

    for i, var in enumerate(vars):
        for j, ds in enumerate(ds_l):
            x = np.array([time.year for time in ds.time.values])
            y = ds[var].values
            if var == "rmax" or var == "r0":
                y /= 1000  # divide by 1000 to go to km
            axs[i].plot(x, y, color=colors[j], label=f"r{members[j]}i1p1f1")
            m = safe_grad(x, y)
            rho = safe_corr(x, y)
            corr_bit = f"ρ = {rho:.2f}"
            m_bit = _m_to_text(m) + " " + units[i] + " yr$^{-1}$"
            print(f"r{members[j]}i1p1f1 " + var + " " + corr_bit + ", " + m_bit)
            axs[i].annotate(
                corr_bit + ", " + m_bit,
                xy=(0.44, 0.21 - j * 0.1),
                xycoords=axs[i].transAxes,
                color=colors[j],
            )

        axs[i].set_xlabel("")
        axs[i].set_ylabel(var_labels[i] + " [" + units[i] + "]")
        if i == len(vars) - 1:
            axs[i].legend()
            axs[i].set_xlabel("Year [A.D.]")
    axs[2].set_xlim(2015, 2100)
    label_subplots(axs)
    axs[0].set_title("New Orleans CESM2 SSP585")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    plt.savefig(os.path.join(FIGURE_PATH, "new_orleans_timeseries.pdf"))
    plt.clf()
    for j, member in enumerate(members):
        del ds_l[j]["time"]
        ds_l[j]["r0"].attrs["units"] = "km"
        ds_l[j]["rmax"].attrs["units"] = "km"
        _, axs = pairplot(ds_l[j][vars])
        plt.savefig(
            os.path.join(FIGURE_PATH, f"new_orleans_pairplot_r{member}i1p1f1.pdf")
        )
    plt.clf()


if __name__ == "__main__":
    # python -m cle.new_ps_plot
    # plot_panels()
    timeseries_plot()
