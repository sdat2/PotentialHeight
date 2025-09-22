"""Plot the new potential size calculation results for PIPS chapter/paper"""

import os
from typing import List, Union, Tuple
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from uncertainties import ufloat
from sithom.io import write_json
from sithom.plot import feature_grid, label_subplots, plot_defaults, get_dim, pairplot
from sithom.curve import fit
from .constants import DATA_PATH, FIGURE_PATH, OFFSET_D
from .cle15m import profile_from_stats
from .utils import coriolis_parameter_from_lat


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

    plt.suptitle("August 2015")

    plt.savefig(os.path.join(FIGURE_PATH, "new_ps_calculation_output_gom.pdf"))


def safe_grad(xt: np.ndarray, yt: np.ndarray) -> ufloat:
    """
    Calculate the gradient of the data with error handling.

    Args:
        xt (np.ndarray): The x data.
        yt (np.ndarray): The y data.

    Returns:
        ufloat: The gradient.
    """

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


def safe_corr(xt: np.ndarray, yt: np.ndarray) -> float:
    """
    Calculate the correlation of the data with error handling.

    Args:
        xt (np.ndarray): The x data.
        yt (np.ndarray): The y data.

    Returns:
        float: The correlation.
    """
    corr = ma.corrcoef(ma.masked_invalid(xt), ma.masked_invalid(yt))
    return corr[0, 1]


def _float_to_latex(x: float, precision: int = 2):
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


def _m_to_text(m: ufloat) -> str:
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


def timeseries_plot(
    name: str = "new_orleans",
    plot_name: str = "New Orleans",
    years: List[int] = [2025, 2097],
    members: List[int] = [4, 10, 11],
) -> None:
    """
    Plot the timeseries for the given location and members.

    Args:
        name (str): The name of the location (default is "new_orleans").
        plot_name (str): The name of the location to use in the plot title (default is "New Orleans").
        years (List[int]): The years to plot (default is [2025, 2097]).
        members (List[int]): The members to plot (default is [4, 10, 11]).
    """
    # plot CESM2 ensemble members for ssp585 near New Orleans
    plot_defaults()
    colors = ["purple", "green", "orange"]
    file_names = [
        os.path.join(DATA_PATH, f"{name}_august_ssp585_r{member}i1p1f1.nc")
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
            ds_l[j][var].attrs["long_name"] = var_labels[i]
            ds_l[j][var].attrs["units"] = units[i]

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
    axs[0].set_title(f"{plot_name} CESM2 SSP585")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    plt.savefig(os.path.join(FIGURE_PATH, f"{name}_timeseries.pdf"))
    plt.clf()
    colors = ["Green", "Blue"]
    vars = ["p", "VV"]
    var_labels = ["Pressure [hPa]", "Velocity [m s$^{-1}$]"]
    plot_defaults()
    for j, member in enumerate(members):
        fig, axs = plt.subplots(2, 1, sharex=True)
        for k, year in enumerate(years):
            tp = ds_l[j].isel(time=[t.year == year for t in ds_l[j].time.values])
            vmax = float(tp.vmax.values.ravel())
            fcor = float(coriolis_parameter_from_lat(tp.lat.values.ravel()))
            r0 = float(tp.r0.values.ravel()) * 1000  # back to m
            p0 = float(tp.msl.values.ravel())  # hPa
            if not np.isnan(vmax) and not np.isnan(r0):
                profile = profile_from_stats(
                    vmax,
                    fcor,
                    r0,
                    p0,
                )
                write_json(
                    profile,
                    os.path.join(
                        DATA_PATH, f"{year}_{name}_profile_r{member}i1p1f1.json"
                    ),
                )
                for i, var in enumerate(vars):
                    axs[i].plot(
                        np.array(profile["rr"]) / 1000,
                        profile[var],
                        color=colors[k],
                        label=f"August {year}",
                    )
                    if k == 0:
                        axs[i].set_ylabel(var_labels[i])
        label_subplots(axs)
        plt.legend(ncols=2)
        plt.xlabel("Radius [km]")
        plt.savefig(
            os.path.join(
                FIGURE_PATH,
                f"{name}_profiles_{years[0]}_{years[1]}_r{member}i1p1f1.pdf",
            )
        )
        plt.clf()

    vars = ["sst", "vmax", "rmax", "r0"]
    for j, member in enumerate(members):
        del ds_l[j]["time"]
        _, axs = pairplot(ds_l[j][vars], label=True)
        plt.savefig(os.path.join(FIGURE_PATH, f"{name}_pairplot_r{member}i1p1f1.pdf"))
        plt.clf()


def plot_seasonal_profiles():
    """
    Plot the seasonal profiles for New Orleans.
    """
    in_path = os.path.join(DATA_PATH, "new_orleans_10year3.nc")
    ds = xr.open_dataset(in_path)
    # plot season for msl, rh,  sst, t0, vmax, rmax, r0
    # shared x -axis = months
    # seperate y-axes = values
    plt.figure(figsize=(6, 6))
    var = ["msl", "rh", "sst", "t0", "vmax", "rmax", "r0"]
    var_labels = [
        "Sea level pressure, $p_a$",
        r"Relative humidity, $\mathcal{H}_e$",
        "Sea surface temperature, $T_s$",
        "Outflow temperature, $T_0$",
        r"Potential intensity, $V_{p}$",
        "Radius of maximum winds, $r_{\max}$",
        "Potential size, $r_a$",
    ]
    # only symbols to save space
    var_labels = [
        "$p_a$",
        r"$\mathcal{H}_e$",
        "$T_s$",
        "$T_0$",
        r"$V_{p}$",
        "$r_{\max}$",
        "$r_a$",
    ]
    ds["rh"] *= 100
    ds["rmax"] /= 1000
    ds["r0"] /= 1000
    ds["t0"] -= 273.15
    # units = ["fraction", "K", "m s$^{-1}$", "hPa", "$^{\circ}$C"]
    units = ["hPa", "%", r"$^{\circ}$C", r"$^{\circ}$C", r"m s$^{-1}$", "km", "km"]
    fig, axs = plt.subplots(len(var), 1, figsize=get_dim(ratio=1.5), sharex=True)

    # 10 dark colors
    colors = [
        "black",
        "red",
        "green",
        "blue",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    for i, v in enumerate(var):
        for j in range(10):
            axs[i].plot(
                ds[v].values[j * 12 : (j + 1) * 12], linewidth=0.5, color=colors[j]
            )
        axs[i].set_ylabel(var_labels[i] + " [" + units[i] + "]")
        # blank x ticks axs[i].set_xticks([])
        # get error message now with new version of matplotlib

    axs[-1].set_xlabel("Month")
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Aug",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    axs[-1].set_xticks(
        np.arange(0, 12),
        months,
    )
    plt.xlim(0, 11)
    label_subplots(axs)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "new_orleans_seasons.pdf"))
    plt.clf()
    plt.close()
    # let's plot the temperature profile and humidity profile
    fig, axs = plt.subplots(1, 2, figsize=get_dim(ratio=0.7), sharey=True)
    # four dark colors
    colors = ["black", "red", "green", "blue"]
    # ds["t"] -= 273.15 # OC has already been taken away
    for i, j in enumerate(list(range(0, 12, 3))):
        k = (j - 2) % 12
        ids = ds.isel(time=k)
        axs[0].plot(
            ids["t"].values,
            ids["p"].values,
            color=colors[i],
            linewidth=0.5,
            label=months[k],
        )
        axs[0].scatter(ids["t0"].values, ids["otl"].values, color=colors[i], marker="x")
        axs[1].plot(ids["q"].values, ids["p"].values, color=colors[i], linewidth=0.5)

    axs[0].set_ylabel("Pressure level [hPa]")
    axs[0].set_xlabel(r"Temperature [$^{\circ}$C]")
    axs[1].set_xlabel("Specific humidity [g/kg]")
    label_subplots(axs, override="outside")
    axs[0].legend()
    # set a legend above the plot
    # axs[0].legend(loc="lower center", bbox_to_anchor=(1, 1.1))
    # reverse the y-axis
    axs[0].invert_yaxis()
    axs[0].set_ylim(1000, 0)  # between 0 and 1000 hPa
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "new_orleans_vertical_profiles.pdf"))
    plt.clf()
    plt.close()


def place_to_position(place: str) -> tuple:
    """
    Convert a place name to a latitude and longitude position.

    Args:
        place (str): The name of the place.

    Returns:
        tuple: A tuple containing the latitude and longitude.
    """
    return (
        OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
        OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
    )


def plot_spatials(
    axs: np.ndarray,
    place: str = "new_orleans",
    vars: Tuple[str, str] = ("vmax_3", "r0_3", "rmax_3", "rmax_1"),
    labels: tuple = (
        "Potential intensity, $V_{p}$ [m s$^{-1}$]",
        "PI potential outer size, $r_{a3}$ [km]",
        "PI potential inner size, $r_{3}$ [km]",
        "Cat1 potential inner size, $r_{1}$ [km]",
    ),
    pi_version: int = 4,
    trial=1,
    pressure_assumption="isothermal",
    model: str = "HADGEM3-GC31-MM",
    member: str = "r1i1p1f3",
) -> None:
    assert len(axs) == len(vars)

    name = f"august_cmip6_{model}_{member}_pi{pi_version}_{pressure_assumption}_trial{trial}.nc"

    if place in {"new_orleans", "galverston", "miami"}:
        name = "gom_" + name
    if place in {"shanghai", "hong_kong", "hanoi"}:
        name = "scs_" + name
    ds = xr.open_dataset(os.path.join(DATA_PATH, name), engine="h5netcdf")
    print("spatial ds", ds)

    ds["lon"].attrs = {"units": "$^{\circ}E$", "long_name": "Longitude"}
    ds["lat"].attrs = {"units": "$^{\circ}N$", "long_name": "Latitude"}
    for i, var in enumerate(vars):
        if var in ds:
            if var in {"r0", "rmax", "rmax_1", "rmax_3", "r0_1", "r0_3"}:
                (ds[var] / 1000).plot(ax=axs[i], cbar_kwargs={"label": ""})
            else:
                ds[var].plot(ax=axs[i], cbar_kwargs={"label": ""})
            axs[i].set_title(labels[i])
        point = place_to_position(place)
        axs[i].scatter(*point, color="black", s=75, marker="x")
        if i != len(vars) - 1:
            axs[i].set_xlabel("")


def plot_two_spatial(
    axs: np.ndarray,
    place: str = "new_orleans",
    pi_version: int = 4,
    trial=1,
    pressure_assumption="isothermal",
) -> None:
    """
    Plot the potential intensity and size for the Gulf of Mexico area.

    Args:
        axs (np.ndarray): axes to plot on. Should be two.
    """
    assert len(axs) == 2
    plot_defaults()

    name = f"august_cmip6_pi{pi_version}_{pressure_assumption}_trial{trial}.nc"

    if place in {"new_orleans", "galverston", "miami"}:
        name = "gom_" + name
    if place in {"shanghai", "hong_kong", "hanoi"}:
        name = "scs_" + name

    ds = xr.open_dataset(os.path.join(DATA_PATH, name))
    # print("ds", ds)
    ds["lon"].attrs = {"units": "$^{\circ}E$", "long_name": "Longitude"}
    ds["lat"].attrs = {"units": "$^{\circ}N$", "long_name": "Latitude"}
    print(ds)
    print(ds["sst"])
    print("two lats", ds["sst"].isel(lat=slice(0, 2)).values.shape)

    vmaxs = ds["vmax"].values
    lats = ds["lat"].values
    ssts = ds["sst"].values
    r0s = ds["r0"].values

    print("vmaxs", vmaxs.shape)
    print("lats", lats.shape)
    print("ssts", ssts.shape)
    print("r0s", r0s.shape)
    lats = np.array([lats.tolist() for _ in range(len(ds.lon))]).T
    assert np.shape(lats) == np.shape(ssts)
    ssts = ssts.ravel()
    r0s = r0s.ravel()
    lats = lats.ravel()
    vmaxs = vmaxs.ravel()
    rho = safe_corr(lats, ssts)
    fit_space_sst_lat = safe_grad(lats, ssts)
    print(
        "space (sst, lat) $m={:.1eL}$ ".format(fit_space_sst_lat)
        + "$^{\circ}$C  $^{\circ}$N$^{-1}$",
    )
    print("space rho (sst, lat): {:.2f}".format(rho))
    lats = lats[~np.isnan(r0s)]
    ssts = ssts[~np.isnan(r0s)]
    vmaxs = vmaxs[~np.isnan(r0s)]
    coriolis_fs = coriolis_parameter_from_lat(lats)
    r0s = r0s[~np.isnan(r0s)]
    fit_space_r0s_lats = safe_grad(lats, r0s / 1000)
    print(
        "space (r0s, lat) $m={:.2eL}$ ".format(fit_space_r0s_lats)
        + "km  $^{\circ}$N$^{-1}$",
    )
    rho = safe_corr(lats, r0s)
    print("space rho (r0s, lat): {:.2f}".format(rho))
    fit_space_vmaxs_lats = safe_grad(lats, vmaxs)
    print(
        "space (vmax, lat) $m={:.1eL}$ ".format(fit_space_vmaxs_lats)
        + "m s$^{-1}$  $^{\circ}$N$^{-1}$",
    )
    fit_space_vmaxs_ssts = safe_grad(ssts, vmaxs)
    print(
        "space (vmax, sst) $m={:.1eL}$ ".format(fit_space_vmaxs_ssts)
        + "m s$^{-1}$  $^{\circ}$C$^{-1}$",
    )
    rho = safe_corr(ssts, vmaxs)
    print("space rho (sst, vmax): {:.2f}".format(rho))
    vmax_div_coriolis = vmaxs / coriolis_fs
    rho = safe_corr(vmax_div_coriolis, r0s)
    print("space rho (vmax/coriolis, r0): {:.2f}".format(rho))
    fit_space_vmaxs_div_coriolis_r0 = safe_grad(vmax_div_coriolis, r0s)

    print(
        "space (vmax/coriolis, r0) $m={:.2eL}$ ".format(fit_space_vmaxs_div_coriolis_r0)
        + "km m$^{-1}$ s$^2$$",
    )
    (ds["r0"] / 1000).plot(ax=axs[1], cbar_kwargs={"label": ""})
    ds["vmax"].plot(ax=axs[0], cbar_kwargs={"label": ""})
    point = place_to_position(place)
    axs[0].scatter(*point, color="black", s=75, marker="x")
    axs[1].scatter(*point, color="black", s=75, marker="x")
    axs[0].set_title("Potential intensity, $V_{p}$ [m s$^{-1}$]")
    axs[1].set_title("Potential size, $r_a$ [km]")
    axs[0].set_xlabel("")


def get_cmip6_timeseries(
    place: str = "new_orleans",
    pressure_assumption: str = "isothermal",
    model: str = "CESM2",
    member: Union[int, str] = 4,
    pi_version: int = 4,
) -> xr.Dataset:
    """Get the CMIP6 timeseries dataset for the given place and potential intensity version.

    Args:
        place (str): The place to get the timeseries for (default is "new_orleans").
        pressure_assumption (str): The pressure assumption to use (default is "isothermal").
        member (int): The ensemble member to use (default is 4).
        pi_version (int): The potential intensity version to use (default is 4).
    Returns:
        xr.Dataset: The timeseries dataset.
    """
    if isinstance(member, int):
        member = f"r{member}i1p1f1"
    ds = xr.concat(
        [
            xr.open_dataset(
                os.path.join(
                    DATA_PATH,
                    f"{place}_august_{exp}_{model}_{member}_{pressure_assumption}_pi{pi_version}new.nc",
                )
            )
            for exp in ["historical", "ssp585"]
        ],
        dim="time",
    )
    print("cmip6_ds", ds)
    if isinstance(ds["time"].values[0], np.datetime64):
        # convert time to year
        ds = ds.assign_coords(
            time=[
                t.astype("datetime64[Y]").astype(int) + 1970 for t in ds["time"].values
            ]
        )
    else:
        ds = ds.assign_coords(time=[t.year for t in ds["time"].values])
    return ds


def plot_timeserii(
    axs: np.ndarray,
    vars: tuple = ("vmax_3", "r0_3", "rmax_3", "rmax_1"),
    labels: tuple = (
        "Potential intensity, $V_{p}$ [m s$^{-1}$]",
        "PI potential outer size, $r_{a3}$ [km]",
        "PI potential inner size, $r_{3}$ [km]",
        "Cat1 potential inner size, $r_{1}$ [km]",
    ),
    linewidth: float = 0.5,
    color: str = "black",
    member: int = 4,
    model: str = "CESM2",
    pressure_assumption: str = "isothermal",
    place: str = "new_orleans",
    pi_version: int = 4,
    year_min: int = 2014,
    year_max: int = 2100,
) -> None:
    assert len(axs) == len(vars)
    assert len(labels) == len(vars)
    if model == "ERA5":
        ds = get_era5_timeseries(place=place, pi_version=pi_version)
    else:
        ds = get_cmip6_timeseries(
            place=place,
            pressure_assumption=pressure_assumption,
            member=member,
            model=model,
            pi_version=pi_version,
        )
    for i in range(len(vars)):
        axs[i].set_title(vars[i])
        var_np = ds[vars[i]].values
        if vars[i] in {"r0", "rmax", "rmax_1", "rmax_3", "r0_1", "r0_3"}:
            var_np /= 1000

        axs[i].plot(
            # np.array(
            #     [t.astype("datetime64[Y]").astype(int) + 1970 for t in ds.time.values]
            # ),
            ds["time"].values,
            var_np,
            color=color,
            linewidth=linewidth,
            alpha=0.5,
        )
        axs[i].set_title(labels[i])
        axs[i].set_xlabel("")
        if i == len(vars) - 1:
            axs[i].set_xlabel("Year")
        axs[i].set_xlim([1850, 2100])
        # vertical black line at year_min
        axs[i].axvline(int(year_min), color="black", linestyle="--", linewidth=0.5)


def plot_two_timeseries(
    axs: np.ndarray,
    text: bool = True,
    color: str = "black",
    member: int = 4,
    model: str = "CESM2",
    pressure_assumption: str = "isothermal",
    place: str = "new_orleans",
    pi_version: int = 4,
    year_min: int = 2014,
    year_max: int = 2100,
) -> None:
    """Plot the potential intensity and size timeseries for the point near New Orleans for different ensemble members.

    Args:
        axs (np.ndarray): The axes to plot on.
        text (bool): Whether to add text to the plot (default is True).
        color (str): The color of the line (default is "black").
        member (int): The ensemble member to plot (default is 4).

    """

    assert len(axs) == 2
    # put historical and ssp585 together
    timeseries_ds = get_cmip6_timeseries(
        place=place,
        pressure_assumption=pressure_assumption,
        member=member,
        model=model,
        pi_version=pi_version,
    )
    print("timeseries_ds", timeseries_ds)
    axs[0].set_title("Potential intensity, $V_{p}$ [m s$^{-1}$]")
    axs[1].set_title("Potential size, $r_a$ [km]")

    if text:
        df = timeseries_relationships(
            timeseries_ds, place, member, year_min=year_min, year_max=year_max
        )
        axs[0].text(
            0.75, 0.9, f"$\\rho$ = {df['rho_vmax']:.2f}", transform=axs[0].transAxes
        )
        axs[1].text(
            0.75, 0.9, f"$\\rho$ = {df['rho_r0']:.2f}", transform=axs[1].transAxes
        )
        axs[0].text(
            0.60,
            0.05,
            f"$m=$  "
            + "${:.1eL}$".format(ufloat(df["fit_vmax"], df["fit_vmax_err"]))
            + "\n \t\t\t m s$^{-1}$ yr$^{-1}$",
            transform=axs[0].transAxes,
        )
        axs[1].text(
            0.60,
            0.1,
            f"$m=$"
            + "${:.2L}$".format(ufloat(df["fit_r0"], df["fit_r0_err"]))
            + " km yr$^{-1}$",
            transform=axs[1].transAxes,
        )

    axs[0].plot(
        timeseries_ds["time"].values, timeseries_ds["vmax"], color=color, linewidth=0.5
    )
    axs[1].plot(
        timeseries_ds["time"].values,
        timeseries_ds["r0"] / 1000,
        color=color,
        linewidth=0.5,
    )
    axs[0].set_xlabel("")
    axs[1].set_xlabel("Year")

    axs[0].set_xlim([1850, 2100])
    axs[1].set_xlim([1850, 2100])
    # axs[0].set_xticks(np.arange(1850, 2101, 10))
    # axs[0].set_xticklabels(["" for _ in range(10)])
    # axs[1].set_xticks(np.arange(1850, 2101, 10))
    # axs[1].set_xticklabels(np.arange(1850, 2101, 10).astype(str))
    # vertical black line at year_min
    axs[0].axvline(int(year_min), color="black", linestyle="--", linewidth=0.5)
    axs[1].axvline(int(year_min), color="black", linestyle="--", linewidth=0.5)


def get_era5_timeseries(place: str = "new_orleans", pi_version: int = 4) -> xr.Dataset:
    """Get the ERA5 timeseries dataset for the given place and potential intensity version.

    Args:
        place (str): The place to get the timeseries for (default is "new_orleans").
        pi_version (int): The potential intensity version to use (default is 4).

    Returns:
        xr.Dataset: The timeseries dataset.
    """

    def _years_from_times(times: xr.DataArray) -> np.ndarray:
        return times.astype("datetime64[Y]").astype(int) + 1970

    timeseries_ds = xr.open_dataset(
        os.path.join(DATA_PATH, f"{place}_august_era5_{'isothermal'}.nc")
    )
    print("loading era5 timeseries", timeseries_ds)
    return timeseries_ds.assign_coords(
        time=_years_from_times(timeseries_ds["time"].values)
    )


def plot_era5_timeseries(
    axs: np.ndarray,
    place: str = "new_orleans",
    color="black",
    pi_version: int = 4,
) -> None:
    """Plot the potential intensity and size timeseries for the point near New Orleans for ERA5 data.

    Args:
        axs (np.ndarray): The axes to plot on.
        color (str): The color of the line (default is "black").
        place (str): The place to plot (default is "new_orleans").
        pi_version (int): The potential intensity version to use (default is 4).
    """
    assert len(axs) == 2
    timeseries_ds = get_era5_timeseries(place, pi_version)
    axs[0].set_title("Potential intensity, $V_{p}$ [m s$^{-1}$]")
    axs[1].set_title("Potential size, $r_{a3}$ [km]")
    ## work out correlation between time and vmax between 2000 and 2099
    axs[0].plot(
        timeseries_ds["time"].values,
        timeseries_ds["vmax_3"].values,
        color=color,
        linewidth=1,
    )
    axs[1].plot(
        timeseries_ds["time"].values,
        timeseries_ds["r0_3"].values / 1000,
        color=color,
        linewidth=1,
    )


def timeseries_relationships(
    timeseries_ds: xr.Dataset,
    place: str,
    member: int,
    year_min: int = 2014,
    year_max: int = 2100,
) -> pd.DataFrame:

    ssts = timeseries_ds["sst"].sel(time=slice(year_min, year_max)).values
    vmaxs = timeseries_ds["vmax"].sel(time=slice(year_min, year_max)).values
    r0s = timeseries_ds["r0"].sel(time=slice(year_min, year_max)).values
    years = np.array(
        [
            time
            for time in timeseries_ds["time"].sel(time=slice(year_min, year_max)).values
        ]
    )

    rho_vmax = safe_corr(vmaxs, years)
    rho_r0 = safe_corr(r0s, years)
    rho_sst = safe_corr(ssts, years)
    rho_sst_vmax = safe_corr(ssts, vmaxs)
    rho_sst_r0 = safe_corr(ssts, r0s)

    # work out gradient with error bars for same period
    fit_vmax = safe_grad(years, vmaxs)
    fit_r0 = safe_grad(years, r0s / 1000)
    fit_r0_sst = safe_grad(ssts, r0s / 1000)

    # print("fit_r0_sst timeseries", fit_r0_sst, "km $^{\circ}$C$^{-1}$")
    fit_vmax_sst = safe_grad(ssts, vmaxs)
    # print("fit_vmax_sst timeseries", fit_vmax_sst, "m s$^{-1}$C$ ^{-1}$")
    # print("fit_vmax_years timeseries", fit_vmax, "m s$^{-1}$ yr$^{-1}$")
    # print("fit_r0_years timeseries", fit_r0, "km yr$^{-1}$")
    df = pd.DataFrame(
        {
            "place": [place],
            "member": member,
            "year_min": [year_min],
            "year_max": [year_max],
            "rho_vmax": [rho_vmax],
            "rho_r0": [rho_r0],
            "rho_sst": [rho_sst],
            "rho_sst_vmax": [rho_sst_vmax],
            "rho_sst_r0": [rho_sst_r0],
            "fit_vmax": [fit_vmax.n],
            "fit_vmax_err": [fit_vmax.s],
            "fit_r0": [fit_r0.n],
            "fit_r0_err": [fit_r0.s],
            "fit_r0_sst": [fit_r0_sst.n],
            "fit_r0_sst_err": [fit_r0_sst.s],
            "fit_vmax_sst": [fit_vmax_sst.n],
            "fit_vmax_sst_err": [fit_vmax_sst.s],
        }
    )
    return df


def figure_two(place: str = "new_orleans") -> None:
    """Plot the solution for the GOM bbox for potential size and intensity.

    Then plot New Orleans timeseries.
    """
    plot_defaults()
    _, axs = plt.subplots(
        2,
        2,
        figsize=get_dim(fraction_of_line_width=1.5),
        width_ratios=[1, 1.5],
        height_ratios=[1, 1],
    )
    plot_two_spatial(axs[:, 0], place=place)
    plot_two_timeseries(axs[:, 1], text=False, color="orange", place=place, member=4)
    plot_two_timeseries(axs[:, 1], text=False, color="orange", member=10, place=place)
    plot_two_timeseries(axs[:, 1], text=False, color="orange", place=place, member=11)
    for member in ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"]:
        plot_two_timeseries(
            axs[:, 1],
            text=False,
            color="green",
            place=place,
            model="HADGEM3-GC31-MM",
            member=member,
        )
    for member in [1, 2, 3]:
        plot_two_timeseries(
            axs[:, 1],
            text=False,
            color="grey",
            place=place,
            model="MIROC6",
            member=member,
        )

    plot_era5_timeseries(
        axs[:, 1],
        color="blue",
        place=place,
        pi_version=4,
    )
    label_subplots(axs)
    if place == "new_orleans":
        plt.savefig(os.path.join(FIGURE_PATH, "figure_two.pdf"))
    else:
        plt.savefig(os.path.join(FIGURE_PATH, f"figure_two_{place}.pdf"))

    plt.clf()
    plt.close()


def multipanel(
    place: str = "new_orleans",
    vars: Tuple[str] = ("vmax_3", "r0_3", "rmax_3", "rmax_1"),
    models: set = {"CESM2", "HADGEM3-GC31-MM", "MIROC6", "ERA5"},
):
    """Plot the multipanel figure for the CMIP6 timeseries.

    Args:
        vars (tuple): The variables to plot (default is ("vmax_3", "r0_3", "rmax_3", "rmax_1")).
    """
    plot_defaults()
    _, axs = plt.subplots(
        len(vars),
        2,
        figsize=get_dim(ratio=1.5),
    )
    plot_spatials(axs[:, 0], place=place, vars=vars)
    if "CESM2" in models:
        members = [4, 10, 11]
        for i, member in enumerate(members):
            plot_timeserii(
                axs[:, 1],
                vars=vars,
                color="orange",
                member=member,
                model="CESM2",
                pressure_assumption="isothermal",
                place=place,
                pi_version=4,
                year_min=2014,
                year_max=2100,
            )
    if "HADGEM3-GC31-MM" in models:
        members = ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"]
        for i, member in enumerate(members):
            plot_timeserii(
                axs[:, 1],
                vars=vars,
                color="green",
                member=member,
                model="HADGEM3-GC31-MM",
                pressure_assumption="isothermal",
                place=place,
                pi_version=4,
                year_min=2014,
                year_max=2100,
            )
    if "MIROC6" in models:
        members = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
        for i, member in enumerate(members):
            plot_timeserii(
                axs[:, 1],
                vars=vars,
                color="grey",
                member=member,
                model="MIROC6",
                pressure_assumption="isothermal",
                place=place,
                pi_version=4,
                year_min=2014,
                year_max=2100,
            )

    if "ERA5" in models:
        plot_timeserii(
            axs[:, 1],
            vars=vars,
            color="blue",
            member="era5",
            model="ERA5",
            linewidth=1,
            place=place,
            pi_version=4,
        )

    label_subplots(axs)
    plt.savefig(os.path.join(FIGURE_PATH, f"{place}_multipanel.pdf"))
    plt.clf()
    plt.close()


def temporal_relationship_data(place: str = "new_orleans", pi_version: int = 4) -> None:
    """Get the temporal relationships data for the given place and potential intensity version.

    Args:
        place (str): The place to get the data for (default is "new_orleans").
        pi_version (int): The potential intensity version to use (default is 4).
    """
    df_l = []
    for member in [4, 10, 11]:
        timeseries_ds = get_cmip6_timeseries(
            place=place, member=member, pi_version=pi_version
        )
        df_l.append(
            timeseries_relationships(
                timeseries_ds,
                place=place,
                member="r" + str(member) + "i1p1f1",
                year_min=2014,
                year_max=2100,
            )
        )
        df_l.append(
            timeseries_relationships(
                timeseries_ds,
                place=place,
                member="r" + str(member) + "i1p1f1",
                year_min=1980,
                year_max=2024,
            )
        )
    timeseries_ds = get_era5_timeseries(place=place, pi_version=pi_version)
    df_l.append(
        timeseries_relationships(
            timeseries_ds,
            place=place,
            member="era5",
            year_min=1980,
            year_max=2024,
        )
    )
    df_l.append(
        timeseries_relationships(
            timeseries_ds,
            place=place,
            member="era5",
            year_min=1940,
            year_max=2024,
        )
    )
    df = pd.concat(df_l, ignore_index=True)
    from .constants import DATA_PATH

    df.to_csv(
        os.path.join(
            DATA_PATH, f"{place}_temporal_relationships_pi{pi_version}new.csv"
        ),
        index=False,
    )
    print("Saved temporal relationships data to CSV.")
    # now let's save this pandas dataframe to a LaTeX table.
    # it needs to be formatted so that the headers are renamed to formal names with units
    # it is in scientific notation with 2 significant figures
    # it combines the nominal and error values into one column like \(nominal \pm error\)
    # the error if it exists in in "{name}_err" where name is the nominal name
    # the headers should be transformed so that "fit_vmax" and "fit_vmax_err" become "\(m\left(t, V_p\right)\) [m s\(^{-1}\)]"
    # the headers should be transformed so that "rho_vmax" becomes "\(\\rho\left(t, V_p\right)\)"
    # the headers should be transformed so that "fit_vmax_sst" and "fit_vmax_sst_err" become "\(m\left(T_s, V_p\right)\) [m s\(^{-1}\) \(^{\circ}\)C\(^{-1}\)]"
    # the labels name, members, year_min, year_max should not be changed
    # dictionary of variable to symbol transformations:
    # vmax -> V_p
    # r0 -> r_a
    # sst -> T_s
    # t0 -> T_0
    # rmax -> r_{\mathrm{max}}
    # rh -> \mathcal{H}_e
    # p -> p_a
    # msl -> p_a
    df.drop(columns=["place"], inplace=True)

    # data frame was getting too large for latex table, so splitting into correlations and fits
    df_str = dataframe_to_latex_table(
        df[[col for col in df.columns if not col.startswith("fit_")]]
    )
    print(df_str)
    with open(
        os.path.join(DATA_PATH, f"{place}_temporal_correlation_pi{pi_version}new.tex"),
        "w",
    ) as f:
        f.write(df_str)
    print("Saved temporal relationships data to LaTeX table.")

    df_str = dataframe_to_latex_table(
        df[[col for col in df.columns if not col.startswith("rho_")]]
    )
    print(df_str)
    with open(
        os.path.join(DATA_PATH, f"{place}_temporal_fit_pi{pi_version}new.tex"), "w"
    ) as f:
        f.write(df_str)
    print("Saved temporal relationships data to LaTeX table.")


def dataframe_to_latex_table(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a publication-quality LaTeX table string.

    This function automatically generates formal LaTeX headers and formats
    numerical values into a human-readable scientific notation suitable for
    academic papers.

    Key features:
    - **Dynamic Header Generation**: Parses column names like 'rho_vmax' or
      'fit_r0_sst' into LaTeX expressions, e.g., '\\(\\rho(V_p)\\)'.
    - **Advanced Number Formatting**:
        - Single numbers are formatted as \\(m \\times 10^{e}\\).
        - Value ± error pairs are formatted as \\((\\textit{m}_n \\pm \\textit{m}_e) \\times 10^{E}\\),
          where the exponent is factored out and values are rounded
          systematically based on the error's magnitude.
    - **Custom Table Style**: Uses '\\topline', '\\midline', '\\bottomline' for
      table rules.

    Args:
        df (pd.DataFrame): The input DataFrame. Column names are expected to
            follow conventions like 'rho_{...}' and 'fit_{...}'.

    Returns:
        str: A string containing the fully formatted LaTeX table.

    Doctest:
        >>> # Example DataFrame mimicking a typical analysis output
        >>> data = {
        ...     'place': ['Atlantic'],
        ...     'member': ['member_01'],
        ...     'rho_vmax': [0.891],
        ...     'fit_vmax': [43.2],
        ...     'fit_vmax_err': [5.73],
        ...     'fit_r0_sst': [-1.54],
        ...     'fit_r0_sst_err': [0.11],
        ... }
        >>> df_test = pd.DataFrame(data)

    """
    # --- Local Helper Functions for Advanced Formatting ---

    def format_single_latex_sci(value, sig_figs=2):
        """Formats a single number as m x 10^e."""
        if value == 0 or not math.isfinite(value):
            return f"\\({0.0:.{sig_figs-1}f}\\)"

        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / 10**exponent

        # Round mantissa to specified significant figures
        mantissa = round(mantissa, sig_figs - 1)

        # Correct for rounding rollovers (e.g., 9.99 -> 10.0)
        if abs(mantissa) >= 10.0:
            mantissa /= 10.0
            exponent += 1

        mantissa_str = f"{{:.{sig_figs-1}f}}".format(mantissa)

        if exponent == 0:
            return f"\\({mantissa_str}\\)"
        return f"\\({mantissa_str} \\times 10^{{{exponent}}}\\)"

    def format_error_latex_sci(nominal, error):
        """Formats a nominal ± error pair with a common exponent."""
        if error == 0 or not math.isfinite(error) or not math.isfinite(nominal):
            return format_single_latex_sci(nominal) + " \\pm 0"

        # Determine the common exponent from the nominal value
        exponent = (
            math.floor(math.log10(abs(nominal)))
            if nominal != 0
            else math.floor(math.log10(abs(error)))
        )

        # Rescale numbers to the common exponent
        nominal_rescaled = nominal / (10**exponent)
        error_rescaled = error / (10**exponent)

        # Determine decimal places for rounding from the error's first significant digit
        if error_rescaled == 0:
            rounding_decimals = 1
        else:
            rounding_decimals = -math.floor(math.log10(abs(error_rescaled)))

        # Round the rescaled numbers to the determined decimal place
        nominal_rounded = round(nominal_rescaled, rounding_decimals)
        error_rounded = round(error_rescaled, rounding_decimals)

        fmt_str = f"{{:.{rounding_decimals}f}}"
        nominal_str = fmt_str.format(nominal_rounded)
        error_str = fmt_str.format(error_rounded)

        if exponent == 0:
            return f"\\({nominal_str} \\pm {error_str}\\)"
        return f"\\(\\left({nominal_str} \\pm {error_str}\\right)\\times 10^{{{exponent}}}\\)"

    # --- Main Function Logic ---
    df_proc = df.copy()

    def _generate_header_map(columns: list[str]) -> dict[str, str]:
        symbol_map = {
            "vmax": "V_p",
            "r0": "r_a",
            "sst": "T_s",
            "t0": "T_0",
            "rmax": r"r_{\mathrm{max}}",
            "rh": r"\mathcal{H}_e",
            "p": "p_a",
            "msl": "p_a",
            "years": "t",
        }
        unit_map = {
            "vmax": r"\text{m s}^{-1}",
            "r0": "\\text{km}",
            "sst": r"^{\circ}\text{C}",
            "years": "\\text{yr}",
        }
        header_map = {}
        for col in columns:
            if col.startswith("rho_"):
                parts = col.split("_")[1:]
                symbols = [symbol_map.get(p, p) for p in parts]
                if len(symbols) == 1:
                    symbols.append("t")
                header_map[col] = f"\\(\\rho({', '.join(symbols)})\\)"
            elif col.startswith("fit_"):
                parts = col.split("_")[1:]
                dep_var, ind_var = parts[0], parts[1] if len(parts) > 1 else "years"
                dep_sym, ind_sym = symbol_map.get(dep_var, dep_var), symbol_map.get(
                    ind_var, ind_var
                )
                dep_unit, ind_unit = unit_map.get(dep_var), unit_map.get(ind_var)
                unit_str = (
                    f" [\\({dep_unit} \;{ind_unit}^{{-1}}\\)]"
                    if dep_unit and ind_unit
                    else ""
                )
                header_map[col] = f"\\(m({ind_sym}, {dep_sym})\\){unit_str}"
        header_map["member"] = "Member"
        header_map["place"] = "Place"
        header_map["year_min"] = "Start"
        header_map["year_max"] = "End"
        return header_map

    header_map = _generate_header_map(list(df_proc.columns))
    err_cols_to_drop = []

    for col in df_proc.columns:
        err_col = f"{col}_err"
        if err_col in df_proc.columns:
            df_proc[col] = df_proc.apply(
                lambda row: (
                    format_error_latex_sci(row[col], row[err_col])
                    if pd.notnull(row[col]) and pd.notnull(row[err_col])
                    else ""
                ),
                axis=1,
            )
            err_cols_to_drop.append(err_col)
        elif col in header_map and not col in [
            "place",
            "member",
            "year_min",
            "year_max",
        ]:
            df_proc[col] = df_proc[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else ""
            )

    df_proc.drop(columns=err_cols_to_drop, inplace=True)

    final_order = [col for col in df.columns if not col.endswith("_err")]
    df_proc.rename(columns=header_map, inplace=True)
    final_renamed_order = [header_map.get(col, col) for col in final_order]
    df_proc = df_proc[final_renamed_order]

    col_format = "l" * len(df_proc.columns)
    latex_str = df_proc.to_latex(
        index=False, escape=False, header=True, column_format=col_format, caption=" "
    )

    return latex_str


if __name__ == "__main__":
    # python -m w22.plot
    # plot_panels()
    #
    # figure_two(place="new_orleans")
    # figure_two(place="hong_kong")
    # temporal_relationship_data(place="new_orleans", pi_version=4)
    # plot_seasonal_profiles()
    # years = [2015, 2100]
    # timeseries_plot(
    #     name="new_orleans",
    #     plot_name="New Orleans",
    #     years=years,
    #     members=[4, 10],
    # )
    # timeseries_plot(name="miami", plot_name="Miami", years=years, members=[4, 10])
    # timeseries_plot(
    #     name="galverston",
    #     plot_name="Galverston",
    #     years=years,
    #     members=[4, 10],
    # )
    multipanel(place="new_orleans")
    multipanel(place="hong_kong", models={"HADGEM3-GC31-MM", "MIROC6", "ERA5"})
