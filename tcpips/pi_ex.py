"""Process and plot example data."""

import os
import xarray as xr
import matplotlib.pyplot as plt
from sithom.time import timeit
from sithom.plot import plot_defaults, get_dim
from sithom.misc import in_notebook
from .pi import calculate_pi
from .pi_old import combined_inout_timestep_cmip6
from .convert import convert
from .plot import plot_features
from .constants import FIGURE_PATH, MONTHS, DATA_PATH, GOM


TCPYPI_SAMPLE_DATA: str = (
    "../tcpypi/data/sample_data.nc"  # Sample data for the tcpyPI package
)


@timeit
def plot_tcpypi_in_out_ex() -> None:
    """
    Plot example data from the sample_data.nc file.
    """
    plot_defaults()
    ds = xr.open_dataset(TCPYPI_SAMPLE_DATA).isel(p=0, month=9)
    for var in ds:
        if "standard_name" in ds[var].attrs:
            ds[var].attrs["long_name"] = ds[var].attrs["standard_name"]
    print(ds)
    plot_features(ds, [["sst", "t"], ["msl", "q"]], super_titles=["", ""])
    plt.suptitle("month=9")
    plt.savefig(os.path.join(FIGURE_PATH, "sample-input.png"))
    plt.clf()
    ds = xr.open_dataset(TCPYPI_SAMPLE_DATA).isel(
        month=9
    )  # August? Wait isn't that October?
    pi_ds = calculate_pi(ds, dim="p")
    for var in pi_ds:
        if "standard_name" in pi_ds[var].attrs:
            pi_ds[var].attrs["long_name"] = pi_ds[var].attrs["standard_name"]
    print(pi_ds)
    plot_features(pi_ds, [["vmax", "pmin"], ["t0", "otl"]], super_titles=["", ""])
    plt.suptitle("month=9")
    plt.savefig(os.path.join(FIGURE_PATH, f"sample-pi.png"))
    plt.clf()


def plot_seasonality_in_gom_tcypi_ex() -> None:
    """
    Process sample data for the Gulf of Mexico. Look at the seasonality, etc.
    """
    ds = xr.open_dataset(TCPYPI_SAMPLE_DATA).sel(
        lon=GOM[1], lat=GOM[0], method="nearest"
    )
    print("ds", ds)

    pi_ds = calculate_pi(ds, dim="p")
    for var in pi_ds:
        if "standard_name" in pi_ds[var].attrs:
            pi_ds[var].attrs["long_name"] = pi_ds[var].attrs["standard_name"]
    print("pi_ds", pi_ds)

    plot_defaults()
    fig, axs = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=get_dim(width=500, fraction_of_line_width=0.6, ratio=0.6180 / 0.6),
    )

    axs[0].invert_yaxis()
    markers = ["x"] * 4 + ["+"] * 4 + ["*"] * 4
    lines = ["-"] * 4 + ["--"] * 4 + [":"] * 4
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"] * 2

    for month in range(0, 12):
        axs[0].plot(
            ds.t.isel(month=month), ds.p, lines[month], color=colors[month]
        )  # , label="t")
        axs[0].plot(
            pi_ds.t0.isel(month=month) - 273.15,
            pi_ds.otl.isel(month=month),
            markers[month],
            color=colors[month],
            label="$T_{o}$ " + f"{MONTHS[month]}",
        )
        axs[1].plot(
            ds.q.isel(month=month),
            ds.p,
            lines[month],
            color=colors[month],
            label=MONTHS[month],
        )

    # axs[0].legend()
    axs[0].set_ylabel("Pressure [hPa]")
    axs[0].set_xlabel("Air Temperature, $T_a$ [$^{\circ}$C]")
    axs[1].set_xlabel("Specific Humidity, $q$ [g/kg]")
    # axs[0].legend()
    axs[1].legend()
    axs[1].set_ylim(1000, 0)
    plt.savefig(os.path.join(FIGURE_PATH, "sample-profile.png"))
    plt.clf()
    fig, axs = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=get_dim(width=500, fraction_of_line_width=0.5, ratio=0.6180 / 0.5),
    )
    axs[0].plot(ds.month - 1, pi_ds.vmax, color="black")
    axs[0].set_ylabel("$V_{\mathrm{max}}$ [m s$^{-1}$]")
    axs[1].plot(ds.month - 1, pi_ds.pmin, color="black")
    axs[1].set_ylabel("$P_{\mathrm{min}}$ [hPa]")
    # axs[2].plot(ds.month, pi_ds.t0 - 273.15, color="black")
    # axs[3].plot(ds.month, pi_ds.otl, color="black")
    axs[3].plot(ds.month - 1, ds.sst, color="black")
    axs[3].set_ylabel("$T_{\mathrm{SST}}$ [$^{\circ}$C]")
    axs[2].plot(ds.month - 1, ds.msl, color="black")
    axs[2].set_ylabel("$P_s$ [hPa]")
    axs[3].set_xlabel("Month")
    axs[0].set_xlim(0, 11)
    axs[0].set_xticks(range(0, 12), MONTHS)
    for month in range(0, 12):
        axs[0].plot(
            month, pi_ds.vmax.isel(month=month), markers[month], color=colors[month]
        )
        axs[1].plot(
            month, pi_ds.pmin.isel(month=month), markers[month], color=colors[month]
        )
        axs[3].plot(
            month, ds.sst.isel(month=month), markers[month], color=colors[month]
        )
        axs[2].plot(
            month, ds.msl.isel(month=month), markers[month], color=colors[month]
        )
    # format y ticks to scientific notation
    # axs[1].yaxis.set_major_formatter(axis_formatter())
    # axs[2].yaxis.set_major_formatter(axis_formatter())
    plt.savefig(os.path.join(FIGURE_PATH, "sample-seasons.png"))
    plt.clf()


@timeit
def calc_pi_example() -> None:
    """Calculate the potential intensity using the initial regridded data using the tcpyPI package.

    Use first 5 time steps of the regridded data.
    """
    ds = xr.open_dataset(os.path.join(DATA_PATH, "all_regridded.nc"), engine="h5netcdf")
    input = ds.isel(time=slice(0, 5))  # .bfill("plev").ffill("plev")
    print("input", input)
    input.tos.isel(time=0).plot(x="lon", y="lat")
    if in_notebook():
        plt.show()
    input.hus.isel(time=0, plev=0).plot(x="lon", y="lat")
    plt.show()
    pi_ds = calculate_pi(input, dim="plev")
    print(pi_ds)
    pi_ds.vmax.plot(x="lon", y="lat", col="time", col_wrap=2)
    plt.show()


@timeit
def plot_pi_inputs_map(time: str = "2015-01-15") -> None:
    """
    Plot the inputs to the potential intensity calculation after conversion.

    Args:
        time (str, optional): Time string. Defaults to "2015-01-15".
    """
    plot_defaults()
    ds = convert(combined_inout_timestep_cmip6(time=time).isel(plev=0))
    plot_features(
        ds.isel(y=slice(20, -20)), [["sst", "t"], ["msl", "q"]], super_titles=["", ""]
    )
    plt.suptitle(time)
    plt.savefig(os.path.join(FIGURE_PATH, f"combined-{time}.png"))
    plt.clf()


def plot_pi_outputs_map(time: str = "2015-01-15") -> None:
    """
    Plot the potential intensity calculation results.

    Args:
        time (str, optional): Time string. Defaults to "2015-01-15".
    """
    plot_defaults()
    ds = convert(combined_inout_timestep_cmip6(time=time))
    print(ds)
    pi_ds = calculate_pi(ds, dim="p")
    for var in pi_ds:
        if "standard_name" in pi_ds[var].attrs:
            pi_ds[var].attrs["long_name"] = pi_ds[var].attrs["standard_name"]
    print(pi_ds)
    plot_features(
        pi_ds.isel(y=slice(20, -20)),
        [["vmax", "pmin"], ["t0", "otl"]],
        super_titles=["", ""],
    )
    plt.suptitle(time)
    plt.savefig(os.path.join(FIGURE_PATH, f"pi-{time}.png"))
    plt.clf()
