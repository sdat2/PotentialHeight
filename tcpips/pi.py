"""Potential Intensity Calculation script."""
from typing import Dict, Tuple
import os
import xarray as xr
from tcpyPI import pi
from matplotlib import pyplot as plt
from sithom.time import timeit
from sithom.plot import feature_grid, plot_defaults, get_dim, axis_formatter
from tcpips.constants import FIGURE_PATH
from tcpips.pangeo import convert, regrid_2d_1degree

CKCD: float = 0.9


@timeit
def calculate_pi(ds: xr.Dataset, dim: str = "p") -> xr.Dataset:
    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1),
        input_core_dims=[
            [],
            [],
            [
                dim,
            ],
            [
                dim,
            ],
            [
                dim,
            ],
        ],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
    )

    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    out_ds = xr.Dataset(
        {
            "vmax": vmax,
            "pmin": pmin,
            "ifl": ifl,
            "t0": t0,
            "otl": otl,
        }
    )

    # add names and units to the structure
    out_ds.vmax.attrs["standard_name"], out_ds.vmax.attrs["units"] = (
        "Maximum Potential Intensity",
        "m/s",
    )
    out_ds.pmin.attrs["standard_name"], out_ds.pmin.attrs["units"] = (
        "Minimum Central Pressure",
        "hPa",
    )
    out_ds.ifl.attrs["standard_name"] = "pyPI Flag"
    out_ds.t0.attrs["standard_name"], out_ds.t0.attrs["units"] = (
        "Outflow Temperature",
        "K",
    )
    out_ds.otl.attrs["standard_name"], out_ds.otl.attrs["units"] = (
        "Outflow Temperature Level",
        "hPa",
    )
    return out_ds


@timeit
def calc_pi_example() -> None:
    ds = xr.open_dataset("data/all_regridded.nc", engine="h5netcdf")
    input = ds.isel(time=slice(0, 5))  # .bfill("plev").ffill("plev")
    print("input", input)
    input.tos.isel(time=0).plot(x="lon", y="lat")
    plt.show()
    input.hus.isel(time=0, plev=0).plot(x="lon", y="lat")
    plt.show()
    pi_ds = calculate_pi(input, dim="plev")
    print(pi_ds)
    pi_ds.vmax.plot(x="lon", y="lat", col="time", col_wrap=2)
    plt.show()


def plot_features(
    plot_ds: xr.Dataset, features, units=None, names=None, vlim=None, super_titles=None
) -> None:
    if names is None:
        names = [
            [
                plot_ds[features[x][y]].attrs["long_name"]
                for y in range(len(features[x]))
            ]
            for x in range(len(features))
        ]
    if units is None:
        units = [
            [plot_ds[features[x][y]].attrs["units"] for y in range(len(features[x]))]
            for x in range(len(features))
        ]
    if vlim is None:
        vlim = [[None for y in range(len(features[x]))] for x in range(len(features))]
    if super_titles is None:
        super_titles = ["" for x in range(len(features))]

    feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )


@timeit
def plot_combined(time: str = "2015-01-15") -> None:
    plot_defaults()
    ds = convert(combined_data_timestep(time=time).isel(plev=0))
    plot_features(
        ds.isel(y=slice(20, -20)), [["sst", "t"], ["msl", "q"]], super_titles=["", ""]
    )
    plt.suptitle(time)
    plt.savefig(os.path.join(FIGURE_PATH, f"combined-{time}.png"))
    plt.clf()


def plot_pi(time: str = "2015-01-15") -> None:
    plot_defaults()
    ds = convert(combined_data_timestep(time=time))
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


def elevate_standards(ds: xr.Dataset) -> xr.Dataset:
    for var in ds:
        if "standard_name" in ds[var].attrs:
            ds[var].attrs["long_name"] = ds[var].attrs["standard_name"]
    return ds


def propagate_names(ds_old: xr.Dataset, ds_new: xr.Dataset) -> xr.Dataset:
    for var in ds_old:
        if var in ds_new:
            if "units" in ds_old[var].attrs:
                ds_new[var].attrs["units"] = ds_old[var].attrs["units"]
            elif "units" not in ds_new[var].attrs:
                ds_new[var].attrs["units"] = ""
            if "long_name" in ds_old[var].attrs:
                ds_new[var].attrs["long_name"] = ds_old[var].attrs["long_name"]
            if "standard_name" in ds_old[var].attrs:
                ds_new[var].attrs["standard_name"] = ds_old[var].attrs["standard_name"]
    return ds_new


@timeit
def plot_diffs(times: Tuple[str, str] = ("1850-09-15", "2099-09-15")) -> None:
    plot_defaults()
    ds_l = [convert(combined_data_timestep(time=time)) for time in times]
    pi_ds_l = [elevate_standards(calculate_pi(ds, dim="p")) for ds in ds_l]
    diff_ds = ds_l[1] - ds_l[0]
    pi_diff_ds = pi_ds_l[1] - pi_ds_l[0]
    diff_ds = propagate_names(ds_l[0], diff_ds)
    pi_diff_ds = propagate_names(pi_ds_l[0], pi_diff_ds)
    print(diff_ds)
    print(pi_diff_ds)
    plot_features(
        pi_diff_ds.isel(y=slice(20, -20)),
        [["vmax", "pmin"], ["t0", "otl"]],
        super_titles=["", ""],
    )
    plt.suptitle(f"{times[1]}-{times[0]}")
    plt.savefig(os.path.join(FIGURE_PATH, f"pi-diff-{times[1]}-{times[0]}.png"))
    plt.clf()
    plot_features(
        diff_ds.isel(y=slice(20, -20)).isel(p=0),
        [["sst", "t"], ["msl", "q"]],
        super_titles=["", ""],
    )
    plt.suptitle(f"{times[1]}-{times[0]}")
    plt.savefig(os.path.join(FIGURE_PATH, f"diff-{times[1]}-{times[0]}.png"))
    plt.clf()


@timeit
def plot_example() -> None:
    plot_defaults()
    ds = xr.open_dataset("../tcpypi/data/sample_data.nc").isel(p=0, month=9)
    for var in ds:
        if "standard_name" in ds[var].attrs:
            ds[var].attrs["long_name"] = ds[var].attrs["standard_name"]
    print(ds)
    plot_features(ds, [["sst", "t"], ["msl", "q"]], super_titles=["", ""])
    plt.suptitle("month=9")
    plt.savefig(os.path.join(FIGURE_PATH, "sample-input.png"))
    plt.clf()
    ds = xr.open_dataset("../tcpypi/data/sample_data.nc").isel(month=9)
    pi_ds = calculate_pi(ds, dim="p")
    for var in pi_ds:
        if "standard_name" in pi_ds[var].attrs:
            pi_ds[var].attrs["long_name"] = pi_ds[var].attrs["standard_name"]
    print(pi_ds)
    plot_features(pi_ds, [["vmax", "pmin"], ["t0", "otl"]], super_titles=["", ""])
    plt.suptitle("month=9")
    plt.savefig(os.path.join(FIGURE_PATH, f"sample-pi.png"))
    plt.clf()


def gom() -> None:
    GOM = (25.443701, -90.013120)
    ds = xr.open_dataset("../tcpypi/data/sample_data.nc").sel(
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
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for month in range(0, 12):
        axs[0].plot(
            ds.t.isel(month=month), ds.p, lines[month], color=colors[month]
        )  # , label="t")
        axs[0].plot(
            pi_ds.t0.isel(month=month) - 273.15,
            pi_ds.otl.isel(month=month),
            markers[month],
            color=colors[month],
            label="$T_{o}$ " + f"{months[month]}",
        )
        axs[1].plot(
            ds.q.isel(month=month),
            ds.p,
            lines[month],
            color=colors[month],
            label=months[month],
        )  # , label="q")

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
    axs[0].set_xticks(range(0, 12), months)
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


def combined_data_timestep(time: str = "2015-01-15") -> xr.Dataset:
    def open(name: str) -> xr.Dataset:
        ds = xr.open_dataset(name, engine="h5netcdf").sel(time=time).isel(time=0)
        return ds.drop_vars([x for x in ["time", "time_bounds", "nbnd"] if x in ds])

    atmos_ds = open("data/atmos_new_regridded.nc")
    ocean_ds = open("data/ocean_regridded.nc")
    return xr.merge([ocean_ds, atmos_ds])


if __name__ == "__main__":
    gom()
    # python tcpips/pi.py
    # get_ocean()
    # get_ocean()
    # get_all()
    # regrid_2d_1degree()
    # calc_pi_example()
    # ds = xr.open_dataset("data/ocean_regridded.nc")
    # ds.tos.isel(time=0).plot(x="lon", y="lat")
    # plt.show()
    # plot_example()

    # plot_diffs()

    # for time in ["1850-09-15", "1950-09-15", "2015-09-15", "2099-09-15"]:
    #    # print(convert(combined_data_timestep(time=time)))
    #    plot_pi(time=time)
    # print(combined_data_timestep(time="2015-01-15"))
    # print(combined_data_timestep(time="1850-01-15"))
    # print(combined_data_timestep(time="2099-01-15"))

    # atmos_ds.psl.isel(time=0).plot(x="lon", y="lat")
    # plt.show()

    # calc_pi_example()
    # regrid_2d()
    # print(xr.open_dataset("data/ocean.nc"))
