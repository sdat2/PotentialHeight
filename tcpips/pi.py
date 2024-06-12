"""Potential Intensity Calculation script."""

from typing import Dict, Tuple, List, Optional
import os
import xarray as xr
from tcpyPI import pi
from matplotlib import pyplot as plt
from sithom.time import timeit
from sithom.plot import (
    feature_grid,
    plot_defaults,
    get_dim,
    label_subplots,
)  # , axis_formatter
from sithom.place import Point, BoundingBox
from sithom.misc import in_notebook
from tcpips.constants import FIGURE_PATH, GOM, MONTHS, DATA_PATH
from tcpips.convert import convert  # , regrid_2d_1degree

TCPYPI_SAMPLE_DATA: str = (
    "../tcpypi/data/sample_data.nc"  # Sample data for the tcpyPI package
)
CKCD: float = 0.9  # Enthalpy exchange coefficient / drag coefficient [dimensionless]
PTOP: float = 50.0  # Top pressure level for the calculation [hPa]


@timeit
def calculate_pi(ds: xr.Dataset, dim: str = "p") -> xr.Dataset:
    """Calculate the potential intensity using the tcpyPI package.

    Data must have been converted to the tcpyPI units by `tcpips.convert'.

    Args:
        ds (xr.Dataset): xarray dataset containing the necessary variables.
        dim (str, optional): Vertical dimension. Defaults to "p" for pressure level.

    Returns:
        xr.Dataset: xarray dataset containing the calculated variables.
    """
    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=PTOP, miss_handle=1),
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
    out_ds.ifl.attrs["standard_name"] = "tcpyPI Flag"
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
    """Calculate the potential intensity using the initial regridded data using the tcpyPI package."""
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


def plot_features(
    plot_ds: xr.Dataset,
    features: List[List[str]],
    units: Optional[List[List[str]]] = None,
    names: Optional[List[List[str]]] = None,
    vlim: Optional[List[List[Tuple[str, float, float]]]] = None,
    super_titles: Optional[List[str]] = None,
) -> None:
    """
    A wrapper around the feature_grid function to plot the features of a dataset for the potential intensity inputs/outputs.

    Args:
        plot_ds (xr.Dataset): The dataset to plot data from.
        features (List[List[str]]): List of feature names to plot.
        units (Optional[List[List[str]]], optional): Units to plot. Defaults to None.
        names (Optional[List[List[str]]], optional): Names to plot. Defaults to None.
        vlim (Optional[List[List[Tuple[str, float, float]]]], optional): Colormap/vlim to plot. Defaults to None.
        super_titles (Optional[List[str]], optional): Supertitles to plot. Defaults to None.
    """

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

    fig, axs = feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    label_subplots(axs)


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


def standard_name_to_long_name(ds: xr.Dataset) -> xr.Dataset:
    """
    Turn the standard_name attribute into a long_name attribute.

    Args:
        ds (xr.Dataset): dataset with standard_name attributes.

    Returns:
        xr.Dataset: dataset with long_name attributes.
    """
    for var in ds:
        if "standard_name" in ds[var].attrs:
            ds[var].attrs["long_name"] = ds[var].attrs["standard_name"]
    return ds


def propagate_attrs(ds_old: xr.Dataset, ds_new: xr.Dataset) -> xr.Dataset:
    """
    Propagate the standard_name and units attributes from one dataset to another.

    Args:
        ds_old (xr.Dataset): dataset with standard_name and units attributes.
        ds_new (xr.Dataset): dataset with standard_name and units attributes.

    Returns:
        xr.Dataset: dataset with standard_name and units attributes.
    """

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
    """
    Plot the difference maps between two dates.

    Args:
        times (Tuple[str, str], optional): Defaults to ("1850-09-15", "2099-09-15").
    """
    plot_defaults()
    ds_l = [convert(combined_inout_timestep_cmip6(time=time)) for time in times]
    pi_ds_l = [standard_name_to_long_name(calculate_pi(ds, dim="p")) for ds in ds_l]
    diff_ds = ds_l[1] - ds_l[0]
    pi_diff_ds = pi_ds_l[1] - pi_ds_l[0]
    diff_ds = propagate_attrs(ds_l[0], diff_ds)
    pi_diff_ds = propagate_attrs(pi_ds_l[0], pi_diff_ds)
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
    ds = xr.open_dataset(TCPYPI_SAMPLE_DATA).isel(month=9)
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
    Process sample data for the Gulf of Mexico.
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


def combined_inout_timestep_cmip6(time: str = "2015-01-15") -> xr.Dataset:
    """
    Combined data from the ocean and atmosphere datasets at a given time.

    Args:
        time (str, optional): _description_. Defaults to "2015-01-15".

    Returns:
        xr.Dataset: combined dataset.
    """

    def open(name: str) -> xr.Dataset:
        ds = xr.open_dataset(name, engine="h5netcdf").sel(time=time).isel(time=0)
        return ds.drop_vars([x for x in ["time", "time_bounds", "nbnd"] if x in ds])

    atmos_ds = open(os.path.join(DATA_PATH, "atmos_new_regridded.nc"))
    ocean_ds = open(os.path.join(DATA_PATH, "ocean_regridded.nc"))
    return xr.merge([ocean_ds, atmos_ds])


def processed_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False
) -> xr.Dataset:
    """
    Process the combined data from the ocean and atmosphere datasets at a given time to produce potential intensity.

    Args:
        time (str, optional): Time string. Defaults to "2015-01-15".
        verbose (bool, optional): Whether to print info. Defaults to False.

    Returns:
        xr.Dataset: Processed dataset.
    """
    ds = combined_inout_timestep_cmip6(time=time)
    lats = ds.lat.values
    lons = ds.lon.values
    ds = ds.drop_vars(["lat", "lon"])
    ds = ds.assign_coords({"lat": ("y", lats[:, 0]), "lon": ("x", lons[0, :])})
    if verbose:
        print("ds with 1d coords", ds)
    ds = ds.swap_dims({"y": "lat", "x": "lon"})
    if verbose:
        print(ds)
    return ds


def gom_combined_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False
) -> xr.Dataset:
    """Get the Gulf of Mexico centre data at a given time.

    Args:
        time (str, optional): Defaults to "2015-01-15".
        verbose (bool, optional): Defaults to False.

    Returns:
        xr.Dataset: Gulf of Mexico centre data at a given time.
    """
    ds = processed_inout_timestep_cmip6(time=time, verbose=verbose)
    # select point closest to GOM centre
    ds = ds.sel(lat=GOM[0], lon=GOM[1], method="nearest")
    ds = convert(ds)
    pi = calculate_pi(ds, dim="p")
    if verbose:
        print(pi)
    return xr.merge([ds, pi])


@timeit
def gom_bbox_combined_inout_timestep_cmip6(
    time: str = "2015-01-15", verbose: bool = False, pad: float = 5
) -> xr.Dataset:
    """Get the Gulf of Mexico bounding box data at a given time.

    Args:
        time (str, optional): Defaults to "2015-01-15".
        verbose (bool, optional): Defaults to False.
        pad (float, optional): Padding around the bounding box. Defaults to 5.

    Returns:
        xr.Dataset: Gulf of Mexico bounding box data at a given time.
    """
    GOM_BBOX: BoundingBox = Point(GOM[1], GOM[0], desc="Gulf of Mexico Centre").bbox(
        pad
    )
    ds = processed_inout_timestep_cmip6(time=time, verbose=verbose)
    ds = ds.sel(lon=slice(*GOM_BBOX.lon), lat=slice(*GOM_BBOX.lat))
    ds = convert(ds)
    pi = calculate_pi(ds, dim="p")
    if verbose:
        print(pi)
    return xr.merge([ds, pi])


def find_atmos_ocean_pairs():
    """
    Find the atmospheric and oceanic data pairs that can be combined to calculate potential intensity.
    """
    import os
    from tcpips.constants import REGRIDDED_PATH

    pairs = {}
    for exp in [
        x
        for x in os.listdir(REGRIDDED_PATH)
        if os.path.isdir(os.path.join(REGRIDDED_PATH, x))
    ]:
        print(exp)
        for model in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean")):
            print(model)
            for member in [
                x.strip(".nc")
                for x in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean", model))
            ]:
                key = f"{exp}.{model}.{member}"
                print(key)
                oc_path = (
                    os.path.join(REGRIDDED_PATH, exp, "ocean", model, member) + ".nc"
                )
                oc_lock = (
                    os.path.join(REGRIDDED_PATH) + f"{exp}.ocean.{model}.{member}.lock"
                )
                at_path = (
                    os.path.join(REGRIDDED_PATH, exp, "ocean", model, member) + ".nc"
                )
                at_lock = (
                    os.path.join(REGRIDDED_PATH) + f"{exp}.at.{model}.{member}.lock"
                )
                if os.path.exists(oc_path) and os.path.exists(at_path):
                    if not os.path.exists(oc_lock) and not os.path.exists(at_lock):
                        pairs[f"{exp}.{model}.{member}"] = {
                            "exp": exp,
                            "model": model,
                            "member": member,
                        }
                    else:
                        print(f"Regrid lock file exists for {key}")
                else:
                    print(f"File missing for {exp}.{model}.{member}")

    return pairs


if __name__ == "__main__":
    # python tcpips/pi.py
    # plot_seasonality_in_gom_tcypi_ex()
    # regrid_2d_1degree()
    # ds = xr.open_dataset("data/ocean_regridded.nc")
    # ds.tos.isel(time=0).plot(x="lon", y="lat")
    # plt.show()
    # plot_tcpypi_in_out_ex()

    # print(
    #    gom_bbox_combined_inout_timestep_cmip6(time="2015-01-15", verbose=True, pad=5)
    # )
    pairs = find_atmos_ocean_pairs()
    print(pairs)

    # print(gom_combined_inout_timestep_cmip6())

    # plot_diffs()

    # for time in ["1850-09-15", "1950-09-15", "2015-09-15", "2099-09-15"]:
    #    # print(convert(combined_inout_timestep_cmip6(time=time)))
    #    plot_pi_outputs_map(time=time)
    # print(combined_inout_timestep_cmip6(time="2015-01-15"))
    # print(combined_inout_timestep_cmip6(time="1850-01-15"))
    # print(combined_inout_timestep_cmip6(time="2099-01-15"))

    # atmos_ds.psl.isel(time=0).plot(x="lon", y="lat")
    # plt.show()

    # calc_pi_example()
    # regrid_2d()
    # print(xr.open_dataset("data/ocean.nc"))
