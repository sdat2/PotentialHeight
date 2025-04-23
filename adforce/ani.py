"""Animate the outputs and inputs of ADCIRC simulations.

TODO: There is a lot of repeated code in the plotting functions. This should be refactored.

TODO: Add geographic plotting options for including cartopy maps with coastlines, etc. if cartopy is installed.
"""

from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import xarray as xr
from sithom.plot import plot_defaults, lim
from sithom.place import BoundingBox, Point
from sithom.plot import axis_formatter
from sithom.time import timeit
import argparse
from .mesh import bbox_mesh, xr_loader
from .fort22datatree import read_fort22
from .constants import NO_BBOX, NEW_ORLEANS
from .config import load_config
from .fort22 import create_fort22

# from sithom.xr import plot_units


@timeit
def ani_heights(
    path_in: str = "mult1",
    step_size: int = 1,
    add_name: str = "",
    bbox: Optional[BoundingBox] = NO_BBOX,
) -> None:
    """
    Plot heights.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        step_size (int, optional): step size. Defaults to 1.
        add_name (str, optional): additional prefix name. Defaults to "".
        bbox (BoundingBox, optional): bounding box. Defaults to NO_BBOX.
    """
    plot_defaults()
    img_folder = os.path.join(path_in, "img", "height")
    gif_folder = os.path.join(path_in, "gif", "height")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)
    figure_names = []
    # path_in = os.path.join(DATA_PATH, path_in)
    if bbox is not None:
        ds = bbox_mesh(
            os.path.join(path_in, "fort.63.nc"), bbox=bbox.pad(0.3), use_dask=True
        )
    else:
        ds = xr_loader(os.path.join(path_in, "fort.63.nc"), use_dask=True)
    vmin, vmax = ds.zeta.min().values, ds.zeta.max().values
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    print(vmin, vmax)
    levels = np.linspace(vmin, vmax, num=1000)
    cbar_levels = np.linspace(vmin, vmax, num=5)
    ckwargs = {
        "label": "",
        "format": axis_formatter(),
        "extend": "neither",
        "extendrect": False,
        "extendfrac": 0,
    }

    for time_i in range(0, len(ds.time.values), step_size):
        ax = plt.gca()
        ax.set_facecolor("#d1ffbd")
        im = ax.tricontourf(
            ds.x.values,
            ds.y.values,
            ds.element.values - 1,
            np.nan_to_num(ds.zeta.isel(time=time_i).values, copy=False, nan=0),
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap="cmo.balance",
        )
        cbar = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            **ckwargs,
        )
        ax.set_title("Water Height [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.1f}".format(x) for x in cbar_levels.tolist()])
        plt.xlabel(r"Longitude [$^{\circ}$E]")
        plt.ylabel(r"Latitude [$^{\circ}$N]")
        time = ds.isel(time=time_i).time.values
        ts = pd.to_datetime(str(time))
        print("time:", ts)
        plt.scatter(
            NEW_ORLEANS.lon,
            NEW_ORLEANS.lat,
            marker=".",
            color="purple",
            label="New Orleans",
        )
        if bbox is not None:
            bbox.ax_lim(plt.gca())
        ax.set_aspect("equal")
        plt.title(ts.strftime("%d  %H:%M"))
        figure_name = os.path.join(img_folder, add_name + f"height_{time_i:04}.png")
        plt.savefig(figure_name)
        figure_names.append(figure_name)
        plt.clf()

    gif_name = os.path.join(gif_folder, add_name + "height.gif")
    print("gif_name", gif_name)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for filename in figure_names:
            image = imageio.imread(filename)
            writer.append_data(image)


def plot_locations(path, ax: plt.Axes, transform=None) -> None:
    """
    Plot the attempted and actual observation locations.

    Args:
        path (str): path the config.yaml file is in.
        ax (plt.Axes): axes to plot on.
    """

    config = load_config(os.path.join(path, "config.yaml"))
    at_lon, at_lat = config.adcirc.attempted_observation_location.value
    ac_lon, ac_lat = config.adcirc.actual_observation_location.value

    ax.scatter(
        at_lon,
        at_lat,
        marker=".",
        color="red",
        label="Attempted Observation Location",
        transform=transform,
    )
    ax.scatter(
        ac_lon,
        ac_lat,
        marker=".",
        color="blue",
        label="Actual Observation Location",
        transform=transform,
    )


@timeit
def ani_heights_and_winds(
    path_in: str = "mult1",
    step_size: int = 1,
    add_name: str = "",
    bbox: Optional[BoundingBox] = NO_BBOX,
    x_pos: float = 0.9,
    y_pos: float = 1.05,
    scale: float = 800,
    coarsen: int = 2,
) -> None:
    """
    Plot heights and winds.

    Assumes that the heights on the mesh are in the fort.63.nc file and the winds on the grid are in the fort.22.nc file.

    Equally we might want to plot the winds on the mesh instead. This might get rid of some of the time synchronization issues.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        step_size (int, optional): step size. Defaults to 1.
        add_name (str, optional): additional prefix name. Defaults to "".
        bbox (BoundingBox, optional): bounding box. Defaults to NO_BBOX.
        x_pos (float, optional): relative x position of quiver label.
        y_pos (float, optional): relative y position of quiver label.
    """
    regenerate_fort22_if_does_not_exist(path_in)
    plot_defaults()
    img_folder = os.path.join(path_in, "img", "height_and_wind")
    gif_folder = os.path.join(path_in, "gif", "height_and_wind")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)
    figure_names = []
    # path_in = os.path.join(DATA_PATH, path_in)
    if bbox is not None:
        f63_ds = bbox_mesh(
            os.path.join(path_in, "fort.63.nc"), bbox=bbox.pad(0.3), use_dask=True
        )
    else:
        f63_ds = xr_loader(os.path.join(path_in, "fort.63.nc"), use_dask=True)
    vmin_eta, vmax_eta = f63_ds.zeta.min().values, f63_ds.zeta.max().values
    vmin_eta, vmax_eta = np.min([-vmax_eta, vmin_eta]), np.max([-vmin_eta, vmax_eta])
    print(vmin_eta, vmax_eta)
    levels = np.linspace(vmin_eta, vmax_eta, num=400)
    cbar_levels = np.linspace(vmin_eta, vmax_eta, num=5)
    f22_main_ds = read_fort22(os.path.join(path_in, "fort.22.nc"))["Main"].to_dataset()
    f22_main_ds = line_up_f22(f22_main_ds, path_in, NEW_ORLEANS)
    f22_main_ds = f22_main_ds.coarsen(xi=coarsen, yi=coarsen, boundary="trim").mean()

    ckwargs = {
        "label": "",
        "format": axis_formatter(),
        "extend": "neither",
        "extendrect": False,
        "extendfrac": 0,
    }

    for time_i in range(0, len(f63_ds.time.values), step_size):
        ax = plt.gca()
        ax.set_facecolor("#d1ffbd")
        im = plt.tricontourf(
            f63_ds.x.values,
            f63_ds.y.values,
            f63_ds.element.values - 1,
            np.nan_to_num(f63_ds.zeta.isel(time=time_i).values, copy=False, nan=0),
            vmin=vmin_eta,
            vmax=vmax_eta,
            levels=levels,
            cmap="cmo.balance",
        )
        ax = plt.gca()
        cbar = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.5,
            **ckwargs,
        )
        ax.set_title("Water Height [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.1f}".format(x) for x in cbar_levels.tolist()])
        time = f63_ds.isel(time=time_i).time.values
        ts = pd.to_datetime(str(time))

        quiver = f22_main_ds.sel(time=time, method="nearest").plot.quiver(
            ax=ax,
            x="lon",
            y="lat",
            u="U10",
            v="V10",
            scale=scale,
            add_guide=False,
        )

        _ = plt.quiverkey(
            quiver,
            # 1.08,
            x_pos,
            y_pos,  # 08,
            40,
            str(str(40) + r" m s$^{-1}$"),  # + "\n"
            labelpos="E",
            coordinates="axes",
            # transform=ccrs.PlateCarree(),
            # coordinates="figure"
            # ,
        )
        print(ts)
        plt.xlabel(r"Longitude [$^{\circ}$E]")
        plt.ylabel(r"Latitude [$^{\circ}$N]")
        plt.scatter(
            NEW_ORLEANS.lon,
            NEW_ORLEANS.lat,
            marker=".",
            color="purple",
            label="New Orleans",
        )
        if bbox is not None:
            bbox.ax_lim(plt.gca())
        ax.set_aspect("equal")
        plt.title(ts.strftime("%m-%d  %H:%M"))
        figure_name = os.path.join(
            img_folder, add_name + f"height_and_wind_{time_i:04}.png"
        )
        plt.savefig(figure_name)
        figure_names.append(figure_name)
        plt.clf()

    gif_name = os.path.join(gif_folder, add_name + "height_and_wind.gif")
    print("gif_name", gif_name)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for filename in figure_names:
            image = imageio.imread(filename)
            writer.append_data(image)


def find_starttime_of_impact(da: xr.DataArray) -> any:
    """Helper function to find first nonzero time in timeseries.

    Args:
        da (xr.DataArray): timeseries dataarray.

    Returns:
        any: start time of nonzeros.
    """
    nonzeros = np.nonzero(da.values)[0]
    assert len(nonzeros) > 0
    first_nonzero = nonzeros[0]
    final_nonzero = nonzeros[-1]
    print(
        first_nonzero,
        final_nonzero,
        da.time.values[first_nonzero],
        da.time.values[final_nonzero],
        da.time.values[final_nonzero] - da.time.values[first_nonzero],
    )
    return da.time.values[first_nonzero]


def line_up_f22(f22_main_ds: xr.Dataset, path_in: str, point: Point) -> xr.Dataset:
    """
    Line up fort.22.

    Args:
        f22_main_ds (xr.Dataset): main fort.22.nc dataset.
        path_in (str): path for inputs.
        point (Point): point to compare at.

    Returns:
        xr.Dataset: f22_main_ds.
    """
    f74_ds = xr_loader(os.path.join(path_in, "fort.74.nc"), use_dask=True)
    f22_lons = f22_main_ds.lon.values
    f22_lats = f22_main_ds.lat.values
    sq_distances_f22 = (f22_lons - point.lon) ** 2 + (f22_lats - point.lat) ** 2
    min_idxs_f22 = np.unravel_index(sq_distances_f22.argmin(), sq_distances_f22.shape)
    sq_distances_f74 = (f74_ds.x.values - point.lon) ** 2 + (
        f74_ds.y.values - point.lat
    ) ** 2
    min_idx_f74 = sq_distances_f74.argmin()
    # perhaps not robust to a rotation of the xarray arrays
    u10no_f22 = f22_main_ds.U10.isel(xi=min_idxs_f22[1], yi=min_idxs_f22[0])
    u10no_f74 = f74_ds.windx.isel(node=min_idx_f74)
    print("fort.22")
    f22_start = find_starttime_of_impact(u10no_f22)
    print("fort.74")
    f74_start = find_starttime_of_impact(u10no_f74)
    f22_new_times = u10no_f22.time.values - f22_start + f74_start
    del f22_main_ds["time"]
    f22_main_ds["time"] = ("time", f22_new_times)
    return f22_main_ds


def plot_u10_windx_at_a_point(path_in: str, point: Point, plot: bool = True) -> None:
    """
    Plot U10 at a point so that we can try to sync up the fort.22 and fort.74 files time coordinate.

    Args:
        path_in (str): path in.
        point (Point): point to compare at (nearest mesh point to).
        plot (bool, optional): Defaults to True.
    """
    ds = xr_loader(os.path.join(path_in, "fort.74.nc"), use_dask=True)
    f22 = read_fort22(os.path.join(path_in, "fort.22.nc"))
    lons = f22["Main"].lon.values
    lats = f22["Main"].lat.values
    sq_distances_f22 = (lons - point.lon) ** 2 + (lats - point.lat) ** 2
    min_idxs = np.unravel_index(sq_distances_f22.argmin(), sq_distances_f22.shape)
    sq_distances_ds = (ds.x.values - point.lon) ** 2 + (ds.y.values - point.lat) ** 2
    min_idx_ds = sq_distances_ds.argmin()
    u10no = f22["Main"].U10.isel(xi=min_idxs[1], yi=min_idxs[0])
    u10no_ds = ds.windx.isel(node=min_idx_ds)
    if plot:
        u10no_ds.plot(color="blue", label="fort.74")
        u10no.plot(color="red", label="fort.22")
        plt.legend()
        plt.show()
    # change times to match start
    print("fort.22")
    f22_start = find_starttime_of_impact(u10no)
    print("fort.74")
    f74_start = find_starttime_of_impact(u10no_ds)
    u10no_ds_new_times = u10no_ds.time.values - f74_start + f22_start
    del u10no_ds["time"]
    u10no_ds["time"] = ("time", u10no_ds_new_times)
    if plot:
        u10no_ds.plot(color="blue", label="fort.74")
        u10no.plot(color="red", label="fort.22")
        plt.legend()
        plt.show()


def run_animation() -> None:
    """Run the animation based on command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_in",
        type=str,
        default="/work/n01/n01/sithom/adcirc-swan/exp/angle_test/exp_004",
        help="path to data",
    )
    parser.add_argument(
        "--step_size", type=int, default=10, help="step size for animation"
    )
    parser.add_argument(
        "--coarsen", type=int, default=2, help="coarsen the wind data by this factor"
    )
    parser.add_argument(
        "--x_pos", type=float, default=0.9, help="relative x position of quiver label"
    )
    # add in option flag for with or without winds
    parser.add_argument("--winds", action="store_true", help="plot winds with heights")

    args = parser.parse_args()

    if args.winds:
        ani_heights_and_winds(
            path_in=args.path_in,
            bbox=NO_BBOX,
            add_name="",
            step_size=args.step_size,
            coarsen=args.coarsen,
            x_pos=args.x_pos,
        )

    else:
        ani_heights(
            path_in=args.path_in,
            bbox=NO_BBOX,
            add_name="",
            step_size=args.step_size,
        )
    # python -m adforce.ani --path_in /work/n01/n01/sithom/adcirc-swan/exp/ani-2d-2/exp_0049 --step_size 10 --coarsen 2


def single_wind_and_height_step(
    path_in: str = "/work/n01/n01/sithom/adcirc-swan/tcpips/exp/8761724-2015/exp_0049",  # "/work/n02/n02/sdat2/adcirc-swan/exp/notide-mid-2025",
    time_i: int = 380,
    coarsen: int = 2,
    bbox: Optional[BoundingBox] = NO_BBOX,
    add_name: str = "",
    x_pos: float = 0.4,
    y_pos: float = 1.05,
    scale: float = 800,
    plot_loc: bool = False,
    figure_name: str = "snapshot.pdf",
) -> None:
    """
    Plot a single snapshot of height and wind.

    Args:
        path_in (str, optional): path to data. Defaults to "/work/n01/n01/sithom/adcirc-swan/exp/ani-2d-2/exp_0049".
        time_i (int, optional): time index. Defaults to 380.
        coarsen (int, optional): coarsen the wind data by this factor. Defaults to 2.
        bbox (Optional[BoundingBox], optional): bounding box. Defaults to NO_BBOX.
        add_name (str, optional): additional prefix name. Defaults to "".
        x_pos (float, optional): relative x position of quiver label. Defaults to 0.9.
        y_pos (float, optional): relative y position of quiver label. Defaults to 1.05.
        scale (float, optional): scale of the quiver plot. Defaults to 800.
        plot_loc (bool, optional): whether to plot the locations. Defaults to False.
        figure_name (str, optional): name of the figure. Defaults to "snapshot.pdf".

    """
    plot_defaults()

    regenerate_fort22_if_does_not_exist(path_in)

    img_folder = os.path.join(path_in, "img")
    os.makedirs(img_folder, exist_ok=True)
    if bbox is not None:
        f63_ds = bbox_mesh(
            os.path.join(path_in, "fort.63.nc"), bbox=bbox.pad(0.3), use_dask=True
        )
    else:
        f63_ds = xr_loader(os.path.join(path_in, "fort.63.nc"), use_dask=True)
    vmin_eta, vmax_eta = f63_ds.zeta.min().values, f63_ds.zeta.max().values
    vmin_eta, vmax_eta = np.min([-vmax_eta, vmin_eta]), np.max([-vmin_eta, vmax_eta])
    print(vmin_eta, vmax_eta)
    levels = np.linspace(vmin_eta, vmax_eta, num=400)
    cbar_levels = np.linspace(vmin_eta, vmax_eta, num=5)
    f22_main_ds = read_fort22(os.path.join(path_in, "fort.22.nc"))["Main"].to_dataset()
    f22_main_ds = line_up_f22(f22_main_ds, path_in, NEW_ORLEANS)
    f22_main_ds = f22_main_ds.coarsen(xi=coarsen, yi=coarsen, boundary="trim").mean()

    ckwargs = {
        "label": "Storm Surge Height [m]",
        "format": axis_formatter(),
        "extend": "neither",
        "extendrect": False,
        "extendfrac": 0,
    }

    try:
        import cartopy
        import cartopy.crs as ccrs

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        # ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.STATES, linestyle=":")
        fd = dict(transform=ccrs.PlateCarree())
        if plot_loc:
            plot_locations(path_in, ax)
    except:
        fd = dict()

    ax = plt.gca()
    ax.set_facecolor("#d1ffbd")
    im = plt.tricontourf(
        f63_ds.x.values,
        f63_ds.y.values,
        f63_ds.element.values - 1,
        np.nan_to_num(f63_ds.zeta.isel(time=time_i).values, copy=False, nan=0),
        vmin=vmin_eta,
        vmax=vmax_eta,
        levels=levels,
        cmap="cmo.balance",
        **fd,
    )
    ax = plt.gca()
    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        shrink=0.8,  # 0.5,
        **ckwargs,
    )
    ax.set_title("Storm Surge Height [m]")
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels(["{:.1f}".format(x) for x in cbar_levels.tolist()])
    time = f63_ds.isel(time=time_i).time.values
    ts = pd.to_datetime(str(time))
    print("time:", ts)
    quiver = f22_main_ds.sel(time=time, method="nearest").plot.quiver(
        ax=ax,
        x="lon",
        y="lat",
        u="U10",
        v="V10",
        scale=scale,
        add_guide=False,
    )

    _ = plt.quiverkey(
        quiver,
        # 1.08,
        x_pos,
        y_pos,  # 08,
        40,
        str(str(40) + r" m s$^{-1}$ $\vec{u}_{10}$"),  # + "\n"
        labelpos="E",
        coordinates="axes",
        # transform=ccrs.PlateCarree(),
        # coordinates="figure"
        # ,
        **fd,
    )
    # print(ts)
    plt.xlabel(r"Longitude [$^{\circ}$E]")
    plt.ylabel(r"Latitude [$^{\circ}$N]")
    plt.scatter(
        NEW_ORLEANS.lon,
        NEW_ORLEANS.lat,
        marker=".",
        color="purple",
        label="New Orleans",
        transform=ccrs.PlateCarree(),
    )
    if bbox is not None:
        bbox.ax_lim(plt.gca())
    ax.set_aspect("equal")
    ax.set_title("")
    plt.title("")

    if fd != {}:
        ax.set_yticks(
            [
                x
                for x in range(
                    int((bbox.lat[0] // 1) + 1),
                    int((bbox.lat[1] // 1) + 1),
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xticks(
            [
                x
                for x in range(
                    int((bbox.lon[0] // 1) + 1),
                    int((bbox.lon[1] // 1) + 1),
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        # ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        # ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    # plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
    from tcpips.constants import FIGURE_PATH

    # figure_name = os.path.join(img_folder, add_name + "snapshot.pdf")
    plt.savefig(figure_name)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name))

    plt.clf()


def regenerate_fort22_if_does_not_exist(path: str) -> None:
    """
    Regenerate fort.22 if it does not exist.

    Args:
        path (str): path to fort.22.nc.
    """
    if not os.path.exists(os.path.join(path, "fort.22.nc")) and os.path.exists(
        os.path.join(path, "config.yaml")
    ):
        print("Regenerating fort.22.nc")
        cfg = load_config(os.path.join(path, "config.yaml"))
        create_fort22(path, cfg.grid, cfg.tc)
        print("fort.22.nc regenerated")
    else:
        print("fort.22.nc already exists or config.yaml not found")


if __name__ == "__main__":
    # python -m adforce.ani
    # run_animation()
    single_wind_and_height_step(
        path_in=".",
        bbox=NO_BBOX.pad(1),
        coarsen=3,
        plot_loc=True,
        figure_name="2015_worst_mid_notide_snapshot.pdf",
    )
    single_wind_and_height_step()
    # single_wind_and_height_step()

# if __name__ == "__main__":
#    drive_heights_and_winds()
# python -m adforce.ani
# line_up_times()
# ani_heights(
#     path_in="/work/n01/n01/sithom/adcirc-swan/NWS13example/",
#     bbox=NO_BBOX.pad(5),
#     add_name="zoomed_out_",
#     step_size=10,
# )
# ani_heights_and_winds(
#     path_in="/work/n01/n01/sithom/adcirc-swan/kat.nws13.2004.wrap2/",
#     bbox=NO_BBOX,
#     step_size=1,
#     add_name="zoomed_in_",
# )

# python -m adforce.ani --path_in /work/n01/n01/sithom/adcirc-swan/exp/ani-2d --step_size 10 --coarsen 2
# python -m adforce.ani --path_in . --step_size 10 --coarsen 2
# python -m adforce.ani --path_in /work/n02/n02/sdat2/adcirc-swan/exp/notide-mid-2025 --step_size 10 --winds --coarsen 2
