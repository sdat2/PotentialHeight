"""Read output files fort.63.nc, maxele.63.nc etcs.

Should probably add more plotting and trimming files here.

Fort.63 files output to unstructured grid netCDF4 files.

However this is a general feature, so perhaps a grid script would be useful.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
from .constants import NO_BBOX, NEW_ORLEANS
from sithom.time import timeit
from sithom.place import BoundingBox, Point
from sithom.xr import plot_units
from sithom.plot import label_subplots, plot_defaults
from .mesh import select_nearby, bbox_mesh  # select_coast, filter_mesh,
from .fort22 import read_fort22


plot_defaults()


@timeit
def plot_nearby(
    data_folder: str = "../../NWS13set4/",
    bbox: BoundingBox = NO_BBOX,
    point: Point = NEW_ORLEANS,
    number: int = 10,
    pad: float = 2,
    plot_mesh: bool = True,
    overtopping: bool = False,
) -> None:
    """
    Plot zeta timeseries points from the mesh near a point.
    Designed for comparison with tidal gauge data.


    Args:
        data_folder (str, optional): _description_. Defaults to "../../NWS13set4/".
        bbox (BoundingBox, optional): _description_. Defaults to NO_BBOX.
        point (Point, optional): _description_. Defaults to NEW_ORLEANS.
        number (int, optional): _description_. Defaults to 10.
        pad (float, optional): _description_. Defaults to 2.
        plot_mesh (bool, optional): _description_. Defaults to True.
        overtopping (bool, optional): _description_. Defaults to False.

    """
    plot_defaults()
    maxele_ds = bbox_mesh(os.path.join(data_folder, "maxele.63.nc"), bbox=bbox.pad(pad))
    timeseries_ds = bbox_mesh(
        os.path.join(data_folder, "fort.63.nc"), bbox=bbox.pad(pad)
    )
    coastal_da = select_nearby(
        timeseries_ds,
        point.lon,
        point.lat,
        number=number,
        verbose=True,
        overtopping=overtopping,
    )
    coastal_zeta_da = coastal_da.zeta

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # axs[0].tricontourf(maxele_ds.x.values, maxele_ds.y.values, maxele_ds.element.values -1, np.nan_to_num(maxele_ds.zeta.isel(time=0), nan=0), levels=20)
    im = axs[0].tricontourf(
        maxele_ds.x.values,
        maxele_ds.y.values,
        maxele_ds.element.values - 1,
        maxele_ds.depth,
        levels=1000,
    )
    if plot_mesh:
        axs[0].triplot(
            maxele_ds.x.values,
            maxele_ds.y.values,
            maxele_ds.element.values - 1,
            color="grey",
            linewidth=0.2,
        )

    axs[0].scatter(point.lon, point.lat, marker="x", color="red")

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", label="Depth [m]")

    im = axs[1].tricontourf(
        maxele_ds.x.values,
        maxele_ds.y.values,
        maxele_ds.element.values - 1,
        np.nan_to_num(maxele_ds.zeta_max.values, 0),
        levels=1000,
    )
    if plot_mesh:
        axs[1].triplot(
            maxele_ds.x.values,
            maxele_ds.y.values,
            maxele_ds.element.values - 1,
            color="grey",
            linewidth=0.2,
        )
    axs[1].scatter(point.lon, point.lat, marker="x", color="red")

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", label="Max water level [m]")

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "black",
        "yellow",
        "pink",
        "brown",
        "grey",
    ]
    for i in range(len(coastal_zeta_da.x.values)):
        axs[0].scatter(
            coastal_zeta_da.x.values[i],
            coastal_zeta_da.y.values[i],
            marker="x",
            c=colors[i],
        )
        axs[1].scatter(
            coastal_zeta_da.x.values[i],
            coastal_zeta_da.y.values[i],
            marker="x",
            c=colors[i],
        )
        coastal_zeta_da.isel(node=i).plot.line(
            ax=axs[2], x="time", label=f"Node {i+1}", color=colors[i]
        )

    axs[0].set_ylabel(r"Latitude [$^\circ$N]")
    axs[0].set_xlabel(r"Longitude [$^\circ$E]")

    axs[1].set_ylabel(r"Latitude [$^\circ$N]")

    bbox.ax_lim(axs[0])
    bbox.ax_lim(axs[1])

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Water level [m]")
    axs[2].set_title("")
    axs[2].legend()

    label_subplots(axs, override="inside")
    return fig, axs


def kat2() -> None:
    from src.constants import KATRINA_TIDE_NC

    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)

    from tcpips.constants import FIGURE_PATH

    figure_dir = os.path.join(FIGURE_PATH, "kat2")
    os.makedirs(figure_dir, exist_ok=True)

    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)
    for station in range(len(tide_ds.lon)):
        fig, axs = plot_nearby(
            data_folder="/work/n01/n01/sithom/adcirc-swan/kat2/",
            point=Point(
                tide_ds.isel(stationid=station).lon.values,
                tide_ds.isel(stationid=station).lat.values,
            ),
            bbox=NO_BBOX.pad(0.5),
            plot_mesh=False,
            overtopping=True,
        )
        axs[2].plot(
            tide_ds.date_time,
            tide_ds.isel(stationid=station).water_level,
            # color="",
        )
        # plt.plot(tide_ds.date_time, tide_ds.isel(stationid=node).water_level)
        plt.savefig(os.path.join(figure_dir, f"station_{station}.png"))
        plt.clf()
        plt.close()


if __name__ == "__main__":
    # python -m adforce.fort63
    from src.constants import KATRINA_TIDE_NC

    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)

    from tcpips.constants import FIGURE_PATH

    figure_dir = os.path.join(FIGURE_PATH, "hresmeshnws12")
    os.makedirs(figure_dir, exist_ok=True)

    tide_ds = xr.open_dataset(KATRINA_TIDE_NC)
    for station in range(len(tide_ds.lon)):
        fig, axs = plot_nearby(
            data_folder="/work/n01/n01/sithom/adcirc-swan/testsuite/adcirc/adcirc_katria_2d-highres",
            point=Point(
                tide_ds.isel(stationid=station).lon.values,
                tide_ds.isel(stationid=station).lat.values,
            ),
            bbox=NO_BBOX.pad(0.5),
            plot_mesh=True,
            overtopping=False,
        )
        axs[2].plot(
            tide_ds.date_time,
            tide_ds.isel(stationid=station).water_level,
        )
        # plt.plot(tide_ds.date_time, tide_ds.isel(stationid=node).water_level)
        plt.savefig(os.path.join(figure_dir, f"station_{station}.png"))
        plt.clf()
        plt.close()


def plot_quiver_height(
    path_in: str = "mult1", time_i: int = 160, x_pos: float = 0.95, y_pos: float = -0.15
) -> None:
    """
    Plot quiver height.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        time_i (int, optional): time_i. Defaults to 185.
        x_pos (float, optional): x_pos. Defaults to 0.95.
        y_pos (float, optional): y_pos. Defaults to -0.15.
    """
    # path_in = os.path.join(DATA_PATH, path_in)
    ds = bbox_mesh(
        os.path.join(path_in, "fort.63.nc"), bbox=NO_BBOX.pad(0.3), use_dask=True
    )
    print(ds)
    vmin, vmax = ds.zeta.min().values, ds.zeta.max().values
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    print(vmin, vmax)
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=5)
    plt.tricontourf(
        ds.x.values,
        ds.y.values,
        ds.element.values - 1,
        np.nan_to_num(ds.zeta.isel(time=time_i).values, copy=False, nan=0),
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap="cmo.balance",
    )
    ax = plt.gca()
    cbar = plt.colorbar(label="Height [m]")
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
    plt.xlabel(r"Longitude [$^{\circ}$E]")
    plt.ylabel(r"Latitude [$^{\circ}$N]")
    time = ds.isel(time=time_i).time.values
    ts = pd.to_datetime(str(time))
    print(ts)
    # plt.savefig(os.path.join(output_path, str(time_i) + ".png"))
    # plt.clf()
    ds = read_fort22(os.path.join(path_in, "fort.22.nc"))["Main"].to_dataset()
    print(ds)
    quiver = plot_units(
        ds.sel(time=time, method="nearest"), x_dim="lon", y_dim="lat"
    ).plot.quiver(
        ax=ax,
        x="lon",
        y="lat",
        u="U10",
        v="V10",
        add_guide=False,
    )
    _ = plt.quiverkey(
        quiver,
        x_pos,
        y_pos,
        40,
        str(r"$40$ m s$^{-1}$"),  # + "\n"
        labelpos="E",
        coordinates="axes",
        # coordinates="figure"
    )
    NO_BBOX.ax_lim(plt.gca())
    plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
    # plt.savefig(os.path.join(FIGURE_PATH, "example_colision.png"))
    # plt.clf()
