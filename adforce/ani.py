import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.constants import NO_BBOX, NEW_ORLEANS
import pandas as pd
from sithom.plot import plot_defaults, lim
from sithom.place import BoundingBox
from sithom.xr import plot_units
from sithom.plot import axis_formatter
from sithom.time import timeit
from adforce.mesh import bbox_mesh

plot_defaults()


@timeit
def plot_heights(
    path_in: str = "mult1",
    step_size: int = 1,
    add_name: str = "",
    bbox: BoundingBox = NO_BBOX,
) -> None:
    """
    Plot heights.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        step_size (int, optional): step size. Defaults to 1.
        add_name (str, optional): additional prefix name. Defaults to "".
        bbox (BoundingBox, optional): bounding box. Defaults to NO_BBOX.
    """
    img_folder = os.path.join(path_in, "img")
    os.makedirs(img_folder, exist_ok=True)
    figure_names = []
    # path_in = os.path.join(DATA_PATH, path_in)
    ds = bbox_mesh(
        os.path.join(path_in, "fort.63.nc"), bbox=bbox.pad(0.3), use_dask=True
    )
    vmin, vmax = ds.zeta.min().values, ds.zeta.max().values
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    print(vmin, vmax)
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=5)
    ckwargs = {
        "label": "",
        "format": axis_formatter(),
        "extend": "neither",
        "extendrect": False,
        "extendfrac": 0,
    }

    for time_i in range(0, len(ds.time.values), step_size):
        im = plt.tricontourf(
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
        cbar = plt.colorbar(
            im,
            ax=ax,
            **ckwargs,
        )
        ax.set_title("Water Height [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.1f}".format(x) for x in cbar_levels.tolist()])
        plt.xlabel("Longitude [$^{\circ}$E]")
        plt.ylabel("Latitude [$^{\circ}$N]")
        time = ds.isel(time=time_i).time.values
        ts = pd.to_datetime(str(time))
        print(ts)
        plt.scatter(
            NEW_ORLEANS.lon,
            NEW_ORLEANS.lat,
            marker=".",
            color="purple",
            label="New Orleans",
        )
        NO_BBOX.ax_lim(plt.gca())
        ax.set_aspect("equal")
        plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
        figure_name = os.path.join(img_folder, add_name + f"height_{time_i:04}.png")
        plt.savefig(figure_name)
        figure_names.append(figure_name)
        plt.clf()

    gif_name = os.path.join(path_in, add_name + "height.gif")
    print("gif_name", gif_name)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for filename in figure_names:
            image = imageio.imread(filename)
            writer.append_data(image)
