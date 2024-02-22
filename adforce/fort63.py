"""Read output files fort.63.nc, maxele.63.nc etcs.

Should probably add more plotting and trimming files here.

Fort.63 files output to unstructured grid netCDF4 files.

However this is a general feature, so perhaps a grid script would be useful.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.constants import NO_BBOX, NEW_ORLEANS
from sithom.time import timeit
from sithom.place import BoundingBox, Point
from sithom.plot import label_subplots, plot_defaults
from .mesh import select_coast, filter_mesh, select_nearby, bbox_mesh


@timeit
def plot_nearby(
    data_folder: str = "../../NWS13set4/",
    bbox: BoundingBox = NO_BBOX,
    point: Point = NEW_ORLEANS,
    number: int = 10,
    pad: float = 2,
    plot_mesh: bool = True,
    overtopping=False,
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
    axs[0].tricontourf(
        maxele_ds.x.values,
        maxele_ds.y.values,
        maxele_ds.element.values - 1,
        maxele_ds.depth,
        levels=100,
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

    axs[1].tricontourf(
        maxele_ds.x.values,
        maxele_ds.y.values,
        maxele_ds.element.values - 1,
        np.nan_to_num(maxele_ds.zeta_max.values, 0),
        levels=100,
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

    axs[0].set_ylabel("Latitude [$^\circ$N]")
    axs[0].set_xlabel("Longitude [$^\circ$E]")

    axs[1].set_ylabel("Latitude [$^\circ$N]")

    bbox.ax_lim(axs[0])
    bbox.ax_lim(axs[1])

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Water level [m]")
    axs[2].set_title("")
    axs[2].legend()

    label_subplots(axs, override="inside")


if __name__ == "__main__":
    plot_nearby()
