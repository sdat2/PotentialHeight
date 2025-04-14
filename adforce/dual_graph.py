"""Dual graph plotting module."""

import os
import numpy as np
from matplotlib import pyplot as plt
from sithom.plot import plot_defaults, get_dim, label_subplots
from sithom.time import timeit
from .constants import FIGURE_PATH, NO_BBOX, DATA_PATH
from .mesh import bbox_mesh, process_dual_graph


@timeit
def plot_dual_graph() -> None:
    """
    Some test plots for the dual graph.
    """
    figure_path = os.path.join(FIGURE_PATH, "dual_graph")
    os.makedirs(figure_path, exist_ok=True)
    ds = bbox_mesh(
        os.path.join(DATA_PATH, "fort.63.nc"),
        bbox=NO_BBOX,
        use_dask=False,
    )

    dg = process_dual_graph(ds)

    # plot depth and depth gradient as two plots
    plot_defaults()
    fig, axs = plt.subplots(3, 1, figsize=get_dim(ratio=1.1))
    im = axs[0].scatter(
        dg.x.values, dg.y.values, c=dg.depth.values, s=0.1, cmap="viridis_r"
    )
    axs[0].set_title("Depth [m]")
    plt.colorbar(im, ax=axs[0], orientation="vertical")  # , shrink=0.3
    # set equal aspect ratio
    axs[0].set_aspect("equal")
    NO_BBOX.ax_lim(axs[0])
    vmin = dg.depth_grad.values.min()
    vmax = dg.depth_grad.values.max()
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])

    im = axs[1].scatter(
        dg.x.values,
        dg.y.values,
        c=dg.depth_grad.values[0, :],
        s=0.1,
        cmap="cmo.balance",
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title("Depth $x$ Gradient [m $^{\circ}\;^{-1}$]")
    plt.colorbar(
        im,
        ax=axs[1],
        orientation="vertical",
        # shrink=0.3,
    )
    axs[1].set_aspect("equal")
    NO_BBOX.ax_lim(axs[1])
    im = axs[2].scatter(
        dg.x.values,
        dg.y.values,
        c=dg.depth_grad.values[1, :],
        s=0.1,
        cmap="cmo.balance",
        vmin=vmin,
        vmax=vmax,
    )
    axs[2].set_title("Depth $y$ Gradient [m $^{\circ}\;^{-1}$]")
    plt.colorbar(
        im,
        ax=axs[2],
        orientation="vertical",
        # shrink=0.3,
    )
    axs[2].set_aspect("equal")
    NO_BBOX.ax_lim(axs[2])
    label_subplots(axs)
    axs[2].set_xlabel("Longitude [$^{\circ}$E]")
    axs[2].set_ylabel("Latitude [$^{\circ}$N]")
    axs[1].set_ylabel("Latitude [$^{\circ}$N]")
    axs[0].set_ylabel("Latitude [$^{\circ}$N]")
    plt.savefig(os.path.join(figure_path, "depth.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    # python -m adforce.dual_graph
    plot_dual_graph()
    # download_ibtracs_data()
    # ibtracs_to_era5_map()
