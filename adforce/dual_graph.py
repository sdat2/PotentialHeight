"""Dual graph plotting module."""

import os
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
    fig, axs = plt.subplots(3, 1, figsize=get_dim(ratio=1.5))
    im = axs[0].scatter(
        dg.x.values, dg.y.values, c=dg.depth.values, s=0.1, cmap="viridis"
    )
    plt.colorbar(im, ax=axs[0], label="Depth [m]", orientation="vertical", shrink=0.3)
    # set equal aspect ratio
    axs[0].set_aspect("equal")
    NO_BBOX.ax_lim(axs[0])
    im = axs[1].scatter(
        dg.x.values,
        dg.y.values,
        c=dg.depth_grad.values[0, :],
        s=0.1,
        cmap="viridis",
    )
    plt.colorbar(
        im,
        ax=axs[1],
        label="Depth $x$ Gradient [m $^{\circ}\;^{-1}$]",
        orientation="vertical",
        shrink=0.3,
    )
    axs[1].set_aspect("equal")
    NO_BBOX.ax_lim(axs[1])
    im = axs[2].scatter(
        dg.x.values,
        dg.y.values,
        c=dg.depth_grad.values[1, :],
        s=0.1,
        cmap="viridis",
    )
    plt.colorbar(
        im,
        ax=axs[2],
        label="Depth $y$ Gradient [m $^{\circ}\;^{-1}$]",
        orientation="vertical",
        shrink=0.3,
    )
    axs[2].set_aspect("equal")
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
