"""Dual graph plotting module."""

import os
import numpy as np
from matplotlib import pyplot as plt
import imageio
from sithom.plot import plot_defaults, get_dim, label_subplots
from sithom.time import timeit
from .constants import FIGURE_PATH, NO_BBOX, DATA_PATH
from .mesh import bbox_mesh, dual_graph_ds_from_mesh_ds


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

    dg = dual_graph_ds_from_mesh_ds(ds)
    print("ds", ds)
    print("dg", dg)

    @timeit
    def plot_depth_and_gradients():
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
        plt.close()
        plt.clf()

    # plot_depth_and_gradients()

    def animate_zeta_and_gradients():
        # plot zeta and zeta gradients as a set of pngs, then animate them. Use the
        # same colormap and setup as above.
        vmin_zeta = dg.zeta.values.min()
        vmax_zeta = dg.zeta.values.max()
        vmin_zeta, vmax_zeta = np.min([-vmax_zeta, vmin_zeta]), np.max(
            [-vmin_zeta, vmax_zeta]
        )

        vmin_grad = dg.zeta_grad.values.min()
        vmax_grad = dg.zeta_grad.values.max()
        vmin_grad, vmax_grad = np.min([-vmax_grad, vmin_grad]), np.max(
            [-vmin_grad, vmax_grad]
        )
        tmp_path = os.path.join(figure_path, "zeta")
        os.makedirs(tmp_path, exist_ok=True)

        figure_names = []

        @timeit
        def plot_zeta_and_gradients(i: int) -> None:
            fig, axs = plt.subplots(3, 1, figsize=get_dim(ratio=1.1))
            im = axs[0].scatter(
                dg.x.values,
                dg.y.values,
                c=dg.zeta.values[i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_zeta,
                vmax=vmax_zeta,
            )
            axs[0].set_title(f"Zeta [m] at time {i}")
            plt.colorbar(im, ax=axs[0], orientation="vertical")
            axs[0].set_aspect("equal")
            NO_BBOX.ax_lim(axs[0])
            axs[1].scatter(
                dg.x.values,
                dg.y.values,
                c=dg.zeta_grad.values[0, i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_grad,
                vmax=vmax_grad,
            )
            axs[1].set_title(r"Zeta $x$ Gradient [m $^{\circ}\;^{-1}$]")
            plt.colorbar(im, ax=axs[1], orientation="vertical")
            axs[1].set_aspect("equal")
            NO_BBOX.ax_lim(axs[1])
            axs[2].scatter(
                dg.x.values,
                dg.y.values,
                c=dg.zeta_grad.values[1, i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_grad,
                vmax=vmax_grad,
            )
            axs[2].set_title(r"Zeta $y$ Gradient [m $^{\circ}\;^{-1}$]")
            plt.colorbar(im, ax=axs[2], orientation="vertical")
            axs[2].set_aspect("equal")
            NO_BBOX.ax_lim(axs[2])
            label_subplots(axs)
            axs[2].set_xlabel("Longitude [$^{\circ}$E]")
            axs[2].set_ylabel("Latitude [$^{\circ}$N]")
            axs[1].set_ylabel("Latitude [$^{\circ}$N]")
            axs[0].set_ylabel("Latitude [$^{\circ}$N]")
            tmp_file = os.path.join(tmp_path, f"zeta_{i:04d}.png")
            figure_names.append(tmp_file)
            plt.savefig(tmp_file, bbox_inches="tight")
            plt.close()
            plt.clf()

        for i in range(len(dg.zeta.values)):
            plot_zeta_and_gradients(i)

        gif_path = os.path.join(figure_path, "zeta.gif")

        # make an animation of the zeta and zeta gradients
        with imageio.get_writer(gif_path, mode="I") as writer:
            for filename in figure_names:
                image = imageio.imread(filename)
                writer.append_data(image)

    # animate_zeta_and_gradients()
    @timeit
    def plot_mesh_dual_graph():
        # plot the grid for the original triangular component mesh in ds, and the dual graph in dg
        plot_defaults()

        fig, axs = plt.subplots(2, 1, figsize=get_dim(ratio=1.1), sharex=True)
        label_subplots(axs)
        axs[1].set_xlabel("Longitude [$^{\circ}$E]")
        axs[0].set_ylabel("Latitude [$^{\circ}$N]")
        axs[1].set_ylabel("Latitude [$^{\circ}$N]")
        # plot the original mesh
        axs[0].set_title("Original mesh")
        axs[0].triplot(
            ds.x.values, ds.y.values, ds.element.values - 1, color="black", alpha=0.1
        )
        axs[0].set_aspect("equal")
        # plot the dual graph
        axs[1].set_title("Dual graph")
        for edge_i in range(len(dg.edge.values)):
            axs[1].plot(
                dg.x.values[[dg.start.values[edge_i], dg.end.values[edge_i]]],
                dg.y.values[[dg.start.values[edge_i], dg.end.values[edge_i]]],
                color="black",
                alpha=0.1,
            )
        axs[1].set_aspect("equal")
        NO_BBOX.ax_lim(axs[0])
        NO_BBOX.ax_lim(axs[1])

        plt.savefig(
            os.path.join(figure_path, "dual_graph.pdf"),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    plot_mesh_dual_graph()


def test_dual_graph():
    """
    Test dual graph maker.
    """

    from .mesh import dual_graph_starts_ends_from_triangles

    elements = np.array(
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 3, 1]], dtype=int
    )
    x = np.array([0, 1, 2, 3, 4, 3, 1]) + 1
    y = np.array([2, 4, 2, 4, 2, 0, 0])

    starts, ends, lengths, xd, yd, nx, ny = dual_graph_starts_ends_from_triangles(
        elements - 1, x, y
    )
    print("lengths", lengths)
    print("triangles", elements - 1)
    print("starts", starts)
    print("ends", ends)
    print("xd", xd)
    print("yd", yd)
    print("nx", nx)
    print("ny", ny)

    for i in range(len(starts)):
        dot_prod = nx[i] * (xd[ends[i]] - xd[starts[i]]) + ny[i] * (
            yd[ends[i]] - yd[starts[i]]
        )
        print("dot_prod", dot_prod)

    # plot the dual graph
    plot_defaults()

    plt.triplot(x, y, elements - 1, color="black", alpha=0.1, label="Original mesh")

    for edge_i in range(len(starts)):
        if edge_i == 0:
            label_d = {"label": "Dual graph"}
            arrow_d = {"label": "Unit normal vector"}
        else:
            label_d = {}
            arrow_d = {}
        plt.plot(
            xd[[starts[edge_i], ends[edge_i]]],
            yd[[starts[edge_i], ends[edge_i]]],
            color="orange",
            alpha=0.1,
            **label_d,
        )
        plt.arrow(
            xd[starts[edge_i]] + 0.5 * (xd[ends[edge_i]] - xd[starts[edge_i]]),
            yd[starts[edge_i]] + 0.5 * (yd[ends[edge_i]] - yd[starts[edge_i]]),
            nx[edge_i] / 10,
            ny[edge_i] / 10,
            color="green",
            alpha=0.1,
            width=0.01,
            **arrow_d,
        )
    # get rid of x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.legend()

    plt.savefig(
        os.path.join(FIGURE_PATH, "dual_graph_test.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # python -m adforce.dual_graph
    # plot_dual_graph()
    test_dual_graph()
