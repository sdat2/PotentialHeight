"""Dual graph plotting module.

Dual graph functions in .mesh currently"""

import os
import numpy as np
import xarray as xr
import netCDF4 as nc
from matplotlib import pyplot as plt
import imageio
from tqdm import tqdm
import datetime
from sithom.plot import plot_defaults, get_dim, label_subplots
from sithom.time import timeit
from .constants import FIGURE_PATH, NO_BBOX, DATA_PATH
from .mesh import (
    dual_graph_ds_from_mesh_ds_from_path,
    dual_graph_ds_from_mesh_ds,
    xr_loader,
    swegnn_dg_from_mesh_ds_from_path,  # <-- Add this one
)
FIGURE_PATH = os.path.join(FIGURE_PATH, "dual_graph")
os.makedirs(FIGURE_PATH, exist_ok=True)


@timeit
def plot_dual_graph() -> None:
    """
    Some test plots for the dual graph.
    """
    figure_path = FIGURE_PATH

    ds = xr_loader(os.path.join(DATA_PATH, "exp_0049", "fort.64.nc"))
    dg = dual_graph_ds_from_mesh_ds_from_path(
        path=os.path.join(DATA_PATH, "exp_0049"),
        bbox=NO_BBOX,
        take_grad=True,
    )
    # dg = dual_graph_ds_from_mesh_ds(ds)
    print("ds", ds)
    print("dg", dg)

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

    var_units = {
        "zeta": "m",
        "pressure": "m water",
        "windx": "m s$^{-1}$",
        "windy": "m s$^{-1}$",
        "u-vel": "m s$^{-1}$",
        "v-vel": "m s$^{-1}$",
    }

    @timeit
    def animate_var_and_gradients(var="zeta"):
        # plot zeta and zeta gradients as a set of pngs, then animate them. Use the
        # same colormap and setup as above.
        vmin_var = dg[var].min().values
        vmax_var = dg[var].max().values
        vmin_var, vmax_var = np.min([-vmax_var, vmin_var]), np.max(
            [-vmin_var, vmax_var]
        )

        vmin_grad = dg[f"{var}_grad"].min().values
        vmax_grad = dg[f"{var}_grad"].max().values
        vmin_grad, vmax_grad = np.min([-vmax_grad, vmin_grad]), np.max(
            [-vmin_grad, vmax_grad]
        )
        print(
            f"vmin_var {vmin_var}, vmax_var {vmax_var}, vmin_grad {vmin_grad}, vmax_grad {vmax_grad}"
        )
        tmp_path = os.path.join(figure_path, var)
        os.makedirs(tmp_path, exist_ok=True)

        # @timeit
        def plot_var_and_gradients(
            i: int,
            figure_name: str,
            var="zeta",
        ) -> None:

            fig, axs = plt.subplots(3, 1, figsize=get_dim(ratio=1.1))
            im = axs[0].scatter(
                dg.x.values,
                dg.y.values,
                c=dg[var].values[i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_var,
                vmax=vmax_var,
            )
            axs[0].set_title(
                var
                + f" [{var_units[var]}] at {dg.time.values[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d %H:%M')}"
            )
            plt.colorbar(im, ax=axs[0], orientation="vertical")
            axs[0].set_aspect("equal")
            NO_BBOX.ax_lim(axs[0])
            axs[1].scatter(
                dg.x.values,
                dg.y.values,
                c=dg[f"{var}_grad"].values[0, i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_grad,
                vmax=vmax_grad,
            )
            axs[1].set_title(
                var + r" $x$ Gradient [" + var_units[var] + r" $^{\circ}\;^{-1}$]"
            )
            plt.colorbar(im, ax=axs[1], orientation="vertical")
            axs[1].set_aspect("equal")
            NO_BBOX.ax_lim(axs[1])
            axs[2].scatter(
                dg.x.values,
                dg.y.values,
                c=dg[f"{var}_grad"].values[1, i, :],
                s=0.1,
                cmap="cmo.balance",
                vmin=vmin_grad,
                vmax=vmax_grad,
            )
            axs[2].set_title(
                var + r" $y$ Gradient [" + var_units[var] + r" $^{\circ}\;^{-1}$]"
            )
            plt.colorbar(im, ax=axs[2], orientation="vertical")
            axs[2].set_aspect("equal")
            NO_BBOX.ax_lim(axs[2])
            label_subplots(axs)
            axs[2].set_xlabel("Longitude [$^{\circ}$E]")
            axs[2].set_ylabel("Latitude [$^{\circ}$N]")
            axs[1].set_ylabel("Latitude [$^{\circ}$N]")
            axs[0].set_ylabel("Latitude [$^{\circ}$N]")
            plt.savefig(figure_name, bbox_inches="tight")
            plt.close()
            plt.clf()

            # plot the variable and its gradients

        tmp_path = os.path.join(figure_path, var)
        os.makedirs(tmp_path, exist_ok=True)
        figure_names = []
        for i in tqdm(
            range(0, len(dg.time.values), 10), desc=f"Making {var} animation"
        ):
            figure_name = os.path.join(tmp_path, var + f"_{i:04d}.png")
            plot_var_and_gradients(i, figure_name, var=var)
            figure_names.append(figure_name)
            gif_path = os.path.join(figure_path, var + ".gif")
            # make an animation of the zeta and zeta gradients
        with imageio.get_writer(gif_path, mode="I") as writer:
            for filename in figure_names:
                image = imageio.imread(filename)
                writer.append_data(image)

    for var in ["zeta", "pressure", "windx", "windy", "u-vel", "v-vel"]:
        if var not in dg:
            print(f"Variable {var} not in dg")
            continue
        # animate_var_and_gradients(var)

    def make_in_out_ani():
        # make a large animation of the inputs and outputs of the ADCIRC model on the dual graph.
        # Column 1: pressure, windx, windy
        # Column 2: zeta, u-vel, v-vel
        vars = np.array([["pressure", "windx", "windy"], ["zeta", "u-vel", "v-vel"]]).T
        # all should have a balanced colormap apart from pressure which should be viridis and not balanced
        vmin = [
            [dg[vars[i][j]].min().values for j in range(len(vars[i]))]
            for i in range(len(vars))
        ]
        vmax = [
            [dg[vars[i][j]].max().values for j in range(len(vars[i]))]
            for i in range(len(vars))
        ]
        for i in range(len(vars)):
            for j in range(len(vars[i])):
                if vars[i][j] == "pressure":
                    continue
                vmin[i][j], vmax[i][j] = np.min([-vmax[i][j], vmin[i][j]]), np.max(
                    [-vmin[i][j], vmax[i][j]]
                )

        cmap = np.array(
            [
                ["viridis_r", "cmo.balance", "cmo.balance"],
                ["cmo.balance", "cmo.balance", "cmo.balance"],
            ]
        ).T
        tmp_path = os.path.join(figure_path, "in_out")
        os.makedirs(tmp_path, exist_ok=True)

        def plot_in_out_timestep(t, figure_name):
            # make twice previous width, same height (3 rows, 2 columns)
            fig, axs = plt.subplots(
                3,
                2,
                figsize=get_dim(fraction_of_line_width=2, ratio=1.1 / 2),
                sharex=True,
                sharey=True,
            )
            for i in range(len(vars)):
                for j in range(len(vars[i])):
                    for edge_i in range(len(dg.edge.values)):
                        axs[i, j].plot(
                            dg.x.values[
                                [dg.start.values[edge_i], dg.end.values[edge_i]]
                            ],
                            dg.y.values[
                                [dg.start.values[edge_i], dg.end.values[edge_i]]
                            ],
                            color="black",
                            alpha=0.1,
                            linewidth=0.2,
                        )

                    im = axs[i, j].scatter(
                        dg.x.values,
                        dg.y.values,
                        c=dg[vars[i][j]].values[t, :],
                        s=0.2,
                        cmap=cmap[i][j],
                        vmin=vmin[i][j],
                        vmax=vmax[i][j],
                    )
                    axs[i, j].set_title(
                        vars[i][j]
                        + f" [{var_units[vars[i][j]]}] at {dg.time.values[t].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d %H:%M')}"
                    )
                    plt.colorbar(im, ax=axs[i, j], orientation="vertical")
                    axs[i, j].set_aspect("equal")
                    NO_BBOX.ax_lim(axs[i, j])
            for i in range(len(vars)):
                axs[i, 0].set_ylabel("Latitude [$^{\circ}$N]")
            for j in range(len(vars[0])):
                axs[2, j].set_xlabel("Longitude [$^{\circ}$E]")
            plt.savefig(figure_name, bbox_inches="tight")
            plt.close()
            plt.clf()

        figure_names = []
        snapshots = [320]
        for i in tqdm(
            range(0, len(dg.time.values), 10), desc="Making in_out animation"
        ):
            figure_name = os.path.join(tmp_path, f"in_out_{i:04d}.png")
            plot_in_out_timestep(i, figure_name)
            figure_names.append(figure_name)
            if i in snapshots:
                plot_in_out_timestep(
                    i, os.path.join(FIGURE_PATH, f"in_out_{i:04d}_snap.pdf")
                )

        gif_path = os.path.join(figure_path, "in_out.gif")
        with imageio.get_writer(gif_path, mode="I") as writer:
            for filename in figure_names:
                image = imageio.imread(filename)
                writer.append_data(image)

    make_in_out_ani()


def test_dual_graph() -> None:
    """
    Test dual graph maker with a simple hexagon of triangles.
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
        print("dot_prod same direc (should be > 0)", dot_prod)

    # let's do the dot product between the original mesh edge and the dual graph unit normal vector (should be 0 for all)
    for i in range(len(starts)):
        # find x joining traingle starts[i] and ends[i]
        tri_1 = elements[starts[i]] - 1
        tri_2 = elements[ends[i]] - 1
        edge = set(tri_1).intersection(set(tri_2))
        if len(edge) != 2:
            raise ValueError("Edge not found in triangles")
        edge = list(edge)
        delta_x = x[edge[1]] - x[edge[0]]
        delta_y = y[edge[1]] - y[edge[0]]
        # take dot product
        dot_prod = nx[i] * delta_x + ny[i] * delta_y
        print("dot_prod (should be 0)", dot_prod)

    # plot the dual graph
    plot_defaults()

    plt.triplot(x, y, elements - 1, color="black", alpha=0.1, label="Original mesh")

    for edge_i in range(len(starts)):
        if edge_i == 0:
            label_d = {"label": "Dual graph"}
            arrow_d = {"label": "Unit normal vectors"}
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
            0.5 * (xd[ends[edge_i]] + xd[starts[edge_i]]),
            0.5 * (yd[ends[edge_i]] + yd[starts[edge_i]]),
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


@timeit
def make_dual_graph_nc() -> None:
    """Make full example dual graph netcdf file."""

    dual_graph_ds_from_mesh_ds_from_path(
        path=os.path.join(DATA_PATH, "exp_0049")
    ).to_netcdf(os.path.join(DATA_PATH, "exp_0049", "dual_graph.nc"))


@timeit
def plot_swegnn_ghost_cells(path_in: str, output_path: str) -> None:
    """
    Calculates SWE-GNN data on the fly and plots components to verify ghost cells.

    This plot shows:
    1. The original triangular mesh (faint gray).
    2. The dual graph face centers and edges (faint blue).
    3. The original mesh boundary edges (thick green).
    4. The real boundary faces (red dots).
    5. The calculated ghost faces (cyan dots).
    6. The mirroring lines connecting boundary faces to ghost faces (dashed magenta).
    7. The calculated ghost nodes (orange 'x').

    Args:
        path_in (str): Path to the directory containing fort.*.nc files
                       (e.g., .../exp_0049).
        output_path (str): Path to save the output plot (e.g., .../figure.pdf).
    """
    print(f"Generating SWE-GNN data on the fly from {path_in}...")

    swegnn_ds = None  # Initialize for finally block
    try:
        # --- 1. Calculate SWE-GNN Dataset On-the-Fly ---
        # Use use_dask=False to compute immediately for plotting
        swegnn_ds = swegnn_dg_from_mesh_ds_from_path(
            path=path_in, use_dask=True
        )
        print("SWE-GNN dataset calculated.", swegnn_ds)
        print("SWE-GNN dataset variables:", [var for var in swegnn_ds])

        # --- 2. Load Original Mesh Coords ---
        original_mesh_path = os.path.join(path_in, "fort.63.nc")
        print(f"Loading original mesh from {original_mesh_path}...")
        try:
            with nc.Dataset(original_mesh_path, "r") as ds:
                print("Variables in original mesh file:", ds.variables.keys())
                original_x = ds.variables["x"][:]
                original_y = ds.variables["y"][:]
                original_elements = ds.variables["element"][:] - 1
        except FileNotFoundError:
            print(f"Error: Original mesh file not found at {original_mesh_path}")
            return
        except KeyError as e:
            print(f"Error: Missing variable {e} in {original_mesh_path}")
            return

        # --- 3. Setup Plot ---
        plot_defaults()
        fig, ax = plt.subplots(figsize=get_dim(fraction_of_line_width=1.2, ratio=0.7))

        # --- 4. Plot Original Mesh (faint) ---
        ax.triplot(
            original_x,
            original_y,
            original_elements,
            color="gray",
            alpha=0.3,
            linewidth=0.5,
            label="Original Mesh",
        )

        # --- 5. Plot Dual Graph (faint) ---
        edge_index = swegnn_ds["edge_index"].values
        face_x = swegnn_ds["x"].values
        face_y = swegnn_ds["y"].values
        starts, ends = edge_index[0, :], edge_index[1, :]
        # Plot dual edges as segments
        ax.plot(
            [face_x[starts], face_x[ends]],
            [face_y[starts], face_y[ends]],
            color="blue",
            alpha=0.15,
            linewidth=0.4,
            label="Dual Graph Edges",
        )
        # Plot all dual nodes (face centers)
        ax.scatter(
            face_x, face_y, color="blue", s=0.5, alpha=0.2, label="Face Centers"
        )

        # --- 6. Plot Original Boundary Edges (Green) ---
        edge_index_BC = swegnn_ds["edge_index_BC"].values
        if edge_index_BC.shape[0] > 0:
            bc_edge_starts = edge_index_BC[:, 0]
            bc_edge_ends = edge_index_BC[:, 1]
            ax.plot(
                [original_x[bc_edge_starts], original_x[bc_edge_ends]],
                [original_y[bc_edge_starts], original_y[bc_edge_ends]],
                color="green",
                alpha=0.8,
                linewidth=2.0,
                label="Boundary Edges (Original)",
                zorder=4,
            )
        else:
            print("Warning: No 'edge_index_BC' found in dataset.")

        # --- 7. Plot Boundary Faces (Red) ---
        face_BC_indices = swegnn_ds["face_BC"].values
        if face_BC_indices.shape[0] > 0:
            boundary_face_x = face_x[face_BC_indices]
            boundary_face_y = face_y[face_BC_indices]
            ax.scatter(
                boundary_face_x,
                boundary_face_y,
                color="red",
                s=12,
                label="Boundary Faces (Real)",
                zorder=5,
            )
        else:
            print("Warning: No 'face_BC' found in dataset.")
            # Set empty arrays to prevent errors if no BCs
            boundary_face_x, boundary_face_y = np.array([]), np.array([])

        # --- 8. Plot Ghost Faces (Cyan) ---
        ghost_face_x = swegnn_ds["ghost_face_x"].values
        ghost_face_y = swegnn_ds["ghost_face_y"].values
        if ghost_face_x.shape[0] > 0:
            ax.scatter(
                ghost_face_x,
                ghost_face_y,
                color="cyan",
                s=12,
                edgecolors="k",
                linewidth=0.5,
                label="Ghost Faces (Calculated)",
                zorder=5,
            )
        else:
            print("Warning: No 'ghost_face_x' found in dataset.")

        # --- 9. Plot Mirroring Lines (Magenta) ---
        if boundary_face_x.shape[0] > 0 and ghost_face_x.shape[0] > 0:
            # Ensure they have the same length
            if boundary_face_x.shape[0] == ghost_face_x.shape[0]:
                ax.plot(
                    [boundary_face_x, ghost_face_x],
                    [boundary_face_y, ghost_face_y],
                    color="magenta",
                    alpha=0.6,
                    linewidth=0.8,
                    linestyle="--",
                    label="Face Mirroring",
                    zorder=3,
                )
            else:
                print(
                    f"Warning: Mismatch in boundary faces ({boundary_face_x.shape[0]}) "
                    f"and ghost faces ({ghost_face_x.shape[0]}). "
                    "Cannot plot mirroring lines."
                )

        # --- 10. Plot Ghost Nodes (Orange 'x') ---
        ghost_node_x = swegnn_ds["ghost_node_x"].values
        ghost_node_y = swegnn_ds["ghost_node_y"].values
        if ghost_node_x.shape[0] > 0:
            ax.scatter(
                ghost_node_x,
                ghost_node_y,
                color="orange",
                s=15,
                marker="x",
                label="Ghost Nodes (Calculated)",
                zorder=4,
            )
        else:
            print("Warning: No 'ghost_node_x' found in dataset.")

        # --- 11. Finalize Plot ---
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude [$^{\circ}$E]")
        ax.set_ylabel("Latitude [$^{\circ}$N]")
        ax.set_title("SWE-GNN Ghost Cell Calculation Verification")

        # Create legend with a specific order for clarity
        handles, labels = ax.get_legend_handles_labels()
        label_map = {label: handle for handle, label in zip(handles, labels)}
        order_keys = [
            "Original Mesh",
            "Boundary Edges (Original)",
            "Boundary Faces (Real)",
            "Ghost Faces (Calculated)",
            "Face Mirroring",
            "Ghost Nodes (Calculated)",
            "Dual Graph Edges",
            "Face Centers",
        ]

        ordered_handles = [label_map[key] for key in order_keys if key in label_map]
        ordered_labels = [key for key in order_keys if key in label_map]

        ax.legend(ordered_handles, ordered_labels, fontsize="small", loc="best")

        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Ghost cell plot saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

    finally:
        if swegnn_ds is not None:
            swegnn_ds.close()  # Ensure the dataset is closed


if __name__ == "__main__":
    # python -m adforce.dual_graph
    # plot_dual_graph()
    # test_dual_graph()
    # make_dual_graph_nc()
    plot_swegnn_ghost_cells(
        path_in=os.path.join(DATA_PATH, "exp_0049"),
        output_path=os.path.join(FIGURE_PATH, "swegnn_ghost_cells.pdf"),
    )
