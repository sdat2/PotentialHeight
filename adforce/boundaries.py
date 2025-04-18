"""Script to explore the boundaries of the ADCIRC mesh."""

import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from sithom.plot import plot_defaults, label_subplots, get_dim
from .constants import FORT63_EXAMPLE, FIGURE_PATH


def find_boundaries_in_fort63(path: str = FORT63_EXAMPLE):
    figure_path = os.path.join(FIGURE_PATH, "mesh_boundaries")
    os.makedirs(figure_path, exist_ok=True)
    plot_defaults()
    nc_ds = nc.Dataset(path)
    # Get the mesh coordinates
    x = nc_ds.variables["x"][:]
    y = nc_ds.variables["y"][:]
    # Get the mesh elements
    triangles = nc_ds.variables["element"][:] - 1
    ibtype = nc_ds.variables["ibtype"][:]  # type of normal flow boundary.
    ibtypee = nc_ds.variables["ibtypee"][:]  # type of elevvation boundary.
    nvdll = nc_ds.variables["nvdll"][:]  #
    nvell = nc_ds.variables["nvell"][
        :
    ]  # number of nodes in each normal flow boundary segment
    nbvv = (
        nc_ds.variables["nbvv"][:] - 1
    )  # node numbers on normal flow boundary segment
    # Get the mesh boundaries
    nbdv = nc_ds.variables["nbdv"][:] - 1  # node numbers on elevation boundary segment
    print(nc_ds)
    print("ibtype", nc_ds.variables["ibtype"])
    print("ibtypee", nc_ds.variables["ibtypee"])
    print("nvdll", nc_ds.variables["nvdll"])
    print("nvell", nc_ds.variables["nvell"])
    print("nbvv", nc_ds.variables["nbvv"])
    print("nbdv", nc_ds.variables["nbdv"])
    print()
    fig, axs = plt.subplots(1, 1, figsize=get_dim(), sharex=True, sharey=True)
    axs.scatter(x[nbvv], y[nbvv], s=0.1, color="blue", label="Normal Flow Boundary")
    axs.set_aspect("equal")
    axs.set_title("eastcoast_95d_ll_select.grd")  # "EC95d Mesh")
    axs.scatter(
        x[nbdv], y[nbdv], s=0.1, color="red", label="Elevation Specified Boundary"
    )
    axs.triplot(
        x, y, triangles, color="grey", alpha=0.5, linewidth=0.2, label="Mesh Triangles"
    )
    # expand to the side of plot
    # put anchor on middle left of plot, and then put the the right hand side of the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # axs[1].set_aspect("equal")
    # axs[1].set_title("Elevation Boundary Nodes (rivers)")
    axs.set_xlabel("Longitude [$^\circ$E]")
    axs.set_ylabel("Latitude [$^\circ$N]")
    # axs[1].set_xlabel("Longitude [$^\circ$E]")
    plt.savefig(
        os.path.join(figure_path, "mesh_boundaries.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    for var in nc_ds.variables:
        print(var, nc_ds.variables[var])
    # normal flow:  nbou: nvell (number), ibtype (type)
    # neta: nbdv (nodes)
    # nope: nvdll (number)

    plot_defaults()

    plt.scatter(
        np.arange(len(nvell)),
        nvell,
        c=ibtype,
        marker="+",
        cmap="viridis",
        s=0.5,
        label="Normal Flow Boundarys",
    )
    plt.scatter(
        np.arange(len(nvdll)),
        nvdll,
        c=ibtypee,
        marker="x",
        cmap="plasma",
        s=0.5,
        label="Elevation Specified Boundary Segments",
    )
    print("unique normal coast boundary ", np.unique(ibtype))
    print("number of segments", {i: np.sum(ibtype == i) for i in np.unique(ibtype)})

    print(
        "count of nodes in each category",
        {i: np.sum(nvell[np.where(ibtype == i)[0]]) for i in np.unique(ibtype)},
    )

    print("unique elevation boundary ", np.unique(ibtypee))
    print("number of segments", {i: np.sum(ibtypee == i) for i in np.unique(ibtypee)})
    print(
        "count of nodes in each category",
        {i: np.sum(nvdll[np.where(ibtypee == i)[0]]) for i in np.unique(ibtypee)},
    )
    plt.legend()
    plt.xlabel("Boundary Segment Number")
    plt.ylabel("Number of Nodes")

    plt.savefig(
        os.path.join(figure_path, "mesh_boundaries_types.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # python -m adforce.boundaries
    find_boundaries_in_fort63()
