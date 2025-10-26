"""Script to explore the boundaries of the ADCIRC mesh."""
from typing import Optional
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from sithom.plot import plot_defaults, get_dim
from .constants import FORT63_EXAMPLE, FIGURE_PATH, DATA_PATH
from .mesh import elevation_boundary_edges


def find_boundaries_in_fort63(path: str = FORT63_EXAMPLE, typ="medium"):
    """Plot the boundaries in the ADCIRC mesh from a fort.63 netCDF file.

    Args:
        path (str): Path to the fort.63 netCDF file. Defaults to FORT63_EXAMPLE.
        typ (str): Type of mesh for labeling purposes. Defaults to "medium".
    """

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
    coast_type_l = []
    for segment, coast_type in enumerate(ibtype):
        coast_type_l += [coast_type] * nvell[segment]
    assert len(coast_type_l) == len(nbvv)
    fig, axs = plt.subplots(1, 1, figsize=get_dim(), sharex=True, sharey=True)
    coast_type = np.array(coast_type_l)
    # axs.scatter(x[nbvv], y[nbvv], s=0.1, color="blue", label="Normal Flow Boundary")
    # just plot normal flow boundary where coast_type is 0
    axs.scatter(
        x[nbvv[np.where(coast_type == 0)[0]]],
        y[nbvv[np.where(coast_type == 0)[0]]],
        s=0.1,
        color="red",
        label="Normal Flow Boundary (Type 0)",
    )
    # Plot the normal flow boundary where coast_type is 1
    axs.scatter(
        x[nbvv[np.where(coast_type == 1)[0]]],
        y[nbvv[np.where(coast_type == 1)[0]]],
        s=0.1,
        color="green",
        label="Normal Flow Boundary (Type 1)",
    )

    axs.set_aspect("equal")
    axs.set_title(typ)  # "eastcoast_95d_ll_select.grd")  # "EC95d Mesh")

    axs.scatter(
        x[nbdv],
        y[nbdv],
        s=0.1,
        color="blue",
        label="Elevation Specified Boundary (Type 0)",
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
        os.path.join(figure_path, f"mesh_boundaries_{typ}.pdf"),
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
        os.path.join(figure_path, f"mesh_boundaries_types_{typ}.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def extract_elevation_boundary_edges(
    netcdf_path: str,
    boundary_type: Optional[int] = 0
) -> np.ndarray:
    """
    Extracts the edges corresponding to elevation specified boundaries
    from an ADCIRC NetCDF file (like fort.63.nc or maxele.63.nc).

    Args:
        netcdf_path (str): Path to the ADCIRC NetCDF file.
        boundary_type (Optional[int]): The specific elevation boundary type
            (from 'ibtypee') to extract. If None, extracts edges for all
            elevation boundary types. Defaults to 0 based on user's data.

    Returns:
        np.ndarray: An array of boundary edges, shape (num_BC_edges, 2),
                    containing pairs of 0-based node indices. Returns an
                    empty array if no relevant boundaries are found.

    Raises:
        FileNotFoundError: If the netcdf_path does not exist.
        KeyError: If required variables ('nbdv', 'nvdll', 'ibtypee') are
                  missing from the NetCDF file.

    Doctests:
        >>> # Create a dummy NetCDF file for testing
        >>> dummy_path = "dummy_adcirc_boundary.nc"
        >>> with nc.Dataset(dummy_path, 'w') as ds:
        ...     _ = ds.createDimension('node', 5)
        ...     _ = ds.createDimension('neta', 4) # Total nodes in elevation boundaries
        ...     _ = ds.createDimension('nope', 1) # Number of elevation boundary segments
        ...     # Node indices (1-based in file, adjusted to 0-based: 0, 1, 3, 4)
        ...     nbdv_var = ds.createVariable('nbdv', 'i4', ('neta',))
        ...     nbdv_var[:] = np.array([1, 2, 4, 5])
        ...     # Segment lengths (one segment with 4 nodes)
        ...     nvdll_var = ds.createVariable('nvdll', 'i4', ('nope',))
        ...     nvdll_var[:] = np.array([4])
        ...     # Segment types (one segment of type 0)
        ...     ibtypee_var = ds.createVariable('ibtypee', 'i4', ('nope',))
        ...     ibtypee_var[:] = np.array([0])
        >>> extract_elevation_boundary_edges(dummy_path, boundary_type=0)
        masked_array(
        data=[[0, 1],
               [1, 3],
               [3, 4]],
        mask=False,
        fill_value=999999,
        dtype=int32)
        >>> # Test with type mismatch
        >>> extract_elevation_boundary_edges(dummy_path, boundary_type=1)
        array([], shape=(0, 2), dtype=int64)
        >>> # Test with multiple segments (requires adjusting dimensions/data)
        >>> # Cleanup dummy file
        >>> os.remove(dummy_path)
    """
    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found at: {netcdf_path}")

    with nc.Dataset(netcdf_path, 'r') as ds:
        # Check for required variables
        required_vars = ['nbdv', 'nvdll', 'ibtypee']
        for var in required_vars:
            if var not in ds.variables:
                raise KeyError(f"Required variable '{var}' not found in {netcdf_path}")

        be_np = get_elevation_boundary_edges(ds, boundary_type)

    return be_np


def get_elevation_boundary_edges(
        ds: nc.Dataset,
        boundary_type: Optional[int] = 0
        )  -> np.ndarray:
    """
    Helper function to extract elevation boundary edges from an open
    netCDF4 Dataset object.

    Args:
        ds (nc.Dataset): Open netCDF4 Dataset object.
        boundary_type (Optional[int]): The specific elevation boundary type
            (from 'ibtypee') to extract. If None, extracts edges for all
            elevation boundary types. Defaults to 0.

    Returns:
        np.ndarray: An array of boundary edges, shape (num_BC_edges, 2),
                    containing pairs of 0-based node indices. Returns an
                    empty array if no relevant boundaries are found.
    """

    nbdv = ds.variables['nbdv'][:] - 1  # Get 0-based node indices
    nvdll = ds.variables['nvdll'][:]    # Nodes per segment
    ibtypee = ds.variables['ibtypee'][:] # Type per segment
    return elevation_boundary_edges(nbdv, nvdll, ibtypee, boundary_type)



# --- Example Usage (within if __name__ == '__main__': block) ---
if __name__ == "__main__":
    # Example using the path from your script output
    # Make sure DATA_PATH is correctly defined or replace with the full path
    try:
        from .constants import DATA_PATH # Adjust import if needed
        file_path = os.path.join(DATA_PATH, "exp_0049", "maxele.63.nc")

        print(f"Extracting elevation boundary edges (type 0) from: {file_path}")
        ocean_edges = extract_elevation_boundary_edges(file_path, boundary_type=0)

        if ocean_edges.shape[0] > 0:
            print(f"Found {ocean_edges.shape[0]} elevation boundary edges.")
            print("First 5 edges:\n", ocean_edges[:5])
            print("Last 5 edges:\n", ocean_edges[-5:])
        else:
            print("No elevation boundary edges of the specified type found.")

    except ImportError:
        print("Could not import DATA_PATH. Please ensure constants.py is accessible or provide the full path.")
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
    # python -m adforce.boundaries

    # find_boundaries_in_fort63()
    # find_boundaries_in_fort63(os.path.join(DATA_PATH, "exp_0049", "maxele.63.nc"), typ="medium")

    #find_boundaries_in_fort63(os.path.join(DATA_PATH, "tiny.maxele.63.nc"), typ="small")
    # find_boundaries_in_fort63(os.path.join(DATA_PATH, "big.maxele.63.nc"), typ="big")
