"""
Process ADCIRC meshes efficiently. This is the shared functionality for processing ADCIRC meshes.
"""
from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
import netCDF4 as nc
import xarray as xr
from sithom.time import timeit
from sithom.place import BoundingBox
from src.constants import NO_BBOX


@timeit
def xr_loader(file_path: str, verbose: bool = False) -> xr.Dataset:
    """
    Load an xarray dataset from a ADCIRC netCDF4 file.

    Args:
        file_path (str): Path to netCDF4 file.
        verbose (bool, optional): Whether to print. Defaults to False.

    Returns:
        xr.Dataset: loaded xarray dataset (lazy).
    """

    ds_nc = nc.Dataset(file_path)
    for var in ["neta", "nvel"]:  # known problem variables to remove.
        if var in ds_nc.variables:
            if verbose:
                print("removing", ds_nc.variables[var])
            del ds_nc.variables[var]
    # pass to xarray
    return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc))


@timeit
def calculate_adjacency_matrix(triangles: np.ndarray, N: int, sparse=True) -> Union[np.ndarray, csr_matrix]:
    """
    Calculate a boolean adjacency matrix for a mesh of triangles.
    Assumes that nodes are numbered from 0 to N-1.
    Assumes nodes are not self-adjacent.

    TODO: Test sparse matrix implementation.
    TODO: Could go to 3M long rows/cols if called twice. Is this a speedup? Is there a sym=True option?

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.
        N (int): Number of nodes in the mesh.

    Returns:
        Union[np.ndarray, csr_matrix]: NxN symetric boolean adjacency matrix.
    """
    # M is the number of triangles
    # triangles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], ...])
    rows = np.repeat(triangles, 2, axis=0).flatten() # 6M long
    # [0, 0, 1, 1, 2, 2, ...] values 0 to N-1
    cols = np.repeat(np.roll(triangles, shift=1, axis=1), 2, axis=None) # 6M long
    # [2, 1, 0, 2, 1, 0, ...] values 0 to N-1
    if not sparse:
        adjacency_matrix = np.zeros((N, N), dtype=bool) # NxN boolean matrix
        adjacency_matrix[rows, cols] = True  # "smart indexing" in numpy is very fast and efficient
    else:
        adjacency_matrix = csr_matrix((np.ones_like(rows, dtype=bool), (rows, cols)), shape=(N, N), dtype=bool)
    # adjacency_matrix[cols, rows] = True # already symetric without second call
    return adjacency_matrix


@timeit
def select_coast_indices(mesh_ds: xr.Dataset, overtopping: bool = False) -> np.ndarray:
    """
    Select the coastal nodes.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" (and "depth" if overtopping allowed).
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.

    Returns:
        np.ndarray: coastal indices.
    """
    if not overtopping:
        # method 1: find the nodes that are only in 3 triangles or less:
        (uniq, freq) = np.unique(mesh_ds.element.values -1, return_counts=True)
        # maybe we should add a check for the depth to exclude water boundary nodes.
        indices = uniq[freq <= 4] # 3 or less triangles
        adj = calculate_adjacency_matrix(mesh_ds.element.values -1, len(mesh_ds.x), sparse=False)
        return indices, adj[indices, :][:, indices]
    else:
        # method 2: find land, propogate out to adjacent nodes, intersect with not land.
        # float32 "depth" which attribute for every node (length N).
        # if "depth" negative, then it is land.
        # if "depth" positive, then it is water.
        # We have mesh Mx3 array of triangle indices.
        # [[0, 1, 2], [1, 2, 3], [2, 3, 4], ...]
        # From this we can calculate the adjacency matrix.
        # Objective: We want to select the points that are in the water and are adjacent to land.
        # Output: list of indices or boolean array of length N.

        adj = calculate_adjacency_matrix(mesh_ds.element.values -1, len(mesh_ds.x))
        depths = mesh_ds.depth.values # depths vector length N
        land = depths < 0 # boolean land vector length N
        # coast = np.any(adj[land], axis=0) & ~land # probably the same as
        # ?bipartite graph?
        coast = adj.dot(land) & ~land # , which is more efficient?
        return np.where(coast)[0], adj[coast, :][:, coast].todense()


@timeit
def select_coast(mesh_ds: xr.Dataset, overtopping: bool = False) -> xr.Dataset:
    """
    Select the coastal nodes.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" (and "depth" if overtopping allowed).
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.

    Returns:
        xr.Dataset: Filtered xarray dataset.
    """
    indices, adj = select_coast_indices(mesh_ds, overtopping=overtopping)
    new_mesh = mesh_ds.isel(node=indices)
    del new_mesh["element"]
    new_mesh["adj"] = (["node1", "node2"], adj)
    return new_mesh


@timeit
def filter_mesh(adc_ds: xr.Dataset, indices: np.ndarray) -> xr.Dataset:
    """
    Filter an ADCIRC mesh to a subset of nodes.
    Keep the triangular component mesh.

    Args:
        adc_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element".
        indices (np.ndarray): Indices to keep.

    Returns:
        xr.Dataset: Filtered xarray dataset.

    """
    og_indices = np.arange(len(adc_ds.x))
    neg_indices = np.setdiff1d(og_indices, indices)
    new_indices = np.where(indices)[0]
    elements = adc_ds.element.values - 1  # triangular component mesh
    mask = ~np.isin(elements, neg_indices).any(axis=1)
    filtered_elements = elements[mask]
    mapping = dict(zip(indices, new_indices))
    relabelled_elements = np.vectorize(mapping.get)(filtered_elements)
    adc_ds_n = adc_ds.isel(node=indices)
    del adc_ds_n["element"]  # remove old triangular component mesh
    adc_ds_n["element"] = (  # add new triangular component mesh
        ["nele", "nvertex"],
        relabelled_elements + 1,
    )  # add new triangular component mesh
    return adc_ds_n


@timeit
def bbox_mesh(
    file_path: str = "../data/fort.63.nc", bbox: BoundingBox = NO_BBOX
) -> xr.Dataset:
    """
    Load an adcirc output file and filter it to a bounding box.

    Args:
        file_path (str, optional): Path to ADCIRC file. Defaults to "../data/fort.63.nc".
        bbox (BoundingBox, optional): Bounding box to trim to. Defaults to NO_BBOX.

    Returns:
        xr.Dataset: Filtered xarray dataset.
    """
    adc_ds = xr_loader(file_path)  # load file as xarray dataset
    xs = adc_ds.x.values  # longitudes (1d numpy array)
    ys = adc_ds.y.values  # latitudes (1d numpy array)
    ys_in = (bbox.lat[0] < ys) & (ys < bbox.lat[1])
    xs_in = (bbox.lon[0] < xs) & (xs < bbox.lon[1])
    both_in = xs_in & ys_in
    indices = np.where(both_in)[0]  # surviving old labels
    return filter_mesh(adc_ds, indices)
    """
    new_indices = np.where(indices)[0]  # new labels
    neg_indices = np.where(~both_in)[0]  # indices to get rid of
    elements = adc_ds.element.values - 1  # triangular component mesh
    mask = ~np.isin(elements, neg_indices).any(axis=1)
    filtered_elements = elements[mask]
    mapping = dict(zip(indices, new_indices))
    relabelled_elements = np.vectorize(mapping.get)(filtered_elements)
    adc_ds_n = adc_ds.isel(node=indices)
    del adc_ds_n["element"]  # remove old triangular component mesh
    adc_ds_n["element"] = ( # add new triangular component mesh
        ["nele", "nvertex"],
        relabelled_elements + 1,
    )  # add new triangular component mesh
    return adc_ds_n
    """


@timeit
def select_edge_indices(
    mesh_ds: xr.Dataset,
    lon: float,
    lat: float,
    number: int = 10,
    overtopping=False,
    verbose=False
) -> np.ndarray:
    """
    Select edge cells.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y" and "element".
        lon (float): Longitude of central point (degree_East).
        lat (float): Latitude of central point (degree_North).
        number (int, optional): How many to choose initially. Defaults to 10.
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.
        verbose (bool, optional): Whether to print. Defaults to False.

    Returns:
        np.ndarray: coastal indices near central point.
    """

    lons = mesh_ds.x.values # longitudes (1d numpy array)
    lats = mesh_ds.y.values # latitudes (1d numpy array)

    # compute Euclidean (flat-world) distance in degrees**2
    nsq_distances = -(lons - lon) ** 2 - (lats - lat) ** 2

    if verbose:
        print("Central point", lon, lat)
        print("Distances", nsq_distances)

    # Use argpartition to get top K indices
    # Note: The result includes the partition index, so we use -K-1 to get exactly K elements
    indices = np.argpartition(nsq_distances, -number)[-number:]

    # Optional: Sort the top K indices if you need them sorted
    indices = indices[np.argsort(nsq_distances[indices])[::-1]]

    edge_vertices, adj = select_coast_indices(mesh_ds, overtopping=overtopping)

    if verbose:
        print("Nearby indices", indices, len(indices))
        # print("Coastals indices", edge_vertices, len(edge_vertices))

    indices = indices[np.in1d(indices, edge_vertices)]
    return indices


if __name__ == "__main__":
    # python -m adforce.mesh
    bbox_mesh()
