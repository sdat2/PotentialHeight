"""
Process ADCIRC meshes efficiently vectorized/sparses.
This is the shared functionality for processing ADCIRC meshes.

TODO: Add the dual graph calculation for SurgeNet.
"""

from typing import Union, Tuple, List
import numpy as np
import collections
from scipy.sparse import csr_matrix, coo_matrix
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from sithom.time import timeit
from sithom.place import BoundingBox
from .constants import NO_BBOX, FORT63_EXAMPLE


@timeit
def xr_loader(
    file_path: str, verbose: bool = False, use_dask: bool = False
) -> xr.Dataset:
    """
    Load an xarray dataset from a ADCIRC netCDF4 file.

    ADCIRC netCDF4 files have some scalar variables that need to be removed.

    Args:
        file_path (str): Path to netCDF4 file.
        verbose (bool, optional): Whether to print. Defaults to False.
        use_dask (bool, optional): Whether to use dask. Defaults to False.

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
    if use_dask:
        return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc), chunks={"time": 1})
    else:
        return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc))


def starts_ends_to_adjacency_matrix(
    starts: Union[List[int], np.ndarray],
    ends: Union[List[int], np.ndarray],
    size: int,
    sparse: bool = True,
) -> Union[np.ndarray, csr_matrix]:
    """
    Convert start, end indices to adjacency matrix.

    Args:
        starts (Union[List[int], np.ndarray]): start indices.
        ends (Union[List[int], np.ndarray]): end indices.
        size (int): number of nodes.
        sparse (bool, optional): Whether to return a sparse matrix. Defaults to True.

    Returns:
        Union[np.ndarray, csr_matrix]: adjacency matrix.
    """
    if not sparse:
        adjacency_matrix = np.zeros((size, size), dtype=bool)  # NxN boolean matrix
        adjacency_matrix[starts, ends] = True
    else:
        adjacency_matrix = csr_matrix(
            (np.ones_like(starts, dtype=bool), (starts, ends)),
            shape=(size, size),
            dtype=bool,
        )
    return adjacency_matrix


def standard_starts_ends_from_triangles(
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate start, end indices from the triangles for the regular graph.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: start, end indices for the adjacency matrix.

    Examples::
        >>> start, end = standard_starts_ends_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(start == np.array([2, 1, 0, 2, 1, 0, 3, 2, 1, 3, 2, 1]))
        True
        >>> np.all(end == np.array([0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3]))
        True
    """
    # M is the number of triangles
    # triangles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], ...])
    starts = np.repeat(triangles[:, ::-1], 2, axis=0).flatten()  # 6M long
    # [2, 1, 0, 2, 1, 0, ...] values 0 to N-1
    ends = np.repeat(triangles, 2, axis=None)  # 6M long
    # [0, 0, 1, 1, 2, 2, ...] values 0 to N-1
    return starts, ends


def dual_graph_starts_ends_from_triangles(
    triangles: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """
    Generate start, end indices for the dual graph from the triangles.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: start, end indices for the dual graph.

    Examples::
        >>> start, end = dual_graph_starts_ends_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(start == np.array([0, 1]))
        True
        >>> np.all(end == np.array([1, 0]))
        True

    """
    edge_dict = collections.defaultdict(list)

    for i, triangle in enumerate(triangles):
        edge_dict[tuple(triangle[0:2])].append(i)
        edge_dict[tuple(triangle[1:3])].append(i)
        edge_dict[tuple(triangle[[0, 2]])].append(i)

    starts = []
    ends = []

    for edge in edge_dict:
        nodes = edge_dict[edge]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                starts.append(nodes[i])
                starts.append(nodes[j])
                ends.append(nodes[j])
                ends.append(nodes[i])
    return starts, ends


# @timeit
def calculate_adjacency_matrix(
    triangles: np.ndarray, N: int, sparse: bool = True
) -> Union[np.ndarray, csr_matrix]:
    """
    Calculate a boolean adjacency matrix for a mesh of triangles.
    Assumes that nodes are numbered from 0 to N-1.
    Assumes nodes are not self-adjacent.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.
        N (int): Number of nodes in the mesh.
        sparse (bool, optional): Whether to return a sparse matrix. Defaults to True.

    Returns:
        Union[np.ndarray, csr_matrix]: NxN symetric boolean adjacency matrix.

    Examples::
        >>> np.all(calculate_adjacency_matrix(np.array([[0, 1, 2]]), 3, sparse=False) == np.array([[False, True, True], [True, False, True], [True, True, False]]))
        True
        >>> np.all(calculate_adjacency_matrix(np.array([[0, 1, 2], [1, 2, 3]]), 4, sparse=False) == np.array([[False, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, False]]))
        True
    """
    return starts_ends_to_adjacency_matrix(
        *standard_starts_ends_from_triangles(triangles), N, sparse=sparse
    )


def calculate_dual_graph_adjacency_matrix(
    triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, csr_matrix]:
    """
    Calculate the dual graph adjacency matrix for a mesh of triangles.

    A dual graph is a graph where each face of the original graph is a node,
    and the new nodes are adjacent if they share an edge.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.
        sparse (bool, optional): Whether to return a sparse matrix. Defaults to True.

    Returns:
        Union[np.ndarray, csr_matrix]: MxM symetric boolean adjacency matrix.

    Examples::
        >>> np.all(calculate_dual_graph_adjacency_matrix(np.array([[0, 1, 2], [1, 2, 3]]), sparse=False) == np.array([[False, True], [True, False]]))
        True
        >>> np.all(calculate_dual_graph_adjacency_matrix(np.array([[0, 1, 2], [2, 3, 4]]), sparse=False) == np.array([[False, False], [False, False]]))
        True
    """

    M = len(triangles)

    return starts_ends_to_adjacency_matrix(
        *dual_graph_starts_ends_from_triangles(triangles), M, sparse=sparse
    )


def dual_graph_dataset(triangles: np.ndarray) -> xr.Dataset:
    """
    Calculate the dual graph adjacency matrix for a mesh of triangles.

    Nodes are the faces of the original graph, and adjacent if they share an edge.
    There are E shared edges between the faces.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices. Assumes nodes are numbered from 0 to N-1.
            where N is the number of nodes in the original mesh.

    Returns:
        xr.Dataset: Dual graph dataset.
         - "element": Mx3 array of triangle indices.
         - "start": 2E array of start indices. (Double counting)
         - "end": 2E array of end indices. (Double counting)

    Examples::
        >>> ds = dual_graph_dataset(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(ds.start.values == np.array([0, 1]))
        True
        >>> np.all(ds.end.values == np.array([1, 0]))
        True
        >>> np.all(ds.element.values - 1 == np.array([[0, 1, 2], [1, 2, 3]]))
        True
    """
    starts, ends = dual_graph_starts_ends_from_triangles(triangles)
    return xr.Dataset(
        {
            "element": (
                ["nele", "nvertex"],
                triangles + 1,
                {
                    "description": "Original mesh triangles, with each new node corresponding to the face of the old triangle mesh."
                },
            ),
            "start": ("edge", starts, {"description": "Start indices of the edges."}),
            "end": ("edge", ends, {"description": "End indices of the edges."}),
        },
        coords={
            "edge": np.arange(len(starts)),
            "nele": np.arange(len(triangles)),
            "nvertex": np.arange(3),
        },
        attrs={"description": "Dual graph of the mesh."},
    )


def _test_xr_dataset() -> xr.Dataset:
    """
    Test xr.Dataset for dual graph.

    Returns:
        xr.Dataset: Dual graph dataset.
    """
    return xr.Dataset(
        {
            "element": (
                ["nele", "nvertex"],
                np.array([[0, 1, 2], [1, 2, 3]]) + 1,
                {
                    "description": "Original mesh triangles, with each new node corresponding to the face of the old triangle mesh."
                },
            ),
            "depth": (["node"], np.array([1, 2, 3, 4])),
            "x": (["node"], np.array([0, 1, 2, 3])),
            "y": (["node"], np.array([0, -1, -1, -2])),
            "zeta": (["time", "node"], np.array([[1, 2, 3, 4], [2, 3, 4, 5]])),
        },
        coords={
            "nele": np.arange(2),
            "nvertex": np.arange(3),
            "node": np.arange(4),
            "time": np.array(
                [
                    np.datetime64("2021-01-01 00:00:00"),
                    np.datetime64("2021-01-01 01:00:00"),
                ]
            ),
        },
        attrs={"description": "Test original dataset."},
    )


def process_dual_graph(ds: xr.Dataset = xr_loader(FORT63_EXAMPLE)) -> xr.Dataset:
    """
    Create a dual graph dataset from an ADCIRC output dataset.

    Args:
        ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" and "depth".

    Returns:
        xr.Dataset: Dual graph dataset.

    Examples::
        >>> ds = process_dual_graph(_test_xr_dataset())
        >>> np.isclose(ds.depth.values, np.array([2, 3]), atol=1e-6).all()
        True
        >>> np.isclose(ds.y.values, np.array([-1 +1/3, -1 - 1/3]), atol=1e-6).all()
        True
        >>> np.isclose(ds.x.values, np.array([1, 2]), atol=1e-6).all()
        True
        >>> np.all(ds.element.values - 1 == np.array([[0, 1, 2], [1, 2, 3]]))
        True
        >>> np.isclose(ds.zeta.values, np.array([[2, 3], [3, 4]]), atol=1e-6).all()
        True
    """
    # ds = xr_loader(path)
    dg = dual_graph_dataset(ds.element.values - 1)
    for val in ["x", "y", "depth"]:  # static fields
        dg[val] = (
            ["nele"],
            mean_for_triangle(ds[val].values, ds["element"].values - 1),
        )
    for val in ["zeta"]:  # time varying fields
        dg[val] = (
            ["time", "nele"],
            np.mean(ds[val].values[:, ds["element"].values - 1], axis=2),
        )
    # calculate the gradient of the depth
    dg["depth_grad"] = (
        ["direction", "nele"],
        grad_for_triangle(
            ds.x.values, ds.y.values, ds.depth.values, ds.element.values - 1
        ),
    )
    return dg.assign_coords({"direction": ["x", "y"]})


def mean_for_triangle(
    values: np.ndarray, triangles: np.ndarray, mean_axis=1
) -> np.ndarray:
    """
    Calculate the mean value for each triangle.

    Args:
        values (np.ndarray): N array of values.
        triangles (np.ndarray): Mx3 array of triangle indices.

    Returns:
        np.ndarray: M array of mean values for each triangle.

    Examples::
        >>> np.all(mean_for_triangle(np.array([1, 2, 3, 4]), np.array([[0, 1, 2], [1, 2, 3]])) == np.array([2, 3]))
        True
        >>> np.all(mean_for_triangle(np.array([1, 2, 3, 4]), np.array([[0, 1, 2]])) == np.array([2]))
        True
    """
    return np.mean(values[triangles], axis=1)


def grad_for_triangle(
    x_lon: np.ndarray, y_lat: np.ndarray, values: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """
    Calculate the gradient of the values for each triangle.

    Three points define a plane, so we can calculate the gradient of the plane.

    Args:
        x_lon (np.ndarray): N array of longitudes. The x values.
        y_lat (np.ndarray): N array of latitudes. The y values.
        values (np.ndarray): N array of values. The z values.
        triangles (np.ndarray): Mx3 array of triangle indices. The indices of each plane.

    Returns:
        np.ndarray: 2xM array of gradients (dz/dx, dz/dy) for each triangle.

    Examples::
        >>> np.all(np.isclose(grad_for_triangle(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([[0, 1, 2]])), np.array([[-1], [0]]), atol=1e-6))
        True
        >>> np.all(grad_for_triangle(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([[0, 1, 2]])) == np.array([[-1], [-1]]))
        True
        >>> np.all(grad_for_triangle(np.array([0, 0, 2]), np.array([0, 2, 0]), np.array([1, 0, 0]), np.array([[0, 1, 2], [1, 2, 0]])) == np.array([[-0.5, -0.5], [-0.5, -0.5]]))
        True
        >>> np.all(grad_for_triangle(np.array([0, 0, 0.5]), np.array([0, 0.5, 0]), np.array([1, 0, 0]), np.array([[0, 1, 2]])) == np.array([[-2], [-2]]))
        True
        >>> np.all(grad_for_triangle(np.array([0, 0, 0.5]), np.array([0, 0.5, 0]), np.array([2, 0, 0]), np.array([[0, 1, 2]])) == np.array([[-4], [-4]]))
        True
        >>> np.all(grad_for_triangle(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([[0, 1, 2]])) == np.array([[0], [0]]))
        True
        >>> assert np.all(grad_for_triangle(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([[0, 1, 2]])) == np.array([[0], [1]]))
    """
    x = x_lon[triangles]  # Mx3
    y = y_lat[triangles]  # Mx3
    z = values[triangles]  # Mx3

    # get normal vectors
    c = np.stack((x, y, z), axis=2)  # stack the x, y, z values into columns (Mx3x3)
    # (B-A) x (C-A)
    normal = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], axis=1)  # Mx3
    # To avoid dividing by zero in case the triangles are degenerate
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_x = -normal[:, 0] / normal[:, 2]  # dz/dx  # M
        grad_y = -normal[:, 1] / normal[:, 2]  # dz/dy  # M
    return np.stack((grad_x, grad_y))  # 2xM


# print(xr_loader("../data/fort.63.nc", use_dask=False))


@timeit
def select_coast(
    mesh_ds: xr.Dataset, overtopping: bool = False, keep_sparse: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the coastal nodes.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" (and "depth" if overtopping allowed).
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: coastal indices, adjacency matrix.
    """
    if not overtopping:
        # method 1: find the nodes that are only in 3 triangles or less:
        (uniq, freq) = np.unique(mesh_ds.element.values - 1, return_counts=True)
        # maybe we should add a check for the depth to exclude water boundary nodes.
        indices = uniq[freq <= 4]  # 3 or less triangles
        adj = calculate_adjacency_matrix(
            mesh_ds.element.values - 1, len(mesh_ds.x), sparse=False
        )
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

        adj = calculate_adjacency_matrix(mesh_ds.element.values - 1, len(mesh_ds.x))
        depths = mesh_ds.depth.values  # depths vector length N
        land = depths < 0  # boolean land vector length N
        # coast = np.any(adj[land], axis=0) & ~land # probably the same as
        coast = adj.dot(land) & ~land  # , which is more efficient?
        if keep_sparse:
            return np.where(coast)[0], adj[coast, :][:, coast]
        else:
            return np.where(coast)[0], adj[coast, :][:, coast].todense()


@timeit
def select_coast_mesh(mesh_ds: xr.Dataset, overtopping: bool = False) -> xr.Dataset:
    """
    Select the coastal mesh.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" (and "depth" if overtopping allowed).
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.

    Returns:
        xr.Dataset: Filtered xarray dataset.
    """
    indices, adj = select_coast(mesh_ds, overtopping=overtopping)
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
    file_path: str = "../data/fort.63.nc",
    bbox: BoundingBox = NO_BBOX,
    use_dask: bool = True,
) -> xr.Dataset:
    """
    Load an adcirc output file and filter it to a bounding box.

    Args:
        file_path (str, optional): Path to ADCIRC file. Defaults to "../data/fort.63.nc".
        bbox (BoundingBox, optional): Bounding box to trim to. Defaults to NO_BBOX.
        use_dask (bool, optional): Whether to use dask. Defaults to True.

    Returns:
        xr.Dataset: Filtered xarray dataset.
    """

    adc_ds = xr_loader(file_path, use_dask=use_dask)  # load file as xarray dataset
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
def select_nearby(
    mesh_ds: xr.Dataset,
    lon: float,
    lat: float,
    number: int = 10,
    overtopping: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Select edge cells near a point.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y" and "element".
        lon (float): Longitude of central point (degree_East).
        lat (float): Latitude of central point (degree_North).
        number (int, optional): How many to choose initially. Defaults to 10.
        overtopping (bool, optional): Whether overtopping is included in mesh. Defaults to False.
        verbose (bool, optional): Whether to print. Defaults to False.

    Returns:
        xr.Dataset: coastal indices near central point.
    """

    edge_vertices, adj = select_coast(mesh_ds, overtopping=overtopping)

    coast_mesh_ds = mesh_ds.isel(node=edge_vertices)
    lons = coast_mesh_ds.x.values  # longitudes (1d numpy array)
    lats = coast_mesh_ds.y.values  # latitudes (1d numpy array)

    print("Central point", lon, "degrees_East", lat, "degrees_North")

    # compute Euclidean (flat-world) distance in degrees**2
    nsq_distances = -((lons - lon) ** 2) - ((lats - lat) ** 2)

    if verbose:
        print("Central point", lon, "degrees_East", lat, "degrees_North")
        print("Distances", nsq_distances, "degrees**2")

    # Use argpartition to get top K indices
    # Note: The result includes the partition index, so we use -K-1 to get exactly K elements
    indices = np.argpartition(nsq_distances, -number)[-number:]

    # Optional: Sort the top K indices if you need them sorted
    indices = indices[np.argsort(nsq_distances[indices])[::-1]]

    # indices = indices[np.in1d(indices, edge_vertices)]
    return coast_mesh_ds.isel(node=indices)


def plot_contour(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    adj_matrix: Union[np.ndarray, csr_matrix],
) -> None:
    """
    Plot a mesh with edges.

    Args:
        ax: matplotlib axis.
        x_values (np.ndarray): x values of nodes.
        y_values (np.ndarray): y values of nodes.
        adj_matrix (Union[np.ndarray, csr_matrix]): adjacency matrix.
    """

    if isinstance(adj_matrix, csr_matrix):
        adj_coo = coo_matrix(adj_matrix)
        for i, j, _ in zip(adj_coo.start, adj_coo.col, adj_coo.data):
            ax.plot(
                [x_values[i], x_values[j]],
                [y_values[i], y_values[j]],
                "k-",
                linewidth=0.5,
            )  # Plot each edge with a thin line
    else:
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j]:
                    ax.plot(
                        [x_values[i], x_values[j]],
                        [y_values[i], y_values[j]],
                        "k-",
                        linewidth=0.5,
                    )

    # Optionally, plot the nodes with very small size if needed
    ax.scatter(x_values, y_values, s=0.1, color="blue")  # Very small node size

    # Adjust the plot
    ax.set_aspect("equal")


if __name__ == "__main__":
    # python -m adforce.mesh
    # bbox_mesh()
    print(calculate_adjacency_matrix(np.array([[0, 1, 2], [1, 2, 3]]), 4, sparse=False))
    print(calculate_adjacency_matrix(np.array([[0, 1, 2]]), 3, sparse=False))
    print(calculate_adjacency_matrix(np.array([[0, 1, 2]]), 5, sparse=False))
    # print(calculate_adjacency_matrix(np.array([[]]), 2, sparse=False))
    print(process_dual_graph(xr_loader(FORT63_EXAMPLE)))
    print(xr_loader(FORT63_EXAMPLE))
