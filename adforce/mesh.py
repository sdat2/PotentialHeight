"""
Process ADCIRC meshes efficiently vectorized/sparses.
This is the shared functionality for processing ADCIRC meshes.

TODO: Add the dual graph calculation for SurgeNet.

fort.63 format:
dimensions:
    time: (Unlimited) (604 currently)
    node: (31435)
    nele: (58368)
    nvertex: (3)
    nope: (1)
    neta: (103)
    max_nvdll: (103)
    nbou: (59)
    nvel: (4514)
    max_nvell: (3050)
    mesh: (1)
coordinates:
    time (time): (604)
    x (node): (31435) longitude, degrees_east
    y (node): (31435) latitude, degrees_north
    element (nele, nvertex): (58368, 3) connectivity
    node (node): (31435)
data_vars:
    depth (node): (31435) depth, meters
    zeta (time, node): (604, 31435) water surface elevation, meters

"""

from typing import Union, Tuple, List
import os
import numpy as np
import collections
from scipy.sparse import csr_matrix, coo_matrix
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from sithom.time import timeit
from sithom.place import BoundingBox
from .constants import NO_BBOX, FORT63_EXAMPLE, DATA_PATH


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

    TODO: This looks slow.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.

    Returns:
        Tuple[List[int], List[int]]: start, end indices for the dual graph.

    Examples::
        >>> start, end = dual_graph_starts_ends_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(start == np.array([0, 1]))
        True
        >>> np.all(end == np.array([1, 0]))
        True

    """
    edge_dict = collections.defaultdict(list)

    for i, triangle in enumerate(triangles):
        edge_dict[tuple(sorted(triangle[0:2]))].append(i)
        edge_dict[tuple(sorted(triangle[1:3]))].append(i)
        edge_dict[tuple(sorted(triangle[[0, 2]]))].append(i)

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
        >>> np.all(calculate_dual_graph_adjacency_matrix(
        ... np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5],
        ...           [3, 5, 6], [3, 6, 7], [7, 3, 1]]) - 1,
        ...          sparse=False
        ... ) == np.array([
        ... [False, True, False, False, False, True],
        ... [True, False, True, False, False, False],
        ... [False, True, False, True, False, False],
        ... [False, False, True, False, True, False],
        ... [False, False, False, True, False, True],
        ... [True, False, False, False, True, False]]))
        True
    """

    M = len(triangles)

    return starts_ends_to_adjacency_matrix(
        *dual_graph_starts_ends_from_triangles(triangles), M, sparse=sparse
    )


def dual_graph_ds_base_from_triangles(triangles: np.ndarray) -> xr.Dataset:
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
        >>> ds = dual_graph_ds_base_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(ds.start.values == np.array([0, 1]))
        True
        >>> np.all(ds.end.values == np.array([1, 0]))
        True
        >>> np.all(ds.element.values - 1 == np.array([[0, 1, 2], [1, 2, 3]]))
        True
    """
    print("triangles", type(triangles))
    print(triangles.shape)
    print(triangles)
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
    Test xr.Dataset for dual graph. Kind of a mocker for tests to remove io operations.

    This is a synthetic fort.63.nc dataset excluding some variables, which we have chosen
    to ignore in the current setup.

    Returns:
        xr.Dataset: Dual graph dataset.
    """
    return xr.Dataset(
        {
            "element": (
                ["nele", "nvertex"],
                np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]) + 1,
                {
                    "description": "Original mesh triangles, with each new node corresponding to the face of the old triangle mesh."
                },
            ),
            "depth": (["node"], np.array([1, 2, 3, 4, 4])),
            "x": (["node"], np.array([0, 1, 2, 3, 3])),  # degrees_east
            "y": (["node"], np.array([0, -1, -1, -2, -1])),  # degrees_north
            # between timesteps sea surface elevation increases by one meter
            "zeta": (["time", "node"], np.array([[1, 2, 3, 4, 4], [2, 3, 4, 5, 5]])),
        },
        coords={
            "nele": (["nele"], np.arange(3)),  # number of elements (triangles)
            "nvertex": (
                ["nvertex"],
                np.arange(3),
            ),  # triangular elements -> 3 vertices per element
            "node": (["node"], np.arange(5)),  # number of nodes (5)
            "time": (
                ["time"],
                np.array(
                    [
                        np.datetime64("2021-01-01 00:00:00"),
                        np.datetime64("2021-01-01 01:00:00"),
                    ]
                ),
            ),
        },
        attrs={"description": "Test original dataset."},
    )


def dual_graph_ds_from_mesh_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    Create a dual graph dataset from an ADCIRC output dataset.

    Args:
        ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" and "depth".

    Returns:
        xr.Dataset: Dual graph dataset.

    Examples::
        >>> ds = dual_graph_ds_from_mesh_ds(_test_xr_dataset())
        >>> np.isclose(ds.depth.values, np.array([2, 3, 4-1/3]), atol=1e-6).all()
        True
        >>> np.isclose(ds.y.values, np.array([-1 +1/3, -1 - 1/3, -1 - 1/3]), atol=1e-6).all()
        True
        >>> np.isclose(ds.x.values, np.array([1, 2, 3 - 1/3]), atol=1e-6).all()
        True
        >>> np.all(ds.element.values - 1 == np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]))
        True
        >>> np.isclose(ds.zeta.values, np.array([[2, 3, 4-1/3], [3, 4, 5-1/3]]), atol=1e-6).all()
        True
        >>> np.isclose(ds.depth_grad.values, np.array([[1, 1, 1], [0, 0, 0]]), atol=1e-6).all()
        True
        >>> np.isclose(ds.zeta_grad.values, np.array([[[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]]), atol=1e-6).all()
        True
    """
    # ds = xr_loader(path)
    # get base object from the triangular elements

    dg = dual_graph_ds_base_from_triangles(ds.element.values - 1)
    # calculate the mean for static node features
    for val in ["x", "y", "depth"]:  # static fields
        dg[val] = (
            ["nele"],
            mean_for_triangle(ds[val].values, ds["element"].values - 1),
        )

    # calculate the gradient of the depth in x and y
    dg["depth_grad"] = (
        ["direction", "nele"],  # this might be the wrong way round
        grad_for_triangle_static(
            ds.x.values, ds.y.values, ds.depth.values, ds.element.values - 1
        ),
    )
    variable_names = ["zeta", "u-vel", "v-vel", "windx", "windy", "pressure"]
    # calculate the gradient of the zeta in x and y
    for variable in variable_names:
        if variable in ds:
            dg[variable] = (
                ["time", "nele"],
                np.mean(ds[variable].values[:, ds["element"].values - 1], axis=2),
            )
            # calculate the gradient for the variable in x and y
            dg[f"{variable}_grad"] = (
                ["direction", "time", "nele"],
                grad_for_triangle_timeseries(
                    ds.x.values, ds.y.values, ds[variable].values, ds.element.values - 1
                ),
            )

    # not yet implemented: work out features for edges.

    return dg.assign_coords({"direction": ["x", "y"]})


@timeit
def dual_graph_ds_from_mesh_ds_from_path(
    path: str = os.path.join(DATA_PATH, "exp_0049")
) -> xr.Dataset:
    """
    Process the dual graph from a path to the fort.*.nc files.

    Args:
        path (str, optional): Defaults to DATA_PATH.

    Raises:
        FileNotFoundError: If the fort.*.nc files do not exist. for * in [63, 64, 73, 74].

    Returns:
        xr.Dataset: Dual graph dataset.
    """
    # load the set of adcirc data
    # and process to the dual graph
    var_from_file_d = {
        "63": ["zeta", "depth", "element"],
        "64": ["u-vel", "v-vel"],
        "73": ["pressure"],
        "74": ["windx", "windy"],
    }
    paths = {
        var: os.path.join(path, f"fort.{var}.nc") for var in var_from_file_d.keys()
    }
    ds_l = []
    for var in var_from_file_d.keys():
        if not os.path.exists(paths[var]):
            raise FileNotFoundError(f"File {paths[var]} does not exist.")
        else:
            ds_l += [xr_loader(paths[var])[var_from_file_d[var]]]
    print(ds_l)
    print("merged_ds", xr.merge(ds_l))

    return dual_graph_ds_from_mesh_ds(xr.merge(ds_l))


# not yet implemented: some of this may be reversible? perhaps with the mesh we should be able to recover all (or almost all) of the original properties.


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


def grad_for_triangle_static(
    x_lon: np.ndarray, y_lat: np.ndarray, z_values: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """
    Calculate the gradient of z-values for each triangle,
    assuming z_values has shape (N,).

    Three points define a plane, so we can calculate the
    gradient of that plane (dz/dx, dz/dy).

    Args:
        x_lon (np.ndarray): (N,) array of x-coordinates.
        y_lat (np.ndarray): (N,) array of y-coordinates.
        z_values (np.ndarray): (N,) array of z-values.
        triangles (np.ndarray): (M, 3) array of triangle indices.

    Returns:
        np.ndarray of shape (2, M):
            [0, :] -> dz/dx for each triangle
            [1, :] -> dz/dy for each triangle

    Examples:
        >>> import numpy as np
        >>> # First test
        >>> np.all(
        ...   np.isclose(
        ...     grad_for_triangle_static(
        ...         np.array([0, 0, 1]),
        ...         np.array([0, 1, 0]),
        ...         np.array([1, 1, 0]),
        ...         np.array([[0, 1, 2]])
        ...     ),
        ...     np.array([[-1], [0]]),
        ...     atol=1e-6
        ...   )
        ... )
        True

        >>> # Second test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 1]),
        ...     np.array([0, 1, 0]),
        ...     np.array([1, 0, 0]),
        ...     np.array([[0, 1, 2]])
        ...   ) == np.array([[-1], [-1]])
        ... )
        True

        >>> # Third test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 2]),
        ...     np.array([0, 2, 0]),
        ...     np.array([1, 0, 0]),
        ...     np.array([[0, 1, 2], [1, 2, 0]])
        ...   ) == np.array([[-0.5, -0.5], [-0.5, -0.5]])
        ... )
        True

        >>> # Fourth test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 0.5]),
        ...     np.array([0, 0.5, 0]),
        ...     np.array([1, 0, 0]),
        ...     np.array([[0, 1, 2]])
        ...   ) == np.array([[-2], [-2]])
        ... )
        True

        >>> # Fifth test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 0.5]),
        ...     np.array([0, 0.5, 0]),
        ...     np.array([2, 0, 0]),
        ...     np.array([[0, 1, 2]])
        ...   ) == np.array([[-4], [-4]])
        ... )
        True

        >>> # Sixth test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 1]),
        ...     np.array([0, 1, 0]),
        ...     np.array([0, 0, 0]),
        ...     np.array([[0, 1, 2]])
        ...   ) == np.array([[0], [0]])
        ... )
        True

        >>> # Seventh test
        >>> np.all(
        ...   grad_for_triangle_static(
        ...     np.array([0, 0, 1]),
        ...     np.array([0, 1, 0]),
        ...     np.array([0, 1, 0]),
        ...     np.array([[0, 1, 2]])
        ...   ) == np.array([[0], [1]])
        ... )
        True
    """
    # M x 3 arrays of node coordinates/values
    x = x_lon[triangles]  # (M, 3)
    y = y_lat[triangles]  # (M, 3)
    z = z_values[triangles]  # (M, 3)

    # Stack into shape (M, 3, 3): [ (xA,yA,zA), (xB,yB,zB), (xC,yC,zC) ]
    c = np.stack((x, y, z), axis=2)
    # normal = (B - A) x (C - A)
    normal = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], axis=1)  # (M, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_x = -normal[:, 0] / normal[:, 2]  # (M,)
        grad_y = -normal[:, 1] / normal[:, 2]  # (M,)

    return np.stack((grad_x, grad_y))  # (2, M)


def grad_for_triangle_timeseries(
    x_lon: np.ndarray, y_lat: np.ndarray, z_values: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """
    Calculate the gradient of z-values for each triangle,
    assuming z_values has shape (T, N).

    Three points define a plane, so we can calculate the
    gradient (dz/dx, dz/dy). We do it for each of the T time steps.

    Args:
        x_lon (np.ndarray): (N,) array of x-coordinates.
        y_lat (np.ndarray): (N,) array of y-coordinates.
        z_values (np.ndarray): (T, N) array of z-values at each of T times.
        triangles (np.ndarray): (M, 3) array of triangle indices.

    Returns:
        np.ndarray of shape (2, T, M):
            [0, t, m] -> dz/dx for triangle m at time t
            [1, t, m] -> dz/dy for triangle m at time t

    Examples:
        >>> import numpy as np
        >>> # Suppose we have T=2 snapshots and N=3 nodes
        >>> zt = np.array([
        ...     [1, 1, 0],  # time=0
        ...     [0, 1, 0]   # time=1
        ... ])
        >>> tri = np.array([[0, 1, 2]])
        >>> result = grad_for_triangle_timeseries(
        ...     np.array([0, 0, 1]),
        ...     np.array([0, 1, 0]),
        ...     zt,
        ...     tri
        ... )
        >>> result.shape
        (2, 2, 1)
        >>> # We can also test that the first time-step's gradient
        >>> # matches grad_for_triangle_static if we pass in the same z-values
        >>> static_grad = grad_for_triangle_static(
        ...     np.array([0, 0, 1]),
        ...     np.array([0, 1, 0]),
        ...     np.array([1, 1, 0]),
        ...     tri
        ... )
        >>> np.allclose(result[:, 0, :], static_grad)  # compare time=0 vs static
        True
    """
    T = z_values.shape[0]  # number of time steps
    # M, 3 for the triangle indices
    x = x_lon[triangles]
    y = y_lat[triangles]

    # z becomes (T, M, 3) after indexing
    z = z_values[:, triangles]  # shape: (T, M, 3)

    # Broadcast x, y to shape (1, M, 3) so that they match (T, M, 3)
    x3 = np.repeat(x[np.newaxis, :, :], T, axis=0)  # shape: (T, M, 3)
    y3 = np.repeat(y[np.newaxis, :, :], T, axis=0)  # shape: (T, M, 3)

    # Combine into shape (T, M, 3, 3)
    c = np.stack((x3, y3, z), axis=-1)  # (T, M, 3, 3)

    # normal = cross((B - A), (C - A)) for each time and triangle
    normal = np.cross(
        c[..., 1, :] - c[..., 0, :], c[..., 2, :] - c[..., 0, :], axis=-1
    )  # (T, M, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_x = -normal[..., 0] / normal[..., 2]  # (T, M)
        grad_y = -normal[..., 1] / normal[..., 2]  # (T, M)

    # shape: (2, T, M)
    return np.stack((grad_x, grad_y), axis=0)


# unwritten: grad for edges


# print(xr_loader("../data/fort.63.nc", use_dask=False))


# unwritten function process a whole fort.63 file to dual graph format, for training a Graph Neural Network.


def edge_features(ds: xr.Dataset, dg_ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the edge features for the dual graph.
    This is not yet implemented.

    Edge features are defined as

    e_{ij} = (n_{ij}, l_{ij})

    where n_{ij} is the outward unit normal vector and l_{ij} is the cell sides' length. Thus, the edge features represent the geometrical properties of the mesh. We excluded the fluxes Fij as additional features as they depend on the hydraulic variables ui and uj, which are already included in the dynamic node features.

    Args:
        ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" and "depth".
        dg_ds (xr.Dataset): Dual graph dataset.
    """
    raise NotImplementedError("Not yet implemented.")

    return dg_ds


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


def filter_dual_graph(dg_ds: xr.Dataset, indices: np.ndarray) -> xr.Dataset:
    raise NotImplementedError("Not yet implemented.")


@timeit
def bbox_mesh(
    file_path: str = os.path.join(DATA_PATH, "fort.63.nc"),  # "../data/fort.63.nc",
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
    # print(calculate_adjacency_matrix(np.array([[0, 1, 2], [1, 2, 3]]), 4, sparse=False))
    # print(calculate_adjacency_matrix(np.array([[0, 1, 2]]), 3, sparse=False))
    # print(calculate_adjacency_matrix(np.array([[0, 1, 2]]), 5, sparse=False))
    # print(calculate_adjacency_matrix(np.array([[]]), 2, sparse=False))
    # dg_ex = dual_graph_ds_from_mesh_ds(xr_loader(FORT63_EXAMPLE))
    # dg_ex.to_netcdf(os.path.join(DATA_PATH, "fort.63.dual.nc"))
    # print(xr_loader(FORT63_EXAMPLE))
    print(dual_graph_ds_from_mesh_ds_from_path())
