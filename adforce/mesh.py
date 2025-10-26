"""
Process ADCIRC meshes efficiently vectorized/sparses.
This is the shared functionality for processing ADCIRC meshes.

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

fort.74 format: ditto but with u-vel and v-vel instead of zeta

fort.73 format: ditto but "pressure" instead of "zeta"

fort.74 format: ditto but "u-vel" and "v-vel" instead of "zeta"

"""
from typing import Union, Tuple, List, Literal, Optional
import os
import numpy as np
import collections
from scipy.sparse import csr_matrix, coo_matrix
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from sithom.time import timeit
from sithom.place import BoundingBox
from .constants import NO_BBOX, DATA_PATH, GEOD


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
    x: np.ndarray = None,
    y: np.ndarray = None,
    use_pyproj: bool = False,
) -> Union[
    Tuple[List[int], List[int]],
    Tuple[List[int], List[int], List[float], List[float], List[float]],
]:
    """
    Generate start, end indices for the dual graph from the triangles.

    TODO: This looks slow.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices.
        x (np.ndarray, optional): x-coordinates. Defaults to None.
        y (np.ndarray, optional): y-coordinates. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: start, end indices for the dual graph.
        or
        Tuple[List[int], List[int], List[float], List[float], List[float]]: start, end, length, xdual, ydual, unit normal x, unit normal y.

    Examples::
        >>> start, end = dual_graph_starts_ends_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]))
        >>> np.all(start == np.array([0, 1]))
        True
        >>> np.all(end == np.array([1, 0]))
        True
        >>> start, end, length, xd, yd, unx, uny = dual_graph_starts_ends_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]), x=np.array([0, 1, 2, 3]), y=np.array([0, -1, -1, -2]))
        >>> np.all(length == np.array([1.0, 1.0]))
        True
        >>> np.all(unx == np.array([0.0, 0.0]))
        True
        >>> np.all(uny  == np.array([-1.0, 1.0]))
        True

    """

    return_geometry = x is not None and y is not None
    edge_dict = collections.defaultdict(list)

    for i, triangle in enumerate(triangles):
        edge_dict[tuple(sorted(triangle[0:2]))].append(i)
        edge_dict[tuple(sorted(triangle[1:3]))].append(i)
        edge_dict[tuple(sorted(triangle[[0, 2]]))].append(i)

    starts = []
    ends = []

    # could also calculate length of edge and normal vector here if x and y are given.
    if return_geometry:
        len_edges = []
        unit_normal_x = []
        unit_normal_y = []
        # calculate dual graph x and y coordinates
        xd = mean_for_triangle(x, triangles)
        yd = mean_for_triangle(y, triangles)

    for edge in edge_dict:
        nodes = edge_dict[edge]
        if len(nodes) == 2:
            # one way round
            starts.append(nodes[0])
            ends.append(nodes[1])
            # the other way round
            starts.append(nodes[1])
            ends.append(nodes[0])
            if return_geometry:
                if not use_pyproj:
                    # calculate the centre to centre vector for the dual graph edge 0 to 1
                    dg_delta_x = xd[nodes[1]] - xd[nodes[0]]
                    dg_delta_y = yd[nodes[1]] - yd[nodes[0]]
                    # calculate the length of the edge
                    delta_x = x[edge[1]] - x[edge[0]]
                    delta_y = y[edge[1]] - y[edge[0]]
                    len_edges += [np.sqrt((delta_x) ** 2 + (delta_y) ** 2)] * 2
                else:
                    dg_delta_x = GEOD.inv(
                        xd[nodes[0]], yd[nodes[0]], xd[nodes[1]], yd[nodes[0]]
                    )[2]
                    dg_delta_y = GEOD.inv(
                        xd[nodes[0]], yd[nodes[0]], xd[nodes[0]], yd[nodes[1]]
                    )[2]
                    delta_x = GEOD.inv(x[edge[0]], y[edge[0]], x[edge[1]], y[edge[0]])[
                        2
                    ]
                    delta_y = GEOD.inv(x[edge[0]], y[edge[0]], x[edge[0]], y[edge[1]])[
                        2
                    ]
                    len_edges += [
                        GEOD.inv(x[edge[0]], y[edge[0]], x[edge[1]], y[edge[1]])[2]
                    ]

                # unit normal vector
                mx = delta_x / len_edges[-1]
                my = delta_y / len_edges[-1]

                # if dot product is positive then vectors are in the same rough direction
                # delta_dual_graph_center \cdot normal
                # there is some bug here
                # the dual graph 0 to 1 edge is in the same direction as the unit normal
                # so we need to flip the unit normal vector
                dot_prod = dg_delta_x * my - dg_delta_y * mx
                if dot_prod < 0:
                    # the dual graph 0 to 1 edge is in the opposite direction to the unit normal
                    # so we need to flip the unit normal vector
                    mx = -mx
                    my = -my
                elif dot_prod == 0:
                    raise ValueError(
                        "Dot product is zero. Check mesh. This should not happen."
                    )

                # unit normal vector (perpendicular) (edge 0 to 1)
                unit_normal_x += [my]
                unit_normal_y += [-mx]
                # unit normal vector (opposite direction) (edge 1 to 0)
                unit_normal_x += [-my]
                unit_normal_y += [mx]
                # let's hope these happen to be the right way round.
                # this is not guaranteed. The edges have been sorted, so are not necessarily
                # in the same order as the triangles.
        if len(nodes) == 3:
            raise ValueError("An edge is shared by more than 2 triangles. Check mesh.")

    if return_geometry:
        return starts, ends, len_edges, xd, yd, unit_normal_x, unit_normal_y
    else:
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


def dual_graph_ds_base_from_triangles(
    triangles: np.ndarray, x: np.ndarray, y: np.ndarray
) -> xr.Dataset:
    """
    Calculate the dual graph adjacency matrix for a mesh of triangles.

    Nodes are the faces of the original graph, and adjacent if they share an edge.
    There are E shared edges between the faces.

    Args:
        triangles (np.ndarray): Mx3 array of triangle indices. Assumes nodes are numbered from 0 to N-1.
            where N is the number of nodes in the original mesh.
        x (np.ndarray): x-coordinates of the nodes. Degrees east.
        y (np.ndarray): y-coordinates of the nodes. Degrees north.

    Returns:
        xr.Dataset: Dual graph dataset.
         - "element": Mx3 array of triangle indices.
         - "start": 2E array of start indices. (Double counting)
         - "end": 2E array of end indices. (Double counting)
         - "x": M array of x-coordinates of the dual graph nodes.
         - "y": M array of y-coordinates of the dual graph nodes.
         - "length": 2E array of lengths of the edges.
         - "unit_normal_x": 2E array of x-coordinates of the unit normal vectors.
         - "unit_normal_y": 2E array of y-coordinates of the unit normal vectors.

    Examples::
        >>> ds = dual_graph_ds_base_from_triangles(np.array([[0, 1, 2], [1, 2, 3]]), np.array([0, 1, 2, 3]), np.array([0, -1, -1, -2]))
        >>> np.all(ds.start.values == np.array([0, 1]))
        True
        >>> np.all(ds.end.values == np.array([1, 0]))
        True
        >>> np.all(ds.element.values - 1 == np.array([[0, 1, 2], [1, 2, 3]]))
        True
    """

    starts, ends, lengths, xd, yd, unx, uny = dual_graph_starts_ends_from_triangles(
        triangles, x, y
    )
    areas = calculate_triangle_areas(x, y, triangles)

    return xr.Dataset(
        {
            "element": (
                ["nele", "nvertex"],
                triangles + 1,
                {
                    "description": "Original mesh triangles, with each new node corresponding to the face of the old triangle mesh."
                },
            ),
            "x": (
                ["nele"],
                xd,
                {
                    "description": "Longitude of the dual graph nodes.",
                    "units": "degrees_east",
                },
            ),
            "y": (
                ["nele"],
                yd,
                {
                    "description": "Latitude of the dual graph nodes.",
                    "units": "degrees_north",
                },
            ),
            "area": (
                ["nele"],
                areas,
                {"description": "Area of the original triangles.", "units": "degrees^2"},
            ),
            "start": ("edge", starts, {"description": "Start indices of the edges."}),
            "end": ("edge", ends, {"description": "End indices of the edges."}),
            "length": (
                ["edge"],
                lengths,
                {"description": "Length of the edges.", "units": "degrees"},
            ),
            "unit_normal_x": (
                ["edge"],
                unx,
                {"description": "Unit normal vector x-component."},
            ),
            "unit_normal_y": (
                ["edge"],
                uny,
                {"description": "Unit normal vector y-component."},
            ),
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


def dual_graph_ds_from_mesh_ds(ds: xr.Dataset,
                               take_grad: Literal["none", "dynamic", "static", "all"] = "static"
                               ) -> xr.Dataset:
    """
    Create a dual graph dataset from an ADCIRC output dataset.

    Args:
        ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y", "element" and "depth".
        take_grad (take_grad: Literal["none", "dynamic", "static", "all"], optional): Whether to calculate the gradient of the depth, zeta, u-vel, v-vel, windx, windy, pressure. Defaults to "static".

    Returns:
        xr.Dataset: Dual graph dataset.

    Examples::
        >>> ds = dual_graph_ds_from_mesh_ds(_test_xr_dataset(), take_grad="all")
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

    dg = dual_graph_ds_base_from_triangles(
        ds.element.values - 1, ds.x.values, ds.y.values
    )
    # calculate the mean for static node features
    if "depth" in ds:
        for val in ["depth"]:  # static fields # "x", "y",s
            dg[val] = (
                ["nele"],
                mean_for_triangle(ds[val].values, ds["element"].values - 1),
            )
        if take_grad in ["all", "static"]:
            # calculate the gradient of the depth in x and y
            dg["depth_grad"] = (
                ["direction", "nele"],  # this might be the wrong way round
                grad_for_triangle_static(
                    ds.x.values, ds.y.values, ds.depth.values, ds.element.values - 1
                ),
            )

    variable_names = [x for x in ["zeta", "u-vel", "v-vel", "windx", "windy", "pressure"] if x in ds]
    # calculate the gradient of the zeta in x and y
    # ds["time"] = ("time", ds["time"].values)
    # assign time coordinate
    dg = dg.assign_coords({"time": ds.time.values})
    for variable in variable_names:
        if variable in ds:
            dg[variable] = (
                ["time", "nele"],
                np.mean(ds[variable].values[:, ds.element.values - 1], axis=2),
            )
            # calculate the gradient for the variable in x and y
            if take_grad in ["dynamic", "all"]:
                dg[f"{variable}_grad"] = (
                    ["direction", "time", "nele"],
                    grad_for_triangle_timeseries(
                        ds.x.values,
                        ds.y.values,
                        ds[variable].values,
                        ds.element.values - 1,
                    ),
                )

    return dg.assign_coords({"direction": ["x", "y"]})


@timeit
def dual_graph_ds_from_mesh_ds_from_path(
    path: str = os.path.join(DATA_PATH, "exp_0049"),
    bbox: BoundingBox = None,
    use_dask: bool = True,
    take_grad: Literal["none", "dynamic", "static", "all"] = "static",
) -> xr.Dataset:
    """
    Process the dual graph from a path to the fort.*.nc files.

    Args:
        path (str, optional): Defaults to DATA_PATH
        bbox (BoundingBox, optional): Bounding box to filter the data. Defaults to NO_BBOX.
        use_dask (bool, optional): Whether to use dask. Defaults to True.
        take_grad (bool, Literal["none", "dynamic", "static", "all"]): Whether to calculate the gradient of the variables. Defaults to "static".

    Raises:
        FileNotFoundError: If the fort.*.nc files do not exist. for * in [63, 64, 73, 74].

    Returns:
        xr.Dataset: Dual graph dataset.
    """
    # load the set of adcirc data
    # and process to the dual graph
    var_from_file_d = {
        "63": ["zeta", "depth", "element", "x", "y", "time"],
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
            if bbox is not None:
                ds_l += [
                    bbox_mesh(paths[var], bbox, use_dask=use_dask)[var_from_file_d[var]]
                ]
            else:
                ds_l += [xr_loader(paths[var], use_dask=use_dask)[var_from_file_d[var]]]
    # print(ds_l)

    return dual_graph_ds_from_mesh_ds(xr.merge(ds_l), take_grad=take_grad)


def swegnn_dg_from_mesh_ds_from_path(
    path: str = os.path.join(DATA_PATH, "exp_0049"),
    use_dask: bool = True,
) -> xr.Dataset:
    """
    Get the dual graph into the format required for the SWE-GNN model.

    Transforms the output of dual_graph_ds_from_mesh_ds_from_path to match
    the expected variable names and structure for the mSWE-GNN input pipeline.

    Args:
        path (str, optional): Path to ADCIRC fort.*.nc files. Defaults to DATA_PATH/exp_0049.
        use_dask (bool, optional): Whether to use dask for loading. Defaults to True.

    Returns:
        xr.Dataset: Dual graph dataset formatted for mSWE-GNN processing.
    """

    # --- Step 1: Get the initial dual graph dataset ---
    dg = dual_graph_ds_from_mesh_ds_from_path(
        path=path,
        use_dask=use_dask,
        take_grad="static" # Only calculate static depth gradient
    )

    # --- Step 2: Create a new Dataset for SWE-GNN format ---
    swegnn_ds = xr.Dataset()

    # --- Step 3: Rename/Calculate Node (Face) Features ---

    # DEM (Digital Elevation Model / Bathymetry)
    # Assuming 'depth' in dg is positive downwards bathymetry.
    # mSWE-GNN might expect positive elevation, so DEM = -depth.
    # Check mSWE-GNN's convention if unsure. For now, let's assume -depth.
    if 'depth' in dg:
        swegnn_ds['DEM'] = -dg['depth']
        swegnn_ds['DEM'].attrs = {'description': 'Bathymetry/DEM at face centers (m, positive up)', 'units': 'm'}
    else:
         # Add handling if depth is missing - maybe raise error or set to zero
         raise ValueError("Input dataset 'dg' is missing the 'depth' variable.")


    # WD (Water Depth)
    # WD = zeta - DEM = zeta - (-depth) = zeta + depth
    if 'zeta' in dg and 'depth' in dg:
        swegnn_ds['WD'] = dg['zeta'] + dg['depth']
        swegnn_ds['WD'].attrs = {'description': 'Water depth at face centers (time series)', 'units': 'm'}
        # Ensure non-negative water depth
        swegnn_ds['WD'] = swegnn_ds['WD'].where(swegnn_ds['WD'] > 0, 0)
    else:
        # Handle missing zeta or depth for WD calculation
        if 'zeta' not in dg:
             print("Warning: 'zeta' variable not found. Cannot calculate 'WD'.")
        if 'depth' not in dg:
             print("Warning: 'depth' variable not found. Cannot calculate 'WD'.")


    # VX, VY (Velocities) - Rename u-vel, v-vel
    if 'u-vel' in dg:
        swegnn_ds['VX'] = dg['u-vel']
        swegnn_ds['VX'].attrs = {'description': 'X-velocity at face centers (time series)', 'units': 'm/s'}
    if 'v-vel' in dg:
        swegnn_ds['VY'] = dg['v-vel']
        swegnn_ds['VY'].attrs = {'description': 'Y-velocity at face centers (time series)', 'units': 'm/s'}

    # Slopes (from depth_grad)
    if 'depth_grad' in dg:
        # Assuming dg['depth_grad'] has shape [direction(2), nele]
        swegnn_ds['slopex'] = dg['depth_grad'].isel(direction=0)
        # Apply negative sign if DEM = -depth (gradient of DEM = -gradient of depth)
        swegnn_ds['slopex'] = -swegnn_ds['slopex']
        swegnn_ds['slopex'].attrs = {'description': 'Topographic slope in x-direction at face centers', 'units': 'm/degree_east'} # Check units

        swegnn_ds['slopey'] = dg['depth_grad'].isel(direction=1)
        # Apply negative sign
        swegnn_ds['slopey'] = -swegnn_ds['slopey']
        swegnn_ds['slopey'].attrs = {'description': 'Topographic slope in y-direction at face centers', 'units': 'm/degree_north'} # Check units
    else:
        # Handle missing depth_grad if needed (e.g., set slopes to zero)
        print("Warning: 'depth_grad' not found. Slopes ('slopex', 'slopey') will not be included.")


    # Area
    swegnn_ds['area'] = dg["area"]
    swegnn_ds['area'].attrs = {'description': 'Area of each mesh face (triangle)', 'units': 'm^2'}

    # --- Step 4: Rename/Calculate Edge (Dual Edge) Features ---
    # Edge Index (Connectivity of faces)
    # Using 'start' and 'end' from dg. These represent connections between 'nele' indices.
    # mSWE-GNN expects shape [2, num_dual_edges]. Ensure dg.start/end are combined correctly.
    # Note: mSWE-GNN's graph_creation makes edges undirected later.
    edge_index = np.stack([dg['start'].values, dg['end'].values])
    swegnn_ds['edge_index'] = (['two', 'edge'], edge_index)
    swegnn_ds['edge_index'].attrs = {'description': 'Dual graph connectivity (face indices)', 'units': 'index'}

    # Face Distance (Distance between face centers)
    # Calculate from dg.x, dg.y using edge_index
    x_coords = dg['x'].values
    y_coords = dg['y'].values
    dx = x_coords[edge_index[1, :]] - x_coords[edge_index[0, :]]
    dy = y_coords[edge_index[1, :]] - y_coords[edge_index[0, :]]
    face_distance = np.sqrt(dx**2 + dy**2)
    swegnn_ds['face_distance'] = (['edge'], face_distance)
    swegnn_ds['face_distance'].attrs = {'description': 'Distance between centers of connected faces', 'units': 'degrees'} # Check units

    # Relative Face Distance (Vector between face centers)
    face_relative_distance = np.stack([dx, dy], axis=-1) # Shape [num_edges, 2]
    # mSWE-GNN might expect [num_edges, 2]. Let's store it like that.
    swegnn_ds['face_relative_distance'] = (['edge', 'xy'], face_relative_distance)
    swegnn_ds['face_relative_distance'].attrs = {'description': 'Vector (dx, dy) between centers of connected faces', 'units': 'degrees'} # Check units

    # Edge Slope (Slope between connected faces based on DEM)
    dem_values = swegnn_ds['DEM'].values
    dem_diff = dem_values[edge_index[1, :]] - dem_values[edge_index[0, :]]
    # Avoid division by zero if face_distance is zero (shouldn't happen for distinct faces)
    edge_slope = np.divide(dem_diff, face_distance, out=np.zeros_like(dem_diff), where=face_distance!=0)
    swegnn_ds['edge_slope'] = (['edge'], edge_slope)
    swegnn_ds['edge_slope'].attrs = {'description': 'Slope between connected faces based on DEM', 'units': 'm/degree'} # Check units

    # --- Step 5: Add Coordinates and other relevant info ---
    swegnn_ds = swegnn_ds.assign_coords(dg.coords) # Copy coordinates (time, nele, edge, nvertex, direction)
    swegnn_ds = swegnn_ds.drop_vars(['start', 'end', 'direction', 'depth_grad', 'length', 'unit_normal_x', 'unit_normal_y'], errors='ignore') # Drop original/intermediate vars
    swegnn_ds = swegnn_ds.rename({'nele': 'num_nodes'}) # Rename nele dimension to num_nodes (for clarity in PyG context)

    # Add original mesh info if needed for area calc or BCs later
    swegnn_ds['element'] = dg['element'] # Example

    # --- Step 6: Placeholder for Boundary Conditions ---
    # Boundary condition processing (ghost cells) is complex and needs separate implementation.
    # Add placeholders or markers here if needed before the PyG conversion step.
    print("Boundary Condition / Ghost Cell implementation is needed separately.")


    swegnn_ds.attrs['description'] = 'Dual graph formatted for mSWE-GNN input pipeline'

    return swegnn_ds

# not yet implemented: some of this may be reversible? perhaps with the mesh we should be able to recover all (or almost all) of the original properties.


def mean_for_triangle(
    values: np.ndarray,
    triangles: np.ndarray,
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


def calculate_triangle_areas(
    x_coords: np.ndarray, y_coords: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """
    Calculate the area of each triangle using the Shoelace formula.

    Args:
        x_coords (np.ndarray): (N,) array of x-coordinates for all nodes.
        y_coords (np.ndarray): (N,) array of y-coordinates for all nodes.
        triangles (np.ndarray): (M, 3) array of triangle indices (0-based).

    Returns:
        np.ndarray: (M,) array containing the area of each triangle.

    Examples::
        >>> x = np.array([0, 1, 0])
        >>> y = np.array([0, 0, 1])
        >>> tri = np.array([[0, 1, 2]])
        >>> calculate_triangle_areas(x, y, tri)
        array([0.5])
        >>> x = np.array([0, 1, 1, 0])
        >>> y = np.array([0, 0, 1, 1])
        >>> tri = np.array([[0, 1, 2], [0, 2, 3]])
        >>> calculate_triangle_areas(x, y, tri)
        array([0.5, 0.5])
    """
    # Get coordinates for each vertex of each triangle
    # Shape: (M, 3)
    x1 = x_coords[triangles[:, 0]]
    y1 = y_coords[triangles[:, 0]]
    x2 = x_coords[triangles[:, 1]]
    y2 = y_coords[triangles[:, 1]]
    x3 = x_coords[triangles[:, 2]]
    y3 = y_coords[triangles[:, 2]]

    # Apply Shoelace formula
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return areas


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
    # question: if we convert from (lon, lat, z) to local coordinates (x, y, z),
    # would that bet better?
    # i.e. use pyproj to convert to some local coordinate system before taking the
    # gradient, so that the gradient is per metre instead of per degree?
    # in the current setup we implicitly assume that lon at lat are the same size
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


# --- ghost cell functions ---

def get_boundary_edge_midpoints(
    node_xy: np.ndarray,
    edge_index_BC: np.ndarray
) -> np.ndarray:
    """Calculates the midpoint of each boundary edge.

    Args:
        node_xy: Array of node coordinates, shape (num_nodes, 2).
        edge_index_BC: Array of node indices forming boundary edges,
                       shape (num_BC_edges, 2).

    Returns:
        Midpoints of boundary edges, shape (num_BC_edges, 2).

    Doctests:
        >>> node_coords = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        >>> bc_edges = np.array([[0, 1], [1, 2]]) # Edges (0,1) and (1,2)
        >>> get_boundary_edge_midpoints(node_coords, bc_edges)
        array([[1., 0.],
               [2., 1.]])
        >>> bc_edges_empty = np.empty((0, 2), dtype=int)
        >>> get_boundary_edge_midpoints(node_coords, bc_edges_empty)
        array([], shape=(0, 2), dtype=float64)
    """
    if edge_index_BC.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    # Get coordinates of nodes for each edge: shape (num_BC_edges, 2, 2)
    edge_node_coords = node_xy[edge_index_BC]
    # Calculate mean along the axis containing the two nodes: axis=1
    midpoints = np.mean(edge_node_coords, axis=1)
    return midpoints


def calculate_outward_normals(
    node_xy: np.ndarray,
    face_xy: np.ndarray,
    edge_index_BC: np.ndarray,
    face_BC: np.ndarray,
    edge_midpoints: np.ndarray
) -> np.ndarray:
    """Calculates outward-pointing unit normal vectors for boundary edges.

    Args:
        node_xy: Array of node coordinates, shape (num_nodes, 2).
        face_xy: Array of face center coordinates, shape (num_faces, 2).
        edge_index_BC: Indices of nodes forming BC edges, shape (num_BC_edges, 2).
        face_BC: Indices of boundary faces corresponding to edge_index_BC,
                   shape (num_BC_edges,).
        edge_midpoints: Midpoints of boundary edges, shape (num_BC_edges, 2).

    Returns:
        Outward unit normal vectors, shape (num_BC_edges, 2).

    Doctests:
        >>> node_coords = np.array([[0., 0.], [2., 0.], [1., 1.]]) # Triangle 0
        >>> face_coords = np.array([[1., 1./3]]) # Center of triangle 0
        >>> bc_edges = np.array([[0, 1]]) # Edge (0,1) is boundary
        >>> bc_faces = np.array([0]) # Face 0 is the boundary face
        >>> midpts = get_boundary_edge_midpoints(node_coords, bc_edges)
        >>> calculate_outward_normals(node_coords, face_coords, bc_edges, bc_faces, midpts)
        array([[ 0., -1.]])

        >>> node_coords = np.array([[0., 0.], [0., 2.], [1., 1.]]) # Triangle 0
        >>> face_coords = np.array([[1./3, 1.]]) # Center of triangle 0
        >>> bc_edges = np.array([[0, 1]]) # Edge (0,1) is boundary
        >>> bc_faces = np.array([0]) # Face 0 is the boundary face
        >>> midpts = get_boundary_edge_midpoints(node_coords, bc_edges)
        >>> normals = calculate_outward_normals(node_coords, face_coords, bc_edges, bc_faces, midpts)
        >>> np.allclose(normals, np.array([[-1., 0.]]))
        True
    """
    if edge_index_BC.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    # Calculate edge vectors (Node1 -> Node2)
    edge_vec = node_xy[edge_index_BC[:, 1]] - node_xy[edge_index_BC[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1, keepdims=True)
    # Handle potential zero-length edges
    edge_len = np.where(edge_len == 0, 1.0, edge_len)

    # Calculate one possible normal vector (rotate 90 deg: (x,y) -> (-y,x))
    # Note: Choice of (-y, x) vs (y, -x) determines initial direction.
    normal_vec = np.stack([-edge_vec[:, 1], edge_vec[:, 0]], axis=-1) / edge_len

    # Orient normals outwards relative to face centers
    # Vector from edge midpoint to face center
    vec_mid_to_face = face_xy[face_BC] - edge_midpoints
    # Dot product: If positive, normal and mid_to_face point in similar directions (normal is inward)
    dot_prod = np.sum(normal_vec * vec_mid_to_face, axis=1, keepdims=True)
    # Flip normal vector if dot product is positive (currently pointing inward)
    outward_normal_vec = normal_vec * np.where(dot_prod >= 0, -1.0, 1.0)

    # Ensure unit length after potential flipping (shouldn't change length)
    # outward_normal_vec /= np.linalg.norm(outward_normal_vec, axis=1, keepdims=True)

    return outward_normal_vec


def calculate_ghost_face_coords(
    face_xy: np.ndarray,
    face_BC: np.ndarray,
    edge_midpoints: np.ndarray,
    outward_normals: np.ndarray
) -> np.ndarray:
    """Calculates coordinates for ghost faces by mirroring across boundary edges.

    Args:
        face_xy: Array of face center coordinates, shape (num_faces, 2).
        face_BC: Indices of boundary faces, shape (num_BC_edges,).
        edge_midpoints: Midpoints of boundary edges corresponding to face_BC,
                        shape (num_BC_edges, 2).
        outward_normals: Outward unit normal vectors for boundary edges,
                         shape (num_BC_edges, 2).

    Returns:
        Coordinates of the ghost faces, shape (num_BC_edges, 2).

    Doctests:
        >>> face_coords = np.array([[1., 0.5], [3., 1.5]]) # Two face centers
        >>> bc_faces = np.array([0, 1]) # Both are boundary faces
        >>> midpts = np.array([[1., 0.], [3., 2.]]) # Edge midpoints
        >>> normals = np.array([[0., -1.], [0., 1.]]) # Outward normals
        >>> calculate_ghost_face_coords(face_coords, bc_faces, midpts, normals)
        array([[ 1. , -0.5],
               [ 3. ,  2.5]])

        >>> empty_faces = np.empty((0, 2))
        >>> empty_idx = np.empty(0, dtype=int)
        >>> empty_midpts = np.empty((0, 2))
        >>> empty_normals = np.empty((0, 2))
        >>> calculate_ghost_face_coords(empty_faces, empty_idx, empty_midpts, empty_normals)
        array([], shape=(0, 2), dtype=float64)
    """
    if face_BC.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    face_BC_xy = face_xy[face_BC]

    # Distance from boundary face center to edge midpoint (symmetry point)
    distance_face_edge_BC = np.linalg.norm(
        (face_BC_xy - edge_midpoints), axis=1, keepdims=True
    )

    # Calculate Ghost Coordinates: SymmetryPoint + Normal * Distance
    ghost_face_xy = edge_midpoints + outward_normals * distance_face_edge_BC

    return ghost_face_xy


def find_boundary_faces(
    triangles: np.ndarray,
    edge_index_BC: np.ndarray
) -> np.ndarray:
    """
    Finds the indices of faces (triangles) that contain boundary edges using NumPy.

    Compares the edges of each triangle against a provided list of
    boundary edges efficiently.

    Args:
        triangles: Array of triangle definitions, shape (num_faces, 3),
                   containing 0-based node indices for each face.
        edge_index_BC: Array of boundary edges, shape (num_BC_edges, 2),
                       containing pairs of 0-based node indices.

    Returns:
        np.ndarray: A sorted array of unique indices of the faces that
                    contain at least one boundary edge.

    Doctests:
        >>> # Triangle 0: nodes [0, 1, 2], edges (0,1), (1,2), (2,0)
        >>> # Triangle 1: nodes [1, 3, 2], edges (1,3), (3,2), (2,1)
        >>> triangles_data = np.array([[0, 1, 2], [1, 3, 2]])
        >>> # Boundary edge is (0, 1)
        >>> bc_edges_data = np.array([[0, 1]])
        >>> find_boundary_faces(triangles_data, bc_edges_data)
        array([0])
        >>> # Boundary edge is (1, 2) - shared by both triangles
        >>> bc_edges_data = np.array([[1, 2]])
        >>> find_boundary_faces(triangles_data, bc_edges_data)
        array([0, 1])
        >>> # Boundary edge is (2, 1) - reverse direction, shared by both
        >>> bc_edges_data = np.array([[2, 1]])
        >>> find_boundary_faces(triangles_data, bc_edges_data)
        array([0, 1])
        >>> # Boundary edge (3, 2) - only in triangle 1
        >>> bc_edges_data = np.array([[3, 2]])
        >>> find_boundary_faces(triangles_data, bc_edges_data)
        array([1])
        >>> # No matching boundary edges
        >>> bc_edges_data = np.array([[0, 3]])
        >>> find_boundary_faces(triangles_data, bc_edges_data)
        array([], dtype=int64)
        >>> # Empty input edges
        >>> bc_edges_empty = np.empty((0, 2), dtype=int)
        >>> find_boundary_faces(triangles_data, bc_edges_empty)
        array([], dtype=int64)
    """
    if edge_index_BC.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if triangles.shape[0] == 0:
         return np.array([], dtype=np.int64)

    num_faces = triangles.shape[0]

    # 1. Prepare boundary edges: Sort node indices within each edge
    # Shape: (num_BC_edges, 2)
    sorted_bc_edges = np.sort(edge_index_BC, axis=1)

    # 2. Generate all edges for all triangles
    # Edge 1: node 0 -> node 1
    # Edge 2: node 1 -> node 2
    # Edge 3: node 2 -> node 0
    # Shape: (num_faces, 3, 2) -> (num_faces * 3, 2)
    all_triangle_edges = np.stack([
        np.stack([triangles[:, 0], triangles[:, 1]], axis=-1),
        np.stack([triangles[:, 1], triangles[:, 2]], axis=-1),
        np.stack([triangles[:, 2], triangles[:, 0]], axis=-1)
    ], axis=1).reshape(-1, 2)

    # 3. Sort node indices within each triangle edge
    # Shape: (num_faces * 3, 2)
    sorted_triangle_edges = np.sort(all_triangle_edges, axis=1)

    # 4. Efficiently check for matches using broadcasting and comparison
    # Check if each sorted_triangle_edge exists in sorted_bc_edges
    # This creates a boolean array: matches[i, j] is True if triangle_edge[i] == bc_edge[j]
    # Shape: (num_faces * 3, num_BC_edges)
    matches = (sorted_triangle_edges[:, None, :] == sorted_bc_edges).all(axis=2)

    # 5. Find which triangle edges matched *any* boundary edge
    # Shape: (num_faces * 3,) -> boolean mask for edges that are boundary edges
    is_boundary_triangle_edge = np.any(matches, axis=1)

    # 6. Identify the faces (triangles) associated with these matching edges
    # Reshape the boolean mask back to align with faces
    # Shape: (num_faces, 3) -> True if the corresponding edge was a boundary edge
    face_edge_match_mask = is_boundary_triangle_edge.reshape(num_faces, 3)

    # Find the indices of faces where *at least one* edge matched
    # Shape: (num_matching_faces,)
    boundary_face_indices = np.where(np.any(face_edge_match_mask, axis=1))[0]

    return boundary_face_indices.astype(np.int64)



def elevation_boundary_edges(nbdv: np.ndarray,
                             nvdll: np.ndarray,
                             ibtypee: np.ndarray,
                             boundary_type: Optional[int] = 0) -> np.ndarray:
    """
    Extract elevation boundary edges from ADCIRC mesh data.

    Args:
        nbdv (np.ndarray): 1D array of node indices for elevation boundaries (0-based).
        nvdll (np.ndarray): 1D array of number of nodes in each elevation boundary segment.
        ibtypee (np.ndarray): 1D array of types for each elevation boundary segment.
        boundary_type (Optional[int]): Specific elevation boundary type to extract.
            If None, extracts edges for all types. Defaults to 0.
    Returns:
        np.ndarray: Array of boundary edges, shape (num_BC_edges, 2),
                    containing pairs of 0-based node indices. Returns an
                    empty array if no relevant boundaries are found.

    Doctests:
        >>> nbdv = np.array([0, 1, 3, 4])
        >>> nvdll = np.array([4])
        >>> ibtypee = np.array([0])
        >>> elevation_boundary_edges(nbdv, nvdll, ibtypee, boundary_type=0)
        array([[0, 1],
                [1, 3],
                [3, 4]])
        >>> elevation_boundary_edges(nbdv, nvdll, ibtypee, boundary_type=1)
        array([], shape=(0, 2), dtype=int64)
    """

    all_boundary_edges = []
    current_node_index = 0
    # Iterate through each defined elevation boundary segment
    for i, num_nodes_in_segment in enumerate(nvdll):
        segment_type = ibtypee[i]

        # Skip if the type doesn't match the requested type
        if boundary_type is not None and segment_type != boundary_type:
            current_node_index += num_nodes_in_segment
            continue
        # Extract node indices for the current segment
        segment_nodes = nbdv[current_node_index : current_node_index + num_nodes_in_segment]

        # Create edges by pairing consecutive nodes within the segment
        # An open boundary typically doesn't loop back, so we stop at len-1
        segment_edges = np.stack(
           [segment_nodes[:-1], segment_nodes[1:]],
           axis=-1
        )
        all_boundary_edges.append(segment_edges)

        # Move the index pointer for the flat nbdv array
        current_node_index += num_nodes_in_segment

    if not all_boundary_edges:
        return np.empty((0, 2), dtype=np.int64) # Use int64 for consistency

    return np.vstack(all_boundary_edges)



# -- coast functions ---
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
    # ds = dual_graph_ds_from_mesh_ds_from_path(take_grad="static")
    # print(ds)
    # print([v for v in ds])
    # Assuming dual_graph_ds_from_mesh_ds_from_path is defined above or imported
    # Make sure DATA_PATH points to the correct base directory containing exp_0049
    # from adforce.mesh import dual_graph_ds_from_mesh_ds_from_path

    print("Processing ADCIRC data into SWE-GNN format...")
    swegnn_formatted_ds = swegnn_dg_from_mesh_ds_from_path(
        path=os.path.join(DATA_PATH, "exp_0049"), # Adjust path as needed
        use_dask=False # Using use_dask=False might be simpler initially for debugging
    )
    print("\n--- SWE-GNN Formatted Dataset ---")
    print(swegnn_formatted_ds)
    print("\nVariables:", list(swegnn_formatted_ds.data_vars))

    try:
        from .constants import DATA_PATH # Adjust import if needed
        from .boundaries import extract_elevation_boundary_edges

        file_path = os.path.join(DATA_PATH, "exp_0049", "maxele.63.nc")

        with nc.Dataset(file_path, 'r') as ds:
            # Load 0-based triangle node indices
            triangles_from_file = ds.variables['element'][:] - 1

        # 2. Get the boundary edges using the previously defined function
        ocean_edges = extract_elevation_boundary_edges(file_path, boundary_type=0)

        # 3. Find the boundary faces using the NumPy version
        if triangles_from_file is not None and ocean_edges.shape[0] > 0:
            print(f"Finding boundary faces for {ocean_edges.shape[0]} boundary edges (NumPy)...")
            face_BC_np = find_boundary_faces(triangles_from_file, ocean_edges)

            if face_BC_np.shape[0] > 0:
                print(f"Found {face_BC_np.shape[0]} boundary faces.")
                print("First 10 face indices:", face_BC_np[:10])
                print("Last 10 face indices:", face_BC_np[-10:])
                # Verification
                if face_BC_np.shape[0] == ocean_edges.shape[0]:
                    print("Number of boundary faces matches number of boundary edges.")
                else:
                     print(f"Warning: Number of faces ({face_BC_np.shape[0]}) doesn't match edges ({ocean_edges.shape[0]}).")

            else:
                print("No boundary faces found corresponding to the boundary edges.")
        else:
             if triangles_from_file is None: print("Could not load triangle data.")
             if ocean_edges.shape[0] == 0: print("No boundary edges were provided.")


    except ImportError:
        print("Could not import DATA_PATH or other functions.")
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
