"""
Process ADCIRC meshes efficiently.
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
from sithom.time import timeit
from sithom.plot import BoundingBox
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
def filter_mesh(
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
    f63 = xr_loader(file_path)  # load file as xarray dataset
    xs = f63.x.values  # longitudes (1d numpy array)
    ys = f63.y.values  # latitudes (1d numpy array)
    ys_in = (bbox.lat[0] < ys) & (ys < bbox.lat[1])
    xs_in = (bbox.lon[0] < xs) & (xs < bbox.lon[1])
    both_in = xs_in & ys_in
    indices = np.where(both_in)[0]  # surviving old labels
    new_indices = np.where(indices)[0]  # new labels
    neg_indices = np.where(~both_in)[0]  # indices to get rid of
    elements = f63.element.values - 1  # triangular component mesh
    mask = ~np.isin(elements, neg_indices).any(axis=1)
    filtered_elements = elements[mask]
    mapping = dict(zip(indices, new_indices))
    relabelled_elements = np.vectorize(mapping.get)(filtered_elements)
    f63_n = f63.isel(node=indices)
    del f63_n["element"]  # remove old triangular component mesh
    f63_n["element"] = (
        ["nele", "nvertex"],
        relabelled_elements + 1,
    )  # add new triangular component mesh
    return f63_n


@timeit
def select_edge_indices(
    mesh_ds: xr.Dataset, lon: float, lat: float, number: int = 10, verbose=False
) -> np.ndarray:
    """
    Select edge cells.

    Args:
        mesh_ds (xr.Dataset): ADCIRC output xarray dataset with "x", "y" and "element".
        lon (float): Longitude of central point.
        lat (float): Latitude of central point.
        number (int, optional): How many to choose initially. Defaults to 10.
        verbose (bool, optional): Whether to print. Defaults to False.

    Returns:
        np.ndarray: coastal indices near central point.
    """
    # me = Maxele(os.path.join(KAT_EX_PATH, "maxele.63.nc"))
    lons = mesh_ds.x
    lats = mesh_ds.y

    # compute Euclidean (flat-world) distance in degrees
    distances = (lons - lon) ** 2 + (lats - lat) ** 2

    # Use argpartition to get top K indices
    # Note: The result includes the partition index, so we use -K-1 to get exactly K elements
    indices = np.argpartition(distances, -number)[-number:]

    # Optional: Sort the top K indices if you need them sorted
    indices = indices[np.argsort(distances[indices])[::-1]]

    (uniq, freq) = np.unique(mesh_ds.element, return_counts=True)
    edge_vertices = uniq[freq <= 4]
    if verbose:
        print("Coastals indices", edge_vertices, len(edge_vertices))
        print("Nearby indices", indices, len(indices))

    indices = indices[np.in1d(indices, edge_vertices)]
    return indices


if __name__ == "__main__":
    # python -m
    filter_mesh()
