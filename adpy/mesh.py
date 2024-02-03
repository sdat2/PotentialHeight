"""
Process ADCIRC meshes efficiently.
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
from sithom.time import timeit
from sithom.plot import BoundingBox
from src.constants import NO_BBOX


def xr_loader(file_name: str, verbose: bool = False) -> xr.Dataset:
    """Load an xarray dataset from a ADCIRC netCDF4 file."""
    ds_nc = nc.Dataset(file_name)
    for var in ["neta", "nvel"]:  # known problem variables
        if var in ds_nc.variables:
            if verbose:
                print("removing", ds_nc.variables[var])
            del ds_nc.variables[var]
    return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc))


@timeit
def filter_mesh(file_path: str = "../data/fort.63.nc", bbox: BoundingBox = NO_BBOX) -> xr.Dataset:
    """
    Load an adcirc output file and filter it to a bounding box.

    Args:
        file_path (str, optional): Path to ADCIRC file. Defaults to "../data/fort.63.nc".
        bbox (BoundingBox, optional): Bounding box to trim to. Defaults to NO_BBOX.

    Returns:
        xr.Dataset: Filtered xarray dataset.
    """
    f63 = xr_loader(file_path)  # load file as xarray dataset
    xs = f63.x.values  # longitudes
    ys = f63.y.values  # latitudes
    ys_in = (bbox.lat[0] < ys) & (ys < bbox.lat[1])
    xs_in = (bbox.lon[0] < xs) & (xs < bbox.lon[1])
    both_in = xs_in & ys_in
    indices = np.where(both_in)[0]      # surviving old labels
    new_indices = np.where(indices)[0]  # new labels
    neg_indices = np.where(~both_in)[0] # indices to get rid of
    elements = f63.element.values - 1   # triangular component mesh
    mask = ~np.isin(elements, neg_indices).any(axis=1)
    filtered_elements = elements[mask]
    mapping = dict(zip(indices, new_indices))
    relabelled_elements = np.vectorize(mapping.get)(filtered_elements)
    f63_n = f63.isel(node=indices)
    del f63_n["element"]  # remove old triangular component mesh
    f63_n["element"] = (["nele", "nvertex"], relabelled_elements + 1) # add new triangular component mesh
    return f63_n
