"""Read output files."""
import netCDF4 as nc
import xarray as xr


def xr_loader(file_name: str, verbose: bool = False) -> xr.Dataset:
    """Load an xarray dataset from a ADCIRC netCDF4 file."""
    ds_nc = nc.Dataset(file_name)
    for var in ["neta", "nvel"]:
        if var in ds_nc.variables:
            if verbose:
                print("removing", ds_nc.variables[var])
            del ds_nc.variables[var]
    return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc))
