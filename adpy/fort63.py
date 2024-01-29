"""Read output files."""
import netCDF4 as nc
import xarray as xr


def xr_loader(file_name: str) -> xr.Dataset:
    ds_nc = nc.Dataset(file_name)
    for var in ["neta", "nvel"]:
        if var in ds_nc.variables:
            print("removing", ds_nc.variables[var])
            del ds_nc.variables[var]
    return xr.open_dataset(xr.backends.NetCDF4DataStore(ds_nc))
