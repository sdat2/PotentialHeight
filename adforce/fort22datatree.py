from typing import Optional
import os
import xarray as xr
from .constants import DATA_PATH


def _open_datatree(path: str):
    """Open a grouped netCDF file as a datatree, whatever the xarray vintage.

    DataTree was merged into xarray core in 2024.10 (``xr.open_datatree``);
    the standalone ``xarray-datatree`` package is archived. Prefer the core
    implementation when available and only fall back to the archived shim on
    older xarray (e.g. 2024.2).

    Args:
        path (str): Path to a netCDF file with groups (e.g. fort.22.nc).

    Returns:
        DataTree: ``xarray.DataTree`` on xarray >= 2024.10, else
            ``datatree.DataTree`` from the archived package.
    """
    if hasattr(xr, "open_datatree"):
        return xr.open_datatree(path)
    # Lazy import so environments with modern xarray never need (or touch)
    # the archived xarray-datatree package.
    import datatree

    return datatree.open_datatree(path)


def read_fort22(fort22_path: Optional[str] = None) -> xr.Dataset:
    """Read fort.22.nc file.

    Args:
        fort22_path (Optional[str], optional): Path to fort.22.nc file. Defaults to None. If not provided, the default path will be used.

    Returns:
        xr.Dataset: Dataset containing fort.22.nc data.
    """
    if fort22_path is None:
        fort22_path = os.path.join(DATA_PATH, "fort.22.nc")
    return _open_datatree(fort22_path)
