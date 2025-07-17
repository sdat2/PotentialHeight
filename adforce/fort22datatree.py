from typing import Optional
import os
import xarray as xr
import datatree as dt
from .constants import DATA_PATH


def read_fort22(fort22_path: Optional[str] = None) -> xr.Dataset:
    """Read fort.22.nc file.

    Args:
        fort22_path (Optional[str], optional): Path to fort.22.nc file. Defaults to None. If not provided, the default path will be used.

    Returns:
        xr.Dataset: Dataset containing fort.22.nc data.
    """
    if fort22_path is None:
        fort22_path = os.path.join(DATA_PATH, "fort.22.nc")
    return dt.open_datatree(fort22_path)
