"""A driver to run the whole pipeline for CMIP6 data.


Steps:
1. Download data,
2. Regrid data,
3. Calculate potential intensity,
4. Calculate potential sizes
5. Calculate biases (possibly after regridding on to either ERA5 or the regridded CMIP6 grid).


In stages 1-2 The ouputs are stored in netcdf files, in stages 3-5 the outputs are stored in zarr format.
"""
from .files import get_task_dict, find_atmos_ocean_pairs
from .constants import RAW_PATH, CDO_PATH, PI4_PATH, PS_PATH
from .pi_driver import pi_cmip6_part
from .ps_driver import ps_cmip6_part
from .regrid_cdo import regrid_cmip6_part


def loop_through_all() -> None:
    """Run the whole pipeline for CMIP6 data."""
    print("RAW_PATH:", RAW_PATH, "\nCDO_PATH:", CDO_PATH, "\nPI4_PATH:", PI4_PATH, "\nPS_PATH:", PS_PATH)
    # 1. download data
    # 2. regridding
    tasks = get_task_dict(original_root=RAW_PATH, new_root=CDO_PATH)
    print("tasks", tasks)
    # 3.
    pairs = find_atmos_ocean_pairs(
        path=CDO_PATH,
        new_path=PI4_PATH)
    #
    print("pairs", pairs)


if __name__ == "__main__":
    # python -m tcpips.driver
    loop_through_all()
