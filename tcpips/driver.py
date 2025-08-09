"""A driver to run the whole pipeline for CMIP6 data.


Steps:
1. Download data,
2. Regrid data,
3. Calculate potential intensity,
4. Calculate potential sizes
5. Calculate biases (possibly after regridding on to either ERA5 or the regridded CMIP6 grid).


In stages 1-2 The ouputs are stored in netcdf files, in stages 3-5 the outputs are stored in zarr format.

Maybe it could be good to control this function with a config file, so we can specify which steps to run for which experiments and models.
"""

from .files import get_task_dict, find_atmos_ocean_pairs
from .constants import RAW_PATH, CDO_PATH, PI4_PATH, PS_PATH
from .pi_driver import pi_cmip6_part
from .ps_driver import ps_cmip6_part
from .regrid_cdo import regrid_cmip6_part
from .dask import dask_cluster_wrapper


def loop_through_all() -> None:
    """Run the whole pipeline for CMIP6 data."""
    print(
        "RAW_PATH:",
        RAW_PATH,
        "\nCDO_PATH:",
        CDO_PATH,
        "\nPI4_PATH:",
        PI4_PATH,
        "\nPS_PATH:",
        PS_PATH,
    )
    # 1. download data
    # 2. regridding
    tasks = get_task_dict(original_root=RAW_PATH, new_root=CDO_PATH)
    print("tasks", tasks)
    # 3.
    pairs = find_atmos_ocean_pairs(path=CDO_PATH, new_path=PI4_PATH)
    #
    print("pairs", pairs)


if __name__ == "__main__":
    # python -m tcpips.driver
    # loop_through_all()

    for model in ["HADGEM3-GC31-MM"]:
        for member in ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3", "r4i1p1f3"]:
            for exp in ["ssp585", "historical"]:
                for typ in ["ocean", "atmos"]:
                    try:
                        regrid_cmip6_part(exp=exp, typ=typ, model=model, member=member)
                    except Exception as e:
                        print(f"Error in regridding {exp} {typ} {model} {member}: {e}")
                try:
                    dask_cluster_wrapper(
                        pi_cmip6_part, exp=exp, model=model, member=member
                    )
                except Exception as e:
                    print(
                        f"Error in potential intensity calculation {exp} {model} {member}: {e}"
                    )
