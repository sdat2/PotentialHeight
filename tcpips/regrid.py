"""Regrid data.

Add a lock file to highlight that the regridding is in progress, to allow more flexible parallel processing.

CMIP6/{STAGE}/{EXP}{TYPE}/{MODEL}/{MEMBER}.nc

"""

import os
import xarray as xr
from dask.diagnostics import ProgressBar
import xesmf as xe
from matplotlib import pyplot as plt
from sithom.misc import in_notebook
from sithom.plot import feature_grid, plot_defaults, label_subplots
from sithom.time import timeit
from tcpips.constants import FIGURE_PATH, CMIP6_PATH, RAW_PATH, REGRIDDED_PATH, CONVERSION_NAMES


def run_tasks():
    """Run all tasks sequentially."""
    tasks = define_tasks()
    print("tasks", tasks)
    for key in tasks:
        if not tasks[key]["regridded_exists"] and not tasks[key]["locked"]:
            regrid_any(**tasks[key])


@timeit
def define_tasks() -> dict:
    """Find all tasks to be done."""
    tasks = {}
    for exp in os.listdir(RAW_PATH):
        for typ in [x for x in os.listdir(os.path.join(RAW_PATH, exp)) if os.path.isdir(os.path.join(RAW_PATH, exp))]:
            for model in os.listdir(os.path.join(RAW_PATH, exp, typ)):
                for member in os.listdir(os.path.join(RAW_PATH, exp, typ, model)):
                    key = f"{exp}.{typ}.{model}.{member}"
                    tasks[key] = {
                        "exp": exp,
                        "typ": typ,
                        "model": model,
                        "member": member,
                        "regridded_exists": os.path.exists(os.path.join(REGRIDDED_PATH, exp, typ, model, member) + ".nc"),
                        "locked": os.path.exists(os.path.join(REGRIDDED_PATH, key) + ".lock"),
                    }
    return tasks


@timeit
def regrid_any(output_res: float = 1.0, time_chunk: int = 10, exp: str="ssp585", typ: str="ocean", model: str ="CESM2", member: str="r4i1p1f1", **kwargs) -> None:
    """
    Regrid 2d data to a certain resolution using xesmf.

    Args:
        output_res (float, optional): Resolution of the output grid. Defaults to 1.0.
        time_chunk (int, optional): Chunk size for time. Defaults to 10.
    """

    if os.path.exists(os.path.join(REGRIDDED_PATH, f"{exp}.{typ}.{model}.{member}.lock")):
        print(f"Already regridding {exp}.{typ}.{model}.{member}")
        return # already regridding this file.
    else:
        with open(os.path.join(REGRIDDED_PATH, f"{exp}.{typ}.{model}.{member}.lock"), "w") as f:
            f.write("")
    print(f"Regridding {exp}.{typ}.{model}.{member}")
    plot_defaults()


    def open_ds(path: str) -> xr.Dataset:
        """
        Open dataset.

        Args:
            path (str): path to the dataset.

        Returns:
            xr.Dataset: xarray dataset.
        """
        nonlocal time_chunk
        # open netcdf4 file using dask backend
        ds = xr.open_dataset(path, chunks={"time": time_chunk})
        ds = ds.drop_vars(
            [
                x
                for x in [
                    "x",
                    "y",
                    "dcpp_init_year",
                    "member_id",
                ]
                if x in ds
            ]
        )
        return ds

    in_ds = open_ds(os.path.join(RAW_PATH, exp, typ, model, member)+".nc").isel(time=slice(0, 10))
    # atmos_ds = open_ds(os.path.join(RAW_PATH, "ssp585", "atmos", "CESM2", 'r4i1p1f1.nc'))

    new_coords = xe.util.grid_global(
        output_res, output_res
    )  # make regular lat/lon grid

    def regrid_and_save(input_ds: xr.Dataset, output_name: str) -> xr.Dataset:
        """
        Regrid and save the input dataset to the output.

        Args:
            input_ds (xr.Dataset): dataset to regrid.
            output_name (str): of the output file.

        Returns:
            xr.Dataset: regridded dataset.
        """
        regridder = xe.Regridder(
            input_ds, new_coords, "bilinear", periodic=True, ignore_degenerate=True
        )
        print(regridder)
        out_ds = regridder(
            input_ds,
            keep_attrs=True,
            skipna=True,
            # ignore_degenerate=True,
        )
        delayed_obj = out_ds.to_netcdf(
            os.path.join(CMIP6_PATH, output_name),
            format="NETCDF4",
            engine="h5netcdf",  # should be better at parallel writing/dask
            chunks={"time": time_chunk},
            encoding={
                var: {"dtyp": "float32", "zlib": True, "complevel": 6}
                for var in CONVERSION_NAMES.keys()
                if var in out_ds
            },
            compute=False,
        )
        with ProgressBar():
            results = delayed_obj.compute()
        return out_ds  # return for later plotting.

    folder = os.path.join(REGRIDDED_PATH, exp, typ, model)
    os.makedirs(folder, exist_ok=True)
    out_ds = regrid_and_save(
        in_ds, os.path.join(folder, member)+ ".nc"
    )
    print("out_ds", out_ds)
    if typ == "ocean" and in_notebook():
        out_ds.tos.isel(time=0).plot(x="lon", y="lat")
        plt.show()
        out_ds.tos.isel(time=0).plot()
        plt.show()
    elif typ == "atmos" and in_notebook():
        out_ds.tas.isel(time=0, p=0).plot(x="lon", y="lat")
        plt.show()




if __name__ == "__main__":
    # python -m tcpips.regrid
    #regrid_2d_1degree()
    #regrid_2d()
    #regrid_1d()
    # regrid_1d(xesmf=True)
    # define_tasks()
    run_tasks()
