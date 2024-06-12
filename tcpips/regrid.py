"""Regrid data.

Add a lock file to highlight that the regridding is in progress, to allow more flexible parallel processing.

CMIP6/{STAGE}/{EXP}/{TYPE}/{MODEL}/{MEMBER}.nc
locks are created in CMIP6/{STAGE}/{exp}.{typ}.{model}.{member}.lock

"""

import os
import xarray as xr
import argparse
import subprocess
import dask
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar
import xesmf as xe
from matplotlib import pyplot as plt
from sithom.misc import in_notebook
from sithom.plot import plot_defaults  # feature_grid, label_subplots
from sithom.time import timeit, time_stamp
from tcpips.constants import (
    FIGURE_PATH,
    CMIP6_PATH,
    RAW_PATH,
    REGRIDDED_PATH,
    CONVERSION_NAMES,
)
from tcpips.files import locker


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def run_tasks(
    force_regrid: bool = False,
    output_res: float = 0.5,
    worker: int = 10,
    memory_per_worker: str = "20GiB",
    parallel=False,
) -> None:
    """Run all tasks sequentially.

    Args:
        force_regrid (bool, optional): Force regrid. Defaults to False.
        output_res (float, optional): Resolution of the output grid. Defaults to 0.5.
        worker (int, optional): Number of workers. Defaults to 10.
        memory_per_worker (str, optional): Memory per worker. Defaults to "20GiB".
        parallel (bool, optional): Run in parallel. Defaults to False.

    """
    tasks = define_tasks()
    print("tasks", tasks)

    if parallel:
        print("about to create cluster", time_stamp())

        cluster = LocalCluster(n_workers=worker, memory_limit=memory_per_worker)
        client = Client(cluster)
        print(client)
        dask.distributed.print(client)

    for key in tasks:
        if not tasks[key]["locked"]:
            if not tasks[key]["regridded_exists"] or force_regrid:
                regrid_cmip6_part(output_res, **tasks[key])
            else:
                print(f"Already regridded {key}, not regridding.")

    dask.distributed.print("finished", time_stamp())


@timeit
def define_tasks() -> dict:
    """Find all tasks to be done."""
    tasks = {}
    for exp in os.listdir(RAW_PATH):
        for typ in [
            x
            for x in os.listdir(os.path.join(RAW_PATH, exp))
            if os.path.isdir(os.path.join(RAW_PATH, exp))
        ]:
            for model in os.listdir(os.path.join(RAW_PATH, exp, typ)):
                for member in os.listdir(os.path.join(RAW_PATH, exp, typ, model)):
                    member = member.replace(".nc", "")
                    key = f"{exp}.{typ}.{model}.{member}"
                    tasks[key] = {
                        "exp": exp,
                        "typ": typ,
                        "model": model,
                        "member": member,
                        "regridded_exists": os.path.exists(
                            os.path.join(REGRIDDED_PATH, exp, typ, model, member)
                            + ".nc"
                        ),
                        "locked": os.path.exists(
                            os.path.join(REGRIDDED_PATH, key) + ".lock"
                        ),
                    }
    return tasks


@timeit
@locker(REGRIDDED_PATH)
def regrid_cmip6_part(
    output_res: float = 0.5,
    time_chunk: int = 1,
    exp: str = "ssp585",
    typ: str = "ocean",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
    **kwargs,
) -> None:
    """
    Regrid 2d data to a certain resolution using xesmf.

    Args:
        output_res (float, optional): Resolution of the output grid. Defaults to 1.0.
        time_chunk (int, optional): Chunk size for time. Defaults to 1.
        exp (str, optional): Experiment name. Defaults to "ssp585".
        typ (str, optional): Type of data. Defaults to "ocean". Can be "ocean" or "atmos".
        model (str, optional): Model name. Defaults to "CESM2".
        member (str, optional): Member name. Defaults to "r4i1p1f1".

    """
    plot_defaults()

    @timeit
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
        return ds.chunk(chunks={"time": time_chunk})

    in_ds = open_ds(
        os.path.join(RAW_PATH, exp, typ, model, member) + ".nc"
    )  # .isel(time=slice(0, 10))
    print("in_ds", in_ds)
    # atmos_ds = open_ds(os.path.join(RAW_PATH, "ssp585", "atmos", "CESM2", 'r4i1p1f1.nc'))

    new_coords = xe.util.grid_global(
        output_res, output_res
    )  # make a regular lat/lon grid

    print("new_coords", new_coords)

    @timeit
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
            input_ds,
            new_coords,
            "bilinear",
            periodic=True,
            ignore_degenerate=True,
        )
        print(regridder)
        out_ds = regridder(
            input_ds,
            keep_attrs=True,
            skipna=True,
            output_chunks={"time": time_chunk, "lat": 90, "lon": 90},
            # ignore_degenerate=True,
        ).chunk(chunks={"time": time_chunk})
        out_ds.attrs["regrid_info"] = (
            regridder.__str__() + "git " + get_git_revision_hash() + " " + time_stamp()
        )
        print("out_ds", out_ds)
        delayed_obj = out_ds.to_netcdf(
            os.path.join(CMIP6_PATH, output_name),
            format="NETCDF4",
            engine="h5netcdf",  # should be better at parallel writing/dask
            encoding={
                var: {"dtype": "float32"}  # , "zlib": True, "complevel": 6}
                for var in CONVERSION_NAMES.keys()
                if var in out_ds
            },
            compute=False,
        )
        with ProgressBar():
            _ = delayed_obj.compute()
        return out_ds  # return for later plotting.

    folder = os.path.join(REGRIDDED_PATH, exp, typ, model)
    os.makedirs(folder, exist_ok=True)
    out_ds = regrid_and_save(in_ds, os.path.join(folder, member) + ".nc")
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
    # regrid_2d_1degree()
    # regrid_2d()
    # regrid_1d()
    # regrid_1d(xesmf=True)
    # define_tasks()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--worker", type=int, default=10)
    parser.add_argument("-m", "--memory", type=str, default="4GiB")
    parser.add_argument(
        "-f", "--force", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("-r", "--resolution", type=float, default=0.5)
    parser.add_argument(
        "-p", "--parallel", type=lambda x: (str(x).lower() == "true"), default=True
    )

    args = parser.parse_args()
    print("args", args)
    run_tasks(
        worker=args.worker,
        force_regrid=args.force,
        output_res=args.resolution,
        memory_per_worker=args.memory,
        parallel=args.parallel,
    )
