"""Regrid data.

Add a lock file to highlight that the regridding is in progress, to allow more flexible parallel processing.

CMIP6/{STAGE}/{EXP}/{TYPE}/{MODEL}/{MEMBER}.nc
locks are created in CMIP6/{STAGE}/{exp}.{typ}.{model}.{member}.lock

"""

import math
import shutil
import os
from typing import Optional
import xarray as xr
import argparse
import dask
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar
import xesmf as xe
from matplotlib import pyplot as plt
from sithom.misc import in_notebook, get_git_revision_hash
from sithom.plot import plot_defaults
from sithom.time import timeit, time_stamp
from sithom.io import write_json
from .constants import (
    DATA_PATH,
    CMIP6_PATH,
    RAW_PATH,
    REGRIDDED_PATH,
    CONVERSION_NAMES,
)
from .files import locker, get_task_dict


def run_regridding_sequentially(
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
    tasks = get_task_dict()
    write_json(tasks, os.path.join(DATA_PATH, "regridding_tasks.json"))
    print("tasks", tasks)

    if parallel:
        print("about to create cluster", time_stamp())

        cluster = LocalCluster(n_workers=worker, memory_limit=memory_per_worker)
        client = Client(cluster)
        print(client)
        dask.distributed.print(client)
        print(f"Dask dashboard link: {client.dashboard_link}")
    else:
        client = None

    for key in tasks:
        if not tasks[key]["locked"]:
            if not tasks[key]["processed_exists"] or force_regrid:
                regrid_cmip6_part(output_res, **tasks[key], client=client)
            else:
                print(f"Already regridded {key}, not regridding.")

    dask.distributed.print("finished", time_stamp())


@timeit
@locker(REGRIDDED_PATH)
def regrid_cmip6_part(
    output_res: float = 0.5,
    time_chunk: int = 1,
    exp: str = "ssp585",
    typ: str = "ocean",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
    client: Optional[Client] = None,
    **kwargs,
) -> None:
    """
    Regrid 2d data to a certain resolution using xesmf.

    Args:
        output_res (float, optional): Resolution of the output grid. Defaults to 0.5.
        time_chunk (int, optional): Chunk size for time. Defaults to 1.
        exp (str, optional): Experiment name. Defaults to "ssp585".
        typ (str, optional): Type of data. Defaults to "ocean". Can be "ocean" or "atmos".
        model (str, optional): Model name. Defaults to "CESM2".
        member (str, optional): Member name. Defaults to "r4i1p1f1".

    """
    print(f"exp:{exp} typ:{typ} model:{model} member:{member}")
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
        if typ == "ocean":
            chunks = {"time": 150, "y": 128, "x": 128}
        elif typ == "atmos":
            chunks = {"time": 50, "plev": -1, "y": 128, "x": 128}
        ds = xr.open_dataset(
            path,
            chunks=chunks,
        )
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
            ]  ## REDUCING time for exp
        )
        return ds  # .chunk(chunks={"time": time_chunk})

    in_ds = open_ds(
        os.path.join(RAW_PATH, exp, typ, model, member) + ".nc",
    )  # .isel(time=slice(0, 10))
    print("input dataset", in_ds)
    # atmos_ds = open_ds(os.path.join(RAW_PATH, "ssp585", "atmos", "CESM2", 'r4i1p1f1.nc'))

    new_coords = xe.util.grid_global(
        output_res, output_res
    )  # make a regular lat/lon grid

    print("new coordinates", new_coords)

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
        grid_template = input_ds.isel(time=0, drop=True)

        regridder = xe.Regridder(
            grid_template,
            new_coords.chunk({}),
            "bilinear",
            periodic=True,
            ignore_degenerate=True,
            # ignore_nan=True,
            parallel=True,
        )
        print(regridder)

        # 2. Set up parameters for batch processing
        output_path = os.path.join(CMIP6_PATH, output_name)
        if typ == "atmos":
            batch_size = 10  # Process 10 timesteps at a time
        else:
            batch_size = 100  # Process 100 timesteps at a time
        # batch_size = 100  # Process 100 timesteps at a time
        num_timesteps = len(input_ds.time)
        num_batches = math.ceil(num_timesteps / batch_size)

        # Ensure the target directory is clean before starting
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # 3. Loop over the data in time-based batches
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, num_timesteps)
            print(
                f"--> Processing Batch {i+1}/{num_batches} (timesteps {start_index} to {end_index-1})"
            )

            # Select a slice of the data (this is still lazy)
            ds_slice = input_ds.isel(time=slice(start_index, end_index))

            # Apply the regridder to the slice (also lazy)
            out_slice = regridder(
                ds_slice,
                keep_attrs=True,
                skipna=True,
                # output_chunks={"time": -1, "lat": 360, "lon": 360},
            )

            # Define the write mode
            mode = "a" if i > 0 else "w"
            append_dim = "time" if i > 0 else None

            # Compute and write only this batch to the Zarr store
            out_slice.to_zarr(
                output_path,
                mode=mode,
                append_dim=append_dim,
                consolidated=True,
                encoding={
                    var: {"dtype": "float32"}
                    for var in CONVERSION_NAMES.keys()
                    if var in out_slice
                },
            )

        print("--> Finished processing all batches.")
        return xr.open_zarr(output_path, consolidated=True)

    folder = os.path.join(REGRIDDED_PATH, exp, typ, model)
    os.makedirs(folder, exist_ok=True)
    out_ds = regrid_and_save(in_ds, os.path.join(folder, member) + ".zarr")
    print("out_ds saved", out_ds)


if __name__ == "__main__":
    # python -m tcpips.regrid
    # regrid_2d_1degree()
    # regrid_2d()
    # regrid_1d()
    # regrid_1d(xesmf=True)
    # get_task_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--worker", type=int, default=2)  # number of workers
    parser.add_argument("-m", "--memory", type=str, default="6GiB")  # memory per worker
    parser.add_argument(
        "-f", "--force", type=lambda x: (str(x).lower() == "true"), default=False
    )  # force regrid
    parser.add_argument(
        "-r", "--resolution", type=float, default=0.5
    )  # output resolution
    parser.add_argument(
        "-p",
        "--parallel",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        # description="Run in parallel?",
    )

    args = parser.parse_args()
    print("args", args)
    run_regridding_sequentially(
        worker=args.worker,
        force_regrid=args.force,
        output_res=args.resolution,
        memory_per_worker=args.memory,
        parallel=args.parallel,
    )
