"""
Pangeo download and processing scripts.

This script is used to download CMIP6 data from the
Pangeo Google cloud store and process it for use in the TCPIPS project to calculate
potential size and potential intensity.

Folder structure:

${ROOT}/${processing-step}/${experiment}/${atm/oc}/${model}/${member_id}.nc

"""

import os
from typing import Dict, List, Optional
import time
import intake
import dask
from dask.diagnostics import ProgressBar
import xarray as xr
from xmip.preprocessing import combined_preprocessing
from sithom.time import timeit, hr_time, time_stamp
from .constants import RAW_PATH, CONVERSION_NAMES, PANGEO_CMIP6_URL

# from dask.distributed import Client


try:
    print("url", PANGEO_CMIP6_URL)
    cat = intake.open_esm_datastore(PANGEO_CMIP6_URL)
    print("cat", cat)
    unique = cat.unique()
    print("unique", unique)
except Exception as e:
    print("Exception", e)


@timeit
def combined_experiments_from_dset_dict(
    dset_dict: dict, experiments: List[str], oc_or_at: str = "atmos", **kwargs
) -> Optional[xr.Dataset]:
    """
    Function to combine experiments together.

    This is really misnamed and should be called something like `save_experiments`.

    It is used to save the experiments to disk.

    Args:
        dset_dict (dict): dictionary of datasets.
        experiments (List[str]): list of experiments to combine.
        oc_or_at (str, optional): Defaults to "atmos".

    Returns:
        Optional[xr.Dataset]: combined xarray dataset.
    """
    ds_d: Dict[str, xr.Dataset] = {}  # order datasets by experiment order
    # zero_dims = ["member_id", "dcpp_init_year"]
    for k, ds in dset_dict.items():
        assert ds.member_id.size == 1
        ds_member_id = ds.member_id.values[0]
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0)
        if "dcpp_init_year" in ds.dims:
            ds = ds.isel(dcpp_init_year=0)

        print(k, ds)
        ds = ds.chunk({"time": 1})
        for experiment in experiments:
            if experiment in k:
                if "source_id" in kwargs:
                    model_name = kwargs["source_id"]
                else:
                    model_name = ds["intake_esm_attrs"]["source_id"]  # does not work
                ds_d[experiment] = ds
                path = os.path.join(
                    RAW_PATH,
                    experiment,
                    oc_or_at,
                    model_name,
                )
                os.makedirs(path, exist_ok=True)
                new_name = os.path.join(path, f"{ds_member_id}.nc")
                lock = os.path.join(path, f"{ds_member_id}.nc.lock")
                if os.path.exists(lock):
                    print("lock exists", lock)
                    continue

                with open(lock, "w") as f:
                    f.write(time_stamp())

                print("saving", new_name, ds, "ds")
                # print("ds", ds)
                tick = time.perf_counter()

                with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                    output_file = ds.to_netcdf(
                        new_name,
                        format="NETCDF4",
                        engine="h5netcdf",
                        encoding={
                            var: {"dtype": "float32"}  # , "zlib": True, "complevel": 6
                            for var in CONVERSION_NAMES.keys()
                            if var in ds
                        },
                        compute=False,
                    )

                    with ProgressBar():
                        output_file.compute()
                tock = time.perf_counter()
                print(f"Time taken: {hr_time(tock - tick)}")
                os.remove(lock)
                print("output_file", output_file)

    # put the two experiments together
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        if len(ds_d) == len(experiments):
            ds = xr.concat([ds_d[experiment] for experiment in experiments], dim="time")
        else:
            ds = None

    return ds


@timeit
def combined_experiments_from_cat_subset(
    cat_subset: intake.catalog.local.LocalCatalogEntry,
    experiments: List[str],
    oc_or_at: str = "atmos",
    **kwargs,
) -> Optional[xr.Dataset]:
    """
    Combine experiments from a catalog subset.

    Args:
        cat_subset (intake.catalog.local.LocalCatalogEntry): catalog subset.
        experiments (List[str]): experiments to combine.
        oc_or_at (str, optional): Defaults to "atmos".

    Returns:
        Optional[xr.Dataset]: combined xarray dataset.
    """

    print("cat_subset", cat_subset)
    unique = cat_subset.unique()
    print("unique", unique)

    z_kwargs = {"consolidated": True, "decode_times": True}
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        dset_dict = cat_subset.to_dataset_dict(
            zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
        )

    print("dset_dict.keys()", dset_dict.keys())

    ds = combined_experiments_from_dset_dict(dset_dict, experiments, oc_or_at, **kwargs)

    return ds


@timeit
def get_data_part(
    experiments: List[str] = ["historical", "ssp585"],
    institution_id: str = "NCAR",  # model center
    table: str = "Omon",
    oc_or_at: str = "ocean",
    source_id: str = "CESM2-SE",  # particular model
) -> None:
    """
    Get data part from one of the experiments.

    Args:
        experiments (List[str], optional): Defaults to ["historical", "ssp585"].
        table (str, optional): Defaults to "Omon".
        oc_or_at (str, optional): Defaults to "ocean".
        institution_id (str, optional): Defaults to "NCAR".
        source_id (str, optional): Defaults to "CESM2-SE".
    """
    print(
        "experiments",
        experiments,
        "\ntable",
        table,
        "\noc_or_at",
        oc_or_at,
        "\nsource_id",
        source_id,
        "\ninstitution_id",
        institution_id,
        "\nCONVERSION_NAMES.keys()",
        CONVERSION_NAMES.keys(),
    )
    add_kwargs = {}
    if source_id in ["MIROC6", "CESM2"]:
        add_kwargs["member_id"] = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    elif source_id in ["HadGEM3-GC31-MM"]:
        add_kwargs["member_id"] = ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"]

    cat_subset_obj = cat.search(
        experiment_id=experiments,
        table_id=[table],
        institution_id=institution_id,
        # member_id="r10i1p1f1",
        source_id=source_id,
        variable_id=CONVERSION_NAMES.keys(),
        # dcpp_init_year="20200528",
        # grid_label="gn",
        **add_kwargs,
    )
    print("cat_subset_obj", cat_subset_obj)
    print("cat_subset_obj.unique()", cat_subset_obj.unique())
    print("cat_subset_obj.unique()['member_id']", cat_subset_obj.unique()["member_id"])
    for member_id in cat_subset_obj.unique()["member_id"]:
        print("member_id", member_id)
        cat_subset = cat_subset_obj.search(member_id=member_id)
        ds = combined_experiments_from_cat_subset(
            cat_subset, experiments, oc_or_at, source_id=source_id
        )
        print("ds", ds)


@timeit
def get_data_pair(
    exp: str = "ssp585",
    institution_id: str = "THU",  # "MOHC",  # "NCAR",
    source_id: str = "CIESM",  # "HadGEM3-GC31-HH",  # "CESM2-SE",
) -> None:
    """
    Get data from Pangeo for a particular institution source pair.

    Have a look at the CMIP6 data available on Pangeo:

    https://docs.google.com/spreadsheets/d/13DHeTEH_8G08vxTMX1Fs-WbAA6SamBjDdh0fextdcGE/edit#gid=165882553

    Args:
        institution_id (str, optional): Defaults to "MOHC".
        source_id (str, optional): Defaults to "HadGEM3-GC31-HH".
    """
    get_data_part(
        experiments=[exp],
        table="Amon",
        oc_or_at="atmos",
        institution_id=institution_id,
        source_id=source_id,
    )
    get_data_part(
        experiments=[exp],
        table="Omon",
        oc_or_at="ocean",
        institution_id=institution_id,
        source_id=source_id,
    )


def cmd_download_call() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pangeo download and processing scripts."
    )
    parser.add_argument(
        "--institution_id",
        default="MOHC",
        help="Institution id",
    )
    parser.add_argument(
        "--source_id",
        default="UKESM1-0-LL",
        help="Source id",
    )
    parser.add_argument(
        "--exp",
        default="ssp585",
        help="Experiment",
    )
    args = parser.parse_args()

    print("args", args)

    get_data_pair(
        institution_id=args.institution_id, source_id=args.source_id, exp=args.exp
    )


# def unique_model_name(centre: str, model: str, exp: str):


if __name__ == "__main__":
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2 --exp=historical
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2 --exp=historical

    # python -m tcpips.pangeo --institution_id=THU --source_id=CIESM
    # python -m tcpips.pangeo --institution_id=THU --source_id=CIESM --exp=historical
    # let's download MIROC MIROC6 data
    # python -m tcpips.pangeo --source_id=MIROC6 --institution_id=MIROC --exp=ssp585
    #

    # python -m tcpips.pangeo --institution_id=MOHC --source_id=HadGEM3-GC31-MM --exp=ssp585
    # python -m tcpips.pangeo --institution_id=MOHC --source_id=HadGEM3-GC31-LL --exp=historical
    # python -m tcpips.pangeo --institution_id=MOHC --source_id=UKESM1-0-LL --exp=historical
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2 --exp=historical
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2-SE --exp=historical
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2-WACCM --exp=historical
    # regrid_2d()
    # regrid_1d(xesmf=True)
    # regrid_2d_1degree()s
    # get_data_pair(institution_id="MOHC", source_id="HadGEM3-GC31-HH")
    # client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")
    cmd_download_call()
    # cat_subset = cat.search(
    #     experiment_id=["historical", "ssp585"],
    #     table_id=["Amon", "Omon"],
    #     institution_id="MOHC",
    #     # source_id="CIESM",
    #     # member_id="r1i1p1f1",
    #     # member_id="r10i1p1f1",
    #     variable_id=CONVERSION_NAMES.keys(),
    #     # grid_label="gn",
    # ).unique()
    # print("cat_subset", cat_subset)

    # cat_subset.to_csv("cat_subset.csv")

    # get_data_pair(institution_id="THU", source_id="CIESM")
    # get_data_pair(institution_id="THU", source_id="CIESM")
    # regrid_2d()
    # regrid_2d_1degree(output_res=0.25)
    # regrid_2d()
    # regrid_1d(xesmf=True)
