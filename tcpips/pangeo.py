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
import intake
import dask
import xarray as xr
from xmip.preprocessing import combined_preprocessing
from sithom.time import timeit
from tcpips.constants import RAW_PATH
from tcpips.convert import conversion_names


# url = intake_esm.tutorial.get_url('google_cmip6')
url: str = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
try:
    print("url", url)
    cat = intake.open_esm_datastore(url)
    print("cat", cat)
    unique = cat.unique()
    print("cat", cat)
    print("unique", unique)
except Exception as e:
    print("Exception", e)


@timeit
def combined_experiments_from_dset_dict(
    dset_dict: dict, experiments: List[str], oc_or_at: str = "atmos", **kwargs
) -> Optional[xr.Dataset]:
    """
    Function to combine experiments together.

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
                print("saving", new_name, ds, "ds")
                with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                    ds.to_netcdf(
                        new_name,
                        format="NETCDF4",
                        engine="h5netcdf",
                        encoding={
                            var: {"dtype": "float32", "zlib": True, "complevel": 6}
                            for var in conversion_names.keys()
                            if var in ds
                        },
                    )

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
def get_atmos(experiments: List[str] = ["ssp585"]) -> None:  # ["historical", "ssp585"]
    """
    Get atmospheric data.

    Args:
        experiments (List[str], optional): Defaults to ["historical", "ssp585"].
    """

    cat_subset_obj = cat.search(
        experiment_id=experiments,
        table_id=["Amon"],  # , "Omon"],
        institution_id="NCAR",
        # member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_names.keys(),
        # grid_label="gn",
    )
    for member_id in cat_subset_obj.unique()["member_id"]:
        print("member_id", member_id)
        cat_subset = cat_subset_obj.search(member_id=member_id)
        ds = combined_experiments_from_cat_subset(cat_subset, experiments, "atmos")
        print("ds", ds)

    # ds.to_netcdf(os.path.join(CMIP6_PATH, "atmos.nc"))


@timeit
def get_ocean(experiments: List[str] = ["historical", "ssp585"]) -> None:
    """
    Get ocean data.

    Args:
        experiments (List[str], optional): Defaults to ["historical", "ssp585"].
    """
    cat_subset_obj = cat.search(
        experiment_id=experiments,
        table_id=["Omon"],
        institution_id="NCAR",
        # member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_names.keys(),
        # dcpp_init_year="20200528",
        grid_label="gn",
    )

    for member_id in cat_subset_obj.unique()["member_id"]:
        print("member_id", member_id)
        cat_subset = cat_subset_obj.search(member_id=member_id)
        ds = combined_experiments_from_cat_subset(cat_subset, experiments, "ocean")
        print("ds", ds)

    # ds.to_netcdf(os.path.join(CMIP6_PATH, "ocean.nc"))


@timeit
def get_data_part(
    experiments: List[str] = ["historical", "ssp585"],
    institution_id: str = "NCAR",
    table: str = "Omon",
    oc_or_at: str = "ocean",
    source_id: str = "CESM2-SE",
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
    cat_subset_obj = cat.search(
        experiment_id=experiments,
        table_id=[table],
        institution_id=institution_id,
        # member_id="r10i1p1f1",
        source_id=source_id,
        variable_id=conversion_names.keys(),
        # dcpp_init_year="20200528",
        # grid_label="gn",
    )
    print("cat_subset_obj", cat_subset_obj)
    for member_id in cat_subset_obj.unique()["member_id"]:
        print("member_id", member_id)
        cat_subset = cat_subset_obj.search(member_id=member_id)
        ds = combined_experiments_from_cat_subset(
            cat_subset, experiments, oc_or_at, source_id=source_id
        )
        print("ds", ds)


@timeit
def get_data_pair(
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
        experiments=["ssp585"],
        table="Amon",
        oc_or_at="atmos",
        institution_id=institution_id,
        source_id=source_id,
    )
    get_data_part(
        experiments=["ssp585"],
        table="Omon",
        oc_or_at="ocean",
        institution_id=institution_id,
        source_id=source_id,
    )


if __name__ == "__main__":
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2-SE
    # python -m tcpips.pangeo --institution_id=NCAR --source_id=CESM2
    # regrid_2d()
    # regrid_1d(xesmf=True)
    # regrid_2d_1degree()
    # pass
    # get_data_pair(institution_id="MOHC", source_id="HadGEM3-GC31-HH")
    import argparse

    parser = argparse.ArgumentParser(description="Pangeo download and processing scripts.")
    parser.add_argument(
        "--institution_id",
        default="NCAR",
        help="Institution id",
    )
    parser.add_argument(
        "--source_id",
        default="CESM2",
        help="Source id",
    )
    args = parser.parse_args()

    get_data_pair(institution_id=args.institution_id, source_id=args.source_id)
    # get_data_pair(institution_id="THU", source_id="CIESM")
    # get_data_pair(institution_id="THU", source_id="CIESM")
    # regrid_2d()
    # regrid_2d_1degree(output_res=0.25)
    # regrid_2d()
    # regrid_1d(xesmf=True)
