"""
Pangeo download and processing scripts.

This script is used to download CMIP6 data from the
Pangeo Google cloud store and process it for use in the TCPIPS project to calculate
potential size and potential intensity.

"""

import os
from typing import Tuple, Dict, List, Optional
import intake
import dask
import xesmf as xe
import xarray as xr
from matplotlib import pyplot as plt
from xmip.preprocessing import combined_preprocessing
from xmip.postprocessing import interpolate_grid_label
from sithom.plot import feature_grid, plot_defaults, label_subplots
from sithom.time import timeit
from tcpips.constants import FIGURE_PATH, CMIP6_PATH
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


def combined_experiments_from_dset_dict(
    dset_dict: dict, experiments: List[str], name: str = "test"
) -> Optional[xr.Dataset]:
    """
    Function to combine experiments together.

    Args:
        dset_dict (dict): _description_
        experiments (List[str]): _description_
        name (str, optional): _description_. Defaults to "test".

    Returns:
        xr.Dataset: combined xarray dataset.
    """
    ds_d: Dict[str, xr.Dataset] = {}  # order datasets by experiment order
    # zero_dims = ["member_id", "dcpp_init_year"]
    for k, ds in dset_dict.items():
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0)
        if "dcpp_init_year" in ds.dims:
            ds = ds.isel(dcpp_init_year=0)

        print(k, ds)
        for experiment in experiments:
            if experiment in k:
                ds_d[experiment] = ds
                ds.to_netcdf(os.path.join(CMIP6_PATH, f"{name}_{experiment}_{k}.nc"))

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        if len(ds_d) == len(experiments):
            ds = xr.concat([ds_d[experiment] for experiment in experiments], dim="time")
        else:
            ds = None
        # ds = xr.concat([ds_d[experiment] for experiment in experiments], dim="time")

    return ds


def combined_experiments_from_cat_subset(
    cat_subset: intake.catalog.local.LocalCatalogEntry,
    experiments: List[str],
    name: str = "test",
) -> Optional[xr.Dataset]:

    print("cat_subset", cat_subset)
    unique = cat_subset.unique()
    print("unique", unique)

    z_kwargs = {"consolidated": True, "decode_times": True}
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        dset_dict = cat_subset.to_dataset_dict(
            zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
        )

    print("dset_dict.keys()", dset_dict.keys())

    ds = combined_experiments_from_dset_dict(dset_dict, experiments, name)

    return ds


@timeit
def get_atmos(experiments: List[str] = ["historical", "ssp585"]) -> None:

    cat_subset_obj = cat.search(
        experiment_id=["historical", "ssp585"],
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
        ds = combined_experiments_from_cat_subset(cat_subset, experiments, "atmos")
        print("ds", ds)

    # ds.to_netcdf(os.path.join(CMIP6_PATH, "ocean.nc"))


@timeit
def get_data() -> None:
    get_ocean()
    get_atmos()


@timeit
def regrid_2d_1degree(output_res=1.0, time_chunk=10) -> None:
    plot_defaults()

    def open_ds(path: str) -> xr.Dataset:
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

    ocean_ds = open_ds(os.path.join(CMIP6_PATH, "ocean.nc"))
    atmos_ds = open_ds(os.path.join(CMIP6_PATH, "atmos.nc"))

    new_coords = xe.util.grid_global(
        output_res, output_res
    )  # make regular lat/lon grid

    def regrid_and_save(input_ds: xr.Dataset, output_name: str) -> xr.Dataset:
        """
        Regrid and save the input dataset to the output name.

        Args:
            input_ds (xr.Dataset): dataset to regrid.
            output_name (str): name of the output file.

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
        out_ds.to_netcdf(
            os.path.join(CMIP6_PATH, output_name),
            format="NETCDF4",
            engine="h5netcdf",  # should be better at parallel writing/dask
            encoding={
                var: {"dtype": "float32", "zlib": True, "complevel": 6}
                for var in conversion_names.keys()
                if var in out_ds
            },
        )
        return out_ds  # return for later plotting.

    ocean_out = regrid_and_save(
        ocean_ds, f"regrid_2d_{output_res:.1}degree_ocean_regridded.nc"
    )
    regrid_and_save(atmos_ds, f"regrid_2d_{output_res:.1}degree_atmos_regridded.nc")

    print("ocean_out", ocean_out)
    ocean_out.tos.isel(time=0).plot(x="lon", y="lat")
    plt.savefig(
        os.path.join(
            FIGURE_PATH, f"regrid_2d_{output_res:.1}degree_ocean_regridded.png"
        )
    )
    plt.clf()
    ocean_out.tos.isel(time=0).plot()
    plt.savefig(
        os.path.join(
            FIGURE_PATH, f"regrid_2d_{output_res:.1}degree_ocean_regridded_1d.png"
        )
    )
    plt.clf()


@timeit
def regrid_2d() -> None:
    plot_defaults()

    def open_ds(name):
        ds = xr.open_dataset(name)
        ds = ds.drop_vars(
            [
                x
                for x in [
                    "x",
                    "y",
                    # "lat_verticies",
                    # "lon_verticies",
                    # "lon_bounds",
                    # "time_bounds",
                    # "lat_bounds",
                    "dcpp_init_year",
                    "member_id",
                ]
                if x in ds
            ]
        )
        return ds

    ocean_ds = open_ds(os.path.join(CMIP6_PATH, "ocean.nc"))
    atmos_ds = open_ds(os.path.join(CMIP6_PATH, "atmos.nc"))  # .isel(CMIP6_PATH=0)
    print("ocean_ds", ocean_ds)
    # ocean_ds = ocean_ds.rename({"lon_bounds": "lon_b", "lat_bounds": "lat_b"})
    print("atmos_ds", atmos_ds)
    new_coords = atmos_ds[["lat", "lon"]]
    print("new_coords", new_coords)

    plot_ds = xr.Dataset(
        {
            "lat_o": ocean_ds.lat,
            "lon_o": ocean_ds.lon,
        }
    )
    features = [["lat_o", "lon_o"]]  # , ["lat_a", "lon_a"]]
    names = [["lat", "lon"] for x in range(len(features))]
    units = [[None for y in range(len(features[x]))] for x in range(len(features))]
    vlim = [[None for y in range(len(features[x]))] for x in range(len(features))]
    super_titles = ["lat", "lon"]
    fig, axs = feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    plt.savefig(os.path.join(FIGURE_PATH, "o_coords.png"))
    plt.clf()

    plot_ds = xr.Dataset(
        {
            "lat_a": atmos_ds.lat,
            "lon_a": atmos_ds.lon,
        }
    )
    features = [["lat_a", "lon_a"]]  # , ["lat_a", "lon_a"]]
    names = [["lat", "lon"] for x in range(len(features))]
    units = [[None for y in range(len(features[x]))] for x in range(len(features))]
    vlim = [[None for y in range(len(features[x]))] for x in range(len(features))]
    super_titles = ["lat", "lon"]
    feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    plt.savefig(os.path.join(FIGURE_PATH, "a_coords.png"))
    plt.clf()

    regridder = xe.Regridder(ocean_ds, new_coords, "bilinear", periodic=True)
    print(regridder)
    ocean_out = regridder(
        ocean_ds,
        keep_attrs=True,
        skipna=True,
    )
    print("ocean_out", ocean_out)
    ocean_out.to_netcdf(os.path.join(CMIP6_PATH, "regrid_2d_ocean_regridded.nc"))
    ocean_out.tos.isel(time=0).plot(x="lon", y="lat")

    ocean_out.tos.isel(time=0).plot()

    atmos_ds["tos"] = ocean_out["tos"]
    print(atmos_ds)

    plot_ds = atmos_ds.isel(
        time=5, plev=0
    )  # .sel(lon=slice(0, 360), lat=slice(-90, 90))
    features = [["tos", "psl"], ["ta", "hus"]]
    names = [
        [plot_ds[features[x][y]].attrs["long_name"] for y in range(len(features[x]))]
        for x in range(len(features))
    ]
    units = [
        [plot_ds[features[x][y]].attrs["units"] for y in range(len(features[x]))]
        for x in range(len(features))
    ]
    vlim = [[None for y in range(len(features[x]))] for x in range(len(features))]
    super_titles = ["" for x in range(len(features))]

    fig, axs = feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    label_subplots(axs)
    plt.savefig(os.path.join(FIGURE_PATH, "test.png"))
    plt.clf()


@timeit
def regrid_1d(xesmf: bool = False) -> None:
    def open_1d(name):
        ds = xr.open_dataset(name)
        # plt.imshow(ds.lat.values)
        #
        # plt.imshow(ds.lon.values)
        #
        # print("ds", name, ds)
        ds = ds.drop_vars(
            [
                x
                for x in [
                    "lon",
                    "lat",
                    "lat_verticies",
                    "lon_verticies",
                    "lon_bounds",
                    "time_bounds",
                    "lat_bounds",
                    "CMIP6_PATH",
                    "member_id",
                ]
                if x in ds
            ]
        ).isel(y=slice(1, -1))
        if xesmf:
            ds = ds.assign_coords({"lon": ds["x"], "lat": ds["y"]})
            return ds.drop_vars(["x", "y"])
        else:
            return ds.rename({"x": "lon", "y": "lat"})

    ocean_ds = open_1d(os.path.join(CMIP6_PATH, "ocean.nc"))
    print("ocean_ds", ocean_ds)
    ocean_ds.isel(time=0).tos.plot(x="lon", y="lat")

    atmos_ds = open_1d(os.path.join(CMIP6_PATH, "atmos.nc"))  # .isel(CMIP6_PATH=0)
    print("atmos_ds", atmos_ds)
    atmos_ds.isel(time=0).psl.plot(x="lon", y="lat")

    new_coords = atmos_ds[["lon", "lat"]]
    print("new_coords", new_coords)
    plt.plot(new_coords.lat.values, label="new")
    plt.plot(ocean_ds.lat.values, label="ocean")
    plt.legend()
    plt.title("lat")

    plt.plot(new_coords.lon.values, label="new")
    plt.plot(ocean_ds.lon.values, label="ocean")
    plt.legend()
    plt.title("lon")

    if xesmf:
        regridder = xe.Regridder(ocean_ds, new_coords, "nearest_s2d", periodic=True)
        print(regridder)
        ocean_out = regridder(
            ocean_ds,  # .drop_vars(["x", "y"]).set_coords(["lon", "lat"]),
            keep_attrs=True,
            skipna=True,
        )
    else:
        ocean_out = ocean_ds.interp(
            {"lon": new_coords.lon.values, "lat": new_coords.lat.values},
            method="nearest",
        )
    print("ocean_out", ocean_out)
    ocean_out.to_netcdf(os.path.join(CMIP6_PATH, "regrid1d_ocean_regridded.nc"))
    ocean_out.tos.isel(time=0).plot(x="lon", y="lat")
    plt.savefig(os.path.join(FIGURE_PATH, "ocean_regridded_regrid1d.png"))


if __name__ == "__main__":
    # tcpips/pangeo.py
    # regrid_2d()
    # regrid_1d(xesmf=True)
    # regrid_2d_1degree()
    # pass
    get_data()
    # regrid_2d()
    # regrid_2d_1degree(output_res=0.25)
    # regrid_2d()
    # regrid_1d(xesmf=True)
