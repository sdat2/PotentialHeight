from typing import Tuple, Dict
import intake
import dask
import xesmf as xe
import xarray as xr
from matplotlib import pyplot as plt
from xmip.preprocessing import combined_preprocessing
from xmip.postprocessing import interpolate_grid_label
from sithom.plot import feature_grid, plot_defaults
from sithom.time import timeit
from tcpips.constants import FIGURE_PATH

# CMIP6 equivalent names
# tos: Sea Surface Temperature [degC] [same]
# hus: Specific Humidity [kg/kg] [to g/kg]
# ta: Air Temperature [K] [to degC]
# psl: Sea Level Pressure [Pa] [to hPa]
# calculate PI over the whole data set using the xarray universal function
conversion_names: Dict[str, str] = {"tos": "sst", "hus": "q", "ta": "t", "psl": "msl"}
conversion_multiples: Dict[str, float] = {
    "hus": 1000,
    "psl": 0.01,  # "plev": 0.01
}
conversion_additions: Dict[str, float] = {"ta": -273.15}
conversion_units: Dict[str, str] = {
    "hus": "g/kg",
    "psl": "hPa",
    "tos": "degC",
    "ta": "degC",  # "plev": "hPa"
}


@timeit
def convert(ds: xr.Dataset) -> xr.Dataset:
    for var in conversion_multiples:
        if var in ds:
            ds[var] *= conversion_multiples[var]
    for var in conversion_additions:
        if var in ds:
            ds[var] += conversion_additions[var]
    for var in conversion_units:
        if var in ds:
            ds[var].attrs["units"] = conversion_units[var]
    ds = ds.rename(conversion_names)
    if "plev" in ds:
        ds = ds.set_coords(["plev"])
        ds["plev"] = ds["plev"] / 100
        ds["plev"].attrs["units"] = "hPa"
        ds = ds.rename({"plev": "p"})
    return ds


# url = intake_esm.tutorial.get_url('google_cmip6')
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
cat = intake.open_esm_datastore(url)
unique = cat.unique()

print("cat", cat)
print("unique", unique)

@timeit
def get_atmos() -> None:
    cat_subset = cat.search(
        experiment_id=["historical", "ssp585"],
        table_id=["Amon"],  # , "Omon"],
        institution_id="NCAR",
        member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_names.keys(),
        # grid_label="gn",
    )

    print("cat_subset", cat_subset)
    unique = cat_subset.unique()
    print("unique", unique)

    z_kwargs = {"consolidated": True, "decode_times": True}
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        dset_dict = interpolate_grid_label(
            cat_subset.to_dataset_dict(
                zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
            ),
            target_grid_label="gr",
        )

    print(dset_dict.keys())

    ds_l = []
    for k, ds in dset_dict.items():
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0, dcpp_init_year=0)
        print(k, ds)
        ds_l.append(ds)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.concat(ds_l[::-1], dim="time")

    print("merged ds", ds)

    ds.to_netcdf("data/atmos.nc")


@timeit
def get_all() -> None:
    cat_subset = cat.search(
        experiment_id=["historical", "ssp585"],
        table_id=["Amon", "Omon"],
        institution_id="NCAR",
        member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_names.keys(),
        # grid_label="gn",
    )

    print("cat_subset", cat_subset)
    unique = cat_subset.unique()
    print("unique", unique)

    z_kwargs = {"consolidated": True, "decode_times": True}
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        # dset_dict = interpolate_grid_label(
        dset_dict = cat_subset.to_dataset_dict(
            zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
        )  # , target_grid_label="gr", merge_kwargs={"compat": "override", "combine_attrs": "override""})

    print(dset_dict.keys())

    ds_l = []
    for k, ds in dset_dict.items():
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0, dcpp_init_year=0)
        print(k)
        ds_l.append(ds)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.concat(ds_l[::-1], dim="time")

    print("merged ds", ds)

    ds.to_netcdf("data/all.nc")


@timeit
def get_ocean() -> None:
    cat_subset = cat.search(
        experiment_id=["historical", "ssp585"],
        table_id=["Omon"],
        institution_id="NCAR",
        member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_names.keys(),
        # dcpp_init_year="20200528",
        grid_label="gn",
    )

    print("cat_subset", cat_subset)
    unique = cat_subset.unique()
    print("unique", unique)

    z_kwargs = {"consolidated": True, "decode_times": True}
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        dset_dict = cat_subset.to_dataset_dict(
            zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
        )

    print(dset_dict.keys())

    ds_l = []
    for i, (k, ds) in enumerate(dset_dict.items()):
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0)
        print("\n\n", i, k, ds.dims, ds["tos"].attrs, "\n\n")
        ds_l.append(ds.isel(dcpp_init_year=0))
        ds.isel(dcpp_init_year=0).to_netcdf(f"data/ocean-{i}.nc")

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.concat(ds_l[::-1])

    print("merged ds", ds)

    ds.to_netcdf("data/ocean.nc")


@timeit
def get_data() -> None:
    get_ocean()
    get_atmos()


@timeit
def regrid_2d_1degree() -> None:
    plot_defaults()

    def open(name):
        ds = xr.open_dataset(name, chunks={"time": 40})
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

    ocean_ds = open("data/ocean.nc")
    atmos_ds = open("data/atmos.nc").isel(dcpp_init_year=0)

    new_coords = xe.util.grid_global(1, 1)

    regridder = xe.Regridder(ocean_ds, new_coords, "bilinear", periodic=True)
    print(regridder)
    ocean_out = regridder(
        ocean_ds,
        keep_attrs=True,
        skipna=True,
    )
    regridder = xe.Regridder(
        atmos_ds, new_coords, "bilinear", periodic=True, ignore_degenerate=True
    )
    regridder(
        atmos_ds,
        keep_attrs=True,
        skipna=True,
    ).to_netcdf(
        "data/atmos_new_regridded.nc",
        format="NETCDF4",
        engine="h5netcdf",
        encoding={
            var: {"dtype": "float32", "zlib": True, "complevel": 6}
            for var in conversion_names.keys()
            if var in atmos_ds
        },
    )

    # xr.merge([ocean_out, atmos_out], compat="override").to_netcdf(
    #    "data/all_regridded.nc", engine="h5netcdf"
    # )

    print("ocean_out", ocean_out)
    ocean_out.tos.isel(time=0).plot(x="lon", y="lat")
    plt.show()
    ocean_out.tos.isel(time=0).plot()
    plt.show()


@timeit
def regrid_2d() -> None:
    plot_defaults()

    def open(name):
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

    ocean_ds = open("data/ocean.nc")
    atmos_ds = open("data/atmos.nc").isel(dcpp_init_year=0)
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
    feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    plt.savefig("o_coords.png")
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
    plt.savefig("a_coords.png")
    plt.clf()

    regridder = xe.Regridder(ocean_ds, new_coords, "bilinear", periodic=True)
    print(regridder)
    ocean_out = regridder(
        ocean_ds,
        keep_attrs=True,
        skipna=True,
    )
    print("ocean_out", ocean_out)
    ocean_out.to_netcdf("data/ocean_regridded.nc")
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

    feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    plt.savefig("test.png")
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
                    "dcpp_init_year",
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

    ocean_ds = open_1d("data/ocean.nc")
    print("ocean_ds", ocean_ds)
    ocean_ds.isel(time=0).tos.plot(x="lon", y="lat")

    atmos_ds = open_1d("data/atmos.nc").isel(dcpp_init_year=0)
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
    ocean_out.to_netcdf("data/ocean_regridded.nc")
    ocean_out.tos.isel(time=0).plot(x="lon", y="lat")
