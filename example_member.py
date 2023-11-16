from typing import Dict
import xarray as xr
from tcpyPI import pi
import intake
from intake import open_catalog
import intake_esm
import dask
from matplotlib import pyplot as plt
from xmip.preprocessing import combined_preprocessing
from sithom.time import timeit

# CMIP6 equivalent names
# tos: Sea Surface Temperature [degC] [same]
# hus: Specific Humidity [kg/kg] [to g/kg]
# ta: Air Temperature [K] [to degC]
# psl: Sea Level Pressure [Pa] [to hPa]
# calculate PI over the whole data set using the xarray universal function
conversion_dict: Dict[str, str] = {"tos": "sst", "hus": "q", "ta": "t", "psl": "msl"}

# url = intake_esm.tutorial.get_url('google_cmip6')
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
cat = intake.open_esm_datastore(url)
unique = cat.unique()

print("cat", cat)
print("unique", unique)

CKCD: float = 0.9


@timeit
def get_atmos():
    cat_subset = cat.search(
        experiment_id=["historical", "ssp585"],
        table_id=["Amon"],  # , "Omon"],
        institution_id="NCAR",
        member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_dict.keys(),
        # grid_label="gn",
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
def get_ocean():
    cat_subset = cat.search(
        experiment_id=["historical", "ssp585"],
        table_id=["Omon"],
        institution_id="NCAR",
        member_id="r10i1p1f1",
        source_id="CESM2",
        variable_id=conversion_dict.keys(),
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
        ds = xr.concat(ds_l[::-1], dim="time")

    print("merged ds", ds)

    ds.to_netcdf("data/ocean.nc")


@timeit
def get_data():
    get_ocean()
    get_atmos()


@timeit
def load_data():
    ocean_ds = xr.open_dataset("data/ocean.nc")
    atmos_ds = xr.open_dataset("data/atmos.nc").isel(dcpp_init_year=0)
    ocean_ds = ocean_ds  # .interp({"": "", "": ""}, method="nearest")
    # .rename({"nlat": "lat", "nlon": "lon"})
    print("ocean_ds", ocean_ds)
    print("atmos_ds", atmos_ds)
    new_coords = atmos_ds[["lat", "lon"]]
    old_coords = ocean_ds[["lat", "lon"]]
    print("new_coords", new_coords)
    print("old_coords", old_coords)
    #ocean_ds_new = ocean_ds.interp(dict(lat=new_coords.lat, lon=new_coords.lon), method="nearest")
    #print(ocean_ds_new)
    #ocean_ds_new.tos.isel(time=0).plot()
    plt.show()



@timeit
def calculate_pi(ds: xr.Dataset, dim: str = "plev") -> xr.Dataset:
    result = xr.apply_ufunc(
        pi,
        ds["sst"],
        ds["msl"],
        ds[dim],
        ds["t"],
        ds["q"],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1),
        input_core_dims=[
            [],
            [],
            [
                dim,
            ],
            [
                dim,
            ],
            [
                dim,
            ],
        ],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
    )


    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    out_ds = xr.Dataset(
        {
            "vmax": vmax,
            "pmin": pmin,
            "ifl": ifl,
            "t0": t0,
            "otl": otl,
        }
    )

    # add names and units to the structure
    out_ds.vmax.attrs["standard_name"], out_ds.vmax.attrs["units"] = (
        "Maximum Potential Intensity",
        "m/s",
    )
    out_ds.pmin.attrs["standard_name"], out_ds.pmin.attrs["units"] = (
        "Minimum Central Pressure",
        "hPa",
    )
    out_ds.ifl.attrs["standard_name"] = "pyPI Flag"
    out_ds.t0.attrs["standard_name"], out_ds.t0.attrs["units"] = (
        "Outflow Temperature",
        "K",
    )
    out_ds.otl.attrs["standard_name"], out_ds.otl.attrs["units"] = (
        "Outflow Temperature Level",
        "hPa",
    )
    return out_ds


if __name__ == "__main__":
    # get_data()
    load_data()
