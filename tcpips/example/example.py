"""Example for downloading TOS for historical and ssp585 for CESM2 r10i1p1f1."""

import dask
import xarray as xr
import intake
from xmip.preprocessing import combined_preprocessing  # standardise cmip6 data


url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
cat = intake.open_esm_datastore(url)
unique = cat.unique()

cat_subset = cat.search(
    experiment_id=["historical", "ssp585"],
    table_id=["Omon"],
    institution_id="NCAR",
    member_id="r10i1p1f1",
    source_id="CESM2",
    variable_id=["tos"],
    # dcpp_init_year="20200528",
    grid_label="gn",
)


z_kwargs = {"consolidated": True, "decode_times": True}
with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    dset_dict = cat_subset.to_dataset_dict(
        zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
    )

ds_l = []
for i, (k, ds) in enumerate(dset_dict.items()):
    if "member_id" in ds.dims:
        ds = ds.isel(member_id=0, dcpp_init_year=0)
        ds_l.append(ds)

with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    ds = xr.concat(ds_l[::-1], dim="time")

ds.to_netcdf("tos.nc")
