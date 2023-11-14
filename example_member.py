from typing import Dict
import xarray as xr
from tcpyPI import pi
import intake
from intake import open_catalog

# CMIP6 equivalent names
# tos: Sea Surface Temperature [degC] [same]
# hus: Specific Humidity [kg/kg] [to g/kg]
# ta: Air Temperature [K] [to degC]
# psl: Sea Level Pressure [Pa] [to hPa]
# calculate PI over the whole data set using the xarray universal function
conversion_dict: Dict[str, str] = {"tos": "sst", "hus": "q", "ta": "t", "psl": "msl"}

PANGEO_CAT_URL = str(
    "https://raw.githubusercontent.com/pangeo-data/"
    + "pangeo-datastore/master/intake-catalogs/master.yaml"
)


instit = "NCAR"
cat = open_catalog(PANGEO_CAT_URL)["climate"]# ["cmip6_gcs"]
print(cat)

import intake_esm
# url = intake_esm.tutorial.get_url('google_cmip6')
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
cat = intake.open_esm_datastore(url)
unique = cat.unique()

print("cat", cat)
print("unique", unique)

cat_subset = cat.search(
    experiment_id=["historical", "ssp585"],
    table_id="Amon",
    institution_id="NCAR",
    # member_id="r10i1p1f1",
    source_id="CESM2",
    variable_id=conversion_dict.keys(),
    #grid_label="gn",
)
unique = cat_subset.unique()
for i in unique["member_id"]:
    print(i, cat_subset.search(member_id=i).unique())

print("cat_subset", cat_subset)
unique = cat_subset.unique()
print("unique", unique)

CKCD: float = 0.9


def get_pi(ds: xr.Dataset, dim: str = "p") -> xr.Dataset:
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
            # merge the state data into the same data structure
            "sst": ds.sst,
            "t": ds.t,
            "q": ds.q,
            "msl": ds.msl,
            "lsm": ds.lsm,
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
