"""A driver for calculating potential sizes using CMIP6 data.

There are two types of potential size we should calculate:
(1) The potential size corresponding to the storm at its potential intensity, and (2) the potential size corresponding to the storm at the lower bound of 33 m/s for a category 1 tropical cyclone.

Each of these types of potential size requires a calculation with a different "vmax" input. They will produce a different
output for rmax and r0 (the radius of maximum wind and the radius of vanishing winds, respectively).

Need to load Potential intensity data and regridded data.
"""

import os
import xarray as xr
import pandas as pd
from w22.ps import parallelized_ps_dask
from .constants import CDO_PATH, PI4_PATH, PS_PATH
from .files import locker
from .dask_utils import dask_cluster_wrapper
from .convert import convert


@locker(PS_PATH)
def ps_cmip6_part(
    exp: str = "ssp585",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
    time_chunk: int = 1,
) -> None:
    """Potential size calculation for CMIP6 data.

        Args:
            exp (str, optional): Experiment name. Defaults to "ssp585".
            model (str, optional): Model name. Defaults to "CESM2".
            member (str, optional): Member name. Defaults to "r4i1p1
    f1".
            time_chunk (int, optional): Chunk size for time. Defaults to 1.
    """
    print(f"exp:{exp} model:{model} member:{member}")
    oc_ds = xr.open_dataset(
        os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc",
        chunks={"time": time_chunk},
    )
    at_ds = xr.open_dataset(
        os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc",
        chunks={"time": time_chunk, "plev": -1},
    )
    # Align atmospheric data to ocean time axis
    one_day_tolerance = pd.to_timedelta("1D")
    at_ds_aligned = at_ds.reindex(
        time=oc_ds.time,
        method="nearest",
        tolerance=one_day_tolerance,
    )

    pi_ds = os.open_zarr(
        os.path.join(PI4_PATH, exp, model, member) + ".zarr",
        chunks={"time": time_chunk, "plev": -1},
    )
    pi_ds_aligned = pi_ds.reindex(
        time=oc_ds.time,
        method="nearest",
        tolerance=one_day_tolerance,
    )
    ds = xr.merge([oc_ds, at_ds_aligned, pi_ds_aligned]).chunk(
        {"time": time_chunk, "plev": -1}
    )
    ds = convert(ds)

    del ds["t"]
    del ds["q"]
    del ds["plev"]
    ds = parallelized_ps_dask(ds.chunk({"time": 64, "lat": 480, "lon": 480})).chunk(
        {"time": 64, "lat": 480, "lon": 480}
    )  # chunk the data for saving
    ds = ds.rename({"r0": "r0_pi", "rmax": "rmax_pi", "vmax": "vmax_pi"})
    ds["vmax"] = 33  # set the vmax to 33 m/s for the lower limit of category 1.
    ds = parallelized_ps_dask(ds.chunk({"time": 64, "lat": 480, "lon": 480})).chunk(
        {"time": 64, "lat": 480, "lon": 480}
    )  # chunk the data for saving
    ds = ds.rename({"r0": "r0_cat1", "rmax": "rmax_cat1", "vmax": "vmax_cat1"})
    vars = [
        var
        for var in ds.variables
        if var
        not in ["r0_cat1", "rmax_cat1", "vmax_cat1", "r0_pi", "rmax_pi", "vmax_pi"]
    ]
    ds = ds.drop_vars(vars, errors="ignore")

    # del ds["t2m"]
    # del ds[""]
    ds.to_zarr(
        os.path.join(PS_PATH, exp, model, member) + ".zarr",
        mode="w",
        consolidated=True,
    )


if __name__ == "__main__":
    # python -m tcpips.ps_driver
    # ps_cmip6_part(exp="ssp585", model="CESM2", member="r4i1p1f1")
    # ps_cmip6_part(exp="historical", model="CESM2", member="r4i1p1f1")
    def run_loop():
        for model in ["CESM2"]:
            for member in ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"]:
                for exp in ["ssp585", "historical"]:
                    ps_cmip6_part(exp=exp, model=model, member=member)

    dask_cluster_wrapper(run_loop)
