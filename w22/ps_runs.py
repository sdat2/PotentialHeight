"""Run potential size calculations on CMIP6 data to make specific subsets (very expensive computation)."""

import os
import xarray as xr
from sithom.time import timeit
from adforce.constants import NEW_ORLEANS, MIAMI, GALVERSTON
from tcpips.constants import PI2_PATH
from .utils import qtp2rh
from .ps import parallelized_ps
from .constants import DATA_PATH

# store offsets for points to get data for.

OFFSET_D = {
    "galverston": {"point": GALVERSTON, "lon_offset": 0, "lat_offset": -0.9},
    "miami": {"point": MIAMI, "lon_offset": 0.2, "lat_offset": 0},
    "new_orleans": {"point": NEW_ORLEANS, "lon_offset": 0, "lat_offset": -0.5},
}
EX_DATA_PATH = str(
    "/work/n02/n02/sdat2/adcirc-swan/worstsurge/data/cmip6/pi/ssp585/CESM2/r10i1p1f1.nc"
)


@timeit
def trimmed_cmip6_example() -> None:
    in_ds = xr.open_dataset(EX_DATA_PATH).isel(
        y=slice(215, 245), x=slice(160, 205), time=slice(0, 12)
    )
    print(in_ds)
    rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
    in_ds["rh"] = rh
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh"]]
    # get rid of V_reduc accidentally added in for vmax calculation
    in_ds["vmax"] = in_ds["vmax"] / 0.8
    out_ds = parallelized_ps(in_ds, jobs=30)
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, "example_potential_size_output_small_year.nc")
    )


@timeit
def global_august_cmip6_example() -> None:
    in_ds = xr.open_dataset(EX_DATA_PATH).isel(time=7)
    rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
    in_ds["rh"] = rh
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh"]]
    print(in_ds)
    # get rid of V_reduc accidentally added in for vmax calculation
    in_ds["vmax"] = in_ds["vmax"] / 0.8
    out_ds = parallelized_ps(in_ds, jobs=30)
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, "example_potential_size_output_august_2015.nc")
    )


def point_timeseries(member=10, place="new_orleans") -> None:
    file_name = os.path.join(PI2_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.nc")
    point_ds = xr.open_dataset(file_name).sel(
        lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
        lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
        method="nearest",
    )
    rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
    trimmed_ds = point_ds[["sst", "msl", "vmax", "t0"]]
    trimmed_ds["rh"] = rh
    # accidentally added in V_reduc for vmax calculation before
    trimmed_ds["vmax"] = trimmed_ds["vmax"] / 0.8

    print("trimmed", trimmed_ds)
    # select august data from every year of timeseries xarray
    trimmed_ds = trimmed_ds.isel(
        time=[i for i in range(7, len(trimmed_ds.time.values), 12)]
    )
    out_ds = parallelized_ps(trimmed_ds, jobs=25)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, f"{place}_august_ssp585_r{member}i1p1f1.nc")
    )


if __name__ == "__main__":
    # python -m cle.ps_runs
    # trimmed_cmip6_example()
    # global_august_cmip6_example()
    # python -c "from cle.ps_runs import galverston_timeseries as gt; gt(4); gt(10); gt(11)"
    # python -c "from cle.ps_runs import new_orleans_timeseries as no; no(4); no(10); no(11)"
    # python -c "from cle.ps_runs import miami_timeseries as mm; mm(4); mm(10); mm(11)"
    # trimmed_cmip6_example()
    point_timeseries(4, "new_orleans")
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4. "new_orleans"); pt(10, "new_orleans"); pt(11, "new_orleans")"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'miami'); pt(10, 'miami'); pt(11, 'miami')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'galverston'); pt(10, 'galverston'); pt(11, 'galverston')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(10, 'new_orleans'); pt(11, 'new_orleans')"
    # set off global
    # python -c "from w22.ps_runs import global_august_cmip6_example as ga; ga()"
    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc()"
