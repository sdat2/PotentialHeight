import os
import xarray as xr
from sithom.time import timeit
from adforce.constants import NEW_ORLEANS, MIAMI, GALVERSTON
from .utils import qtp2rh
from .ps import parallelized_ps
from .constants import DATA_PATH, PI2_PATH


place_d = {
    "galverston": {"point": GALVERSTON, "lon_offset": 0, "lat_offset": -0.9},
    "miami": {"point": MIAMI, "lon_offset": 0.2, "lat_offset": 0},
    "new_orleans": {"point": NEW_ORLEANS, "lon_offset": 0, "lat_offset": -0.5},
}


@timeit
def trimmed_cmip6_example() -> None:
    ex_data_path = str(
        "/work/n02/n02/sdat2/adcirc-swan/worstsurge/data/cmip6/pi/ssp585/CESM2/r10i1p1f1.nc"
    )
    in_ds = xr.open_dataset(ex_data_path)[["sst", "msl", "vmax", "t0"]].isel(
        y=slice(215, 245), x=slice(160, 205)
    )
    out_ds = parallelized_ps(in_ds)
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, "example_potential_size_output_small_year.nc")
    )
    print(in_ds)


@timeit
def global_august_cmip6_example() -> None:
    ex_data_path = str(
        "/work/n02/n02/sdat2/adcirc-swan/worstsurge/data/cmip6/pi/ssp585/CESM2/r10i1p1f1.nc"
    )
    # /work/n02/n02/sdat2/adcirc-swan/worstsurge/data/cmip6/pi2/ssp585/CESM2/r10i1p1f1.nc
    in_ds = xr.open_dataset(ex_data_path)[["sst", "msl", "vmax", "t0"]].isel(time=7)
    out_ds = parallelized_ps(in_ds, jobs=20)
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, "example_potential_size_output_august_2015.nc")
    )
    print(in_ds)


def new_orleans_timeseries(member=10):

    file_name = os.path.join(PI2_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.nc")
    point_ds = xr.open_dataset(file_name).sel(
        lon=NEW_ORLEANS.lon, lat=NEW_ORLEANS.lat - 0.5, method="nearest"
    )
    print("point_ds", point_ds)
    rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
    trimmed_ds = point_ds[["sst", "msl", "vmax", "t0"]]
    trimmed_ds["rh"] = rh
    point_timeseries_august = trimmed_ds.isel(
        time=[time.month == 8 for time in point_ds.time.values]
    )
    out_ds = parallelized_ps(point_timeseries_august, jobs=25)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, f"new_orleans_august_ssp585_r{member}i1p1f1.nc")
    )


def miami_timeseries(member=10):

    file_name = os.path.join(PI2_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.nc")
    ds = xr.open_dataset(file_name)[["sst", "msl", "vmax", "t0"]]
    ds = ds.sel(lon=MIAMI.lon + 0.2, lat=MIAMI.lat, method="nearest").isel(
        time=[t.month == 8 for t in ds.time.values]
    )
    out_ds = parallelized_ps(ds, jobs=10)
    out_ds.to_netcdf(os.path.join(DATA_PATH, f"miami_august_ssp585_r{member}i1p1f1.nc"))


def galverston_timeseries(member: int = 10):
    file_name = os.path.join(PI2_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.nc")
    ds = xr.open_dataset(file_name)[["sst", "msl", "vmax", "t0"]]
    ds = ds.sel(lon=GALVERSTON.lon, lat=GALVERSTON.lat - 0.9, method="nearest").isel(
        time=[t.month == 8 for t in ds.time.values]
    )
    out_ds = parallelized_ps(ds, jobs=10)
    out_ds.to_netcdf(
        os.path.join(DATA_PATH, f"galverston_august_ssp585_r{member}i1p1f1.nc")
    )


if __name__ == "__main__":
    # python -m cle.ps_runs
    # trimmed_cmip6_example()
    # global_august_cmip6_example()
    # python -c "from cle.ps_runs import galverston_timeseries as gt; gt(4); gt(10); gt(11)"
    # python -c "from cle.ps_runs import new_orleans_timeseries as no; no(4); no(10); no(11)"
    # python -c "from cle.ps_runs import miami_timeseries as mm; mm(4); mm(10); mm(11)"
    trimmed_cmip6_example()
