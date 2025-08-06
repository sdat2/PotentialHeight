"""Run potential size calculations on CMIP6 data to make specific subsets (very expensive computation)."""

import os
import xarray as xr
from sithom.time import timeit
from adforce.constants import NEW_ORLEANS, MIAMI, GALVERSTON, HONG_KONG, SHANGHAI, HANOI
from sithom.xr import mon_increase
from tcpips.constants import PI2_PATH, PI3_PATH, PI4_PATH
from tcpips.era5 import get_all_regridded_data
from .utils import qtp2rh
from .ps import parallelized_ps
from .constants import DATA_PATH

# store offsets for points to get data for.

OFFSET_D = {
    "galverston": {"point": GALVERSTON, "lon_offset": 0, "lat_offset": -0.9},
    "miami": {"point": MIAMI, "lon_offset": 0.2, "lat_offset": 0},
    "new_orleans": {"point": NEW_ORLEANS, "lon_offset": 0, "lat_offset": -0.5},
    # these ones are not checked
    "shanghai": {"point": SHANGHAI, "lon_offset": 0.5, "lat_offset": -0.3},
    "hong_kong": {"point": HONG_KONG, "lon_offset": 0.5, "lat_offset": -0.5},
    "hanoi": {"point": HANOI, "lon_offset": 0.5, "lat_offset": -0.1},
}


def ex_data_path(pi_version=2, member: int = 4) -> str:
    """Get the path to the example data.

    Args:
        pi_version (int, optional): pi_version of the pi calculation. Defaults to 2.
        member (int, optional): member number. Defaults to 4.

    Returns:
        str: path to the example data.
    """
    if pi_version == 2:
        return os.path.join(PI2_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.zarr")
    elif pi_version == 3:
        return os.path.join(PI3_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.zarr")
    elif pi_version == 4:
        return os.path.join(PI4_PATH, "ssp585", "CESM2", f"r{member}i1p1f1.zarr")
    else:
        raise ValueError("pi_version must be 2, 3, or 4")


@timeit
def trimmed_cmip6_example(
    pressure_assumption="isothermal", trial=1, pi_version=2, place="new_orleans",
) -> None:
    """Run potential size calculations on CMIP6 data to get Gulf of Mexico data.

    Args:
        pressure_assumption (str, optional): pressure assumption. Defaults to "isopycnal".
        trial (int, optional): trial number. Defaults to 1.
    """
    # print("input cmip6 data", xr.open_dataset(EX_DATA_PATH))
    # select roughly gulf of mexico
    # for some reason using the bounding box method doesn't work
    in_ds = mon_increase(xr.open_dataset(ex_data_path(pi_version=pi_version, member=4)), x_dim='lon', y_dim='lat').isel(
        #lat=slice(215, 245),
        #lon=slice(160, 205),
        time=7,  # slice(0, 12) # just august 2015

    )
    if place in ["new_orleans", "galverston", "miami"]:
        in_ds = in_ds.sel(
            lon=slice(-100, -73),
            lat=slice(17.5, 32.5),
            # method="nearest"
            ).compute()
    elif place in ["shanghai", "hong_kong", "hanoi"]:
        in_ds = in_ds.sel(
            lon=slice(100, 130),
            lat=slice(17.5, 32.5),
            # method="nearest"
        ).compute()
    else:
        raise ValueError("place must be one of new_orleans, galverston, miami, shanghai, hong_kong, hanoi")
    print("trimmed input data", in_ds)
    rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
    in_ds["rh"] = rh
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh", "otl"]]

    if pi_version == 2:
        # get rid of V_reduc accidentally added in for vmax calculation
        in_ds["vmax"] = in_ds["vmax"] / 0.8

    out_ds = parallelized_ps(in_ds, jobs=20)
    name = f"august_cmip6_pi{pi_version}_{pressure_assumption}_trial{trial}.nc"

    if place in ["new_orleans", "galverston", "miami"]:
        name = "gom_" + name
    if place in ["shanghai", "hong_kong", "hanoi"]:
        name = "scs_" + name

    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            name,
        )
    )


@timeit
def new_orleans_10year(
    pi_version: int = 2, pressure_assumption: str = "isothermal"
) -> None:
    """Run potential size calculations on CMIP6 data to get New Orleans data.

    Args:
        pi_version (int, optional): pi_version of the pi calculation. Defaults to 2.
        pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
    """
    # look at some seasonal data for new orleans
    in_ds = xr.open_dataset(ex_data_path(pi_version=pi_version)).isel(
        time=slice(0, 120)
    )
    in_ds = in_ds.sel(
        lon=OFFSET_D["new_orleans"]["point"].lon
        + OFFSET_D["new_orleans"]["lon_offset"],
        lat=OFFSET_D["new_orleans"]["point"].lat
        + OFFSET_D["new_orleans"]["lat_offset"],
        method="nearest",
    )
    print(in_ds)
    rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
    in_ds["rh"] = rh
    qt_ds = in_ds[["q", "t", "otl"]]
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh"]]
    if pi_version == 2:
        # get rid of V_reduc accidentally added in for vmax calculation
        in_ds["vmax"] = in_ds["vmax"] / 0.8
    out_ds = parallelized_ps(in_ds, jobs=20)
    out_ds["q"] = qt_ds["q"]
    out_ds["t"] = qt_ds["t"]
    out_ds["otl"] = qt_ds["otl"]
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH, f"new_orleans_10year_pi{pi_version}_{pressure_assumption}new.nc"
        )
    )


@timeit
def global_cmip6(part="nw", year: int = 2015, pi_version=2) -> None:
    """Run potential size calculations on CMIP6 data to make specific subsets.
    Args:
        part (str, optional): segment of the world to calculate. Defaults to "nw".
        year (int, optional): year to calculate. Defaults to 2015.
        pi_version (int, optional): pi_version of the data. Defaults to 2.
    """
    # just get North Hemisphere August
    # and South Hemisphere February
    year_offset = (year - 2015) * 12  # number of months since 2015
    southern_hemisphere_month = 1
    northern_hemisphere_month = 7
    full_ds = xr.open_dataset(ex_data_path(pi_version=pi_version, member=4))
    if part == "nw":
        in_ds = full_ds.isel(
            time=northern_hemisphere_month + year_offset,
            lat=slice(180, 300),
            lon=slice(0, 360),
        )
    elif part == "ne":
        in_ds = full_ds.isel(
            time=northern_hemisphere_month + year_offset,
            lat=slice(180, 300),
            lon=slice(360, 720),
        )
    elif part == "sw":  # just February
        in_ds = full_ds.isel(
            time=southern_hemisphere_month + year_offset,
            lat=slice(0, 180),
            lon=slice(0, 360),
        )
    elif part == "se":
        in_ds = full_ds.isel(
            time=southern_hemisphere_month + year_offset,
            lat=slice(0, 180),
            lon=slice(360, 720),
        )
    else:
        raise ValueError("part must be one of nw, ne, sw, se")
    print("input cmip6 data", in_ds)

    rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
    in_ds["rh"] = rh
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh", "otl"]]
    print(in_ds)
    if pi_version == 2:
        # get rid of V_reduc accidentally added in for vmax calculation
        in_ds["vmax"] = in_ds["vmax"] / 0.8
    out_ds = parallelized_ps(in_ds, jobs=30)
    print(out_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH, f"potential_size_global_{part}_{year}_pi{pi_version}new.nc"
        )
    )


def load_global(year: int = 2015) -> xr.Dataset:
    """Load all parts of the global data and stitch together.

    Args:
        year (int, optional): year to load. Defaults to 2015.

    Returns:
        xr.Dataset: stitched together dataset.
    """
    return xr.concat(
        [
            xr.concat(
                [
                    os.path.join(DATA_PATH, f"potential_size_global_{x}_{year}new.nc")
                    for x in y
                ],
                dim="lon",
            )
            for y in [["sw", "se"], ["nw", "ne"]]
        ],
        dim="lat",
    )


def point_timeseries(
    member: int = 10,
    exp: str = "ssp585",
    place: str = "new_orleans",
    pressure_assumption="isothermal",
    pi_version=3,
) -> None:
    """
    Point timeseries.

    Args:
        member (int, optional): member number. Defaults to 10.
        place (str, optional): location. Defaults to "new_orleans".
        pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
    """
    if pi_version == 2:
        file_name = os.path.join(PI2_PATH, exp, "CESM2", f"r{member}i1p1f1.zarr")
    elif pi_version == 3:
        file_name = os.path.join(PI3_PATH, exp, "CESM2", f"r{member}i1p1f1.zarr")
    elif pi_version == 4:
        file_name = os.path.join(PI4_PATH, exp, "CESM2", f"r{member}i1p1f1.zarr")
    else:
        raise ValueError("pi_version must be 2, 3, or 4")
    point_ds = xr.open_zarr(file_name).sel(
        lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
        lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
        method="nearest",
    )
    print("point ds", point_ds)
    rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
    trimmed_ds = point_ds[["sst", "msl", "vmax", "t0"]]
    trimmed_ds["rh"] = rh
    if pi_version == 2:
        # accidentally added in V_reduc for vmax calculation before
        trimmed_ds["vmax"] = trimmed_ds["vmax"] / 0.8

    print("trimmed", trimmed_ds)
    # select august data from every year of timeseries xarray
    trimmed_ds = trimmed_ds.isel(
        time=[i for i in range(7, len(trimmed_ds.time.values), 12)]
    )
    out_ds = parallelized_ps(trimmed_ds, jobs=25)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            f"{place}_august_{exp}_r{member}i1p1f1_{pressure_assumption}_pi{pi_version}new.nc",
        )
    )

def point_era5_timeseries(
    place: str = "new_orleans",
    pressure_assumption: str = "isothermal",
) -> None:
    """Point timeseries for ERA5 data.

    Args:
        member (int, optional): member number. Defaults to 10.
        place (str, optional): location. Defaults to "new_orleans".
        pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
    """
    point_ds = get_all_regridded_data(start_year=1940, end_year=2024,
    ).sel(
        lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
        lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"] -0.5,
        method="nearest",
    ).compute()
    print("point ds era5", point_ds)
    point_ds = point_ds.rename({"pressure_level": "p"})
    print("point ds", point_ds)
    if point_ds["t"].max().values > 100:
        point_ds["t"].attrs["units"] = "K"
    elif point_ds["t"].max().values <= 100:
        point_ds["t"].attrs["units"] = "degC"
    point_ds["t"].attrs["units"] = "K"
    if point_ds["q"].max().values > 10:
        point_ds["q"].attrs["units"] = "g/kg"
    elif point_ds["q"].max().values <= 1:
        point_ds["q"].attrs["units"] = "kg/kg"

    rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
    trimmed_ds = point_ds[["sst", "msl", "vmax", "t0"]]
    trimmed_ds["rh"] = rh

    print("trimmed_ds", trimmed_ds)
    # select august data from every year of timeseries xarray
    trimmed_ds = trimmed_ds.isel(
        time=[i for i in range(7, len(trimmed_ds.time.values), 12)]
    )
    out_ds = parallelized_ps(trimmed_ds, jobs=10)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            f"{place}_august_era5_{pressure_assumption}_pi4new.nc",
        )
    )


if __name__ == "__main__":
    # python -m w22.ps_runs
    print("Running as main")
    # point_timeseries(4, "new_orleans", pi_version=4)
    # point_timeseries(10, "new_orleans", pi_version=4)
    # point_timeseries(11, "new_orleans", pi_version=4)

    def data_for_place(place: str, pressure_assumption: str = "isothermal") -> None:
        """Run point timeseries for a specific place.

        Args:
            place (str): place to run for.
            pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
        """
        print(f"Running point timeseries for {place} with pressure assumption {pressure_assumption}")
        for i in [4, 10, 11]:
            point_timeseries(
                member=i,
                place=place,
                pressure_assumption=pressure_assumption,
                pi_version=4,
            )
        point_era5_timeseries(
            place=place, pressure_assumption=pressure_assumption
        )

    # data_for_place("hong_kong")
    # trimmed_cmip6_example(
    #     pressure_assumption="isothermal", trial=1, pi_version=4, place="new_orleans"
    # )

    trimmed_cmip6_example( trial=1, pi_version=4, place="hong_kong"
    )

    # for i in [4, 10, 11]:
    #     for exp in ["historical", "ssp585"]:
    #         point_timeseries(member=i, place="new_orleans", pi_version=4, exp=exp)
    # point_timeseries(member=i, place="new_orleans", pi_version=4, exp="historical")
    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc('isopycnal', 1)"
    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc('isothermal', 1)"
    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc('isopycnal', 2)"
    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc('isothermal', 2)"
    # trimmed_cmip6_example()
    # new_orleans_10year()
    # global_cmip6()
    # python -c "from cle.ps_runs import galverston_timeseries as gt; gt(4); gt(10); gt(11)"
    # python -c "from cle.ps_runs import new_orleans_timeseries as no; no(4); no(10); no(11)"
    # python -c "from cle.ps_runs import miami_timeseries as mm; mm(4); mm(10); mm(11)"
    # trimmed_cmip6_example()
    # point_timeseries(4, "new_orleans")
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, "new_orleans"); pt(10, "new_orleans"); pt(11, "new_orleans")"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'miami'); pt(10, 'miami'); pt(11, 'miami')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'galverston'); pt(10, 'galverston'); pt(11, 'galverston')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(10, 'new_orleans'); pt(11, 'new_orleans')"
    #
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'shanghai'); pt(10, 'shanghai'); pt(11, 'shanghai')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'hanoi'); pt(10, 'hanoi'); pt(11, 'hanoi')"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'hong_kong'); pt(10, 'hong_kong'); pt(11, 'hong_kong')"
    # retry with pi_version = 4 for hong_kong only
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'hong_kong', pi_version=4); pt(10, 'hong_kong', pi_version=4); pt(11, 'hong_kong', pi_version=4)"
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'shanghai', pi_version=4); pt(10, 'shanghai', pi_version=4); pt(11, 'shanghai', pi_version=4)"
    # lets do it for New Orleans too
    # python -c "from w22.ps_runs import point_timeseries as pt; pt(4, 'new_orleans', pi_version=4); pt(10, 'new_orleans', pi_version=4); pt(11, 'new_orleans', pi_version=4)"

    # set off global
    # python -c "from w22.ps_runs import global_cmip6 as ga; ga()"
    # python -c "from w22.ps_runs import global_cmip6 as ga; ga('ne')" &> ne2.log
    # python -c "from w22.ps_runs import global_cmip6 as ga; ga('nw')" &> nw2.log
    # python -c "from w22.ps_runs import global_cmip6 as ga; ga('sw')" &> sw2.log
    # python -c "from w22.ps_runs import global_cmip6 as ga; ga('se')" &> se2.log

    # python -c "from w22.ps_runs import trimmed_cmip6_example as tc; tc()
    # python -c "from w22.ps_runs import new_orleans_10year as no; no()"
    # trimmed_cmip6_example(trial=2, pressure_assumption="isothermal", pi_version=4)
