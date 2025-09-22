"""Run potential size calculations on CMIP6 data to make specific subsets (very expensive computation)."""

import os
from typing import Union
import numpy as np
import xarray as xr
from sithom.time import timeit
from sithom.xr import mon_increase
from tcpips.constants import PI2_PATH, PI3_PATH, PI4_PATH, CDO_PATH
from tcpips.pi import calculate_pi
from tcpips.convert import convert
from tcpips.era5 import get_all_regridded_data
from .utils import qtp2rh
from .ps import parallelized_ps13_dask
from .constants import DATA_PATH, OFFSET_D

CAT_1_WIND_SPEED = (
    33 / 0.8
)  # m/s at 10m, divided by 0.8 to convert to gradient wind speed

# store offsets for points to get data for.


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


def get_regional_processed_data(place="new_orleans") -> xr.Dataset:
    """Get processed data for a specific region.

    Args:
        place (str, optional): place to get data for. Defaults to "new_orleans
    """

    ds_list = [
        xr.open_dataset(os.path.join(CDO_PATH, "ssp585", typ, "CESM2", "r4i1p1f1.nc"))
        for typ in ["ocean", "atmos"]
    ]
    for i, ds in enumerate(ds_list):
        ds_list[i] = ds.drop_vars([x for x in ["time_bounds"] if x in ds])
    ds = xr.merge(ds_list)
    ds = convert(ds)
    if place in ["new_orleans", "galverston", "miami"]:
        ds = ds.sel(
            lon=slice(-100, -73),
            lat=slice(17.5, 32.5),
            time="2015-08",
            # method="nearest"
        ).compute()
    elif place in ["shanghai", "hong_kong", "hanoi"]:
        ds = ds.sel(
            lon=slice(100, 130),
            lat=slice(17.5, 32.5),
            time="2015-08",
            # method="nearest"
        ).compute()
    else:
        raise ValueError(
            "place must be one of new_orleans, galverston, miami, shanghai, hong_kong, hanoi"
        )
    return ds


@timeit
def spatial_example(
    pressure_assumption="isothermal",
    model="CESM2",
    trial=1,
    pi_version=2,
    member: str = "r4i1p1f1",
    recalculate_pi: bool = True,
    place="new_orleans",
) -> None:
    """Run potential size calculations on CMIP6 data to get a region.

    Args:
        pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
        model (str, optional): model name. Defaults to "CESM2".
        member (str, optional): member number. Defaults to "r4i1p1f1".
        trial (int, optional): trial number. Defaults to 1.
        pi_version (int, optional): pi_version of the pi calculation. Defaults to 2.
        recalculate_pi (bool, optional): whether to recalculate pi. Defaults to True.
        place (str, optional): place to get data for. Defaults to "new_orleans
    """
    # print("input cmip6 data", xr.open_dataset(EX_DATA_PATH))
    # select roughly gulf of mexico
    # for some reason using the bounding box method doesn't work

    def select_region(tds: xr.Dataset, place) -> xr.Dataset:
        """Select region of interest.

        Args:
            in_ds (xr.Dataset): input dataset.

        Returns:
            xr.Dataset: dataset for region of interest.
        """
        if place in ["new_orleans", "galverston", "miami"]:
            tds = tds.sel(
                lon=slice(-100, -73),
                lat=slice(17.5, 32.5),
                time="2015-08",
                # method="nearest"
            )
        elif place in ["shanghai", "hong_kong", "hanoi"]:
            tds = tds.sel(
                lon=slice(100, 130),
                lat=slice(17.5, 32.5),
                time="2015-08",
                # method="nearest"
            )
        else:
            raise ValueError(
                "place must be one of new_orleans, galverston, miami, shanghai, hong_kong, hanoi"
            )
        return tds

    if recalculate_pi:
        # get the data from the example path
        ds_list = [
            xr.open_dataset(
                os.path.join(CDO_PATH, "ssp585", typ, model, member + ".nc"),
                chunks={"time": 24, "lat": 90, "lon": 90},
            )
            for typ in ["ocean", "atmos"]
        ]
        for i, ds in enumerate(ds_list):
            ds_list[i] = ds.drop_vars([x for x in ["time_bounds"] if x in ds])
        print("ds_list", ds_list)
        ds = mon_increase(
            xr.merge(ds_list),
            x_dim="lon",
            y_dim="lat",
        )
        print("merged ds", ds)
        ds = select_region(ds, place)
        print("selected region ds", ds)
        ds = convert(ds).chunk(dict(p=-1))
        print("converted ds", ds)
        if "rh" not in ds:
            rh = qtp2rh(ds["q"], ds["t"], ds["msl"])
            ds["rh"] = rh * 100
        ds_pi = calculate_pi(
            ds,
            dim="p",
            fix_temp=True,
        )
        ds["vmax"] = ds_pi["vmax"]
        ds["t0"] = ds_pi["t0"]
        if ds["rh"].max().values > 1:
            ds["rh"] = ds["rh"] / 100  # convert to dimensionless
        ds["rh"].attrs["units"] = "dimensionless"
        in_ds = ds
    else:
        ds = xr.open_dataset(ex_data_path(pi_version=pi_version))
        in_ds = select_region(
            mon_increase(
                ds,
                x_dim="lon",
                y_dim="lat",
            ),
            place,
        )
        if pi_version == 2:
            # get rid of V_reduc accidentally added in for vmax calculation
            in_ds["vmax"] = in_ds["vmax"] / 0.8

        if "rh" not in in_ds:
            rh = qtp2rh(in_ds["q"], in_ds["t"], in_ds["msl"])
            in_ds["rh"] = rh
        print("input cmip6 data", in_ds)

    print("trimmed input data", in_ds)
    in_ds = in_ds[["sst", "msl", "vmax", "t0", "rh"]]

    in_ds = in_ds.rename({"vmax": "vmax_3"})
    in_ds["vmax_1"] = (
        in_ds["vmax_3"].dims,
        CAT_1_WIND_SPEED * np.ones_like(in_ds["vmax_3"]),
    )
    in_ds = in_ds.chunk(
        {"lat": 2, "lon": 2}
    )  # make chunks smaller to improve parallelization over many cores
    print("input ds for ps calculation", in_ds)
    out_ds = parallelized_ps13_dask(in_ds)
    print("output ds from ps calculation", out_ds)

    name = f"august_cmip6_{model}_{member}_pi{pi_version}_{pressure_assumption}_trial{trial}.nc"

    if place in ["new_orleans", "galverston", "miami"]:
        name = "gom_" + name
    if place in ["shanghai", "hong_kong", "hanoi"]:
        name = "scs_" + name

    print("final output before saving", out_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            name,
        ),
        engine="h5netcdf",
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

    in_ds = in_ds.rename({"vmax": "vmax_3"})
    in_ds["vmax_1"] = (
        in_ds["vmax_3"].dims,
        CAT_1_WIND_SPEED * np.ones_like(in_ds["vmax_3"].values),
    )
    out_ds = parallelized_ps13_dask(in_ds)
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
    in_ds = in_ds.rename({"vmax": "vmax_3"})
    in_ds["vmax_1"] = (
        in_ds["vmax_3"].dims,
        CAT_1_WIND_SPEED * np.ones_like(in_ds["vmax_3"]),
    )
    out_ds = parallelized_ps13_dask(in_ds)
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


def get_processed_data_for_point(
    place: str = "new_orleans",
    member: str = "r10i1p1f1",
    model: str = "CESM2",
    exp: str = "ssp585",
):
    """Get processed data for a specific point.

    Args:
        place (str, optional): place to get data for. Defaults to "new_orleans".
        member (str, optional): member number. Defaults to "r10i1p1f1".
        model (str, optional): model name. Defaults to "CESM2".
        exp (str, optional): experiment name. Defaults to "ssp585".
    """
    file_names = [
        os.path.join(CDO_PATH, exp, typ, model, f"{member}.nc")
        for typ in ["ocean", "atmos"]
    ]

    ds_list = [
        xr.open_mfdataset(
            file_name,
        )
        .sel(
            lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
            lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
            method="nearest",
        )
        .compute()
        for file_name in file_names
    ]
    for i, ds in enumerate(ds_list):
        ds_list[i] = ds.drop_vars([x for x in ["time_bounds"] if x in ds])

    ds = xr.merge(ds_list)
    ds = convert(ds)
    ds = ds.isel(time=[i for i in range(8, len(ds.time.values), 12)])
    return ds


def point_timeseries(
    member: Union[int, str] = 10,
    model: str = "CESM2",
    recalculate_pi: bool = True,
    exp: str = "ssp585",
    place: str = "new_orleans",
    pressure_assumption="isothermal",
    pi_version=3,
) -> None:
    """
    Point timeseries.

    Args:
        member (int, optional): member number. Defaults to 10.
        model (str, optional): model name. Defaults to "CESM2".
        recalculate_pi (bool, optional): whether to recalculate pi. Defaults to True.
        exp (str, optional): experiment name. Defaults to "ssp585".
        pi_version (int, optional): pi version. Defaults to 3.xxeexexe
        place (str, optional): location. Defaults to "new_orleans".
        pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
    """
    if isinstance(member, int):
        member = f"r{member}i1p1f1"
    if recalculate_pi:
        ds = get_processed_data_for_point(
            place=place, member=member, model=model, exp=exp
        )
        pi_ds = calculate_pi(
            ds,
            dim="p",
            fix_temp=True,
        )
        print("pi ds", pi_ds)
        trimmed_ds = pi_ds[["vmax", "t0"]]
        trimmed_ds["rh"] = ds["rh"] / 100  # convert to dimensionless
        trimmed_ds["rh"].attrs["units"] = "dimensionless"
        trimmed_ds["sst"] = ds["sst"]
        trimmed_ds["msl"] = ds["msl"]

    else:
        if pi_version == 2:
            file_name = os.path.join(PI2_PATH, exp, model, f"{member}.zarr")
        elif pi_version == 3:
            file_name = os.path.join(PI3_PATH, exp, model, f"{member}.zarr")
        elif pi_version == 4:
            file_name = os.path.join(PI4_PATH, exp, model, f"{member}.zarr")
        else:
            raise ValueError("pi_version must be 2, 3, or 4")
        point_ds = (
            xr.open_zarr(file_name)
            .sel(
                lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
                lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
                method="nearest",
            )
            .compute()
        )
        trimmed_ds = trimmed_ds.isel(
            time=[i for i in range(8, len(trimmed_ds.time.values), 12)]
        )
        print("point ds", point_ds)
        if "rh" not in point_ds:
            point_ds["rh"] = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
            rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
            trimmed_ds["rh"] = rh
            trimmed_ds = point_ds[["sst", "msl", "vmax", "t0", "rh"]]

        if pi_version == 2:
            # accidentally added in V_reduc for vmax calculation before
            trimmed_ds["vmax"] = trimmed_ds["vmax"] / 0.8

    print("trimmed", trimmed_ds)
    trimmed_ds = trimmed_ds.rename({"vmax": "vmax_3"})
    trimmed_ds["vmax_1"] = (
        trimmed_ds["vmax_3"].dims,
        CAT_1_WIND_SPEED * np.ones_like(trimmed_ds["vmax_3"]),
    )

    out_ds = parallelized_ps13_dask(trimmed_ds)
    print("out_ds", out_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            f"{place}_august_{exp}_{model}_{member}_{pressure_assumption}_pi{pi_version}new.nc",
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
    point_ds = (
        get_all_regridded_data(
            start_year=1940,
            end_year=2024,
        )
        .sel(
            lon=OFFSET_D[place]["point"].lon + OFFSET_D[place]["lon_offset"],
            lat=OFFSET_D[place]["point"].lat + OFFSET_D[place]["lat_offset"],
            method="nearest",
        )
        .compute()
    )
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

    if "rh" not in point_ds:
        rh = qtp2rh(point_ds["q"], point_ds["t"], point_ds["msl"])
        point_ds["rh"] = rh

    print("trimmed_ds", trimmed_ds)
    # select august data from every year of timeseries xarray
    trimmed_ds = trimmed_ds.isel(
        time=[i for i in range(7, len(trimmed_ds.time.values), 12)]
    )
    trimmed_ds = trimmed_ds.rename({"vmax": "vmax_3"})
    trimmed_ds["vmax_1"] = (
        out_ds["vmax_3"].dims,
        CAT_1_WIND_SPEED * np.ones_like(trimmed_ds["vmax_3"]),
    )
    out_ds = parallelized_ps13_dask(trimmed_ds)
    out_ds.to_netcdf(
        os.path.join(
            DATA_PATH,
            f"{place}_august_era5_{pressure_assumption}_pi4new.nc",
        )
    )


if __name__ == "__main__":
    # python -m w22.ps_runs
    # point_timeseries(4, "new_orleans", pi_version=4)
    # point_timeseries(10, "new_orleans", pi_version=4)
    # point_timeseries(11, "new_orleans", pi_version=4)

    def ps_for_place(
        place: str,
        pressure_assumption: str = "isothermal",
        models={"CESM2", "MIROC6", "HADGEM3-GC31-MM", "ERA5"},
    ) -> None:
        """Run point timeseries for a specific place.

        Args:
            place (str): place to run for.
            pressure_assumption (str, optional): pressure assumption. Defaults to "isothermal".
        """
        print(
            f"Running point timeseries for {place} with pressure assumption {pressure_assumption}"
        )
        # CESM2
        for exp in ["historical", "ssp585"]:
            if "CESM2" in models:
                for i in [4, 10, 11]:
                    point_timeseries(
                        member=i,
                        place=place,
                        pressure_assumption=pressure_assumption,
                        pi_version=4,
                        exp=exp,
                        model="CESM2",
                        recalculate_pi=True,
                    )
            # HadGEM3-GC31-MM
            if "HADGEM3-GC31-MM" in models:
                for i in [1, 2, 3]:
                    point_timeseries(
                        member="r" + str(i) + "i1p1f3",
                        place=place,
                        pressure_assumption=pressure_assumption,
                        model="HADGEM3-GC31-MM",
                        pi_version=4,
                        exp=exp,
                        recalculate_pi=True,
                    )
            # MIROC6
            if "MIROC6" in models:
                for i in [1, 2, 3]:
                    point_timeseries(
                        member="r" + str(i) + "i1p1f1",
                        place=place,
                        pressure_assumption=pressure_assumption,
                        model="MIROC6",
                        pi_version=4,
                        exp=exp,
                        recalculate_pi=True,
                    )
        if "ERA5" in models:
            point_era5_timeseries(place=place, pressure_assumption=pressure_assumption)
        if "CESM2" in models:
            spatial_example(
                pressure_assumption=pressure_assumption,
                trial=1,
                pi_version=4,
                place=place,
            )

    from tcpips.dask_utils import dask_cluster_wrapper

    # dask_cluster_wrapper(ps_for_place, "new_orleans")
    # dask_cluster_wrapper(ps_for_place, "hong_kong")
    # dask_cluster_wrapper(
    #     spatial_example,
    #     place="new_orleans",
    #     pressure_assumption="isothermal",
    #     trial=1,
    #     pi_version=4,
    #     model="MIROC6",
    #     member="r1i1p1f1",
    # )
    # dask_cluster_wrapper(
    #     spatial_example,
    #     place="hong_kong",
    #     pressure_assumption="isothermal",
    #     trial=1,
    #     pi_version=4,
    #     model="HADGEM3-GC31-MM",
    #     member="r1i1p1f3",
    # )
    dask_cluster_wrapper(
        spatial_example,
        place="new_orleans",
        pressure_assumption="isothermal",
        trial=1,
        pi_version=4,
        model="HADGEM3-GC31-MM",
        member="r1i1p1f3",
    )
    # spatial_example(
    #     pressure_assumption="isothermal",
    #     trial=1,
    #     pi_version=4,
    #     place="hong_kong",
    # )
