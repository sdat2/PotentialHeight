"""Regrid data."""

import os
import xarray as xr
import xesmf as xe
from matplotlib import pyplot as plt
from sithom.plot import feature_grid, plot_defaults, label_subplots
from sithom.time import timeit
from tcpips.constants import FIGURE_PATH, CMIP6_PATH
from tcpips.convert import conversion_names


@timeit
def regrid_2d_1degree(output_res: float = 1.0, time_chunk: int = 10) -> None:
    """
    Regrid 2d data to 1 degree resolution.

    Args:
        output_res (float, optional): Resolution of the output grid. Defaults to 1.0.
        time_chunk (int, optional): Chunk size for time. Defaults to 10.
    """
    plot_defaults()

    def open_ds(path: str) -> xr.Dataset:
        """
        Open dataset.

        Args:
            path (str): path to the dataset.

        Returns:
            xr.Dataset: xarray dataset.
        """
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
        Regrid and save the input dataset to the output.

        Args:
            input_ds (xr.Dataset): dataset to regrid.
            output_name (str): of the output file.

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
    """
    Regrid 2d data.
    """
    plot_defaults()

    def open_ds(name: str) -> xr.Dataset:
        ds = xr.open_dataset(name)
        return ds.drop_vars(
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
        )  # return ds

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
    """
    Regrid 1d data.

    Args:
        xesmf (bool, optional): Defaults to False.
    """

    def open_1d(path: str):
        ds = xr.open_dataset(path)
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
