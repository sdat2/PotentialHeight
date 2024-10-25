"""Fort.22.nc file reader and writer for NWS=13 format (netCDF):

https://wiki.adcirc.org/NWS13

TODO: Distance calculations still assume Flat Platee Caree Earth.

TODO: Tropical cyclone is not yet allowed to evolve with time.

TODO: Could do with some unit tests.

TODO: is 1015 mb as normal too high?
"""

from typing import Optional, Tuple, Callable
import os
import numpy as np
import xarray as xr
import datatree as dt
from netCDF4 import Dataset
from tcpips.constants import DATA_PATH, FIGURE_PATH
from cle.constants import DATA_PATH as CLE_DATA_PATH
from sithom.time import timeit
from sithom.plot import plot_defaults
from .profile import read_profile


def read_fort22(fort22_path: Optional[str] = None) -> xr.Dataset:
    """Read fort.22.nc file.

    Args:
        fort22_path (Optional[str], optional): Path to fort.22.nc file. Defaults to None. If not provided, the default path will be used.

    Returns:
        xr.Dataset: Dataset containing fort.22.nc data.
    """
    if fort22_path is None:
        fort22_path = os.path.join(DATA_PATH, "fort.22.nc")
    return dt.open_datatree(fort22_path)


def trim_fort22() -> None:
    """Trim the fort.22.nc file."""
    fort22 = read_fort22()
    print(fort22)
    f22 = fort22.drop_nodes(["2004223N11301", "2004227N09314"])
    print(fort22.groups)
    print(f22.groups)
    f22["2004217N13306"].name = "TC1"
    f22[""].attrs["group_order"] = "Main TC1"
    f22[""].attrs["institution"] = "Oceanweather Inc. (OWI)"
    f22[""].attrs["conventions"] = "CF-1.6 OWI-NWS13"

    # f22.rename({"2004217N13306": "TC"})
    f22.to_netcdf(os.path.join(DATA_PATH, "test.nc"))
    ds = Dataset(os.path.join(DATA_PATH, "test.nc"))
    print(ds)
    print(ds.groups["TC1"])
    print(ds.groups["Main"])

    ds = Dataset(os.path.join(DATA_PATH, "fort.22.nc"))
    print(ds)
    print(ds.groups["Main"])


@timeit
def blank_fort22() -> None:
    """Create a blank fort.22.nc file."""

    plot_defaults()
    fort22 = read_fort22()
    print(fort22)
    f22 = fort22.drop_nodes(["2004223N11301", "2004227N09314"])
    print(fort22.groups)
    print(f22.groups)
    key = "2004217N13306"
    f22[key]["PSFC"][:] = f22[key]["PSFC"].mean()

    f22["2004217N13306"].name = "TC1"
    f22[""].attrs["group_order"] = "Main TC1"
    f22[""].attrs["institution"] = "Oceanweather Inc. (OWI)"
    f22[""].attrs["conventions"] = "CF-1.6 OWI-NWS13"
    f22["Main"]["PSFC"][:] = f22["Main"]["PSFC"].mean()
    print(f22)
    # f22["TC1"]["PSFC"][:] = f22["TC1"]["PSFC"].mean()
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, sharey=True)
    f22["Main"]["PSFC"].isel(time=0).plot(ax=axs[0], x="lon", y="lat")
    f22[f22.groups[1]].isel(time=0)["PSFC"].plot(ax=axs[1], x="lon", y="lat")
    plt.savefig(os.path.join(FIGURE_PATH, "blank.png"))
    plt.clf()

    # f22.rename({"2004217N13306": "TC"})
    f22.to_netcdf(os.path.join(DATA_PATH, "blank.nc"))
    ds = Dataset(os.path.join(DATA_PATH, "blank.nc"))

    print(ds)
    print(ds.groups["TC1"])
    print(ds.groups["Main"])

    ds = Dataset(os.path.join(DATA_PATH, "fort.22.nc"))
    print(ds)
    print(ds.groups["Main"])


def data_free_f22_coords() -> None:
    f22 = read_fort22(os.path.join(DATA_PATH, "blank.nc"))
    print(f22)
    del f22["Main"]["PSFC"]
    del f22["Main"]["U10"]
    del f22["Main"]["V10"]
    del f22["TC1"]["PSFC"]
    del f22["TC1"]["U10"]
    del f22["TC1"]["V10"]

    f22.to_netcdf(os.path.join(DATA_PATH, "coords.nc"))
    print(f22)


@timeit
def trajectory_ds_from_time(
    angle: float,
    trans_speed: float,
    impact_lon: float,
    impact_lat: float,
    impact_time: np.datetime64,
    time_array: np.datetime64,
) -> xr.Dataset:
    """
    Create a trajectory dataset for the center eye of the tropical cylone.

    Args:
        time_array (np.datetime64): time array for the trajectory.

    Returns:
        xr.Dataset: trajectory dataset with variables lon, lat and time.
    """
    long_angles = (
        (time_array - impact_time) / np.timedelta64(1, "s") * trans_speed
    ) / 111e3
    point = np.array([[impact_lon], [impact_lat]])
    slope = np.array([[np.sin(np.radians(angle))], [np.cos(np.radians(angle))]])
    point_array = (point + slope * long_angles).T
    return xr.Dataset(
        data_vars=dict(
            clon=(["time"], point_array[:, 0]),
            clat=(["time"], point_array[:, 1]),
        ),
        coords=dict(
            time=time_array,
        ),
        attrs=dict(description="Tropical Cyclone trajectory."),
    )


def gen_ps_f(
    profile_path: str = os.path.join(
        CLE_DATA_PATH, "outputs.json"
    )  # "/work/n02/n02/sdat2/adcirc-swan/tcpips/cle/data/outputs.json",
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate the interpolation function from the wind profile (from Chavas et al. 2015).

    Maybe I should also make options for any other azimuthally symetric model automatically.

    This should potentially be changed to allow each timestep to have a different profile.

    That would require some clever way of interpolating each time step.

    TODO: Make it possible to feed a profile file in some way.
    """
    profile = read_profile(profile_path)
    radii = profile["radii"].values
    windspeeds = profile["windspeeds"].values
    pressures = profile["pressures"].values
    # print(radii[0:10], windspeeds[0:10], pressures[0:10])

    def interp_func(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the pressure and wind fields from the wind profile file.

        Args:
            distances (np.ndarray): distances from the center of the storm.

        Returns:
            Tuple[np.ndarray, np.ndarray]: pressure, wind speed.
        """
        pressure_cube = np.interp(distances, radii, pressures)
        velo_cube = np.interp(distances, radii, windspeeds)
        return np.nan_to_num(pressure_cube, nan=pressures[-1]), np.nan_to_num(
            velo_cube, nan=0.0
        )

    return interp_func


@timeit
def moving_coords_from_tj(
    coords: xr.DataArray,
    tj: xr.DataArray,
    profile_path: str = os.path.join(CLE_DATA_PATH, "outputs.json"),
) -> xr.Dataset:
    """
    Make a moving grid from the tropical cyclone trajectory.

    Args:
        coords (xr.DataArray): dataarray to take the lon and lat from.
        tj (xr.DataArray): dataarray with the tropical cyclone trajectory.

    Returns:
        xr.Dataset: Dataset with the moving grid with tropical cyclone surface velocities and pressure. The dataset has the following data variables:
            - clon: center longitude of feature.
            - clat: center latitude of feature.
            - PSFC: Surface pressure [mb].
            - U10: Surface zonal wind speed [m/s].
            - V10: Surface meridional wind speed [m/s].
            The dataset has the following coordinates:
            - lon: Longitude [degrees_east].
            - lat: Latitude [degrees_north].
            - time: Time [minutes since 1990-01-01T01:00:00].
    """
    coords.lon[:] = coords.lon[:] - coords.lon.mean()
    coords.lat[:] = coords.lat[:] - coords.lat.mean()
    clon = tj.clon.values.reshape(-1, 1, 1)
    clat = tj.clat.values.reshape(-1, 1, 1)
    lats = np.expand_dims(coords.lon.values, 0) + clat
    lons = np.expand_dims(coords.lat.values, 0) + clon

    ifunc: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]] = gen_ps_f(
        profile_path=profile_path
    )

    distances = np.sqrt((lons - clon) ** 2 + (lats - clat) ** 2) * 111e3
    psfc, wsp = ifunc(distances)

    rad = np.arctan2(np.radians(lons - clon), np.radians(lats - clat)) - np.pi / 2
    u10, v10 = np.sin(rad) * wsp, np.cos(rad) * wsp

    def transpose(npa: np.ndarray) -> np.ndarray:
        # transpose the first and second axes
        return np.moveaxis(npa, 1, -1)

    # print(lats, lons)
    return xr.Dataset(
        data_vars=dict(
            clon=(
                ["time"],
                tj.clon.values,
                {"units": "degrees_east", "long_name": "center longitude of feature"},
            ),
            clat=(
                ["time"],
                tj.clat.values,
                {"units": "degrees_north", "long_name": "center latitude of feature"},
            ),
            PSFC=(
                [
                    "time",
                    "yi",
                    "xi",
                ],
                transpose(psfc.astype("float32")),
                {"units": "mb"},
            ),
            U10=(
                ["time", "yi", "xi"],
                transpose(u10.astype("float32")),
                {"units": "m s-1"},
            ),
            V10=(
                ["time", "yi", "xi"],
                transpose(v10.astype("float32")),
                {"units": "m s-1"},
            ),
        ),
        coords=dict(
            lon=(
                ["time", "yi", "xi"],
                transpose(lons),
                {"axis": "X", "standard_name": "longitude", "units": "degrees_east"},
            ),
            lat=(
                ["time", "yi", "xi"],
                transpose(lats),
                {"axis": "Y", "standard_name": "latitude", "units": "degrees_north"},
            ),
            time=(
                ["time"],
                tj.time.values,
                {
                    "axis": "T",  # "units": "minutes since 1990-01-01T01:00:00"}
                },
            ),
            # reference_time=self.impact_time,
        ),
        attrs=dict(rank=2, description="Tropical cyclone moving grid."),
    )


@timeit
def static_coords_from_tj(
    orig: xr.DataArray,
    tj: xr.DataArray,
    profile_path: str = os.path.join(CLE_DATA_PATH, "outputs.json"),
) -> xr.Dataset:
    """
    Make a static input grid from the tropical cyclone trajectory.

    Args:
        orig (xr.DataArray): Original dataarray to take the lon and lat from.
        tj (xr.DataArray): dataarray with the tropical cyclone trajectory.

    Returns:
        xr.Dataset: Dataset with the static grid with tropical cyclone surface velocities and pressure.
    """
    lats = np.expand_dims(orig.lat.values, 0)
    lons = np.expand_dims(orig.lon.values, 0)
    clon = tj.clon.values.reshape(-1, 1, 1)
    clat = tj.clat.values.reshape(-1, 1, 1)

    ifunc = gen_ps_f(profile_path=profile_path)

    distances = np.sqrt((lons - clon) ** 2 + (lats - clat) ** 2) * 111e3
    psfc, wsp = ifunc(distances)

    rad = np.arctan2(np.radians(lons - clon), np.radians(lats - clat)) - np.pi / 2
    u10, v10 = np.sin(rad) * wsp, np.cos(rad) * wsp

    # print(lats, lons)
    return xr.Dataset(
        data_vars=dict(
            PSFC=(
                [
                    "time",
                    "yi",
                    "xi",
                ],
                psfc.astype("float32"),
                {"units": "mb"},
            ),
            U10=(["time", "yi", "xi"], u10.astype("float32"), {"units": "m s-1"}),
            V10=(
                ["time", "yi", "xi"],
                v10.astype("float32"),
                {"units": "m s-1"},
            ),
        ),
        coords=dict(
            lon=(
                ["yi", "xi"],
                orig.lon.values,
                {"axis": "X", "standard_name": "longitude", "units": "degrees_east"},
            ),
            lat=(
                ["yi", "xi"],
                orig.lat.values,
                {"axis": "Y", "standard_name": "latitude", "units": "degrees_north"},
            ),
            time=(["time"], tj.time.values, {"axis": "T"}),
            # reference_time=self.impact_time,
        ),
        attrs=dict(rank=1),  # description="Tropical cyclone static grid.",
    )


@timeit
def return_new_input(
    profile_path: str = os.path.join(CLE_DATA_PATH, "outputs.json"),
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -89.4715,  # NO+0.6
    impact_lat: float = 29.9511,  # NO
    impact_time=np.datetime64("2004-08-13T12", "ns"),
) -> dt.DataTree:
    """
    Return a new input file for ADCIRC.

    Args:
        angle (float, optional): Angle of the storm. Defaults
        to 0.
        trans_speed (float, optional): Translation speed of the storm. Defaults to 7.71.
        impact_lon (float, optional): Longitude of the storm. Defaults to -89.4715.
        impact_lat (float, optional): Latitude of the storm. Defaults to 29.9511.
        impact_time ([type], optional): Time of the storm. Defaults to np.datetime64("2004-08-13T12", "ns").

    Returns:
        dt.DataTree: DataTree with the new input file data.
    """
    hard_path: str = os.path.join(DATA_PATH, "blank.nc")
    # f22_dt = dt.open_datatree(os.path.join(DATA_PATH, "blank.nc"))
    f22_dt = dt.open_datatree(hard_path)
    print("f22_dt", f22_dt)

    tj_ds_mv = trajectory_ds_from_time(
        angle,
        trans_speed,
        impact_lon,
        impact_lat,
        impact_time,
        f22_dt["TC1"]["time"].values,
    )
    mc = moving_coords_from_tj(
        f22_dt["TC1"].to_dataset().isel(time=0)[["lon", "lat"]],
        tj_ds_mv,
        profile_path=profile_path,
    )
    tj_ds_sc = trajectory_ds_from_time(
        angle,
        trans_speed,
        impact_lon,
        impact_lat,
        impact_time,
        f22_dt["Main"]["time"].values,
    )
    sc = static_coords_from_tj(
        f22_dt["Main"].to_dataset()[["lon", "lat"]], tj_ds_sc, profile_path=profile_path
    )
    node0 = dt.DataTree(name=None)
    node1 = dt.DataTree(name="Main", parent=node0, data=sc)
    node2 = dt.DataTree(name="TC1", parent=node0, data=mc)

    node0[""].attrs["group_order"] = "Main TC1"
    node0[""].attrs["institution"] = "Oceanweather Inc. (OWI)"
    node0[""].attrs["conventions"] = "CF-1.6 OWI-NWS13"

    return node0


def save_forcing(
    path: str = "/work/n02/n02/sdat2/adcirc-swan/NWS13set3",
    profile_path: str = os.path.join(CLE_DATA_PATH, "outputs.json"),
    angle: float = 0,
    trans_speed: float = 7.71,
    impact_lon: float = -89.4715,
    impact_lat: float = 29.9511,
    impact_time: np.datetime64 = np.datetime64("2004-08-13T12", "ns"),
) -> None:
    """
    Save the forcing file for ADCIRC.

    Args:
        path (str, optional): Defaults to "/work/n02/n02/sdat2/adcirc-swan/NWS13set3".
        profile_path (str, optional): Defaults to os.path.join(CLE_DATA_PATH, "outputs.json").
        angle (float, optional): Defaults to 0 [deg].
        trans_speed (float, optional): Defaults to 7.71 [m/s].
        impact_lon (float, optional): Defaults to -89.4715 [deg].
        impact_lat (float, optional): Defaults to 29.9511 [deg].
        impact_time (np.datetime64, optional): Defaults to np.datetime64("2004-08-13T12", "ns").
    """
    node0 = return_new_input(
        profile_path=profile_path,
        angle=angle,
        trans_speed=trans_speed,  # impact_lon=impact_lon
        impact_lon=impact_lon,
        impact_lat=impact_lat,
        impact_time=impact_time,
    )
    # node0.to_netcdf(os.path.join(DATA_PATH, "ex.nc"))
    enc = {"time": {"units": "minutes since 1990-01-01T01:00:00"}}
    node0.to_netcdf(
        os.path.join(path, "fort.22.nc"), encoding={"/Main": enc, "/TC1": enc}
    )
    print("new", node0)


if __name__ == "__main__":
    # python -m adforce.fort22
    # blank_fort22()
    data_free_f22_coords()
    # node0 = return_new_input()
    # # node0.to_netcdf(os.path.join(DATA_PATH, "ex.nc"))
    # path = "/work/n02/n02/sdat2/adcirc-swan/NWS13set3"
    # enc = {"time": {"units": "minutes since 1990-01-01T01:00:00"}}
    # node0.to_netcdf(
    #     os.path.join(path, "fort.22.nc"), encoding={"/Main": enc, "/TC1": enc}
    # )

    # print("new", node0)

    # print(chavas_profile)
    # print(timedeltas)

    # blank_fort22()
    # jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser
