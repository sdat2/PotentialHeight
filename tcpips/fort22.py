"""Fort.22.nc file reader and writer.

NWS 13 format:

https://wiki.adcirc.org/NWS13


"""
from typing import Optional, List, Tuple
import os
import numpy as np
import xarray as xr
import datatree as dt
from netCDF4 import Dataset
import datetime
from tcpips.constants import DATA_PATH, FIGURE_PATH
from sithom.time import timeit
from sithom.plot import plot_defaults
from sithom.place import Point
from sithom.io import read_json
from src.conversions import angle_between_points


def read_fort22(fort22_path: Optional[str] = None) -> xr.Dataset:
    """Read fort.22.nc file.

    Parameters
    ----------
    fort22_path : str, optional
        Path to fort.22.nc file. If not provided, the default path will be used.

    Returns
    -------
    xr.Dataset
        Dataset containing fort.22.nc data.

    """
    if fort22_path is None:
        fort22_path = os.path.join(DATA_PATH, "fort.22.nc")
    return dt.open_datatree(fort22_path)


def trim_fort22():
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
def blank_fort22():
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


class Trajectory:
    def __init__(
        self,
        point: Point,
        angle: float,
        trans_speed: float,
        impact_time: np.datetime64 = np.datetime64("2005-08-29T12", "ns"),
    ) -> None:
        """
        Tropical cylone to hit coast at point.

        Args:
            point (Point): Point to impact (lon, lat).
            angle (float): Angle to point [degrees].
            trans_speed (float): Translation speed [m s**-1].
        """
        # print(point, angle, trans_speed)
        self.point = point
        self.angle = angle
        self.trans_speed = trans_speed
        self.impact_time = impact_time

        # time axis to fill in
        self.time_delta: Optional[np.datetime64] = None  # = datetime.timedelta(hours=3)
        self.time_axis: any = None

    def __repr__(self) -> str:
        return str(
            "point: "
            + str(self.point)
            + "\n"
            + "angle: "
            + str(self.angle)
            + " degrees\n"
            + "trans_speed: "
            + str(self.trans_speed)
            + " ms-1\n"
        )

    def new_point(self, distance: float) -> List[float]:
        """
        Line. Assumes 111km per degree.

        Args:
            distance (float): Distance in meters.

        Returns:
            List[float, float]: lon, lat.
        """
        return [
            self.point.lon + np.sin(np.radians(self.angle)) * distance / 111e3,
            self.point.lat + np.cos(np.radians(self.angle)) * distance / 111e3,
        ]

    def timeseries_to_timedelta(self, timeseries: np.ndarray) -> np.ndarray:
        print(timeseries)
        return timeseries - self.impact_time

    def trajectory_from_distances(
        self,
        run_up: float = 1e6,
        run_down: float = 3.5e5,
        time_delta: np.timedelta64 = np.timedelta64(3, "h"),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trajectory.

        Args:
            run_up (int, optional): Run up afterwards. Defaults to 1000 km in meteres.
            run_down (int, optional): Run down after point. Defaults to 350 km im meters.
        """
        self.time_delta = time_delta
        distance_per_timestep = (
            self.trans_speed * self.time_delta / np.timedelta64(1, "s")
        )

        # let's change this.
        time_steps_before = int(abs(run_up) / distance_per_timestep)
        time_steps_after = int(abs(run_down) / distance_per_timestep)
        indices = np.linspace(
            -time_steps_before,
            time_steps_after,
            num=time_steps_before + time_steps_after + 1,
            dtype="int16",
        )
        print("indices", indices)
        print("time_delta", time_delta, type(time_delta), time_delta.dtype)
        time_deltas = indices * np.array(time_delta)  # .astype("timedelta64[h]")
        print("time_delta_list", time_deltas, type(time_deltas), time_deltas.dtype)
        print("impact_time", self.impact_time, type(self.impact_time))
        time_array = np.array(self.impact_time) + time_deltas
        print("time_list", time_array[:4], type(time_array), time_array.dtype)
        self.time_axis = time_array
        print(time_steps_before + time_steps_after + 1)

        # work out point array assuming flat world (pretty terrible, but fast,
        # should change) TODO: The world is round.
        long_angles = indices * distance_per_timestep / 111e3
        point = np.array([[self.point.lon], [self.point.lat]])
        slope = np.array(
            [[np.sin(np.radians(self.angle))], [np.cos(np.radians(self.angle))]]
        )
        print(slope, slope.dtype)

        point_array = (point + slope * long_angles).T
        print("point_array", point_array.shape, point_array[0:4, :], point_array.dtype)

        return point_array, time_array

    # def time_traj(self, )
    def trajectory_ds(self, run_up=1e6, run_down=3.5e5) -> xr.Dataset:
        """
        Create a trajectory dataset for the center eye of the tropical cylone.

        Args:
            run_up (float, optional): How many meters to run up. Defaults to 1e6.
            run_down (float, optional): How many meters to run down. Defaults to 3.5e5.

        Returns:
            xr.Dataset: trajectory dataset with variables lon, lat and time.
        """
        point_array, dates = self.trajectory_from_distances(
            run_up=run_up, run_down=run_down
        )
        print(point_array.shape)
        print(dates.shape)
        return xr.Dataset(
            data_vars=dict(
                clon=(["time"], point_array[:, 0]),
                clat=(["time"], point_array[:, 1]),
            ),
            coords=dict(
                time=dates,
                # reference_time=self.impact_time,
            ),
            attrs=dict(description="Tropcial Cyclone trajectory."),
        )

    @timeit
    def trajectory_ds_from_time(self, time_array: np.datetime64) -> xr.Dataset:
        """
        Create a trajectory dataset for the center eye of the tropical cylone.

        Args:
            run_up (float, optional): How many meters to run up. Defaults to 1e6.
            run_down (float, optional): How many meters to run down. Defaults to 3.5e5.

        Returns:
            xr.Dataset: trajectory dataset with variables lon, lat and time.
        """
        long_angles = (
            (time_array - self.impact_time) / np.timedelta64(1, "s") * self.trans_speed
        ) / 111e3
        point = np.array([[self.point.lon], [self.point.lat]])
        slope = np.array(
            [[np.sin(np.radians(self.angle))], [np.cos(np.radians(self.angle))]]
        )
        print(slope, slope.dtype)

        point_array = (point + slope * long_angles).T

        print(point_array.shape)
        return xr.Dataset(
            data_vars=dict(
                clon=(["time"], point_array[:, 0]),
                clat=(["time"], point_array[:, 1]),
            ),
            coords=dict(
                time=time_array,
                # reference_time=self.impact_time,
            ),
            attrs=dict(description="Tropcial Cyclone trajectory."),
        )

    def angle_at_points(
        self, lats: np.ndarray, lons: np.ndarray, point: Point
    ) -> np.ndarray:
        """
        Angles from each point.

        Args:
            lats (np.ndarray): Latitudes.
            lons (np.ndarray): Longitudes.
            point (Point): Point around which to go.

        Returns:
            np.ndarray: Angles in degrees from North.
        """
        return angle_between_points(point, lons, lats)

    def velocity_at_points(
        self, windspeed: np.ndarray, lats: np.ndarray, lons: np.ndarray, point: Point
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Velocity at grid points based on windspeed field.

        Args:
            windspeed (np.ndarray): Wind [m/s]
            lats (np.ndarray): Latitudes [degrees_North].
            lons (np.ndarray): Longitudes [degrees_East].
            point (Point): point (lon, lat).

        Returns:
            Tuple[np.ndarray, np.ndarray]: u_vel [m s**-1], v_vel [m s**-1]
        """
        angle = np.radians(self.angle_at_points(lats, lons, point) - 90.0)
        return np.sin(angle) * windspeed, np.cos(angle) * windspeed


@timeit
def trajectory_ds_from_time(
    angle,
    trans_speed: float,
    impact_lon: float,
    impact_lat: float,
    impact_time: np.datetime64,
    time_array: np.datetime64,
) -> xr.Dataset:
    """
    Create a trajectory dataset for the center eye of the tropical cylone.

    Args:
        run_up (float, optional): How many meters to run up. Defaults to 1e6.
        run_down (float, optional): How many meters to run down. Defaults to 3.5e5.

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
        attrs=dict(description="Tropcial Cyclone trajectory."),
    )


def pressures_profile(profile: dict) -> dict:
    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    # could speed up but is very quick anyway
    fcor = 5e-5  # should vary with latitude
    p0 = 1015 * 100  # [Pa], should be settable
    rho0 = 1.15  # [kg m-3]
    rr = np.array(profile["rr"])  # [m]
    vv = np.array(profile["VV"])  # [m/s]
    p = np.zeros(rr.shape)  # [Pa]
    # rr ascending
    assert np.all(rr == np.sort(rr))
    p[-1] = p0
    for j in range(len(rr) - 1):
        i = -j - 2
        # Assume Coriolis force and pressure-gradient balance centripetal force.
        p[i] = p[i + 1] - rho0 * (
            vv[i] ** 2 / (rr[i + 1] / 2 + rr[i] / 2) + fcor * vv[i]
        ) * (rr[i + 1] - rr[i])
        # centripetal pushes out, pressure pushes inward, coriolis pushes inward

    profile["p"] = p / 100

    return profile


def gen_ps_f() -> callable:
    """Generate the interpolation file from the Chavas15 profile.

    This should be changed to take a file name as input,
    and to allow each timestep to have a different profile.
    """
    chavas_profile = read_json("cle/outputs.json")
    # print(chavas_profile.keys())
    chavas_profile = pressures_profile(chavas_profile)
    radii = np.array(chavas_profile["rr"], dtype="float32")
    velocities = np.array(chavas_profile["VV"], dtype="float32")
    pressures = np.array(chavas_profile["p"], dtype="float32")
    # print(radii[0:10], velocities[0:10], pressures[0:10])
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(radii, pressures)
    axs[1].plot(radii, velocities)
    plt.savefig("prof.png")

    def f(distances):
        pressure_cube = np.interp(distances, radii, pressures)
        velo_cube = np.interp(distances, radii, velocities)
        return pressure_cube, velo_cube

    return f


@timeit
def moving_coords_from_tj(coords: xr.DataArray, tj: xr.DataArray):
    coords.lon[:] = coords.lon[:] - coords.lon.mean()
    coords.lat[:] = coords.lat[:] - coords.lat.mean()
    clon = tj.clon.values.reshape(-1, 1, 1)
    clat = tj.clat.values.reshape(-1, 1, 1)
    lats = np.expand_dims(coords.lon.values, 0) + clat
    lons = np.expand_dims(coords.lat.values, 0) + clon

    f = gen_ps_f()

    distances = np.sqrt((lons - clon) ** 2 + (lats - clat) ** 2) * 111e3
    psfc, u = f(distances)

    rad = np.arctan2(np.radians(lons - clon), np.radians(lats - clat)) - np.pi / 2
    u10, v10 = np.sin(rad) * u, np.cos(rad) * u

    # print(lats, lons)
    return xr.Dataset(
        data_vars=dict(
            clon=(["time"], tj.clon.values),
            clat=(["time"], tj.clat.values),
            PSFC=(
                [
                    "time",
                    "xi",
                    "yi",
                ],
                psfc.astype("float32"),
            ),
            U10=(["time", "xi", "yi"], u10.astype("float32")),
            V10=(["time", "xi", "yi"], v10.astype("float32")),
        ),
        coords=dict(
            lon=(["time", "xi", "yi"], lons),
            lat=(["time", "xi", "yi"], lats),
            time=tj.time.values,
            # reference_time=self.impact_time,
        ),
        attrs=dict(description="Tropical cyclone moving grid."),
    )


@timeit
def static_coords_from_tj(orig: xr.DataArray, tj: xr.DataArray):
    lats = np.expand_dims(orig.lat.values, 0)
    lons = np.expand_dims(orig.lon.values, 0)
    clon = tj.clon.values.reshape(-1, 1, 1)
    clat = tj.clat.values.reshape(-1, 1, 1)

    f = gen_ps_f()

    distances = np.sqrt((lons - clon) ** 2 + (lats - clat) ** 2) * 111e3
    psfc, u = f(distances)

    rad = np.arctan2(np.radians(lons - clon), np.radians(lats - clat)) - np.pi / 2
    u10, v10 = np.sin(rad) * u, np.cos(rad) * u

    # print(lats, lons)
    return xr.Dataset(
        data_vars=dict(
            clon=(["time"], tj.clon.values),
            clat=(["time"], tj.clat.values),
            PSFC=(
                [
                    "time",
                    "yi",
                    "xi",
                ],
                psfc.astype("float32"),
            ),
            U10=(["time", "yi", "xi"], u10.astype("float32")),
            V10=(["time", "yi", "xi"], v10.astype("float32")),
        ),
        coords=dict(
            lon=(["yi", "xi"], orig.lon.values),
            lat=(["yi", "xi"], orig.lat.values),
            time=tj.time.values,
            # reference_time=self.impact_time,
        ),
        attrs=dict(description="Tropical cyclone static grid."),
    )


@timeit
def return_new_input(
    angle=30,
    trans_speed=2,
    impact_lon=-90,
    impact_lat=40,
    impact_time=np.datetime64("2004-08-19T12", "ns"),
):
    f22_dt = dt.open_datatree(os.path.join(DATA_PATH, "blank.nc"))

    tj_ds_mv = trajectory_ds_from_time(
        angle,
        trans_speed,
        impact_lon,
        impact_lat,
        impact_time,
        f22_dt["TC1"]["time"].values,
    )
    mc = moving_coords_from_tj(
        f22_dt["TC1"].to_dataset().isel(time=0)[["lon", "lat"]], tj_ds_mv
    )
    tj_ds_sc = trajectory_ds_from_time(angle,
        trans_speed,
        impact_lon,
        impact_lat,
        impact_time,
        f22_dt["Main"]["time"].values)
    sc = static_coords_from_tj(f22_dt["Main"].to_dataset()[["lon", "lat"]], tj_ds_sc)
    node0 = dt.DataTree(name=None)
    node1 = dt.DataTree(name="Main", parent=node0, data=sc)
    node2 = dt.DataTree(name="TC1", parent=node0, data=mc)

    node0[""].attrs["group_order"] = "Main TC1"
    node0[""].attrs["institution"] = "Oceanweather Inc. (OWI)"
    node0[""].attrs["conventions"] = "CF-1.6 OWI-NWS13"

    return node0


if __name__ == "__main__":
    # python -m tcpips.fort22
    # trim_fort22()
    # print(f22_dt)
    # dt1 = datetime.datetime(year=2004, month=8, day=12)
    # dt1 = np.datetime64("2004-08-12", "ns")
    # print(dt1)
    node0 = return_new_input()
    node0.to_netcdf(os.path.join(DATA_PATH, "ex.nc"))

    print(node0)

    # print(chavas_profile)
    # print(timedeltas)

    # blank_fort22()
    # jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser
