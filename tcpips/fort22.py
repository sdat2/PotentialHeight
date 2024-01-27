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
from src.conversions import angle_between_points, distance_between_points


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
def moving_coords_from_tj(coords: xr.DataArray, tj: xr.DataArray):
    coords.lon[:] = coords.lon[:] - coords.lon.mean()
    coords.lat[:] = coords.lat[:] - coords.lat.mean()
    lats = np.expand_dims(coords.lon.values, -1) + tj.clon.values.reshape(1, 1, -1)
    lons = np.expand_dims(coords.lon.values, -1) + tj.clat.values.reshape(1, 1, -1)
    # print(lats, lons)
    return xr.Dataset(
        data_vars=dict(
            clon=(["time"], tj.clon.values),
            clat=(["time"], tj.clat.values),
        ),
        coords=dict(
            lon=(["xi", "yi", "time"], lons),
            lat=(["xi", "yi", "time"], lats),
            time=tj.time.values,
            # reference_time=self.impact_time,
        ),
        attrs=dict(description="Tropcial cyclone moving grid."),
    )


if __name__ == "__main__":
    # python -m tcpips.fort22
    # trim_fort22()
    tj = Trajectory(
        Point(-90, 40), 30, 2, impact_time=np.datetime64("2004-08-19T12", "ns")
    )
    tj_ds = tj.trajectory_ds()
    tj_ds.to_netcdf(os.path.join(DATA_PATH, "traj.nc"))
    # print(tj_ds)

    f22_dt = dt.open_datatree(os.path.join(DATA_PATH, "blank.nc"))
    # print(f22_dt)
    # dt1 = datetime.datetime(year=2004, month=8, day=12)
    # dt1 = np.datetime64("2004-08-12", "ns")
    # print(dt1)
    tj_ds_new = tj.trajectory_ds_from_time(f22_dt["Main"]["time"].values)
    print("new_tj", tj_ds_new)
    mc = moving_coords_from_tj(
        f22_dt["TC1"].to_dataset().isel(time=0)[["lon", "lat"]], tj_ds_new
    )
    print(mc)

    from sithom.io import read_json

    chavas_profile = read_json("cle/outputs.json")
    print(chavas_profile.keys())
    radii = np.array(chavas_profile["rr"])
    velocities = np.array(chavas_profile["VV"])
    print(radii[0:10], velocities[0:10])
    # print(chavas_profile)

    # print(timedeltas)

    # blank_fort22()
