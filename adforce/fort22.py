"""fort.22.nc file creation functions.

TODO: Add option for Lin and Chavas 2012 Asymmetric wind profile part based on trajectory.

TODO: Is there a faster package than pyproj for distance and bearing calculations?
"""

import os
from typing import Union, Optional, Callable, Tuple
import netCDF4 as nc
import numpy as np
from sithom.time import timeit
from w22.constants import DATA_PATH as CLE_DATA_PATH
from .time import unknown_to_time
from .constants import DATA_PATH
from .geo import (
    distances_bearings_to_center_pyproj,
    line_with_impact_pyproj,
)
from .profile import read_profile


def clon_clat_from_config_and_times(
    cfg: dict, itime: int, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the center of the storm from the config and times.

    Args:
        cfg (dict): A dictionary containing the information necesary to reconstruct the tropical cyclone trajectory.
        itime (int): time in minutes from start of calendar for impacts.
        times (np.ndarray): times in minutes from start of calendar.

    Returns:
        clons, clats (np.ndarray, np.ndarray): center of the storm at each time step.
    """
    angle = cfg["angle"]["value"]
    speed = cfg["translation_speed"]["value"]
    ilon = cfg["impact_location"]["value"][0]
    ilat = cfg["impact_location"]["value"][1]

    return line_with_impact_pyproj(
        impact_time=itime * 60,  # convert to seconds.
        impact_lat=ilat,
        impact_lon=ilon,
        translation_speed=speed,
        bearing=angle,
        times=times * 60,  # convert to seconds.
    )


@timeit
def rectilinear_square(ds: nc.Dataset, grid_config: dict) -> nc.Dataset:
    """
    Create a rectilinear grid with square cells based on grid_config.

    Args:
        grid_config (dict): A dictionary containing the information necesary to reconstruct the coordinates.
    """
    lons = np.linspace(
        grid_config["bottom_left_corner"][0],
        grid_config["bottom_left_corner"][0]
        + grid_config["lateral_spacing"] * (grid_config["xlen"] - 1),
        grid_config["xlen"],
        dtype=np.float64,
    )
    lats = np.linspace(
        grid_config["bottom_left_corner"][1],
        grid_config["bottom_left_corner"][1]
        + grid_config["lateral_spacing"] * (grid_config["ylen"] - 1),
        grid_config["ylen"],
        dtype=np.float64,
    )

    lons, lats = np.meshgrid(lons, lats)
    # currently assumes time is in minutes from 1990-01-01 00:00:00
    # should be able to change this so that it can also accept date time strings.
    tlen = grid_config["tlen"]
    start = unknown_to_time(
        grid_config["start"], grid_config["time_unit"], grid_config["time_calendar"]
    )

    times = np.linspace(
        start, start + (tlen - 1) * (grid_config["timestep"]), tlen, dtype=np.int32
    )  # time in minutes from start of calendar

    # time is an unlimited dimension
    ds.createDimension("time", None)
    ds.createDimension("yi", grid_config["ylen"])
    ds.createDimension("xi", grid_config["xlen"])
    ds.createVariable("time", "i4", ("time",), fill_value=-2147483647)
    ds.createVariable("lat", "f8", ("yi", "xi"), fill_value=9.969209968386869e36)
    ds.createVariable("lon", "f8", ("yi", "xi"), fill_value=9.969209968386869e36)
    # add data to netcdf dataset
    ds["time"][:] = times
    ds["lat"][:] = lats
    ds["lon"][:] = lons
    ds["time"].units = grid_config["time_unit"]
    ds["time"].calendar = grid_config["time_calendar"]
    # set standard coordinate attributes
    ds["lat"].units = "degrees_north"
    ds["lon"].units = "degrees_east"
    ds["lat"].axis = "Y"
    ds["lon"].axis = "X"
    ds["time"].axis = "T"
    ds["time"].coordinates = "time"
    ds["lat"].coordinates = "lat lon"
    ds["lon"].coordinates = "lat lon"
    ds["lat"].standard_name = "latitude"
    ds["lon"].standard_name = "longitude"
    ds["time"].standard_name = "time"
    # add attribute rank=1 to dataset
    ds.rank = 1
    return ds


@timeit
def moving_rectilinear_square(
    ds: nc.Dataset,
    grid_config: dict,
    tc_config: dict,
) -> nc.Dataset:
    """
    Create a rectilinear grid with square cells based on config.
    The grid moves based on the translation speed and angle of the tropical cyclone.

    Args:
        ds (nc.Dataset): reference to input netcdf4 dataset.
        grid_config (dict): A dictionary containing the information necesary to reconstruct the coordinates.
        tc_config (dict): A dictionary containing the information necesary to reconstruct the tropical cyclone trajectory.

    Returns:
        ds (nc.Dataset): reference to transformed netcdf4 dataset.
    """
    # create a rectilinear grid with square cells based on config.
    lons = np.linspace(
        0,
        0 + grid_config["lateral_spacing"] * (grid_config["xlen"] - 1),
        grid_config["xlen"],
        dtype=np.float64,
    )
    lons = lons - lons.mean()  # make the center of the grid the origin
    lats = np.linspace(
        0,
        0 + grid_config["lateral_spacing"] * (grid_config["ylen"] - 1),
        grid_config["ylen"],
        dtype=np.float64,
    )
    lats = lats - lats.mean()  # make the center of the grid the origin
    lons, lats = np.meshgrid(lons, lats)

    tlen = grid_config["tlen"]
    start = unknown_to_time(
        grid_config["start"], grid_config["time_unit"], grid_config["time_calendar"]
    )

    times = np.linspace(
        start, start + (tlen - 1) * (grid_config["timestep"]), tlen, dtype=np.int32
    )  # time in minutes from start of calendar

    itime = unknown_to_time(
        tc_config["impact_time"]["value"],
        grid_config["time_unit"],
        grid_config["time_calendar"],
    )

    clons, clats = clon_clat_from_config_and_times(tc_config, itime, times)

    lons = lons + clons.reshape(tlen, 1, 1)
    lats = lats + clats.reshape(tlen, 1, 1)

    # time is an unlimited dimension
    ds.createDimension("time", None)
    ds.createDimension("yi", grid_config["ylen"])
    ds.createDimension("xi", grid_config["xlen"])
    ds.createVariable("time", "i4", ("time",), fill_value=-2147483647)
    ds.createVariable("clat", "f8", ("time",), fill_value=9.969209968386869e36)
    ds.createVariable("clon", "f8", ("time",), fill_value=9.969209968386869e36)
    ds.createVariable(
        "lat", "f8", ("time", "yi", "xi"), fill_value=9.969209968386869e36
    )
    ds.createVariable(
        "lon", "f8", ("time", "yi", "xi"), fill_value=9.969209968386869e36
    )

    # add data to netcdf dataset
    ds["time"][:] = times
    ds["lat"][:] = lats
    ds["lon"][:] = lons
    ds["clon"][:] = clons
    ds["clat"][:] = clats
    ds["time"].units = grid_config["time_unit"]
    ds["time"].calendar = grid_config["time_calendar"]

    # set standard coordinate attributes
    ds["lat"].units = "degrees_north"
    ds["lon"].units = "degrees_east"
    ds["clat"].units = "degrees_north"
    ds["clon"].units = "degrees_east"
    ds["clat"].coordinates = "time"
    ds["clon"].coordinates = "time"
    ds["lat"].axis = "Y"
    ds["lon"].axis = "X"
    ds["time"].axis = "T"
    ds["time"].coordinates = "time"
    ds["lat"].coordinates = "time lat lon"
    ds["lon"].coordinates = "time lat lon"
    ds["lat"].standard_name = "latitude"
    ds["lon"].standard_name = "longitude"
    ds["time"].standard_name = "time"
    ds["clat"].long_name = "center latitude of feature"
    ds["clon"].long_name = "center longitude of feature"
    # add attribute rank=2 to dataset
    ds.rank = 2
    return ds


def gen_ps_f(
    profile_path_or_dict: Union[str, dict] = os.path.join(
        CLE_DATA_PATH, "outputs.json"
    )  # "/work/n02/n02/sdat2/adcirc-swan/tcpips/cle/data/outputs.json",
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate the interpolation function from an azimuthally symetric windprofile. (Gradient winds)

    This should potentially be changed to allow each timestep to have a different profile.

    That would require some clever way of interpolating each time step.

    Args:
        profile_path_or_dict (Union[str, dict], optional): path to wind profile file or dictionary containing the wind profile. Defaults to os.path.join(CLE_DATA_PATH, "outputs.json").

    Returns:
        Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]: interpolation function.
    """
    print(
        "profile_path_or_dict,",
        profile_path_or_dict,
    )
    if isinstance(profile_path_or_dict, str):
        if not profile_path_or_dict.endswith(".json"):
            profile_path_or_dict += ".json"
        profile = read_profile(profile_path_or_dict)
    elif isinstance(profile_path_or_dict, dict):
        profile = profile_path_or_dict
    else:
        raise ValueError("profile_path_or_dict must be a string or a dictionary.")

    radii = profile["radii"].values
    windspeeds = profile["windspeeds"].values
    pressures = profile["pressures"].values
    # write_json(profile, os.path.join(DATA_PATH, "profile.json"))

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
def add_psfc_u10(
    ds: nc.Dataset,
    tc_config: Optional[dict] = None,
    background_pressure: float = 1010,
    v_reduc: float = 0.8,
) -> nc.Dataset:
    """Add pressure and velocity fields to an existing netcdf dataset.

    Args:
        ds (nc.Dataset): reference to input netcdf4 dataset.
        tc_config (Optional[dict], optional): A dictionary containing the information necesary to reconstruct the tropical cyclone trajectory. Defaults to None. If None, the fields are left blank (velocity fields are zero and pressure is set to background_pressure).
        background_pressure: background pressure to assume in mb. Defaults to 1010 mb.

    Returns:
        ds (nc.Dataset): reference to transformed netcdf4 dataset.
    """
    ds.createVariable("PSFC", "f4", ("time", "yi", "xi"), fill_value=9.96921e36)
    ds.createVariable("U10", "f4", ("time", "yi", "xi"), fill_value=9.96921e36)
    ds.createVariable("V10", "f4", ("time", "yi", "xi"), fill_value=9.96921e36)
    shape = (
        len(ds.dimensions["time"]),
        len(ds.dimensions["yi"]),
        len(ds.dimensions["xi"]),
    )
    if tc_config is None:  # blank fields
        ds["U10"][:] = np.zeros(shape, dtype="float32")
        ds["V10"][:] = np.zeros(shape, dtype="float32")
        ds["PSFC"][:] = np.zeros(shape, dtype="float32") + background_pressure
    else:  # add fields based on tc_config
        times = ds["time"][:]
        tlen = len(times)
        # we're dealing with the static grid.
        itime = unknown_to_time(
            tc_config["impact_time"]["value"],
            ds["time"].units,
            ds["time"].calendar,
        )
        if "clon" not in ds.variables or "clat" not in ds.variables:
            clons, clats = clon_clat_from_config_and_times(tc_config, itime, times)
        else:
            clons = ds["clon"][:].astype("float32")
            clats = ds["clat"][:].astype("float32")
        # distance from the center of the storm
        lons = ds["lon"][:].astype("float32")
        lats = ds["lat"][:].astype("float32")
        # if lons starts off as 2D change to 3D, adding a new axis at the start
        if lons.shape == (len(ds.dimensions["yi"]), len(ds.dimensions["xi"])):
            lons = np.expand_dims(lons, axis=0)
        if lats.shape == (len(ds.dimensions["yi"]), len(ds.dimensions["xi"])):
            lats = np.expand_dims(lats, axis=0)
        dist, bearing = distances_bearings_to_center_pyproj(
            lons, lats, clons.reshape(tlen, 1, 1), clats.reshape(tlen, 1, 1)
        )
        del lons, lats
        assert dist.shape == shape
        assert bearing.shape == shape
        # generate the interpolation function from the wind profile
        interp_func = gen_ps_f(profile_path_or_dict=tc_config["profile_path"]["value"])
        # interpolate the pressure and gradient wind fields from the wind profile file
        psfc, wsp = interp_func(dist)
        assert psfc.shape == shape
        assert wsp.shape == shape
        # add the pressure field to the dataset
        ds["PSFC"][:] = psfc
        del psfc
        # calculate the u10 and v10 from the windspeed
        # rad = np.arctan2(dist_lon, dist_lat) - np.pi / 2
        # reduce by v_reduc to go from gradient wind to 10m wind
        u10, v10 = (
            np.sin(np.deg2rad(bearing) + np.pi / 2) * wsp * v_reduc,
            np.cos(np.deg2rad(bearing) + np.pi / 2) * wsp * v_reduc,
        )
        del wsp
        assert u10.shape == shape
        assert v10.shape == shape
        # add the u10 and v10 fields to the dataset
        ds["U10"][:] = u10
        ds["V10"][:] = v10
        del u10, v10

    ds["PSFC"].units = "mb"
    ds["U10"].units = "m s-1"
    ds["V10"].units = "m s-1"
    ds["PSFC"].coordinates = "time lat lon"
    ds["U10"].coordinates = "time lat lon"
    ds["V10"].coordinates = "time lat lon"
    # ds["PSFC"].standard_name = "Surface pressure"
    # ds["U10"].standard_name = "10m zonal windspeed"
    # ds["V10"].standard_name = "10m meridional windspeed"
    return ds


@timeit
def create_fort22(nc_path: str, grid_config: dict, tc_config: dict) -> None:
    """
    Create a blank fort.22.nc file with the specified grid and TC configurations.

    Args:
        nc_path (str): Path to the netCDF4 file.
        grid_config (dict): Grid configuration dictionary.
        tc_config (dict): Idealized TC configuration dictionary.
    """
    # Create a new netCDF4 file
    ds = nc.Dataset(os.path.join(nc_path, "fort.22.nc"), "w", format="NETCDF4")
    if "profile_name" in tc_config:
        tc_config["profile_path"]["value"] = os.path.join(
            CLE_DATA_PATH, f"{tc_config['profile_name']['value']}.json"
        )
        print("TC profile path", tc_config["profile_path"]["value"])
    # Create the "Main" group (rank 1)
    main_group = ds.createGroup("Main")
    main_group = rectilinear_square(main_group, grid_config["Main"])
    main_group = add_psfc_u10(
        main_group, tc_config, v_reduc=tc_config["v_reduc"]["value"]
    )
    main_group.description = "Main grid"
    # Create the "TC1" group within root (rank 2)
    tc1_group = ds.createGroup("TC1")
    tc1_group = moving_rectilinear_square(tc1_group, grid_config["TC1"], tc_config)
    tc1_group = add_psfc_u10(
        tc1_group, tc_config=tc_config, v_reduc=tc_config["v_reduc"]["value"]
    )
    tc1_group.description = "TC1 grid"

    # institution: Oceanweather Inc. (OWI)
    ds.institution = "Oceanweather Inc. (OWI)"
    ds.conventions = "CF-1.6 OWI-NWS13"
    ds.group_order = "Main TC1"
    # conventions: CF-1.6 OWI-NWS13

    # print(ds)
    ds.close()


if __name__ == "__main__":
    # python -m adforce.fort22
    from .constants import CONFIG_PATH
    import yaml

    tc_config = yaml.safe_load(
        open(os.path.join(CONFIG_PATH, "tc", "tc_param_config.yaml"))
    )
    grid_config = yaml.safe_load(
        open(os.path.join(CONFIG_PATH, "grid", "grid_fort22_config.yaml"))
    )

    create_fort22(DATA_PATH, grid_config, tc_config)
