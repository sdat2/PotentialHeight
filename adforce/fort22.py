"""fort.22.nc file creation functions."""

from typing import Union, Optional
import os
import netCDF4 as nc
import numpy as np
from sithom.time import timeit
from adforce.time import unknown_to_time
from .constants import DATA_PATH


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


def add_blank_psfc_u10(ds: nc.Dataset, background_pressure: float = 1010) -> nc.Dataset:
    """Add blank pressure and velocity fields to an existing netcdf dataset.

    Args:
        ds (nc.Dataset): reference to input netcdf4 dataset.
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
    ds["U10"][:] = np.zeros(shape, dtype="float32")
    ds["V10"][:] = np.zeros(shape, dtype="float32")
    ds["PSFC"][:] = np.zeros(shape, dtype="float32") + background_pressure
    ds["PSFC"].units = "mb"
    ds["U10"].units = "m s-1"
    ds["V10"].units = "m s-1"
    ds["PSFC"].coordinates = "time lat lon"
    ds["U10"].coordinates = "time lat lon"
    ds["V10"].coordinates = "time lat lon"
    ds["PSFC"].standard_name = "Surface pressure"
    ds["U10"].standard_name = "10m zonal windspeed"
    ds["V10"].standard_name = "10m meridional windspeed"
    return ds


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

    angle = tc_config["angle"]["value"]
    speed = tc_config["translation_speed"]["value"]
    ilon = tc_config["impact_location"]["value"][0]
    ilat = tc_config["impact_location"]["value"][1]
    itime = unknown_to_time(
        tc_config["impact_time"]["value"],
        grid_config["time_unit"],
        grid_config["time_calendar"],
    )

    angular_distances = (times - itime) / 60 * speed / 111e3
    point = np.array([[ilon], [ilat]])
    slope = np.array([[np.sin(np.radians(angle))], [np.cos(np.radians(angle))]])
    clon_clat_array = point + slope * angular_distances
    lons = lons + clon_clat_array[0, :].reshape(tlen, 1, 1)
    lats = lats + clon_clat_array[1, :].reshape(tlen, 1, 1)

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
    ds["clon"][:] = clon_clat_array[0]
    ds["clat"][:] = clon_clat_array[1]
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


@timeit
def create_blank_fort22(grid_config: dict, tc_config: dict) -> None:
    """
    Create a blank fort.22.nc file with the specified grid and TC configurations.

    Args:
        grid_config (dict): Grid configuration dictionary.
        tc_config (dict): Idealized TC configuration dictionary.
    """
    # Create a new netCDF4 file
    ds = nc.Dataset(os.path.join(DATA_PATH, "blank_fort22.nc"), "w", format="NETCDF4")
    # Create the "Main" group (rank 1)
    main_group = ds.createGroup("Main")
    main_group = rectilinear_square(main_group, grid_config["Main"])
    main_group = add_blank_psfc_u10(main_group)
    # Create the "TC1" group within root (rank 2)
    tc1_group = ds.createGroup("TC1")
    tc1_group = moving_rectilinear_square(tc1_group, grid_config["TC1"], tc_config)
    tc1_group = add_blank_psfc_u10(tc1_group)

    # institution: Oceanweather Inc. (OWI)
    ds.institution = "Oceanweather Inc. (OWI)"
    ds.conventions = "CF-1.6 OWI-NWS13"
    # conventions: CF-1.6 OWI-NWS13

    print(ds)
    ds.close()


if __name__ == "__main__":
    # python -m adforce.fort22
    from adforce.constants import CONFIG_PATH
    import yaml

    tc_config = yaml.safe_load(open(os.path.join(CONFIG_PATH, "tc_param_wrap.yaml")))
    grid_config = yaml.safe_load(
        open(os.path.join(CONFIG_PATH, "grid_config_fort22.yaml"))
    )
    create_blank_fort22(grid_config, tc_config)
