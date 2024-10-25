"""fort.22.nc file creation functions."""

from typing import Union, Optional
import os
import netCDF4 as nc
import numpy as np
from sithom.time import timeit
from cle.constants import DATA_PATH as CLE_DATA_PATH
from typing import Callable, Tuple
from adforce.time import unknown_to_time
from .constants import DATA_PATH
from .profile import read_profile


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


def gen_ps_f(
    profile_path_or_dict: Union[str, dict] = os.path.join(
        CLE_DATA_PATH, "outputs.json"
    )  # "/work/n02/n02/sdat2/adcirc-swan/tcpips/cle/data/outputs.json",
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate the interpolation function from the wind profile (from Chavas et al. 2015).

    Maybe I should also make options for any other azimuthally symetric model automatically.

    This should potentially be changed to allow each timestep to have a different profile.

    That would require some clever way of interpolating each time step.

    Args:
        profile_path_or_dict (Union[str, dict], optional): path to wind profile file or dictionary containing the wind profile. Defaults to os.path.join(CLE_DATA_PATH, "outputs.json").

    Returns:
        Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]: interpolation function.
    """
    if isinstance(profile_path_or_dict, str):
        profile = read_profile(profile_path_or_dict)
    elif isinstance(profile_path_or_dict, dict):
        profile = profile_path_or_dict
    else:
        raise ValueError("profile_path_or_dict must be a string or a dictionary.")
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
def add_psfc_u10(
    ds: nc.Dataset, tc_config: Optional[dict] = None, background_pressure: float = 1010
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
        if "clat" in ds.variables and "clon" in ds.variables:
            # we're dealing with the moving grid.
            # assume that the right TC config has been given previously.
            dist_lon = ds["lon"][:] - ds["clon"][:].reshape(tlen, 1, 1)
            dist_lat = ds["lat"][:] - ds["clat"][:].reshape(tlen, 1, 1)
        else:
            # we're dealing with the static grid.

            angle = tc_config["angle"]["value"]
            speed = tc_config["translation_speed"]["value"]
            ilon = tc_config["impact_location"]["value"][0]
            ilat = tc_config["impact_location"]["value"][1]
            itime = unknown_to_time(
                tc_config["impact_time"]["value"],
                ds["time"].units,
                ds["time"].calendar,
            )
            angular_distances = (times - itime) / 60 * speed / 111e3
            point = np.array([[ilon], [ilat]])
            slope = np.array([[np.sin(np.radians(angle))], [np.cos(np.radians(angle))]])
            clon_clat_array = point + slope * angular_distances
            lons = ds["lon"][:]
            lats = ds["lat"][:]
            dist_lon = lons - clon_clat_array[0, :].reshape(tlen, 1, 1)
            dist_lat = lats - clon_clat_array[1, :].reshape(tlen, 1, 1)
            del lons, lats
        assert dist_lat.shape == dist_lat.shape
        assert dist_lon.shape == (
            len(ds.dimensions["time"]),
            len(ds.dimensions["yi"]),
            len(ds.dimensions["xi"]),
        )
        interp_func = gen_ps_f(profile_path_or_dict=tc_config["profile"])
        dist = np.sqrt(dist_lon**2 + dist_lat**2)
        psfc, wsp = interp_func(dist)
        ds["PSFC"][:] = psfc
        del psfc
        rad = np.arctan2(np.radians(dist_lon), np.radians(dist_lat)) - np.pi / 2
        del dist_lon, dist_lat
        u10, v10 = np.sin(rad) * wsp, np.cos(rad) * wsp
        ds["U10"][:] = u10
        ds["V10"][:] = v10
        del u10, v10

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
def create_fort22(nc_path: str, grid_config: dict, tc_config: dict) -> None:
    """
    Create a blank fort.22.nc file with the specified grid and TC configurations.

    Args:
        nc_path (str): Path to the netCDF4 file.
        grid_config (dict): Grid configuration dictionary.
        tc_config (dict): Idealized TC configuration dictionary.
    """
    # Create a new netCDF4 file
    ds = nc.Dataset(nc_path, "w", format="NETCDF4")
    # Create the "Main" group (rank 1)
    main_group = ds.createGroup("Main")
    main_group = rectilinear_square(main_group, grid_config["Main"])
    main_group = add_psfc_u10(main_group, tc_config)
    # Create the "TC1" group within root (rank 2)
    tc1_group = ds.createGroup("TC1")
    tc1_group = moving_rectilinear_square(tc1_group, grid_config["TC1"], tc_config)
    tc1_group = add_psfc_u10(tc1_group, tc_config=tc_config)

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
    if "profile_name" in tc_config:
        tc_config["profile"] = os.path.join(
            CLE_DATA_PATH, f"{tc_config['profile_name']['value']}.json"
        )
    create_fort22(os.path.join(DATA_PATH, "default_fort22.nc"), grid_config, tc_config)
