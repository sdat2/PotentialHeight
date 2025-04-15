"""Process IBTrACS data to calculate along-track potential intensity and size from ERA5.

We only want tracks from 1980-2024 (inclusive).

wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc

"""

import os
import numpy as np
import xarray as xr
import ujson
from sithom.time import timeit
from .constants import DATA_PATH
from .era5 import get_era5_coordinates, get_era5_combined


IBTRACS_DATA_PATH = os.path.join(DATA_PATH, "ibtracs")
os.makedirs(IBTRACS_DATA_PATH, exist_ok=True)
IBTRACS_DATA_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"
IBTRACS_DATA_FILE = os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")


def download_ibtracs_data() -> None:
    """Download IBTrACS data if not already downloaded

    Currently gets the post 1980 data."""
    if not os.path.exists(IBTRACS_DATA_FILE):
        print(f"Downloading IBTrACS data from {IBTRACS_DATA_URL}...")
        os.system(f"wget {IBTRACS_DATA_URL} -O {IBTRACS_DATA_FILE}")
        print("Download complete.")
    else:
        print("IBTrACS data already downloaded.")


@timeit
def ibtracs_to_era5_map():
    """Find the corresponding ERA5 data for each IBTrACS storm.

    This function will create a unique counter for each unique (x, y, t) point in the ERA5 data
    that corresponds to an IBTrACS storm. The unique counter will be added to the IBTrACS data
    as a new variable. The unique points will be saved to a JSON file.

    Currenty takes around 36 seconds to run on laptop.

    Paralelization could be implemented if we ignored having a unique counter for each point, however this is not the rate limting step.
    """
    # load ibtracs
    ibtracs_ds = xr.open_dataset(IBTRACS_DATA_FILE)
    # next get era5 coordinates (longitude, latitude, time)

    era5_coords_ds = get_era5_coordinates()
    lats_e = era5_coords_ds.latitude.values
    lons_e = era5_coords_ds.longitude.values
    times_e = era5_coords_ds.valid_time.values

    # set up hashmap for unique x/y/t points
    point_d = {}
    # create a unique counter
    u = 0
    # create a list to keep track of the order of the unique points
    unique_points_list = []
    u_lol = []

    for sid_i in range(len(ibtracs_ds.sid)):
        u_lol.append([])

        storm = ibtracs_ds.isel(storm=sid_i)
        # get the start and end time of the storm

        # get the latitude and longitude of the storm centre
        lats_i = storm.lat.values
        lons_i = storm.lon.values
        times_i = storm.time.values

        # loop through the time steps of the storm
        for i in range(len(storm.time)):
            # get the latitude and longitude of the storm centre at this time step
            lat_i = lats_i[i]
            lon_i = lons_i[i]
            time = times_i[i]
            # if not nan
            # find closest era5 data point to the cyclone centre at this time step
            if not np.isnan(lat_i) and not np.isnan(lon_i):
                # find the closest era5 data point to the cyclone centre at this time step
                # (this part is not implemented yet)
                # it's a rectilinear grid, so we can just find the closest point in each dimension separately
                x = np.argmin(np.abs(lons_e % 360 - lon_i % 360))  # lon dimension

                y = np.argmin(np.abs(lats_e - lat_i))  # lat dimension
                t = np.argmin(np.abs(times_e - time))  # time dimension

                # print out the coordinates of cyclone centre and closest found era5 point
                # print(f"lat_i: {lat_i}, lon_i: {lon_i}, time: {time}")
                # print(
                #    f"lat_e: {lats_e[x]}, lon_e: {lons_e[y]}, time: {times_e[t]}"
                # )

                # check if this point is already in the hashmap
                if (x, y, t) not in point_d:
                    # add it to the hashmap
                    point_d[(x, y, t)] = u
                    unique_points_list.append([x, y, t])
                    u_lol[-1].append(u)
                    u += 1
                else:
                    # if it is already in the hashmap, get the unique counter
                    u_lol[-1].append(point_d[(x, y, t)])
            else:
                u_lol[-1].append(np.nan)

    ibtracs_ds["unique_counter"] = (("storm", "time"), u_lol)
    # save the ibtracs data with the unique counter
    ibtracs_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.unique.nc")
    )
    # save the dict of unique points
    # write_json(
    #    point_d,
    #    os.path.join(IBTRACS_DATA_PATH, "unique_points_dict.json"),
    # )
    print(f"unique points list is {len(unique_points_list)} long")
    with open(os.path.join(IBTRACS_DATA_PATH, "unique_points_dict.json"), "w") as file:
        file.write(ujson.dumps(point_d))
    # save the unique points list
    unique_points_list = np.array(unique_points_list)
    xr.Dataset(
        data_vars={
            "unique_dim": (["u", "coords"], unique_points_list),
            "longitude": (["u"], lons_e[unique_points_list[:, 0]]),
            "latitude": (["u"], lats_e[unique_points_list[:, 1]]),
            "time": (["u"], times_e[unique_points_list[:, 2]]),
        },
        coords={
            "u": (
                ["u"],
                np.arange(len(unique_points_list)),
                {"description": "unique index"},
            ),
            "coords": ["x", "y", "t"],
        },
    ).to_netcdf(os.path.join(IBTRACS_DATA_PATH, "unique_era5_points.nc"))
    print(
        f"Found {u} unique points from ERA5 that correspond to IBTrACS data since 1980."
    )


@timeit
def plot_unique_points():
    # plot geographic distribution of unique points
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import os
    from .constants import PROJECT_PATH
    from sithom.plot import plot_defaults, get_dim, label_subplots

    ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "unique_era5_points.nc"))

    figure_path = os.path.join(PROJECT_PATH, "img", "ibtracs")
    os.makedirs(figure_path, exist_ok=True)
    plot_defaults()

    fig, ax = plt.subplots(
        1,
        1,  # figsize=get_dim(ratio=1.1),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="green", edgecolor="black")
    ax.scatter(
        ds.longitude.values,
        ds.latitude.values,
        c="blue",
        s=0.1,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title("Unique points from ERA5 that correspond to IBTrACS data since 1980")
    plt.savefig(
        os.path.join(figure_path, "unique_points_era5_ibtracs.png"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    plot_defaults()
    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].hist(
        ds.longitude.values,
        bins=360,  # range=(-180, 180),
        alpha=0.5,
        label="Longitude [$^\circ$E]",
    )
    axs[1].hist(
        ds.latitude.values,
        bins=180,
        range=(-90, 90),
        alpha=0.5,
        label="Latitude [$^\circ$N]",
    )
    axs[0].set_xlabel("Longitude [$^\circ$E]")
    axs[1].set_xlabel("Latitude [$^\circ$N]")
    axs[0].set_ylabel("Count")
    label_subplots(axs)
    plt.savefig(
        os.path.join(figure_path, "unique_points_era5_ibtracs_hist.pdf"),
        bbox_inches="tight",
    )


@timeit
def era5_unique_points_raw() -> None:
    """Get the data from the unique points in the ERA5 data, in order to be able to calculate potential intensity and size.

    This function relies on os.path.join(IBTRACS_DATA_PATH, "unique_era5_points.nc") already existing.

    Need to do vectorized indexing, which in xarray only seems to work with sel rather
    than isel, hence we need to index by longitude, latitude, time.
    """
    if os.path.exists(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc")):
        print("ERA5 unique points data already exists.")
        return
    # open the unique points file
    unique_points_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "unique_era5_points.nc")
    )
    # open the era5 data file
    era5_ds = get_era5_combined()
    print("combined era5 data loaded")
    print(era5_ds)
    # get the unique points from the unique points file
    # unique_points = unique_points_ds.unique_dim.values
    # print(f"unique points shape is {unique_points.shape}")
    print("unique_points_ds:", unique_points_ds)
    # get the data from the era5 data file at the unique points
    # we want to vector select the era5 data with the unique points which are the indices of the era5 data
    # this is done by using the isel method of xarray
    era5_unique_data = era5_ds.sel(
        longitude=unique_points_ds.longitude,  # unique_points[:, 0],
        latitude=unique_points_ds.latitude,
        valid_time=unique_points_ds.time,
        method="nearest",  # maybe this doesn't need to be done
    )
    # the new dataset will have dimensions (u, p) where u is the number of unique points and p is the number of pressure levels
    era5_unique_data.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc"), engine="h5netcdf"
    )
    print(era5_unique_data)


@timeit
def example_plot_raw():
    # Let's make a 3 panel plot of the era5_unique_points_raw.nc data
    # plot it on global plattee carree projection with coastlines
    from sithom.plot import plot_defaults, label_subplots, get_dim
    import matplotlib.pyplot as plt
    import os
    from .constants import PROJECT_PATH
    from cartopy import crs as ccrs
    from cartopy.feature import NaturalEarthFeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    figure_path = os.path.join(PROJECT_PATH, "img", "ibtracs")
    os.makedirs(figure_path, exist_ok=True)
    era5_unique_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc")
    )
    print(era5_unique_data)

    plot_defaults()

    fig, axs = plt.subplots(
        3, 1, figsize=get_dim(ratio=1.1), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    label_subplots(axs)
    # (a) sst, (b) , (c) msl
    axs[0].add_feature(
        NaturalEarthFeature("physical", "land", "110m"),
        edgecolor="black",
        facecolor="green",
    )
    im = axs[0].scatter(
        era5_unique_data.longitude.values,
        era5_unique_data.latitude.values,
        c=era5_unique_data.sst.values[:],
        s=0.1,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    axs[0].set_title("SST [K]")
    plt.colorbar(im, ax=axs[0], orientation="vertical")  # , shrink=0.3
    # plot coastline
    # axs[0].coastlines()
    axs[1].add_feature(
        NaturalEarthFeature("physical", "land", "110m"),
        edgecolor="black",
        facecolor="green",
    )
    im = axs[1].scatter(
        era5_unique_data.longitude.values,
        era5_unique_data.latitude.values,
        c=era5_unique_data.msl.values[:] / 100,  # convert to hPa
        s=0.1,
        cmap="viridis_r",
        transform=ccrs.PlateCarree(),
    )
    axs[1].set_title("MSL [hPa]")
    plt.colorbar(im, ax=axs[1], orientation="vertical")

    axs[2].add_feature(
        NaturalEarthFeature("physical", "land", "110m"),
        edgecolor="black",
        facecolor="green",
    )
    im = axs[2].scatter(
        era5_unique_data.longitude.values,
        era5_unique_data.latitude.values,
        c=era5_unique_data.t2m.values[:],
        s=0.1,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    axs[2].set_title("T2M [K]")
    plt.colorbar(im, ax=axs[2], orientation="vertical")
    # facecolor green

    plt.savefig(os.path.join(figure_path, "era5_unique_points_raw.png"), dpi=300)


# open ibtracs data and era5 data
# loop through ibtracs data (by track, then by time (only if start time is after 1980, and end time is before end of 2024)) and find the closest era5 data point to the centre of the cyclone at that timestep.
# create a unique counter before looping through the ibtracs data
# collect the unique x/y/t era5 box points referenced in a hashmap where x,y, t are the x y and t coordinates of the cyclone centre with respect to the era5 grid
# index the unique counter each time a new unique x/y/t is found
# therefore hashmap (x, y, t) -> counter
# Also have a list [(x, y, t), ...] to keep track of the order of the unique points.
# add an additional variable to the ibracs data for this unique counter at each timestep
# (is nan if there was no valid centre data, or an integer otherwise)
# vector select the era5 data with the unique counter list so that the new era5 data has dimensions u (unique counter) and p (pressure level).
# take this era5 data and calculate the potential intensity and size of the cyclone at each timestep.
# should scale O(n*t) where n is the number of storms and t is the number of timesteps in each storm.


if __name__ == "__main__":
    # python -m tcpips.ibtracs
    # download_ibtracs_data()
    # print("IBTrACS data downloaded and ready for processing.")
    # ibtracs_to_era5_map()
    plot_unique_points()
    # era5_unique_points_raw()
    # example_plot_raw()
