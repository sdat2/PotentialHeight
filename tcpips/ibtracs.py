"""Process IBTrACS data to calculate along-track potential intensity and size from ERA5.

We only want tracks from 1980-2024 (inclusive). We currently calculate the potential intensity and size using monthly ERA5 data.

wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc

"""

from typing import Optional, Tuple, Union, List
import os
import numpy as np
from numpy.typing import ArrayLike
import scipy
import scipy.interpolate
import xarray as xr
import ujson
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import imageio
from datetime import datetime

try:
    import cartopy
    from cartopy import crs as ccrs
    from cartopy.feature import NaturalEarthFeature
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    CARTOPY_DIR = os.getenv("CARTOPY_DIR")
    if CARTOPY_DIR is not None and CARTOPY_DIR != "":
        print(f"Using Cartopy with CARTOPY_DIR: {CARTOPY_DIR}")
        cartopy.config["data_dir"] = CARTOPY_DIR
        os.makedirs(CARTOPY_DIR, exist_ok=True)  # Ensure the directory exists
    else:
        print("CARTOPY_DIR not set. Using default Cartopy data directory.")
except ImportError:
    print("Cartopy not installed. Geographic plotting will not work.")

from sithom.time import timeit
from sithom.plot import plot_defaults, label_subplots, get_dim
from w22.ps import parallelized_ps
from .constants import DATA_PATH, PROJECT_PATH
from .era5 import get_era5_coordinates, get_era5_combined
from .pi import calculate_pi
from .era5 import (
    preprocess_pressure_level_data,
    preprocess_single_level_data,
    get_era5_coordinates,
)


IBTRACS_DATA_PATH = os.path.join(DATA_PATH, "ibtracs")
os.makedirs(IBTRACS_DATA_PATH, exist_ok=True)
IBTRACS_DATA_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"
IBTRACS_DATA_FILE = os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
FIGURE_PATH = os.path.join(PROJECT_PATH, "img", "ibtracs")
os.makedirs(FIGURE_PATH, exist_ok=True)


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

    Currently takes around 36 seconds to run on laptop.

    Parallelization could be implemented if we ignored having a unique counter for each point, however this is not the rate limting step.

    TODO: Currently I only get ERA5 data up until the end of 2024, but IBTRACS data goes until mid 2025. This means that the last few storms will erroneously select december 2024 as the closest time point.
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

    ibtracs_ds["unique_counter"] = (("storm", "date_time"), u_lol)
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


def calculate_grid_avg_over_unique_points(
    ds: xr.Dataset, variables=("sst", "msl", "t2m")
) -> xr.Dataset:
    """Create for each gridpoints the average over all unique points.

    Args:
        ds (xr.Dataset): The unique point dataset to average over.
        variables (tuple): The variables to average over.

    Returns:
        xr.Dataset: The averaged dataset.

    This function will take the unique points dataset and average over the gridpoints
    of the ERA5 coordinates.
    """

    coords_ds = get_era5_coordinates()
    # get rid of time dimension
    del coords_ds["valid_time"]
    del coords_ds["expver"]

    # ---------------------------------------------------------------------
    # 1.  Handy aliases
    lon_grid = coords_ds.longitude.values  # (1440,)
    lat_grid = coords_ds.latitude.values  # (721,)   ─ descending in ERA5

    lon_pt = ds.longitude.values  # (N,)
    lat_pt = ds.latitude.values  # (N,)
    # ---------------------------------------------------------------------
    # 2.  Vectorised mapping (O(log n) each thanks to binary search in C)
    lon_idx = np.searchsorted(lon_grid, lon_pt)  # 0 … 1439
    lat_idx = np.searchsorted(lat_grid[::-1], lat_pt)  # 0 … 720   (grid is descending)
    lat_idx = lat_grid.size - 1 - lat_idx  # flip back so 0 == 90 °N
    # 3.  Pre-allocate target grids
    shape = (lon_grid.size, lat_grid.size)
    cnt = np.zeros(shape, dtype=np.int32)
    var_sum = {var: np.zeros(shape, dtype=np.float64) for var in variables}
    # 4.  One pass over the data – all heavy lifting in C
    np.add.at(cnt, (lon_idx, lat_idx), 1)
    for var in variables:
        np.add.at(var_sum[var], (lon_idx, lat_idx), ds[var].values)

    # 5.  Means (avoid /0 warnings)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_mean = {var: var_sum[var] / cnt for var in variables}
    # 6.  Build the Dataset
    data_vars = {var: (("longitude", "latitude"), var_mean[var]) for var in variables}
    data_vars["cnt"] = (("longitude", "latitude"), cnt)
    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(longitude=lon_grid, latitude=lat_grid),
        attrs=ds.attrs,  # keep global metadata
    )


def calculate_grid_avg_over_ibtracs_points(
    ibtracs_ds: xr.Dataset,
    vars: tuple = ("normalized_rmax", "normalized_vmax"),
) -> xr.Dataset:
    """
    Average the ibtracs data onto the era5 grid.

    Similar to the calculate_grid_avg_over_unique_points function, but
    not assuming the latitude and longitudes given are era5 grid points.

    Args:
        ibtracs_ds (xr.Dataset): The IBTrACS dataset to average. Each variable should have dimensions ("storm", "date_time"). The latitude and longitude coordinates similarly are have dimensions ("storm", "date_time"). The tracks are of variable lengths, and for those points the lons and lats are nans.
        vars (tuple, optional): The variables to average. Defaults to ("normalized_rmax", "normalized_vmax"). The variables should be in the ibtracs dataset.

    Returns:
        xr.Dataset: The averaged dataset with dimensions ("latitude", "longitude"). Will also return the count of points in each grid cell.
    """
    # average the ibtracs data onto the era5 grid
    lats_ibtracs = ibtracs_ds.lat.values
    lons_ibtracs = ibtracs_ds.lon.values
    # both shape (storm, time), but can be nan

    era5_ds = (
        get_era5_coordinates()
    )  # get the ERA5 coordinates so we can get the lon, lat grid
    lats_era5 = era5_ds["latitude"].values  # shape (Y,)
    lons_era5 = era5_ds["longitude"].values  # shape (X,)

    # count ibtracs points onto the grid of the era5 data
    era5_counts = np.zeros((len(lats_era5), len(lons_era5)), dtype=int)

    vars_sum = {
        var: np.zeros((len(lats_era5), len(lons_era5)), dtype=float) for var in vars
    }
    lat_res = (
        np.abs(lats_era5[1] - lats_era5[0])
        if len(lats_era5) > 1
        else 2 * np.abs(90 - lats_era5[0])
    )  # Handle single latitude case
    lon_res = (
        np.abs(lons_era5[1] - lons_era5[0])
        if len(lons_era5) > 1
        else 2 * np.abs(180 - lons_era5[0])
    )  # Handle single longitude case

    # Edges will be halfway between the centers.
    # For latitudes (Y axis):
    lat_edges = np.zeros(len(lats_era5) + 1)
    lat_edges[1:-1] = (lats_era5[:-1] + lats_era5[1:]) / 2.0
    lat_edges[0] = lats_era5[0] + lat_res / 2.0  # Upper edge of the first cell
    lat_edges[-1] = lats_era5[-1] - lat_res / 2.0  # Lower edge of the last cell
    # Ensure descending order for latitudes if lats_era5 is descending (typical for ERA5)
    if lats_era5[0] > lats_era5[-1]:  # If latitudes are decreasing (e.g., 90 to -90)
        lat_edges = np.sort(lat_edges)[::-1]
    else:  # If latitudes are increasing
        lat_edges = np.sort(lat_edges)

    # For longitudes (X axis):
    lon_edges = np.zeros(len(lons_era5) + 1)
    lon_edges[1:-1] = (lons_era5[:-1] + lons_era5[1:]) / 2.0
    lon_edges[0] = lons_era5[0] - lon_res / 2.0  # Left edge of the first cell
    lon_edges[-1] = lons_era5[-1] + lon_res / 2.0  # Right edge of the last cell
    # Ensure ascending order for longitudes
    if (
        lons_era5[0] > lons_era5[-1]
    ):  # If longitudes are decreasing (e.g. for some specific datasets)
        lon_edges = np.sort(lon_edges)[::-1]
    else:
        lon_edges = np.sort(lon_edges)

    # Handle longitude wrapping if your lons_era5 go from e.g. 0 to 360 and
    if np.min(lons_era5) >= 0 and np.max(lons_era5) <= 360:
        lons_ibtracs_processed = np.where(
            lons_ibtracs < 0, lons_ibtracs + 360, lons_ibtracs
        )
    else:
        lons_ibtracs_processed = lons_ibtracs

    # 3. Flatten the IBTrACS data and remove NaNs
    flat_lats_ibtracs = lats_ibtracs.flatten()
    flat_lons_ibtracs = lons_ibtracs_processed.flatten()

    valid_indices = ~np.isnan(flat_lats_ibtracs) & ~np.isnan(flat_lons_ibtracs)
    valid_lats = flat_lats_ibtracs[valid_indices]
    valid_lons = flat_lons_ibtracs[valid_indices]
    vars_valid = {var: ibtracs_ds[var].values.flatten()[valid_indices] for var in vars}

    if lat_edges[0] > lat_edges[-1]:
        # We reverse the edges and search for -valid_lats.

        lat_indices = np.digitize(valid_lats, lat_edges[::-1], right=True)
        lat_bin_indices = np.digitize(
            valid_lats, lat_edges[::-1], right=False
        )  # right=False: bins[i-1] <= x < bins[i]
        lat_indices = len(lats_era5) - lat_bin_indices

    else:  # lats_era5 is ascending (e.g., -90 to 90)
        # lat_edges is already ascending.
        # lat_indices = np.digitize(valid_lats, lat_edges, right=False) -1 # gives 0 to N-1 if point is within first to last bin
        lat_bin_indices = np.digitize(valid_lats, lat_edges, right=False)
        lat_indices = (
            lat_bin_indices - 1
        )  # Convert 1-based bin index to 0-based array index

    # For longitudes (assuming lons_era5 is 0 to 360, ascending)
    # lon_edges will be ascending.
    # lon_indices = np.digitize(valid_lons, lon_edges, right=False) -1
    lon_bin_indices = np.digitize(valid_lons, lon_edges, right=False)
    lon_indices = (
        lon_bin_indices - 1
    )  # Convert 1-based bin index to 0-based array index

    # Filter out points that fall outside the grid
    # np.digitize returns 0 if x < bins[0] and len(bins) if x >= bins[-1] (for 1-based from digitize)
    # So, for 0-based indices, we'd check for -1 and len(lats_era5) / len(lons_era5)
    valid_lat_idx = (lat_indices >= 0) & (lat_indices < len(lats_era5))
    valid_lon_idx = (lon_indices >= 0) & (lon_indices < len(lons_era5))
    valid_overall_idx = valid_lat_idx & valid_lon_idx

    final_lat_indices = lat_indices[valid_overall_idx]
    final_lon_indices = lon_indices[valid_overall_idx]

    # 5. Increment the counts in the era5_counts grid
    # np.add.at is useful for adding values to specific indices.
    # It handles cases where multiple points fall into the same cell by summing them up.
    np.add.at(era5_counts, (final_lat_indices, final_lon_indices), 1)

    for var in vars:
        np.add.at(
            vars_sum[var],
            (final_lat_indices, final_lon_indices),
            vars_valid[var][valid_overall_idx],
        )

    data_vars = {"count": (("latitude", "longitude"), era5_counts)}
    with np.errstate(invalid="ignore", divide="ignore"):
        for var in vars:
            data_vars[var] = (
                ("latitude", "longitude"),
                vars_sum[var] / era5_counts,
            )

    return xr.Dataset(
        data_vars=data_vars, coords={"latitude": lats_era5, "longitude": lons_era5}
    )


def plot_var_on_map(
    ax: plt.axis, da: xr.DataArray, label: str, cmap: str, shrink: float = 1, **kwargs
) -> None:
    """Plot a variable on a geographic map.

    Args:
        ax (plt.axis): The axis to plot on.
        da (xr.DataArray): The data array to plot.
        label (str): The label for the colorbar.
        cmap (str): The colormap to use.
        shrink (float): The size of the colorbar.
    """
    # add feature at back of plot
    ax.add_feature(
        NaturalEarthFeature("physical", "land", "110m"),
        edgecolor="black",
        facecolor="green",
        alpha=0.3,
        linewidth=0.5,
    )
    # add in LATITUDE_FORMATTER and LONGITUDE_FORMATTER
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xlocator = plt.MaxNLocator(4)
    gl.ylocator = plt.MaxNLocator(5)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    # only on bottom and left of plot
    gl.top_labels = False
    gl.right_labels = False
    # set the extent of the plot
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    # plot the data
    da = da.where(da > 0, np.nan)
    da.plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cmap=cmap,
        cbar_kwargs={
            "label": label,
            "cmap": cmap,
            "shrink": shrink,
        },
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        add_labels=True,
        rasterized=True,
        **kwargs,
    )
    ax.set_title("")  # remove title


@timeit
def plot_unique_points():
    # plot geographic distribution of unique points

    ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "unique_era5_points.nc"))

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
        os.path.join(FIGURE_PATH, "unique_points_era5_ibtracs.png"),
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
        os.path.join(FIGURE_PATH, "unique_points_era5_ibtracs_hist.pdf"),
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
def plot_example_raw():
    # Let's make a 3 panel plot of the era5_unique_points_raw.nc data
    # plot it on global plattee carree projection with coastlines

    era5_unique_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc")
    )
    # era5_unique_data = before_2025(era5_unique_data)

    era5_coords_ds = calculate_grid_avg_over_unique_points(
        era5_unique_data, variables=("sst", "msl", "t2m")
    )

    # create histogram of unique points back onto the lon/lat grid of ERA5
    # take all of the era5 unique data points
    # and work out if the longitude and latitudes correspond to a point
    plot_defaults()

    _, ax = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    plot_var_on_map(
        ax,
        era5_coords_ds["cnt"],
        "Number of Unique Gridpoints",
        "viridis_r",
        shrink=0.63,
    )
    plt.title(
        "Number of unique points from ERA5 that correspond to IBTrACS data since 1980"
    )
    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_count.pdf"),
        bbox_inches="tight",
        dpi=400,
    )
    plt.clf()
    plt.close()

    # let's plot the sst mean from era5_coords_ds
    _, ax = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    plot_var_on_map(
        ax,
        era5_coords_ds["sst"],
        "Mean SST [K]",
        "viridis",
        shrink=0.63,
    )
    plt.title("Mean Sea Surface Temperature from ERA5")

    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_sst.pdf"),
        bbox_inches="tight",
        dpi=400,
    )
    plt.clf()
    plt.close()
    # let's plot the msl mean from era5_coords_ds

    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )

    era5_coords_ds["msl"] /= 100  # convert to hPa
    era5_coords_ds["t2m"] -= 273.15  # convert to C
    era5_coords_ds["sst"] -= 273.15  # convert to C

    plot_var_on_map(
        axs[0], era5_coords_ds["sst"], "Mean $T_s$ [$^{\circ}$C]", "viridis", shrink=1
    )
    plot_var_on_map(
        axs[1], era5_coords_ds["msl"], "Mean $p_A$ [hPa]", "viridis_r", shrink=1
    )
    plot_var_on_map(
        axs[2], era5_coords_ds["t2m"], "Mean $T_2$ [$^{\circ}$C]", "viridis", shrink=1
    )
    label_subplots(axs)

    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_raw.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


@timeit
def process_era5_raw() -> None:
    """Process the raw ERA5 unique datapoints selected to include the key variables
    for calculating potential intensity and size."""

    era5_unique_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc")
    )
    era5_unique_data = preprocess_pressure_level_data(era5_unique_data)
    era5_unique_data = preprocess_single_level_data(era5_unique_data)
    print(era5_unique_data)
    era5_unique_data.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_processed.nc"),
        engine="h5netcdf",
    )
    print("ERA5 unique points data processed and saved.")


@timeit
def plot_era5_processed() -> None:
    """Plot the processed ERA5 unique datapoints selected to include the key variables
    for calculating potential intensity and size."""

    # lets do a 3 panel (rh, sst, msl) plot of the processed data
    # plot it on global plattee carree projection with coastlines

    era5_unique_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_processed.nc")
    )
    era5_unique_data = before_2025(era5_unique_data)
    era5_ds = calculate_grid_avg_over_unique_points(
        era5_unique_data, variables=("sst", "msl", "rh")
    )
    # create 3 panel plot of sst, msl, rh
    plot_defaults()
    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )

    plot_var_on_map(
        axs[0], era5_ds["sst"], "Mean SST [$^{\circ}$C]", "viridis", shrink=1
    )  # plot count of unique points
    plot_var_on_map(
        axs[1], era5_ds["msl"], "Mean MSL [hPa]", "viridis_r", shrink=1
    )  # plot count of unique points
    plot_var_on_map(
        axs[2], era5_ds["rh"], "Mean RH [fraction]", "viridis", shrink=1
    )  # plot count of unique points

    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_processed.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


@timeit
def calculate_potential_intensity():
    """Calculate the potential intensity of the cyclone at each timestep using the ERA5 data."""
    # open the processed data
    processed_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_processed.nc")
    )
    # calculate the potential intensity at the gradient wind level
    pi_ds = calculate_pi(processed_data, dim="pressure_level", V_reduc=1.0)

    print(pi_ds)
    pi_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_pi.nc"),
        engine="h5netcdf",
    )


@timeit
def plot_potential_intensity():
    """Provide example plots using the potential intensity data."""
    # plot the potential intensity data
    # plot it on global plattee carree projection with coastlines

    pi_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_pi.nc"))
    # get coords
    pi_ds = before_2025(pi_ds)  # remove data after 2024

    # create 3 panel plot of avg vmax, avg t0, avg otl
    grid_avg_ds = calculate_grid_avg_over_unique_points(
        pi_ds, variables=("vmax", "t0", "otl")
    )

    plot_defaults()

    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        sharex=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )  # specify 3 rows for the subplots

    plot_var_on_map(
        axs[0],
        grid_avg_ds["vmax"],
        "Mean Potential Intensity [m s$^{-1}$]",
        "viridis",
        shrink=1,
    )  # plot count of unique points
    plot_var_on_map(
        axs[1], grid_avg_ds["t0"], "Mean Outflow Temperature [K]", "viridis_r", shrink=1
    )  # plot count of unique points
    plot_var_on_map(
        axs[2], grid_avg_ds["otl"], "Mean Outflow Level [hPa]", "plasma", shrink=1
    )  # plot count of unique points
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_pi.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


@timeit
def calculate_potential_size(chunks: int = 20) -> None:
    """Calculate the size of the cyclone at each timestep using the ERA5 data.

    Args:
        chunks (int): The number of chunks to divide the unique points into for checkpointing.
    """
    proc_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_processed.nc")
    )
    pi_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_pi.nc"))
    combined_ds = xr.merge([proc_ds, pi_ds])
    combined_ds["pressure_assumption"] = "isothermal"
    combined_ds["ck_cd"] = 0.9
    combined_ds["w_cool"] = 0.002
    combined_ds["Cdvary"] = 0
    combined_ds["cd_vary"] = 0
    combined_ds["cd"] = 0.0015
    del combined_ds["t"]
    del combined_ds["q"]
    del combined_ds["pressure_level"]
    combined_ds = combined_ds.rename({"latitude": "lat", "longitude": "lon"})

    # This crashed the computer, so maybe we should do it in bits (say 20 bits), saving after each bit and then stich it together at the end.
    # we want to divide the unique data along the unique counter dimension "u", into 20 chunks
    # and then run the parallelized_ps function on each chunk
    # and save the results to a new netcdf file
    # this loop is in serial, and within each loop the task is parallelized

    os.makedirs(os.path.join(IBTRACS_DATA_PATH, "tmp"), exist_ok=True)
    chunk_size = len(combined_ds.u) // chunks

    for i in tqdm(range(0, chunks), desc="Calculating potential size in chunks"):
        # get the start and end index of the chunk
        file_name = os.path.join(
            IBTRACS_DATA_PATH, "tmp", f"era5_unique_points_ps_{i}.nc"
        )
        if os.path.exists(file_name):
            print(f"File {file_name} already exists, skipping.")
            continue
        start = i * chunk_size
        end = (i + 1) * chunk_size

        if i == chunks - 1:  # at end include all remaining data
            end = len(combined_ds.u)
        # get the chunk of data
        chunk_ds = combined_ds.isel(u=slice(start, end))
        # calculate the potential size for this chunk
        ps_chunk = parallelized_ps(chunk_ds, dryrun=False)
        # save the chunk to a netcdf file
        ps_chunk.to_netcdf(
            file_name,
            engine="h5netcdf",
        )

    # # load them all together and save them as one file
    ps_ds = xr.open_mfdataset(
        os.path.join(IBTRACS_DATA_PATH, "tmp", "era5_unique_points_ps_*.nc"),
        combine="by_coords",
        parallel=True,
    )
    # save the combined dataset
    ps_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps.nc"),
        engine="h5netcdf",
    )
    print("Potential size calculated and saved.")


def calculate_potential_size_cat1(vmin=33, v_reduc=0.8) -> None:
    """Calculate potential size, assuming vmax@10m is 33 m/s, the minimum for a category 1 cyclone. Only calculate the potential size where the potential intensity is above this threshold.

    Args:
        vmin (float): The minimum potential intensity to consider a cyclone as category 1. Defaults to 33 m/s.
        v_reduc (float): The reduction factor for the potential size. Defaults to 0.8.
    """
    pi_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_pi.nc"))
    proc_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_processed.nc")
    )
    combined_ds = xr.merge([pi_ds, proc_ds])
    combined_ds["pressure_assumption"] = "isothermal"
    combined_ds["ck_cd"] = 0.9
    combined_ds["w_cool"] = 0.002
    combined_ds["Cdvary"] = 0
    combined_ds["cd_vary"] = 0
    combined_ds["cd"] = 0.0015
    del combined_ds["t"]
    del combined_ds["q"]
    del combined_ds["pressure_level"]
    combined_ds = combined_ds.rename({"latitude": "lat", "longitude": "lon"})
    # if vmax < vmin/v_reduc then set to np.nan
    # if vmax > vmin/v_reduc, set to vmin/v_reduc
    combined_ds["vmax"] = combined_ds["vmax"].where(
        combined_ds["vmax"] >= vmin / v_reduc, np.nan
    )  # set to nan if below vmin (i.e. the potential intensity is below category 1)
    combined_ds["vmax"] = combined_ds["vmax"].where(
        combined_ds["vmax"] < vmin / v_reduc, vmin / v_reduc
    )  # set to vmin/v_reduc if above vmin

    ps_ds = parallelized_ps(
        combined_ds,
        dryrun=False,
    )
    ps_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps_cat1.nc"),
        engine="h5netcdf",
    )


def before_2025(ds: xr.Dataset) -> xr.Dataset:
    """Filter the dataset to only include data before 2025."""
    if "time" not in ds:
        raise ValueError("Dataset must contain a 'time' dimension.")
    return ds.where(ds.time < np.datetime64("2024-12-30"))


@timeit
def plot_potential_size() -> None:
    """Plot the potential size data."""

    ps_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps.nc"))
    ps_ds = before_2025(ps_ds)  # filter to only include data before 2025

    ps_ds = ps_ds.rename({"lat": "latitude", "lon": "longitude"})

    # calculate averages
    grid_avg_ds = calculate_grid_avg_over_unique_points(
        ps_ds,
        variables=("rmax", "r0", "pm"),
    )

    plot_defaults()
    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        sharex=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )  # specify 3 rows for the subplots

    plot_var_on_map(
        axs[0],
        grid_avg_ds["rmax"] / 1000,
        "Mean $r'_{max}$ [km]",
        "viridis",
        shrink=1,
        vmin=10,
        vmax=150,
    )
    plot_var_on_map(
        axs[1],
        grid_avg_ds["r0"] / 1000,
        "Mean $r_a$ [km]",
        "viridis",
        shrink=1,
        vmin=500,
        vmax=5000,
    )

    plot_var_on_map(
        axs[2],
        grid_avg_ds["pm"] / 100,
        "Mean $p_m$ [hPa]",
        "plasma",
        shrink=1,
    )
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_ps.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    print("Potential size plotted and saved.")

    # let's make a histogram of rmax and r0
    fig, axs = plt.subplots(1, 2, figsize=get_dim())
    axs[0].hist(
        grid_avg_ds["rmax"].values.ravel()[
            ~np.isnan(grid_avg_ds["rmax"].values.ravel())
        ]
        / 1000,  # convert to km
        bins=100,
        alpha=0.5,
        label=r"$r_3$ [km]",
    )
    axs[1].hist(
        grid_avg_ds["r0"].values.ravel()[~np.isnan(grid_avg_ds["r0"].values.ravel())]
        / 1000,  # convert to km
        bins=100,
        alpha=0.5,
        label=r"$r_a$ [km]",
    )
    axs[0].set_xlabel(r"$r_3$ [km]")
    axs[1].set_xlabel(r"$r_a$ [km]")
    axs[0].set_ylabel("Count")
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "era5_unique_points_ps_hist.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    plt.clf()


# Now that we have calculated potential size and potential intensity, we can compare them to the true size and intensity in the IBTrACS data.
# In particular, we will initially study the rmax (IBTrACS) / rmax (Potential Size) and vmax (IBTrACS) / vmax (Potential Intensity) relationship.
# We are initially inspired by the relatively simple study of normalized intensity in Emanuel 2000.


def add_pi_ps_back_onto_tracks(
    variables_to_map: Tuple[str] = ("rmax", "vmax", "r0", "pm", "otl", "t0"),
    input_name: Union[str, tuple, list] = "IBTrACS.since1980.v04r01.unique.nc",
    era5_name: str = "era5_unique_points_ps.nc",
    output_name: str = "IBTrACS.since1980.v04r01.pi_ps.nc",
):
    """We first need to use the unique counter to
    map the unique points back onto the IBTrACS data.

    Args:
        variables_to_map (Tuple[str], optional): The variables to map back onto the IBTrACS data. Defaults to ("rmax", "vmax", "r0", "pm", "otl", "t0").
        input_name (str, optional): The name of the input IBTrACS dataset with unique counter. Defaults to "IBTrACS.since1980.v04r01.unique.nc".
        output_name (str, optional): The name of the output IBTrACS dataset with potential intensity and size data added. Defaults to "IBTrACS.since1980.v04r01.pi_ps.nc".
    """
    # open dataset with the indexing
    ibtracs_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, input_name))
    # open the potential size and intensity datasets
    if isinstance(era5_name, str):
        ps_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, era5_name))
    elif isinstance(era5_name, Union[list, tuple]):
        # If era5_name is a list or tuple, open all datasets and concatenate them
        ps_ds = xr.open_mfdataset(
            [os.path.join(IBTRACS_DATA_PATH, name) for name in era5_name],
            parallel=True,
        )
    else:
        raise ValueError(
            "era5_name must be a string or a list/tuple of strings representing file names."
        )

    # Get the unique_counter numpy array from the xarray Dataset
    # This array has dimensions ("storm", "date_time") and contains integers or np.nan
    unique_counter_np = ibtracs_ds["unique_counter"].data

    for var in variables_to_map:
        if var not in ps_ds:
            print(f"Warning: Variable '{var}' not found in ps_ds. Skipping.")
            continue

        source_data = ps_ds[var].data  # This is a 1D numpy array

        if source_data.ndim != 1:
            raise ValueError(f"Source data for variable '{var}' is not 1-dimensional.")

        # 1. Initialize an output array with NaNs, matching the shape of unique_counter_np
        #    Ensure the dtype can hold NaNs (float). If source_data is already float, use its dtype.
        output_dtype = (
            source_data.dtype
            if np.issubdtype(source_data.dtype, np.floating)
            else np.float64
        )
        mapped_values = np.full(unique_counter_np.shape, np.nan, dtype=output_dtype)

        # 2. Create a mask for non-NaN values in unique_counter_np
        #    This mask will have the same shape as unique_counter_np (e.g., ("storm", "date_time"))
        non_nan_mask = ~np.isnan(unique_counter_np)

        # 3. Extract the actual integer indices from unique_counter_np where it's not NaN
        #    These are the values from unique_counter_np that will be used to index source_data.
        #    This will be a 1D array of integers.
        valid_indices = unique_counter_np[non_nan_mask].astype(int)

        # Basic check for index bounds.
        # Assumes valid_indices should be within the bounds of source_data.
        if source_data.size > 0:  # Proceed only if source_data is not empty
            if np.any(valid_indices < 0) or np.any(valid_indices >= source_data.size):
                # This indicates a potential issue with the unique_counter values or ps_ds data.
                # For instance, a non-NaN value in unique_counter might point outside the bounds of ps_ds[var].
                # You might want to handle this more gracefully, e.g., by logging a warning,
                # raising a more specific error, or setting corresponding values to NaN.
                # For this example, we'll raise an error if out of bounds.
                problematic_indices = valid_indices[
                    (valid_indices < 0) | (valid_indices >= source_data.size)
                ]
                raise IndexError(
                    f"Variable '{var}': unique_counter contains out-of-bounds indices: {np.unique(problematic_indices)}. "
                    f"Max valid index for ps_ds['{var}'] is {source_data.size - 1}."
                )

            # 4. Fetch data from source_data using the valid integer indices
            #    This will be a 1D array.
            fetched_values = source_data[valid_indices]

            # 5. Place the fetched values into the mapped_values array at the non-NaN positions
            mapped_values[non_nan_mask] = fetched_values
        # If source_data is empty, mapped_values remains all NaNs, which is likely the correct behavior.

        # Add the new variable (with mapped values) to the ibtracs_ds Dataset
        ibtracs_ds[var] = (("storm", "date_time"), mapped_values)

    # Save the updated dataset
    ibtracs_ds.to_netcdf(os.path.join(IBTRACS_DATA_PATH, output_name))
    print("Successfully added PI and PS data back onto tracks and saved the dataset.")


def create_normalized_variables(v_reduc: float = 0.8) -> None:
    """Create normalized variables for rmax and vmax.

    Args:
        v_reduc (float, optional): The reduction factor for vmax to go from potential intensity at the gradient wind to potential intensity and the 10m wind. Defaults to 0.8.
    """

    # open the dataset with the pi and ps data
    pi_ps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.pi_ps.nc")
    )
    # set vmax to nan if it is 0
    pi_ps_ds["vmax"] = pi_ps_ds["vmax"].where(pi_ps_ds["vmax"] > 0)
    print(pi_ps_ds["usa_rmw"])
    print(pi_ps_ds["usa_wind"])
    pi_ps_ds["normalized_rmax"] = (
        ("storm", "date_time"),
        pi_ps_ds["usa_rmw"].values * 1852 / pi_ps_ds["rmax"].values,
    )  # Asume usa_rmw is in nautical miles, rmax is in m

    pi_ps_ds["normalized_vmax"] = (
        ("storm", "date_time"),
        pi_ps_ds["usa_wind"].values * 0.514444 / (pi_ps_ds["vmax"].values * v_reduc),
    )  # Assume usa_wind is in knots at u10, vmax is in m/s at gradient wind
    print("nanmean vmax:", pi_ps_ds["normalized_vmax"].mean())
    print("nanmean rmax:", pi_ps_ds["normalized_rmax"].mean())
    print("nanmax vmax:", pi_ps_ds["normalized_vmax"].max())
    print("nanmax rmax:", pi_ps_ds["normalized_rmax"].max())

    pi_ps_ds["normalized_rmax"].attrs["units"] = "dimensionless"
    pi_ps_ds["normalized_vmax"].attrs["units"] = "dimensionless"
    pi_ps_ds["normalized_rmax"].attrs["description"] = "Normalized Rmax"
    pi_ps_ds["normalized_vmax"].attrs["description"] = "Normalized Vmax"

    # save the dataset
    pi_ps_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.normalized.nc")
    )


def plot_normalized_variables() -> None:
    """Plot the normalized variables."""
    # open the dataset with the pi and ps data
    pi_ps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.normalized.nc")
    )
    pi_ps_ds = before_2025(pi_ps_ds)  # filter to only include data before 2025
    avg_ds = calculate_grid_avg_over_ibtracs_points(
        pi_ps_ds, vars=("normalized_rmax", "normalized_vmax")
    )

    # let's plot two histograms of the normalized rmax and vmax
    # create 2 panel plot of normalized rmax and vmax
    plot_defaults()
    _, axs = plt.subplots(1, 2, figsize=get_dim(), sharex=True, sharey=True)
    # select nature as "TC"
    # print(pi_ps_ds["nature"])
    # print(pi_ps_ds["nature"].attrs)
    pi_ps_ds = pi_ps_ds.where(pi_ps_ds["nature"] == b"TS")

    axs[0].hist(
        pi_ps_ds["normalized_rmax"].values.ravel()[
            ~np.isnan(pi_ps_ds["normalized_rmax"].values.ravel())
        ],
        bins=100,
        range=(0, 5),
        alpha=0.5,
    )
    axs[1].hist(
        pi_ps_ds["normalized_vmax"].values.ravel()[
            ~np.isnan(pi_ps_ds["normalized_vmax"].values.ravel())
        ],
        bins=100,
        range=(0, 5),
        alpha=0.5,
    )
    axs[0].set_xlabel(r"Normalized $r_{\mathrm{Obs.}}/r_3$")
    axs[1].set_xlabel(r"Normalized ${V_{\mathrm{Obs.}}}/{V_{p}}$")
    axs[0].set_ylabel("Count")

    label_subplots(axs)

    plt.savefig(
        os.path.join(FIGURE_PATH, "normalized_rmax_vmax_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # plot the normalized rmax and vmax
    # create 2 panel plot of averaged normalized rmax and vmax on a map
    # first we need to calculate the average over the unique points

    plot_defaults()
    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )  # specify 2 rows for the subplots

    plot_var_on_map(
        axs[0],
        avg_ds["count"],
        "Count of TC Points",
        "viridis_r",
        shrink=1,
    )

    plot_var_on_map(
        axs[1],
        avg_ds["normalized_rmax"],
        r"Mean Normalized Size, $\frac{r_{\mathrm{Obs.}}}{r_3}$",
        "viridis",
        shrink=1,
    )
    plot_var_on_map(
        axs[2],
        avg_ds["normalized_vmax"],
        r"Mean Normalized Intensity, $\frac{V_\mathrm{{max}}}{V_{p}}$",
        "viridis_r",
        shrink=1,
    )
    plt.tight_layout()
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "normalized_rmax_vmax.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    print("Normalized variables plotted and saved.")

    # now let's remake the count histograms with this point data from the avg_ds
    # I just want to plot histograms of occurences of TCs in each latitude and longitude
    fig, axs = plt.subplots(1, 2, figsize=get_dim())
    axs[0].plot(avg_ds["latitude"], avg_ds["count"].sum(dim="longitude"))

    axs[1].plot(avg_ds["longitude"], avg_ds["count"].sum(dim="latitude"))

    axs[0].set_xlabel(r"Latitude [$^{\circ}$N]")
    axs[1].set_xlabel(r"Longitude [$^{\circ}$E]")
    axs[0].set_xlim(-80, 80)
    axs[1].set_xlim(0, 360)
    axs[0].set_ylabel("Count of IBTrACS Points per 1/4$^{\circ}$")
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "ibtracs_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    print(avg_ds)
    print(pi_ps_ds)


def calculate_cps(v_reduc: float = 0.8, test=False) -> None:
    """Calculate the corresponding potential size, assuming the windspeed from usa_wind/V_reduc instead of the potential intensity. This will mean that we have to do many more potential size calculations.

    Args:
        v_reduc (float, optional): The reduction factor for vmax to go from potential intensity at the gradient wind to potential intensity and the 10m wind. Defaults to 0.8.
        test (bool, optional): If True, will run an autofail run to check dimensions match etc.
    """

    add_pi_ps_back_onto_tracks(
        era5_name=("era5_unique_points_pi.nc", "era5_unique_points_processed.nc"),
        output_name="IBTrACS.since1980.v04r01.cps_inputs.nc",
        variables_to_map=("t0", "sst", "msl", "rh"),
    )

    ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps_inputs.nc")
    )

    ds["pressure_assumption"] = "isothermal"
    ds["ck_cd"] = 0.9
    ds["w_cool"] = 0.002
    ds["cd_vary"] = 0
    ds["cd"] = 0.0015
    print(ds)
    print(ds.variables)
    # knots to m/s, then divide by v_reduc to get gradient wind speed
    ds["vmax"] = ds["usa_wind"] * 0.514444 / v_reduc  # convert to m/s
    ds["vmax"].attrs = {
        "units": "m/s",
        "description": "Vmax at gradient wind, converted from usa_wind",
    }
    if "rmax" in ds:
        del ds["rmax"]
    if "r0" in ds:
        del ds["r0"]
    print(ds["vmax"])
    print("non-nan count:", np.sum(~np.isnan(ds["vmax"])))
    for var in ["sst", "msl", "rh", "vmax"]:
        if var not in ds:
            print(f"Warning: Variable '{var}' not found in ds. Skipping.")
            continue
        else:
            print(f"Variable '{var}' found in ds.")
            print(
                f"Variable '{var}' has shape {ds[var].shape} and dtype {ds[var].dtype}"
            )

    # let's process the dataset.
    ds = parallelized_ps(
        ds[
            [
                "sst",
                "msl",
                "rh",
                "vmax",
                "t0",
                "pressure_assumption",
                "ck_cd",
                "w_cool",
                "cd_vary",
                "cd",
            ]
        ],
        dryrun=False,
        autofail=test,  # if test is True, will run an autofail run to check dimensions match etc.
    )
    if not test:  # save the dataset if not in test mode
        ds.to_netcdf(
            os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc"),
            engine="h5netcdf",
        )

    return None


def plot_cps():
    """Plot the corresponding potential size, assuming the windspeed from usa_wind/V_reduc instead of the potential intensity."""
    # open the dataset with the cps data
    cps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc")
    )
    cps_ds = before_2025(cps_ds)  # filter to only include data before 2025
    avg_ds = calculate_grid_avg_over_ibtracs_points(
        cps_ds, vars=("rmax", "vmax", "r0", "pm", "t0")
    )

    # let's plot two histograms of the normalized rmax and vmax
    # create 2 panel plot of normalized rmax and vmax
    plot_defaults()
    _, axs = plt.subplots(1, 2, figsize=get_dim(), sharey=True)
    axs[0].hist(
        avg_ds["rmax"].values.ravel()[~np.isnan(avg_ds["rmax"].values.ravel())] / 1000,
        bins=100,
        alpha=0.5,
        label="Rmax",
    )
    axs[1].hist(
        avg_ds["vmax"].values.ravel()[~np.isnan(avg_ds["vmax"].values.ravel())],
        bins=100,
        alpha=0.5,
        label="Vmax",
    )
    axs[0].set_xlabel(r"$r_{\mathrm{Obs.}}$ [km]")
    axs[1].set_xlabel(r"$V_{\mathrm{Obs.}}$ [m s$^{-1}$]")
    axs[0].set_ylabel("Count")

    label_subplots(axs)

    plt.savefig(
        os.path.join(FIGURE_PATH, "cps_rmax_vmax_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # plot the rmax and vmax
    # create 2 panel plot of averaged rmax and vmax on a map
    # first we need to calculate the average over the unique points

    plot_defaults()
    _, axs = plt.subplots(
        3,
        1,
        figsize=get_dim(ratio=1.1),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )  # specify 2 rows for the subplots

    plot_var_on_map(
        axs[0],
        avg_ds["count"],
        "Count of TC Points",
        "viridis_r",
        shrink=1,
    )
    plot_var_on_map(
        axs[1],
        avg_ds["rmax"] / 1000,
        r"Mean $r_2$ [km]",
        "viridis",
        shrink=1,
    )
    plot_var_on_map(
        axs[2],
        avg_ds["vmax"],
        r"Mean $V_{\mathrm{Obs.}}$ [m/s]",
        "viridis_r",
        shrink=1,
    )
    plt.tight_layout()
    label_subplots(axs)
    plt.savefig(
        os.path.join(FIGURE_PATH, "cps_rmax_vmax.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    print("CPS variables plotted and saved.")


def plot_normalized_cps():
    """Plot the normalized rmax for the CPS data."""
    plot_defaults()
    cps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc")
    )
    cps_ds = before_2025(cps_ds)  # filter to only include data before 2025
    print("cps_ds", cps_ds.sizes)

    # load usa_rmw data
    ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
    )[
        "usa_rmw"
    ]  # this has the usa_rmw data
    print("ds", ds.sizes)
    print(ds.values.shape)
    cps_ds["usa_rmw"] = (("storm", "date_time"), ds.values)

    # let's now calculate the normalized rmax for cps
    cps_ds["normalized_rmax"] = (
        ("storm", "date_time"),
        cps_ds["usa_rmw"].values * 1852 / cps_ds["rmax"].values,
    )  # Asume usa_rmw is in nautical miles, rmax is in m

    # let's plot the histogram of the normalized rmax
    fig, ax = plt.subplots(figsize=get_dim())
    ax.hist(
        cps_ds["normalized_rmax"].values.ravel()[
            ~np.isnan(cps_ds["normalized_rmax"].values.ravel())
        ],
        bins=100,
        range=(0, 5),
        alpha=0.5,
        label="Normalized Rmax",
    )
    ax.set_xlabel(r"Normalized $r_{\mathrm{Obs.}}/r_2$")
    ax.set_ylabel("Count")
    plt.savefig(
        os.path.join(FIGURE_PATH, "cps_normalized_rmax_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def _union(lst1: list, lst2: list) -> list:
    """
    Union of lists.
    """
    return list(set(lst1) | set(lst2))


def _intersection(lst1: list, lst2: list) -> list:
    """
    Intersection of lists.
    """
    return list(set(lst1).intersection(set(lst2)))


def filter_by_labels(
    ds: xr.Dataset,
    filter: List[Tuple[str, List[str]]] = [
        ("basin", [b"NA"]),
        ("subbasin", [b"GM"]),
        ("nature", [b"TS"]),
        ("usa_record", [b"L"]),
    ],
) -> xr.Dataset:
    """
    Filter by labels for IBTrACS.

    Args:
        ds (xr.DataArray): Input ibtracs datarray.
        filter (List[Tuple[str,List[str]]], optional): Filters to apply.
            Defaults to [("basin", [b"NA"]), ("nature", [b"SS", b"TS"])].

    Returns:
        xr.Dataset: Filtered dataset.
    """
    storm_list = None
    for filter_part in filter:
        # print(filter_part)
        storm_list_part = None
        for value in filter_part[1]:
            truth_array = ds[filter_part[0]] == value
            # print(truth_array.values.shape)
            if len(truth_array.shape) != 1:
                compressed_array = np.any(truth_array, axis=1)
            else:
                compressed_array = truth_array
            # print(compressed_array.shape)
            storm_list_temp = ds.storm.values[compressed_array]
            if storm_list_part is None:
                storm_list_part = storm_list_temp
            else:
                storm_list_part = _union(storm_list_temp, storm_list_part)
            # print(len(storm_list_part))
        if storm_list is None:
            storm_list = storm_list_part
        else:
            storm_list = _intersection(storm_list_part, storm_list)
    # print(len(storm_list))
    return ds.sel(storm=storm_list)


def landings_only(ds: xr.Dataset) -> Optional[xr.Dataset]:
    """
    Extract a reduced dataset based on those points at which a landing occurs.

    Args:
        ds (xr.Dataset): Individual storm.

    Returns:
        Optional[xr.Dataset]: Clipped storm. If there are no tropical cyclones
            hitting the coatline the coastline then None is returned.
    """
    date_times = np.all(
        [(ds["usa_record"].values == b"L"), (ds["usa_sshs"].values > 0)], axis=0
    ).ravel()
    if np.any(date_times):
        return ds.isel(date_time=date_times)
    else:
        return None


def select_tc_from_ds(
    ds: xr.Dataset, name: str = b"KATRINA", basin: str = b"NA", subbasin: str = b"GM"
) -> Optional[xr.Dataset]:
    """
    Select a tropical cyclone from the dataset.

    Args:
        ds (xr.Dataset): Input dataset.
        name (str, optional): Name of the tropical cyclone. Defaults to "KATRINA".
        basin (str, optional): Basin of the tropical cyclone. Defaults to "NA".
        subbasin (str, optional): Subbasin of the tropical cyclone. Defaults to "GM".

    Returns:
        xr.Dataset: Dataset containing only a particular tropical cyclone's data.
    """
    out_ds = filter_by_labels(
        ds,
        filter=[
            ("name", [name]),
            ("basin", [basin]),
            # ("subbasin", [subbasin]),
            ("nature", [b"TS"]),
            # ("usa_record", [b"L"]),
        ],
    )
    if len(out_ds.storm) == 0:
        print(
            f"No tropical cyclone found with name {name} in basin {basin} and subbasin {subbasin}."
        )
        print()
        return out_ds  # will return None
    elif len(out_ds.storm) == 1:
        print(
            f"Found tropical cyclone {name} in basin {basin} and subbasin {subbasin}."
        )
        return out_ds
    else:
        print(
            f"Multiple tropical cyclones found with name {name} in basin {basin} and subbasin {subbasin}. Found {len(out_ds.storm)} storms."
        )
        print("selecting last storm, as assumed most important.")
        # start dates of the three storms
        for i in range(len(out_ds.storm)):
            print(
                f"Storm {i}: {out_ds.storm.values[i]} starts at {out_ds.isel(storm=i).time.values[0]}"
            )

        # for some reason they are not ordered by time, so to select the last storm we form list
        start_list = [
            out_ds.isel(storm=i).time.values[0] for i in range(len(out_ds.storm))
        ]
        # find argmax of start_list
        max_index = np.argmax(start_list)

        return out_ds.isel(
            storm=max_index
        )  # Select the storm with the latest start time


def add_saffir_simpson_boundaries(
    ax, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1
):
    """
    Adds dashed horizontal lines to a Matplotlib Axes object at the
    Saffir-Simpson hurricane category boundaries (wind speeds in m/s).

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        color (str, optional): Color of the lines. Defaults to 'gray'.
        linestyle (str, optional): Style of the lines. Defaults to '--'.
        linewidth (float, optional): Width of the lines. Defaults to 0.8.
        alpha (float, optional): Transparency of the lines. Defaults to 0.7.
        zorder (int, optional): The zorder for drawing the lines. Lines with lower
                               zorder are drawn first. Defaults to 1 (typically
                               behind most plotted data).
    """
    saffir_simpson_boundaries_ms = {
        "Cat 1": 33,
        "Cat 2": 43,
        "Cat 3": 50,
        "Cat 4": 58,
        "Cat 5": 70,
        "Cat 6": 86,  # This is not an official category, but used for illustration, https://www.pnas.org/doi/epub/10.1073/pnas.2322597121
    }

    # Get the current y-axis limits to decide where to place text labels
    ymin, ymax = ax.get_ylim()

    for category, wind_speed_ms in saffir_simpson_boundaries_ms.items():
        ax.axhline(
            y=wind_speed_ms,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        if ymin < wind_speed_ms < ymax:
            ax.text(
                ax.get_xlim()[0]
                + (ax.get_xlim()[1] - ax.get_xlim()[0])
                * 0.05,  # x-position (slightly inside right edge)
                wind_speed_ms
                - (ymax - ymin) * 0.01,  # y-position (slightly above the line)
                f"{category}",
                color=color,
                fontsize=5,
                verticalalignment="bottom",
                horizontalalignment="left",
                # transform=ax,  # Use Axes coordinates for text placement
            )


def highlight_rapid_intensification(
    ax: plt.Axes,
    times: ArrayLike,
    wind_speeds: ArrayLike,
    ri_threshold_knots: float = 30,
    ri_period_hours: float = 24,
    color: str = "red",
    alpha: float = 0.3,
    zorder: int = 0,
) -> None:
    """
    Highlights periods of rapid intensification (RI) on a Matplotlib Axes object.

    RI is defined as an increase in wind speed by at least `ri_threshold_knots`
    over a `ri_period_hours` duration. The highlighted period on the graph
    will be from (t - ri_period_hours) to t, where t is the time RI is achieved.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        times (array-like): 1D array of time values (e.g., in hours).
                           Must be sorted in ascending order.
        wind_speeds (array-like): 1D array of wind speeds (e.g., in m/s),
                                  corresponding to `times`.
        ri_threshold_knots (float, optional): The wind speed increase threshold
                                             for RI in knots. Defaults to 30 knots.
        ri_period_hours (float, optional): The time period over which the
                                          intensification is measured, in hours.
                                          Defaults to 24 hours.
        color (str, optional): Color for the highlighted regions. Defaults to 'gold'.
        alpha (float, optional): Transparency for the highlighted regions. Defaults to 0.3.
        zorder (int, optional): The zorder for drawing the highlighted regions.
                                Lower zorder is drawn first. Defaults to 0 (behind grid/plots).
    """
    if not len(times) == len(wind_speeds):
        raise ValueError("times and wind_speeds must have the same length.")
    if len(times) < 2:
        return  # Not enough data to calculate RI

    # Ensure times is a numpy array for interpolation and calculations
    times_np = np.asarray(times).astype(float)
    wind_speeds_np = np.asarray(wind_speeds)
    # strip out nans from times
    valid_indices = ~np.isnan(times_np) & ~np.isnan(wind_speeds_np)
    times_np = times_np[valid_indices]
    wind_speeds_np = wind_speeds_np[valid_indices]

    if not np.all(np.diff(times_np) >= 0):
        # np.interp requires monotonically increasing x-values.
        # If diff is 0 for some, interp still works but it's good practice for time to be strictly increasing.
        # For simplicity, we'll allow non-decreasing. If strictly increasing is needed, use > 0.
        print(
            "Warning: 'times' array should ideally be strictly increasing for reliable interpolation."
        )

    knots_to_ms = 0.514444  # Conversion factor
    ri_threshold_ms = ri_threshold_knots * knots_to_ms

    identified_ri_periods = []

    for j in range(len(times_np)):
        t_current = times_np[j]
        ws_current = wind_speeds_np[j]

        t_lookback = t_current - ri_period_hours

        if t_lookback >= times_np[0]:

            ws_at_lookback = np.interp(t_lookback, times_np, wind_speeds_np)

            wind_increase = ws_current - ws_at_lookback

            if wind_increase >= ri_threshold_ms:
                # Additional check: Ensure not actively weakening at the current time step
                actively_weakening_at_current_step = False
                if (
                    j > 0
                ):  # Check against previous step if not the first point considered for RI
                    if wind_speeds_np[j] < wind_speeds_np[j - 1]:
                        actively_weakening_at_current_step = True
                # If j == 0, this specific check for weakening is not applicable,
                # but the t_lookback >= times_np[0] condition usually means j > 0
                # by the time we can evaluate a full ri_period_hours.

                if not actively_weakening_at_current_step:
                    identified_ri_periods.append((t_lookback, t_current))

    if not identified_ri_periods:
        return

    identified_ri_periods.sort(key=lambda x: x[0])
    merged_ri_periods = []
    if identified_ri_periods:
        current_start, current_end = identified_ri_periods[0]
        for i in range(1, len(identified_ri_periods)):
            next_start, next_end = identified_ri_periods[i]
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_ri_periods.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_ri_periods.append((current_start, current_end))

    # Draw the merged RI periods
    for i, (start_time, end_time) in enumerate(merged_ri_periods):
        ax.axvspan(
            start_time,
            end_time,
            color=color,
            alpha=alpha,
            zorder=zorder,
            label="Rapid Intensification" if i == 0 else "_nolegend_",
        )


@timeit
def plot_tc_example(
    name: str = b"KATRINA",
    basin: str = b"NA",
    subbasin: str = b"GM",
    bbox: Optional[Tuple[float, float, float, float]] = (-92.5, -72.5, 22.5, 37.5),
    landing_no: int = -1,  # -1 means last landing
) -> None:
    """Plot an example of a TC's track with the potential intensity
    and potential size, and corresponding potential size.

    Args:
        name (bytes, optional): Name of the tropical cyclone to plot. Defaults to b"KATRINA".
        basin (bytes, optional): Basin of the tropical cyclone to plot. Defaults to b"NA".
        subbasin (bytes, optional): Subbasin of the tropical cyclone to plot. Defaults to b"GM".
        bbox (Optional[Tuple[float, float, float, float]], optional): Bounding box for the map. Defaults to (-92.5, -72.5, 22.5, 37.5).
        landing_no (int, optional): Index of the landing to plot. Defaults to -1 (last landing).
    """
    # Load the IBTrACS dataset with potential intensity and size data
    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.pi_ps.nc")
    )
    ibtracs_orig = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
    )
    cps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc")
    )
    ps_cat1_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.ps_cat1.nc")
    )
    # add name, basin, subbasin, nature, and usa_record to the cps_ds

    for var in ["name", "basin", "subbasin", "nature", "usa_record"]:
        cps_ds[var] = ibtracs_orig[var]

    # Select Katrina's data
    tc_ds = select_tc_from_ds(ibtracs_ds, name=name, basin=basin, subbasin=subbasin)
    # print("tc_ds", tc_ds)
    tc_cps_ds = select_tc_from_ds(cps_ds, name=name, basin=basin, subbasin=subbasin)

    ps_cat1_ds = select_tc_from_ds(
        ps_cat1_ds, name=name, basin=basin, subbasin=subbasin
    )
    # print("tc_cps_ds", tc_cps_ds)
    tc_ds["ps_vobs"] = tc_cps_ds["rmax"]  # take the rmax from the cps dataset
    tc_ds["ps_cat1"] = ps_cat1_ds["rmax"]  # take the rmax from the ps_cat1 dataset
    track_dir = os.path.join(FIGURE_PATH, "tracks")
    os.makedirs(track_dir, exist_ok=True)
    plot_name = os.path.join(track_dir, f"{name.decode().lower()}_track.pdf")
    # print(plot_name)
    _plot_tc_track(tc_ds, plot_name, bbox=bbox, landing_no=landing_no)


def plot_tc_track_by_index(index, plot_name):
    """Plot a tropical cyclone track by index.

    Args:
        index (int): Index of the tropical cyclone to plot.
        plot_name (str): Name of the plot file to save.
    """
    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.pi_ps.nc")
    )
    cps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc")
    )
    ps_cat1_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.ps_cat1.nc")
    )
    tc_ds = ibtracs_ds.isel(storm=index)
    tc_cps_ds = cps_ds.isel(storm=index)
    tc_cat1_ds = ps_cat1_ds.isel(storm=index)
    # add rmax from cps_ds to tc_ds
    tc_ds["ps_vobs"] = tc_cps_ds["rmax"]  # take the rmax from the cps dataset
    tc_ds["ps_cat1"] = tc_cat1_ds["rmax"]  # take the rmax from the ps_cat1 dataset

    _plot_tc_track(tc_ds, plot_name)


def _plot_tc_track(
    tc_ds: xr.Dataset,
    plot_name: str,
    bbox: Optional[tuple] = None,
    landing_no: int = -1,
) -> None:
    """Plot the track of a tropical cyclone.

    Args:
        tc_ds (xr.Dataset): Dataset containing the tropical cyclone data.
        plot_name (str): Name of the plot file to save.
        bbox (Optional[tuple], optional): Bounding box for the map. Defaults to None.
        landing_no (int, optional): Index of the landing to plot. Defaults to -1 (last landing).
    """

    if bbox is None:
        perc = 0.1
        min_lon = np.nanmin(tc_ds["lon"].values)
        max_lon = np.nanmax(tc_ds["lon"].values)
        min_lat = np.nanmin(tc_ds["lat"].values)
        max_lat = np.nanmax(tc_ds["lat"].values)
        min_lon = min_lon - perc * (max_lon - min_lon)
        max_lon = max_lon + perc * (max_lon - min_lon) / (1 + perc)
        min_lat = min_lat - perc * (max_lat - min_lat)
        max_lat = max_lat + perc * (max_lat - min_lat) / (1 + perc)
        bbox = (min_lon, max_lon, min_lat, max_lat)

    # Plotting
    plot_defaults()
    fig = plt.figure(figsize=get_dim(ratio=1.1))

    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[3, 1, 1], width_ratios=[100, 1]
    )

    landings = landings_only(tc_ds)
    if landings is not None:
        tc_impact = landings.isel(date_time=landing_no)
    else:
        # find last time that is not nan
        non_nan_times = ~np.isnan(tc_ds["usa_wind"].values.ravel())
        # last true value in non_nan_times
        last_true_index = np.where(non_nan_times)[0][-1]
        tc_impact = tc_ds.isel(date_time=last_true_index)
    print("time at start:", tc_ds["time"].values.ravel()[0])
    print("time of landfall:", tc_impact["time"].values)

    times = (
        (tc_ds["time"].values - tc_impact["time"].values) / 1e9 / 60 / 60
    )  # nanoseconds to hours
    # --- Top Subplot: Geographic Map ---
    # The projection was in subplot_kw, now specified directly for this subplot
    ax_map = fig.add_subplot(
        gs[0, :], projection=ccrs.PlateCarree(central_longitude=(bbox[0] + bbox[1]) / 2)
    )

    ax_map.set_extent(list(bbox), crs=ccrs.PlateCarree())

    ax_map.plot(
        tc_ds["lon"].values.ravel(),
        tc_ds["lat"].values.ravel(),
        color="grey",
        linewidth=0.5,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax_map.scatter(
        tc_impact["lon"].values,
        tc_impact["lat"].values,
        color="green",
        marker="+",
        s=20,
        transform=ccrs.PlateCarree(),
    )
    vmin = np.nanmin(times)
    vmax = np.nanmax(times)
    if vmin >= 0:
        vmin = -1
    if vmax <= 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    img = ax_map.scatter(
        tc_ds["lon"],
        tc_ds["lat"],
        c=times,
        s=4,
        marker="x",
        cmap="cmo.balance",
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(img, ax=ax_map, shrink=0.94, label="Time since impact [hours]")

    # let's plot the times on the track
    flat_times = times.ravel().astype(float)
    flat_lons = tc_ds["lon"].values.ravel()
    flat_lats = tc_ds["lat"].values.ravel()

    # Remove NaN values from flat_times, flat_lons, and flat_lats
    valid_indices = ~np.isnan(flat_times) & ~np.isnan(flat_lons) & ~np.isnan(flat_lats)
    flat_times = flat_times[valid_indices]
    flat_lons = flat_lons[valid_indices]
    flat_lats = flat_lats[valid_indices]

    pre_impact_indices = np.where(flat_times < 0)[0]

    if len(pre_impact_indices) > 0:
        num_annotations = min(6, len(pre_impact_indices))
        # Select evenly spaced indices from the pre-impact track portion
        # Ensure at least one point if num_annotations is 1 to avoid empty array in linspace
        if num_annotations == 1:
            annotation_indices_in_pre_impact = np.array(
                [0], dtype=int
            )  # Take the first pre-impact point
        elif num_annotations > 1:
            annotation_indices_in_pre_impact = np.linspace(
                0, len(pre_impact_indices) - 1, num_annotations, dtype=int
            )
        else:  # num_annotations is 0
            annotation_indices_in_pre_impact = np.array([], dtype=int)

        if annotation_indices_in_pre_impact.size > 0:
            selected_track_indices = pre_impact_indices[
                annotation_indices_in_pre_impact
            ]

            for j, i in enumerate(selected_track_indices):
                lon_point = flat_lons[i]
                lat_point = flat_lats[i]
                time_val = flat_times[i]

                if np.isnan(lon_point) or np.isnan(lat_point) or np.isnan(time_val):
                    continue

                # we're getting rid of the first and last label
                if j != annotation_indices_in_pre_impact.size - 1 and j != 0:
                    ax_map.text(
                        lon_point,
                        lat_point,
                        f"{time_val:.0f}h",
                        color="black",
                        fontsize=7,  # Adjusted fontsize slightly
                        transform=ccrs.PlateCarree(),
                        ha="right",  # Horizontal alignment (right of the point)
                        va="bottom",  # Vertical alignment (text bottom is at point's y, then offset)
                    )
    # let's add the coastline in
    ax_map.coastlines(resolution="50m", color="grey", linewidth=0.5)
    # add rivers
    ax_map.add_feature(cfeature.RIVERS, linewidth=0.5)
    # add lakes
    ax_map.add_feature(cfeature.LAKES, linewidth=0.5)
    # make the land green again
    ax_map.add_feature(
        NaturalEarthFeature("physical", "land", "50m"),
        edgecolor="black",
        facecolor="green",
        alpha=0.3,
        linewidth=0.5,
    )
    # add grid lines
    gl = ax_map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    # get rid of the top and right labels
    gl.top_labels = False
    gl.right_labels = False

    ax_line1 = fig.add_subplot(gs[1, 0])
    ax_line1.plot(
        times.ravel(),
        tc_ds["usa_wind"].values.ravel() * 0.514444,  # convert knots to m/s
        label=r"$V_{\mathrm{Obs.}}$ [m s$^{-1}$]",
        color="blue",
    )
    ax_line1.plot(
        times.ravel(),
        tc_ds["vmax"].values.ravel()
        * 0.8,  # already in m/s, but at gradient wind not 10m, so need to reduce by v_reduc
        label=r"$V_{\mathrm{p}}$@10m [m s$^{-1}$]",
        color="green",
    )
    ax_line1.set_ylabel(r"$V$ [m s$^{-1}$]")
    add_saffir_simpson_boundaries(
        ax_line1,
    )
    highlight_rapid_intensification(
        ax_line1,
        times.ravel(),
        tc_ds["usa_wind"].values.ravel() * 0.514444,  # convert knots to m/s
        ri_threshold_knots=30,
        ri_period_hours=24,
        color="red",
        alpha=0.2,
    )
    # ax_line1.set_xlabel("Time since impact [hours]")
    ax_line1.set_xlim(np.nanmin(times), np.nanmax(times))
    ax_line1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    ax_line2 = fig.add_subplot(gs[2, 0])
    ax_line2.plot(
        times.ravel(),
        tc_ds["usa_rmw"].values.ravel() * 1852 / 1000,  # convert nautical miles to km
        label=r"$r_{\mathrm{Obs.}}$ [km]",
        color="blue",
    )
    ax_line2.plot(
        times.ravel(),
        tc_ds["rmax"].values.ravel() / 1000,  # convert m to km
        label=r"$r_3$ PS [km]" + "\n" + r"($V=V_{\mathrm{p}}$)",
        color="green",
    )
    ax_line2.plot(
        times.ravel(),
        tc_ds["ps_vobs"].values.ravel() / 1000,  # convert m to km
        label=r"$r_2$ CPS [km]" + "\n" + r"($V=V_{\mathrm{Obs.}}$)",
        color="orange",
    )
    ax_line2.plot(
        times.ravel(),
        tc_ds["ps_cat1"].values.ravel() / 1000,  # convert m to km
        label=r"$r_1$ PS Cat 1 [km]" + "\n" + r"($V=33$ m s$^{-1}$)",
        color="red",
    )
    ax_line2.set_ylabel(r"$r$ [km]")
    ax_line2.set_xlabel("Time since impact [hours]")
    # expand legend to the right of the plot
    ax_line2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    ax_line2.set_xlim(np.nanmin(times), np.nanmax(times))

    label_subplots([ax_map])
    label_subplots([ax_line1, ax_line2], override="outside", start_from=1)
    plt.tight_layout()
    plt.savefig(
        plot_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.clf()
    plt.close()


def get_tc_ds(basin: str = b"NA", subbasin: str = b"GM", name: str = b"KATRINA") -> xr.Dataset:
    """
    Get a tropical cyclone from the IBTrACS dataset.

    Args:
        basin (str, optional): Basin of the tropical cyclone. Defaults to b"NA".
        subbasin (str, optional): Subbasin of the tropical cyclone. Defaults to b"GM".
        name (str, optional): Name of the tropical cyclone. Defaults to b"KATRINA".

    Returns:
        xr.Dataset: Dataset containing the tropical cyclone data.
    """
    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.pi_ps.nc")
    )
    ibtracs_orig = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
    )
    cps_inputs = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps_inputs.nc")
    )
    # add name, basin, subbasin, nature, and usa_record to the cps_inputs
    for var in ["name", "basin", "subbasin", "nature", "usa_record"]:
        # add in the new variables
        cps_inputs[var] = ibtracs_orig[var]
        ibtracs_ds[var] = ibtracs_orig[var]
    # Select Katrina's data
    tc_pi_ps_ds = select_tc_from_ds(
        ibtracs_ds, name=name, basin=basin, subbasin=subbasin
    )
    print("tc_pi_ps_ds", tc_pi_ps_ds)
    tc_inputs_ds = select_tc_from_ds(
        cps_inputs, name=name, basin=basin, subbasin=subbasin
    )
    print("tc_inputs_ds", tc_inputs_ds)
    return tc_pi_ps_ds, tc_inputs_ds

def calculate_cps_ds(tc_inputs_ds: xr.Dataset) -> xr.Dataset:
    """ Calculate the potential size dataset from the tropical cyclone inputs.
    Args:
        tc_inputs_ds (xr.Dataset): Dataset containing the tropical cyclone inputs.
    Returns:
        xr.Dataset: Dataset containing the potential size data.
    """

    # trim to get rid of unnecessary dimensions
    trimmed_ds = tc_inputs_ds[["vmax", "msl", "rh", "sst", "t0"]]

    # add in default param values
    print("trimmed_ds", trimmed_ds)
    return parallelized_ps(
        trimmed_ds,
    )


def vary_v_cps(
    v_reduc: float = 0.8,
    name: str = b"KATRINA",
    basin=b"NA",
    subbasin=b"GM",
    timestep: int = 30,
    steps: int = 60,
    v33 = 33, # m/s, threshold for category 1 hurricane
) -> None:
    """
    I want to vary the velocity input to the potential size calculation from the
    threshold for category 1 hurricane to the potential intensity, and plot the corresponding potential size in terms of rmax.

    Args:
        v_reduc (float, optional): Reduction factor for the velocity. Defaults to 0.8.
        name (str, optional): Name of the tropical cyclone to plot. Defaults to b"KATRINA".
        basin (str, optional): Basin of the tropical cyclone to plot. Defaults to b"NA".
        subbasin (str, optional): Subbasin of the tropical cyclone to plot. Defaults to b"GM".
        timestep (int, optional): Timestep to select for the potential intensity and size calculation. Defaults to 20.
        steps (int, optional): Number of steps to vary the velocity. Defaults to 60.
        v33 (float, optional): Threshold for category 1 hurricane in m/s. Defaults to 33.
    """
    # category 1 to potential intensity
    # go from 33 m/s to the potential intensity vmax
    # at gradient wind height
    tc_pi_ps_ds, tc_inputs_ds = get_tc_ds(name=name, basin=basin, subbasin=subbasin)
    tc_pi_ps_ds = tc_pi_ps_ds.isel(date_time=timestep)
    tc_inputs_ds = tc_inputs_ds.isel(date_time=timestep)
    vs = np.linspace(v33 / v_reduc, tc_pi_ps_ds.vmax.values, num=steps)
    tc_inputs_ds["vmax"] = ("v", vs)
    ps_ds = calculate_cps_ds(tc_inputs_ds)

    # plot the potential size
    r1 = ps_ds.rmax.values[0] / 1000  # convert m to km
    r2 = scipy.interpolate.interp1d(
        ps_ds.vmax.values * v_reduc,
        ps_ds.rmax.values / 1000,  # convert m to km
    )(tc_inputs_ds.usa_wind.values * 0.514444)
    radii = ps_ds.rmax.values / 1000  # convert m to km
    velocities = vs * v_reduc # convert to 10m height
    r3 = tc_pi_ps_ds.rmax.values / 1000  # convert m to km
    vp = tc_pi_ps_ds.vmax.values * v_reduc  # convert to 10m height
    robs = tc_inputs_ds.usa_rmw.values * 1852 / 1000  # convert nautical miles to km
    vobs = tc_inputs_ds.usa_wind.values * 0.514444  # convert knots to m/s
    ymin = min(
            [
                min(
                    [
                        np.nanmin(radii),
                        robs,
                    ],
                ),
                0,
            ]
        )
    ymax = np.nanmax(radii) + 1
    xmin = v33
    xmax = np.nanmax(ps_ds.vmax.values * v_reduc)
    _plot_vary_v_cps(velocities, radii, v33, vobs, vp, r1, r2, r3, robs, ymin, ymax, xmin, xmax)
    plt.savefig(
        os.path.join(FIGURE_PATH, f"vary_v_ps_{name.decode().lower()}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def ani_vary_v_cps(
    v_reduc: float = 0.8,
    name: str = b"KATRINA",
    basin=b"NA",
    subbasin=b"GM",
    timestep_end: int = 80,
    steps: int = 20,
    v33 = 33, # m/s, threshold for category 1 hurricane
) -> None:
    """
    Animate the potential size trade off for varying velocities.

    Args:
        v_reduc (float, optional): Reduction factor for the velocity. Defaults to 0.8.
        name (str, optional): Name of the tropical cyclone to plot. Defaults to b"KATRINA".
        basin (str, optional): Basin of the tropical cyclone to plot. Defaults to b"NA".
        subbasin (str, optional): Subbasin of the tropical cyclone to plot. Defaults to b"GM".
        timestep_end (int, optional): Timestep to select for the potential intensity and size calculation. Defaults to 20.
        steps (int, optional): Number of steps to vary the velocity. Defaults to 20.
        v33 (float, optional): Threshold for category 1 hurricane in m/s. Defaults to 33.
    """
    # category 1 to potential intensity
    # go from 33 m/s to the potential intensity vmax
    # at gradient wind height
    tc_pi_ps_ds, tc_inputs_ds = get_tc_ds(name=name, basin=basin, subbasin=subbasin)
    tc_pi_ps_ds = tc_pi_ps_ds.isel(date_time=slice(0, timestep_end))
    tc_inputs_ds = tc_inputs_ds.isel(date_time=slice(0, timestep_end))
    vs = np.array([np.linspace(v33 / v_reduc, vmax, num=steps) for vmax in tc_pi_ps_ds.vmax.values])
    ps_ds_file_path = os.path.join(DATA_PATH, f"vary_v_ps_{name.decode().lower()}.nc")
    tc_inputs_ds["vmax"] = (("date_time", "v"), vs)

    if os.path.exists(ps_ds_file_path):
        ps_ds = xr.open_dataset(ps_ds_file_path)
    else:
        ps_ds = calculate_cps_ds(tc_inputs_ds)
        ps_ds.to_netcdf(ps_ds_file_path)

    img_folder = os.path.join(FIGURE_PATH, "vary_v_cps", name.decode().lower())
    os.makedirs(img_folder, exist_ok=True)
    figure_name_l = []

    ymin = 0
    ymax = np.nanmax(ps_ds.rmax.values / 1000) + 1  # convert m to km
    xmin = min(v33, np.nanmin(tc_inputs_ds.usa_rmw.values * 1852 / 1000 ))
    xmax = np.nanmax(ps_ds.vmax.values * v_reduc)

    for i in range(timestep_end):
        ps_ds_i = ps_ds.isel(date_time=i)
        tc_pi_ps_ds_i = tc_pi_ps_ds.isel(date_time=i)
        tc_inputs_ds_i = tc_inputs_ds.isel(date_time=i)

        robs = tc_inputs_ds_i.usa_rmw.values * 1852 / 1000  # convert nautical miles
        vobs = tc_inputs_ds_i.usa_wind.values * 0.514444  # convert knots to m/s
        # plot the potential size
        r1 = ps_ds_i.rmax.values[0] / 1000  # convert m to km
        if vobs >= v33:
            r2 = scipy.interpolate.interp1d(
                ps_ds_i.vmax.values * v_reduc,
                ps_ds_i.rmax.values / 1000,  # convert m to km
            )(vobs)
        else:
            r2 = np.nan
        radii = ps_ds_i.rmax.values / 1000  # convert m to km
        velocities = ps_ds_i.vmax.values * v_reduc  # convert to 10m height
        r3 = tc_pi_ps_ds_i.rmax.values / 1000  # convert m to km
        vp = tc_pi_ps_ds_i.vmax.values * v_reduc  # convert to 10m height
        _plot_vary_v_cps(
            velocities,
            radii,
            v33,
            vobs,  # convert knots to m/s
            vp,
            r1,
            r2,
            r3,
            robs,
            ymin,
            ymax,
            xmin,
            xmax,
        )
        figure_name = os.path.join(img_folder, f"vary_v_ps_{name.decode().lower()}_{i:03d}.png")
        figure_name_l.append(figure_name)
        time_str = tc_pi_ps_ds_i["time"].dt.strftime('%Y-%m-%d %H').item()
        plt.title(name.decode().capitalize() + " " + time_str)
        plt.savefig(
            figure_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    with imageio.get_writer(
        os.path.join(FIGURE_PATH, f"vary_v_ps_{name.decode().lower()}.gif"),
        mode="I",
        duration=0.1,
    ) as writer:
        for figure_name in figure_name_l:
            image = imageio.imread(figure_name)
            writer.append_data(image)


def _plot_vary_v_cps(velocities: np.ndarray, radii: np.ndarray, v33, vobs, vp, r1, r2, r3, robs, ymin, ymax, xmin, xmax) -> None:
    """
    Plot the potential size trade off for varying velocities.

    Args:
        velocities (np.ndarray): Array of velocities.
        radii (np.ndarray): Array of radii corresponding to the velocities.
        v33 (float): Threshold for category 1 hurricane in m/s.
        vobs (float): Observed velocity in m/s.
        vp (float): Potential intensity velocity in m/s.
        r1 (float): Radius for category 1 hurricane in km.
        r2 (float): Radius for observed velocity in km.
        r3 (float): Radius for potential intensity in km.
        robs (float): Observed radius in km.
        ymin (float): Minimum y-axis limit.
        ymax (float): Maximum y-axis limit.
    """
    # plot the potential size trade off
    plot_defaults()
    plt.plot(
        velocities,
        radii,
        label=f"CPS",
        color="orange",
    )
    # plot vertical dashed lines as 33 m/s, usa_wind, and  v_p
    plt.axvline(v33, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axvline(
        vp,
        color="green",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    plt.axvline(
        vobs,
        linestyle="--",
        color="blue",
        linewidth=0.8,
        alpha=0.7,
    )  # convert knots to m/s
    # plot horizontal dashed line at cps rmax
    plt.axhline(
        r1,  # convert m to km
        color="orange",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    plt.axhline(
        r2,
        color="blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    plt.axhline(
        r3,  # convert m to km
        color="green",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )  # convert knots to m/s

    plt.plot(
        vp,
        r3,
        marker="o",
        color="green",
        label=r"$r_3$ PS [km]",
    )
    plt.plot(
        vobs,
        robs,  # convert nautical miles to km
        marker="+",
        color="blue",
        label=r"$V_{\mathrm{Obs.}}, $r_{\mathrm{Obs.}}$ [km]",
    )

    plt.plot(
        vobs,
        r2,  # convert knots to m/s
        marker="o",
        color="blue",
        label=r"$r_2$ CPS [km]",
    )
    plt.plot(
        v33,
        r1,  # convert m to km
        marker="o",
        color="orange",
        label=r"$r_1$ PS @cat1 [km]",
    )
    # I want to grey out the area above the v curve.
    plt.fill_between(
        velocities,
        radii,  # lower bound
        ymax,  # shade until here
        color="gray",
        alpha=0.3,
        hatch="///",
        label="Impossible Area",
    )
    plt.fill_between(
        [vp, xmax],
        [ymin, ymin],
        [ymax, ymax],
        color="gray",
        alpha=0.3,
        hatch="///",
        # label="Impossible Area",
    )

    plt.ylim(
        ymin,
        ymax,
    )
    plt.xlim(xmin, xmax)
    # xticks at 33 m/s, V_max Obs, and V_p
    plt.xticks(
        [
            v33,
            vobs,  # convert knots to m/s
            vp,
        ],
        labels=[
            r"$V \mathrm{cat1}$" + f"={v33:.1f}" + r" m s$^{-1}$",
            r"$V_{\mathrm{Obs.}}$"
            + f"={vobs:.1f}"
            + r" m s$^{-1}$",
            r"$V_{\mathrm{p}}$ @10m"
            + f"={vp:.1f}"
            + r" m s$^{-1}$",
        ],
    )

    ax1 = plt.gca()  # Get the current axis
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlabel(r"$V$ @ 10m [m s$^{-1}$]")
    ax1.set_ylabel(r"$r\left(V\right)$ [km]")
    ax2 = ax1.twinx()
    ax2.set_ylim([ymin, ymax])
    ax2.set_yticks([r1, r2, r3],
                   labels=["$r_1$", "$r_2$", "$r_3$"])
    print("r1", r1)
    print("r2", r2)
    print("r3", r3)
    ax2.set_ylim([ymin, ymax])


def check_sizes():
    """Loop through all of the IBTrACS datasets used as inputs and outputs and print their sizes."""
    datasets: List[str] = [
        "IBTrACS.since1980.v04r01.nc",
        "IBTrACS.since1980.v04r01.unique.nc",
        "IBTrACS.since1980.v04r01.pi_ps.nc",
        "IBTrACS.since1980.v04r01.normalized.nc",
        "IBTrACS.since1980.v04r01.cps_inputs.nc",
        "IBTrACS.since1980.v04r01.cps.nc",
        "era5_unique_points_raw.nc",
        "era5_unique_points_processed.nc",
        "era5_unique_points_pi.nc",
        "era5_unique_points_ps.nc",
    ]
    for ds in datasets:
        if os.path.exists(os.path.join(IBTRACS_DATA_PATH, ds)):
            print(f"{ds}: {xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, ds)).sizes}")


def save_basin_names():
    """
    Save the tropical cyclones in each basin to a file.
    """
    # NI - North Indian
    # SI - South Indian
    # WP - Western Pacific
    # SP - Southern Pacific
    # EP - Eastern Pacific
    # NA - North Atlantic
    basin_names = [
        b"NA",  # North Atlantic
        b"EP",  # Eastern Pacific
        b"WP",  # Western Pacific
        b"SP",  # South Pacific
        b"SI",  # Indian Ocean
        b"NI",  # North Indian
        b"SA",  # South Atlantic
    ]
    basin_names_d = {}
    for i, basin in enumerate(basin_names):
        tc_ds = filter_by_labels(
            xr.open_dataset(
                os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
            ),
            filter=[("basin", [basin]), ("nature", [b"TS"])],
        )
        names = [x.decode() for x in tc_ds.name.values]
        start_dates = (
            tc_ds.time.values[:, 0].astype("datetime64[ns]").astype("str").tolist()
        )
        max_winds = (np.nanmax(tc_ds["wmo_wind"].values, axis=1) * 0.514444).tolist()
        # convert knots to m/s
        basin_names_d[basin.decode()] = [
            {name: {"start_date": start_date, "max_wind": max_wind}}
            for name, start_date, max_wind in zip(names, start_dates, max_winds)
        ]

    # Save the basin names to a file
    if not os.path.exists(IBTRACS_DATA_PATH):
        os.makedirs(IBTRACS_DATA_PATH, exist_ok=True)
    # Write the basin names to a yaml file

    with open(os.path.join(IBTRACS_DATA_PATH, "basin_names.yaml"), "w") as f:
        yaml.dump(basin_names_d, f, default_flow_style=False, allow_unicode=True)


@timeit
def get_normalized_data(
    lower_wind_vp: float = 33.0, lower_wind_obs: float = 33.0, v_reduc: float = 0.8
) -> xr.Dataset:
    """
    Get the normalized data from the IBTrACS dataset.

    Args:
        lower_wind_vp (float, optional): Lower wind speed threshold for potential intensity. Defaults to 33.0 m/s.
        lower_wind_obs (float, optional): Lower wind speed threshold for observed wind speed. Defaults to 33.0 m/s.
        v_reduc (float, optional): Reduction factor for the potential intensity. Defaults to 0.8.

    Returns:
        xr.Dataset: The normalized dataset.
    """

    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
    )
    pi_ps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.pi_ps.nc")
    )
    cps_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps.nc")
    )
    ps_cat1_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.ps_cat1.nc")
    )
    # only count where vp > lower_wind_vp
    cps_ds = cps_ds.where(pi_ps_ds.vmax * v_reduc > lower_wind_vp, drop=False)
    ibtracs_ds = ibtracs_ds.where(pi_ps_ds.vmax * v_reduc > lower_wind_vp, drop=False)
    pi_ps_ds = pi_ps_ds.where(pi_ps_ds.vmax * v_reduc > lower_wind_vp, drop=False)
    ps_cat1_ds = ps_cat1_ds.where(pi_ps_ds.vmax * v_reduc > lower_wind_vp, drop=False)
    # only count where usa_wind > lower_wind_obs
    ibtracs_ds = ibtracs_ds.where(
        ibtracs_ds["usa_wind"] * 0.514444 > lower_wind_obs, drop=False
    )
    pi_ps_ds = pi_ps_ds.where(
        ibtracs_ds["usa_wind"] * 0.514444 > lower_wind_obs, drop=False
    )
    cps_ds = cps_ds.where(
        ibtracs_ds["usa_wind"] * 0.514444 > lower_wind_obs, drop=False
    )
    ps_cat1_ds = ps_cat1_ds.where(
        ibtracs_ds["usa_wind"] * 0.514444 > lower_wind_obs, drop=False
    )
    # calculate normalized variables
    pi_ps_ds["normalized_intensity"] = (
        ("storm", "date_time"),
        ibtracs_ds["usa_wind"].values * 0.514444 / pi_ps_ds.vmax.values / 0.8,
    )  # normalized intensity
    pi_ps_ds["normalized_size"] = (
        ("storm", "date_time"),
        ibtracs_ds["usa_rmw"].values * 1852 / pi_ps_ds.rmax.values,
    )
    pi_ps_ds["normalized_size_obs"] = (
        ("storm", "date_time"),
        ibtracs_ds["usa_rmw"].values * 1852 / cps_ds.rmax.values,
    )
    pi_ps_ds["normalized_size_cat1"] = (
        ("storm", "date_time"),
        ibtracs_ds["usa_rmw"].values * 1852 / ps_cat1_ds.rmax.values,
    )
    # adding the potential size variables back in
    pi_ps_ds["potential_size_cat1"] = ps_cat1_ds["rmax"]
    pi_ps_ds["potential_size_vp"] = pi_ps_ds["rmax"]
    pi_ps_ds["vmax_obs_10m"] = ibtracs_ds["usa_wind"] * 0.514444
    pi_ps_ds["vmax_vp_10m"] = pi_ps_ds["vmax"] * v_reduc
    pi_ps_ds = before_2025(pi_ps_ds)
    return pi_ps_ds


def perc_gt_1(x: np.ndarray) -> float:
    """Calculate the percentage of values greater than 1.
    Args:
        x (np.ndarray): Input array to calculate the percentage from.

    Returns:
        float: Percentage of non nanvalues greater than 1.
    """
    x = x.ravel()  # Flatten the array to 1D
    x = x[~np.isnan(x)]  # Remove NaN values
    return np.sum(x > 1) / len(x) * 100


def plot_normalized_quad(
    lower_wind_vp: float = 33.0, lower_wind_obs: float = 33.0, plot_storms: bool = False
) -> None:
    """
    Plot the normalized variables from the IBTrACS dataset:

    Args:
        lower_wind_vp (float, optional): Lower wind speed threshold for filtering. Defaults to 33.0 m/s.
        lower_wind_obs (float, optional): Lower wind speed threshold for observed wind speed filtering. Defaults to 33.0 m/s.
        plot_storms (bool, optional): If True, plot the storms with their normalized variables. Defaults to False.
    """

    # vmax/vp, rmax/r'max (size/PS), and rmax/r''max (size/CPS)
    plot_defaults()
    pi_ps_ds = get_normalized_data(
        lower_wind_vp=lower_wind_vp, lower_wind_obs=lower_wind_obs
    )

    fig, axs = plt.subplots(
        1,
        4,
        figsize=get_dim(fraction_of_line_width=4 / 3, ratio=(5**0.5 - 1) / 2 * 3 / 4),
        sharey=True,
        sharex=True,
    )
    axs[0].hist(pi_ps_ds["normalized_intensity"].values.ravel(), bins=100)
    axs[0].set_xlabel(r"$V_{\mathrm{Obs.}} / V_{\mathrm{p}}$")
    axs[0].set_ylabel("Count")
    axs[0].set_title(
        f"{perc_gt_1(pi_ps_ds['normalized_intensity'].values):.1f} % exceedance"
    )
    axs[1].hist(pi_ps_ds["normalized_size"].values.ravel(), bins=400)
    axs[1].set_xlabel(r"$r_{\mathrm{Obs.}} / r_3$")
    axs[1].set_title(
        f"{perc_gt_1(pi_ps_ds['normalized_size'].values):.1f} % exceedance"
    )
    axs[2].hist(pi_ps_ds["normalized_size_obs"].values.ravel(), bins=600)
    axs[2].set_title(
        f"{perc_gt_1(pi_ps_ds['normalized_size_obs'].values):.1f} % exceedance"
    )
    axs[2].set_xlabel(r"$r_{\mathrm{Obs.}} / r_2$")
    axs[3].hist(pi_ps_ds["normalized_size_cat1"].values.ravel(), bins=600)
    axs[3].set_title(
        f"{perc_gt_1(pi_ps_ds['normalized_size_cat1'].values):.1f} % exceedance"
    )
    axs[3].set_xlabel(r"$r_{\mathrm{Obs.}} / r_1$")
    label_subplots(axs)
    plt.xlim(0, 3)
    plt.savefig(
        os.path.join(FIGURE_PATH, "normalized_quad.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # let's get the highest normalized intenisity and size for each storm
    max_normalized_intensity = pi_ps_ds["normalized_intensity"].max(dim="date_time")
    max_normalized_ps = pi_ps_ds["normalized_size"].max(dim="date_time")
    max_normalized_cps = pi_ps_ds["normalized_size_obs"].max(dim="date_time")
    max_normalized_ps_cat1 = pi_ps_ds["normalized_size_cat1"].max(dim="date_time")
    # let's plot the CDFs for these arrays
    fig, axs = plt.subplots(
        1,
        4,
        figsize=get_dim(fraction_of_line_width=4 / 3, ratio=(5**0.5 - 1) / 2 * 3 / 4),
        sharey=True,
        sharex=True,
    )

    def length_of_array(x: np.ndarray) -> int:
        """Return the length of the array, ignoring NaNs."""
        return np.sum(~np.isnan(x.ravel()))

    def processed_array(x: np.ndarray) -> np.ndarray:
        """Return the array with NaNs removed."""
        return x[~np.isnan(x.ravel())]

    axs[0].plot(
        np.sort(processed_array(max_normalized_intensity.values)),
        1
        - np.arange(1, length_of_array(max_normalized_intensity.values) + 1)
        / length_of_array(max_normalized_intensity.values),
    )
    axs[0].set_xlabel(
        r"$\max_{\mathrm{storm}}\left(V_{\mathrm{Obs.}} / V_{\mathrm{p}}\right)$"
    )
    axs[0].set_ylabel("Survival Function (1 - CDF)")

    axs[1].plot(
        np.sort(processed_array(max_normalized_ps.values)),
        1
        - np.arange(1, length_of_array(max_normalized_ps.values) + 1)
        / length_of_array(max_normalized_ps.values),
    )
    axs[1].set_xlabel(r"$\max_{\mathrm{storm}}\left(r_{\mathrm{Obs.}} / r_3\right)$")
    axs[2].plot(
        np.sort(processed_array(max_normalized_cps.values)),
        1
        - np.arange(1, length_of_array(max_normalized_cps.values) + 1)
        / length_of_array(max_normalized_cps.values),
    )
    axs[2].set_xlabel(r"$\max_{\mathrm{storm}}\left(r_{\mathrm{Obs.}} / r_2\right)$")
    axs[3].plot(
        np.sort(processed_array(max_normalized_ps_cat1.values)),
        1
        - np.arange(1, length_of_array(max_normalized_ps_cat1.values) + 1)
        / length_of_array(max_normalized_ps_cat1.values),
    )
    axs[3].set_xlabel(r"$\max_{\mathrm{storm}}\left(r_{\mathrm{Obs.}} / r_1\right)$")
    axs[3].set_title(
        f"{perc_gt_1(max_normalized_ps_cat1.values):.1f} % exceedance"
    )  # add the percentage of exceedance
    label_subplots(axs, override="outside")
    plt.xlim(0, 3)
    plt.ylim(0, 1)

    def calcuate_survival_function(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the survival function."""
        x = processed_array(x)
        sorted_x = np.sort(x)
        survival_function = 1 - np.arange(1, len(sorted_x) + 1) / len(sorted_x)
        return sorted_x, survival_function

    def find_survival_value_at_norm(x: np.ndarray, threshold: float = 1.0) -> float:
        """Find the value at which the survival function is equal to 1."""
        sorted_x, survival_function = calcuate_survival_function(x)
        return scipy.interpolate.interp1d(
            sorted_x, survival_function, bounds_error=False, fill_value="extrapolate"
        )(threshold)

    axs[0].set_title(
        f"{find_survival_value_at_norm(max_normalized_intensity.values)*100:.1f} % exceedance"
    )
    axs[1].set_title(
        f"{find_survival_value_at_norm(max_normalized_ps.values)*100:.1f} % exceedance"
    )
    axs[2].set_title(
        f"{find_survival_value_at_norm(max_normalized_cps.values)*100:.1f} % exceedance"
    )
    plt.savefig(
        os.path.join(FIGURE_PATH, "normalized_quad_cdf.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    # I want to plot a 2d histogram of the normalized intensity and normalized size
    # using category 1 potential size
    fig, ax = plt.subplots(1, 1, figsize=get_dim())
    plt.hexbin(
        pi_ps_ds["normalized_intensity"].values.ravel(),
        pi_ps_ds["normalized_size_cat1"].values.ravel(),
        gridsize=100,
        cmap="cmo.thermal",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log Count")
    plt.xlabel(r"Normalized intensity, $V_{\mathrm{Obs.}} / V_{\mathrm{p}}$")
    plt.ylabel(r"Normalized size, $r_{\mathrm{Obs.}} / r_1$")
    plt.savefig(
        os.path.join(FIGURE_PATH, "normalized_2d_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # let's find the most extreme values in this normalized space.
    super_storms = np.any(
        (pi_ps_ds["normalized_intensity"].values > 1)
        & (pi_ps_ds["normalized_size_cat1"].values > 1),
        axis=1,
    )
    print(
        f"Number of storms with normalized intensity > 1 and normalized size > 1: {np.sum(super_storms)}"
    )

    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")
    )
    # get the names of the storms which have these
    storm_names = [x.decode() for x in ibtracs_ds.name.values[super_storms]]
    print(
        f"Storms with normalized intensity > 1 and normalized size > 1: {', '.join(storm_names)}"
    )

    # get top 10 storms by normalized intensity
    top_intensity_storms = np.argsort(
        -pi_ps_ds["normalized_intensity"].max(dim="date_time").values
    )[:10]
    # Get indices of top 10 storms
    top_intensity_names = [
        ibtracs_ds.name.values[i].decode() for i in top_intensity_storms
    ]
    print(f"Top 10 storms by normalized intensity: {', '.join(top_intensity_names)}")
    print(
        f"Top 10 storms normalized intensities: {', '.join([str(x) for x in pi_ps_ds['normalized_intensity'].max(dim='date_time').values[top_intensity_storms]])}"
    )
    if plot_storms:
        dir = os.path.join(FIGURE_PATH, "top_10_superintense")
        os.makedirs(dir, exist_ok=True)
        for i, storm in enumerate(top_intensity_storms):
            start_time = ibtracs_ds["time"].isel(storm=storm).values[0]
            plot_name = os.path.join(
                dir,
                f"{i:02}-{np.datetime_as_string(start_time, unit='M')}-{ibtracs_ds.name.values[storm].decode()}.pdf",
            )
            plot_tc_track_by_index(
                storm,
                plot_name,
            )

    top_size_storms = np.argsort(
        -pi_ps_ds["normalized_size_cat1"].max(dim="date_time").values
    )[:10]
    # Get indices of top 10 storms
    top_size_names = [ibtracs_ds.name.values[i].decode() for i in top_size_storms]
    print(f"Top 10 storms by normalized size: {', '.join(top_size_names)}")
    print(
        f"Top 10 storms normalized sizes: {', '.join([str(x) for x in pi_ps_ds['normalized_size_cat1'].max(dim='date_time').values[top_size_storms]])}"
    )
    dir = os.path.join(FIGURE_PATH, "top_10_supersize")
    os.makedirs(dir, exist_ok=True)
    if plot_storms:
        for i, storm in enumerate(top_size_storms):
            start_time = ibtracs_ds["time"].isel(storm=storm).values[0]
            plot_name = os.path.join(
                dir,
                f"{i:02}-{np.datetime_as_string(start_time, unit='M')}-{ibtracs_ds.name.values[storm].decode()}.pdf",
            )
            plot_tc_track_by_index(
                storm,
                plot_name,
            )

    # let's select and argmax those storms so we can make a latex table summary
    top_size_storms_ds = pi_ps_ds.isel(storm=top_size_storms)
    top_intensity_storms_ds = pi_ps_ds.isel(storm=top_intensity_storms)

    # print(top_intensity_storms_ds)
    # print(top_size_storms_ds)
    # take max indexes
    max_normalized_intensity_indices_da = top_intensity_storms_ds[
        "normalized_intensity"
    ].argmax(dim="date_time")
    max_normalized_size_indices_da = top_size_storms_ds["normalized_size_cat1"].argmax(
        dim="date_time"
    )
    var = [
        "time",
        "name",
        "basin",
        "lon",
        "lat",
        "normalized_intensity",
        "normalized_size_cat1",
        "usa_wind",
        "usa_rmw",
    ]
    # select the storms at these indices and convert to dataframe
    max_normalized_intensity_storms_df = (
        top_intensity_storms_ds[var]
        .isel(date_time=max_normalized_intensity_indices_da)
        .to_dataframe()
    )
    max_normalized_size_storms_df = (
        top_size_storms_ds[var]
        .isel(date_time=max_normalized_size_indices_da)
        .to_dataframe()
    )

    # save these to csv and latex files
    max_normalized_intensity_storms_df.to_csv(
        os.path.join(DATA_PATH, "max_normalized_intensity_storms.csv")
    )
    max_normalized_size_storms_df.to_csv(
        os.path.join(DATA_PATH, "max_normalized_size_storms.csv")
    )
    formatters = {
        "time": lambda x: x.round(freq="H").strftime("%Y-%m-%d %H:00"),
        "usa_wind": lambda x: f"{x*0.514444:.2f}",  # convert knots to m/s
        "usa_rmw": lambda x: f"{x*1852/1000:.2f}",  # convert nautical miles to km
        "name": lambda x: x.decode().title(),
        "basin": lambda x: x.decode(),
    }
    publication_headers = [
        "Time (UTC)",
        "Name",
        "Basin",
        r"Longitude [$^\circ$]",
        r"Latitude [$^\circ$]",
        r"\(V_{{ \mathrm{{max}} }} / V_{{ \mathrm{{p}} }}\)",  # CORRECTED
        r"\(r_{{ \mathrm{{max}} }} / r'''_{{ \mathrm{{max}} }}\)",  # CORRECTED
        r"\(V_{{ \mathrm{{max}} }}\) [m s$^{{-1}}$]",  # CORRECTED
        r"\(r_{{ \mathrm{{max}} }}\) [km]",  # CORRECTED
    ]
    # save to latex files
    max_normalized_intensity_storms_df.to_latex(
        os.path.join(DATA_PATH, "max_normalized_intensity_storms.tex"),
        index=False,
        float_format="%.2f",
        formatters=formatters,
        header=publication_headers,
        # booktabs=True,
        caption=r"Top storms by maximum normalized intensity, \(V_{{ \mathrm{{max}} }} / V_{{ \mathrm{{p}}}}\), selected at the point of their maximum normalized intensity.",
        label="tab:norm_intensity_storms",
        # column_format="lcccccc",
    )
    max_normalized_size_storms_df.to_latex(
        os.path.join(DATA_PATH, "max_normalized_size_storms.tex"),
        index=False,
        float_format="%.2f",
        formatters=formatters,
        header=publication_headers,
        caption=r"Top storms by normalized size \(r_{{ \mathrm{{max}} }} / r'''_{{ \mathrm{{max}}}}\), selected at the point of their maximum normalized intensity.",
        label="tab:norm_size_storms",
    )


# Question: does the velocity size tradeoff look the same shape for all storms in this normalized space?


def compare_normalized_potential_size(
    lower_wind_vp: float = 33, lower_wind_obs: float = 33
) -> None:
    """
    Compare normalized potential size.

    Args:
        lower_wind_vp (float, optional): Lower wind bound for potential intensity. Defaults to 33 m/s.
        lower_wind_obs (float, optional): Lower wind bound for potential size. Defaults to 33 m/s.
    """
    # For relevant examples, I want to see if r'max / r'''max depends on other parameters, or is a constant value.

    fig, ax = plt.subplots(1, 1, figsize=get_dim())

    pi_ps_ds = get_normalized_data(
        lower_wind_vp=lower_wind_vp, lower_wind_obs=lower_wind_obs
    )
    pi_ps_ds["norm_potential_size"] = (
        pi_ps_ds["potential_size_vp"] / pi_ps_ds["potential_size_cat1"]
    )
    before_2025(pi_ps_ds)
    plt.hexbin(
        pi_ps_ds["potential_size_cat1"].values.ravel() / 1000,
        pi_ps_ds["norm_potential_size"].values.ravel(),
        gridsize=100,
        cmap="cmo.thermal",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log Count")
    plt.xlabel(r"Potential size for cat1 $r_1$ [km]")
    plt.ylabel(r"Normalized potential size $r_3 / r_1$")
    plt.savefig(os.path.join(FIGURE_PATH, "norm_ps.pdf"))
    plt.clf()
    plt.close()
    # find out if potential intensity ever below 33 m/s

    weird_storms = np.any(
        pi_ps_ds["vmax_vp_10m"].values < lower_wind_vp, axis=1
    )  # v_reduc to surface wind.
    print(
        f"Number of storms with potential intensity below {lower_wind_vp} m/s: {np.sum(weird_storms)} out of {len(weird_storms)}"
    )


def get_vary_vp_lower_data(
    num: int = 6,
    lower_vp_min: float = 33,
    lower_vp_max: float = 83,
    lower_wind_obs: float = 33,
) -> xr.Dataset:
    """Vary the lower limits for the normalized variables, and see how exceedance changes.

    Args:
        num (int, optional): Number of points to plot. Defaults to 6.
        lower_vp_min (float, optional): Minimum lower limit for potential intensity. Defaults to 33.
        lower_vp_max (float, optional): Maximum lower limit for potential intensity. Defaults to 83.
        lower_wind_obs (float, optional): Lower wind speed threshold for observed wind speed. Defaults to 33.

    Returns:
        xr.Dataset: Dataset containing the exceedance of normalized variables for varying lower limits of potential intensity wind speed.
    """

    file_name = os.path.join(IBTRACS_DATA_PATH, "vary_vp_lower_exceedance.nc")
    if os.path.exists(file_name):
        print(f"Loading existing dataset from {file_name}")
        return xr.open_dataset(os.path.join(file_name))
    else:
        vp_lower_array = np.linspace(lower_vp_min, lower_vp_max, num=num)
        ps_exceedance = np.zeros(num)
        ps_cat1_exceedance = np.zeros(num)
        ps_obs_exceedance = np.zeros(num)
        intensity_exceedance = np.zeros(num)

        for i, vp_lower in tqdm(enumerate(vp_lower_array)):
            pi_ps_ds = get_normalized_data(
                lower_wind_vp=vp_lower, lower_wind_obs=lower_wind_obs
            )
            ps_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size"].values)
            ps_cat1_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size_cat1"].values)
            ps_obs_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size_obs"].values)
            intensity_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_intensity"].values)

        # save the data to a xarray dataset
        vp_lower_ds = xr.Dataset(
            {
                "vp_lower": (("vp_lower"), vp_lower_array),
                "ps_exceedance": (("vp_lower"), ps_exceedance),
                "ps_cat1_exceedance": (("vp_lower"), ps_cat1_exceedance),
                "ps_obs_exceedance": (("vp_lower"), ps_obs_exceedance),
                "intensity_exceedance": (("vp_lower"), intensity_exceedance),
            }
        )
        vp_lower_ds.attrs["description"] = (
            "Exceedance of normalized variables for varying lower limits of potential intensity wind speed."
        )
        vp_lower_ds.attrs["lower_wind_vp_min"] = lower_vp_min
        vp_lower_ds.attrs["lower_wind_vp_max"] = lower_vp_max
        vp_lower_ds.attrs["lower_wind_obs"] = lower_wind_obs
        vp_lower_ds.attrs["units"] = "m/s"
        vp_lower_ds.to_netcdf(
            file_name,
            mode="w",
            format="NETCDF4",
        )
        return vp_lower_ds


def get_vary_vobs_lower_data(
    num: int = 6,
    lower_vobs_min: float = 18,
    lower_vobs_max: float = 83,
    lower_wind_vp: float = 33,
) -> xr.Dataset:
    """
    Get the exceedance of normalized variables for varying lower limits of observed wind speed.
    Args:
        num (int, optional): Number of points to plot. Defaults to 6.
        lower_vobs_min (float, optional): Minimum lower limit for observed wind speed. Defaults to 18 m/s.
        lower_vobs_max (float, optional): Maximum lower limit for observed wind speed. Defaults to 83 m/s.
        lower_wind_vp (float, optional): Lower limit for potential intensity wind speed. Defaults to 33 m/s.
    Returns:
        xr.Dataset: Dataset containing the exceedance of normalized variables for varying lower limits of observed wind speed.
    """
    file_name = os.path.join(IBTRACS_DATA_PATH, "vary_vobs_lower_exceedance.nc")
    if os.path.exists(file_name):
        print(f"Loading existing dataset from {file_name}")
        return xr.open_dataset(os.path.join(file_name))
    else:
        vobs_lower_array = np.linspace(lower_vobs_min, lower_vobs_max, num=num)
        ps_exceedance = np.zeros(num)
        ps_cat1_exceedance = np.zeros(num)
        ps_obs_exceedance = np.zeros(num)
        intensity_exceedance = np.zeros(num)

        for i, vobs_lower in tqdm(enumerate(vobs_lower_array)):
            pi_ps_ds = get_normalized_data(
                lower_wind_vp=lower_wind_vp, lower_wind_obs=vobs_lower
            )
            pi_ps_ds = before_2025(pi_ps_ds)
            ps_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size"].values)
            ps_cat1_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size_cat1"].values)
            ps_obs_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_size_obs"].values)
            intensity_exceedance[i] = perc_gt_1(pi_ps_ds["normalized_intensity"].values)

        # save the data to a xarray dataset
        vobs_lower_ds = xr.Dataset(
            {
                "vobs_lower": (("vobs_lower"), vobs_lower_array),
                "ps_exceedance": (("vobs_lower"), ps_exceedance),
                "ps_cat1_exceedance": (("vobs_lower"), ps_cat1_exceedance),
                "ps_obs_exceedance": (("vobs_lower"), ps_obs_exceedance),
                "intensity_exceedance": (("vobs_lower"), intensity_exceedance),
            }
        )
        vobs_lower_ds.attrs["description"] = (
            "Exceedance of normalized variables for varying lower limits of observed wind speed."
        )
        vobs_lower_ds.attrs["lower_wind_vp"] = lower_wind_vp
        vobs_lower_ds.attrs["lower_wind_obs_min"] = lower_vobs_min
        vobs_lower_ds.attrs["lower_wind_obs_max"] = lower_vobs_max
        vobs_lower_ds.attrs["units"] = "m/s"
        vobs_lower_ds.to_netcdf(
            file_name,
            mode="w",
            format="NETCDF4",
        )
        return vobs_lower_ds


def vary_limits(num=6):
    """Vary the lower limits for the normalized variables, and see how exceedance changes.

    Args:
        num (int, optional): Number of points to plot. Defaults to 6.
    """
    vp_lower_ds = get_vary_vp_lower_data(num=num)

    fig, ax = plt.subplots(1, 1, figsize=get_dim())
    ax.plot(
        vp_lower_ds["vp_lower"].values,
        vp_lower_ds["ps_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_3$",
        color="green",
    )
    ax.plot(
        vp_lower_ds["vp_lower"].values,
        vp_lower_ds["ps_obs_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_2$",
        color="blue",
    )
    ax.plot(
        vp_lower_ds["vp_lower"].values,
        vp_lower_ds["ps_cat1_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_1$",
        color="orange",
    )
    ax.plot(
        vp_lower_ds["vp_lower"].values,
        vp_lower_ds["intensity_exceedance"].values,
        label=r"$V_{\mathrm{Obs.}} / V_{\mathrm{p}}$",
        color="red",
    )
    ax.set_xlabel(r"$V_{\mathrm{p}}$ lower limit [m s$^{-1}$]")
    ax.set_ylabel("Exceedance [%]")
    ax.legend()
    ax.set_xlim(
        np.nanmin(vp_lower_ds["vp_lower"].values),
        np.nanmax(vp_lower_ds["vp_lower"].values),
    )
    plt.savefig(
        os.path.join(FIGURE_PATH, "vary_vp_lower_exceedance.pdf"),
    )
    plt.clf()
    plt.close()

    vobs_lower_ds = get_vary_vobs_lower_data(num=num)

    _, ax = plt.subplots(1, 1, figsize=get_dim())
    ax.plot(
        vobs_lower_ds["vobs_lower"].values,
        vobs_lower_ds["ps_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_3$",
        color="green",
    )
    ax.plot(
        vobs_lower_ds["vobs_lower"].values,
        vobs_lower_ds["ps_obs_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_2$",
        color="blue",
    )
    ax.plot(
        vobs_lower_ds["vobs_lower"].values,
        vobs_lower_ds["ps_obs_exceedance"].values,
        label=r"$r_{\mathrm{Obs.}} / r_1$",
        color="orange",
    )

    ax.plot(
        vobs_lower_ds["vobs_lower"].values,
        vobs_lower_ds["intensity_exceedance"].values,
        label=r"$V_{\mathrm{Obs.}} / V_{\mathrm{p}}$",
        color="red",
    )
    ax.set_xlabel(r"$V_{\mathrm{Obs.}}$ lower limit [m s$^{-1}$]")
    ax.set_ylabel("Exceedance [%]")
    ax.legend()
    ax.set_xlim(
        np.nanmin(vobs_lower_ds["vobs_lower"].values),
        np.nanmax(vobs_lower_ds["vobs_lower"].values),
    )
    plt.savefig(
        os.path.join(FIGURE_PATH, "vary_vobs_lower_exceedance.pdf"),
    )
    plt.clf()
    plt.close()


def run_all_plots():
    """Re-run all the plot (not the calculations) in the correct order."""
    plot_defaults()
    plot_unique_points()
    plot_example_raw()
    plot_era5_processed()
    plot_potential_intensity()
    plot_potential_size()
    plot_normalized_variables()
    vary_v_cps(v_reduc=0.8)
    plot_cps()
    plot_normalized_cps()
    plot_tc_example()
    plot_tc_example(
        name=b"KATRINA", bbox=(-92.5, -72.5, 22.5, 37.5)
    )  # Katrina's landfall in Louisiana
    plot_tc_example(
        name=b"IDA", bbox=(-92.5 - 5, -72.5, 22.5 - 5, 37.5)
    )  # Ida's landfall in Louisiana
    plot_tc_example(
        name=b"HELENE",
        bbox=(-92.5 - 5, -72.5, 22.5 - 5, 37.5),  # bbox=(-90, -80, 25, 35)
    )
    plot_tc_example(name=b"IAN", bbox=(-92.5 + 7.5, -72.5 + 10, 22.5 - 10, 37.5 - 5))
    plot_tc_example(name=b"HARVEY", bbox=(-92.5 - 7.5, -72.5 + 20, 22.5 - 10, 37.5 + 5))
    plot_tc_example(
        name=b"FIONA",
        bbox=None,
    )
    plot_tc_example(
        name=b"SAOLA",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=(108, 130, 15, 30),
    )
    plot_tc_example(
        name=b"MANGKHUT",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=(100, 170, 10, 30),
    )
    plot_tc_example(
        name=b"HATO",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=(90, 135, 10, 30),
    )
    plot_tc_example(
        name=b"VICENTE",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=None,
    )
    plot_tc_example(
        name=b"YORK",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=None,
    )
    plot_tc_example(
        name=b"ELLEN",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=None,
    )
    plot_tc_example(
        name=b"BEBINCA",  # Bebinca
        basin=b"WP",
        subbasin=b"WPAC",
        # bbox=(100, 170, 10, 30),
        bbox=None,
    )
    plot_tc_example(
        name=b"JEBI",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=None,
    )
    plot_tc_example(
        name=b"MERANTI",
        basin=b"WP",
        subbasin=b"WPAC",
        bbox=None,
    )
    plot_tc_example(
        name=b"FREDDY",
        basin=b"SI",
        bbox=None,
    )
    plot_normalized_quad(lower_wind_vp=33, lower_wind_obs=33, plot_storms=True)
    compare_normalized_potential_size()
    vary_limits(num=50)


if __name__ == "__main__":
    # python -m tcpips.ibtracs &> helene_debug.txt
    # download_ibtracs_data()
    plot_defaults()
    # print("IBTrACS data downloaded and ready for processing.")
    # ibtracs_to_era5_map()
    # plot_unique_points()
    # era5_unique_points_raw()
    # plot_example_raw()
    # process_era5_raw()
    # plot_era5_processed()
    # calculate_potential_intensity()
    # plot_potential_intensity()
    # calculate_potential_size()
    # plot_potential_size()
    # add_pi_ps_back_onto_tracks()
    # create_normalized_variables()
    # plot_normalized_variables()
    # ## calculate_cps(v_reduc=0.8, test=True)
    # #calculate_cps(v_reduc=0.8, test=False)
    # plot_cps()
    # plot_normalized_cps()
    # check_sizes()
    # plot_tc_example()
    # plot_tc_example(
    #     name=b"KATRINA", bbox=(-92.5, -72.5, 22.5, 37.5)
    # )  # Katrina's landfall in Louisiana
    # plot_tc_example(
    #     name=b"IDA", bbox=(-92.5 - 5, -72.5, 22.5 - 5, 37.5)  # bbox=(-90, -80, 25, 35)
    # )  # Ida's landfall in Louisiana
    # plot_tc_example(
    #     name=b"HELENE",
    #     bbox=(-92.5 - 5, -72.5, 22.5 - 5, 37.5),  # bbox=(-90, -80, 25, 35)
    # )
    # plot_tc_example(name=b"IAN", bbox=(-92.5 + 7.5, -72.5 + 10, 22.5 - 10, 37.5 - 5))
    # plot_tc_example(name=b"HARVEY", bbox=(-92.5 - 7.5, -72.5 + 20, 22.5 - 10, 37.5 + 5))
    # plot_tc_example(
    #     name=b"FIONA",
    #     bbox=None,
    # )
    # plot_tc_example(
    #     name=b"SAOLA",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=(108, 130, 15, 30),
    # )
    # plot_tc_example(
    #     name=b"MANGKHUT",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     # bbox=None,
    #     # bbox=None,
    #     bbox=(100, 170, 10, 30),
    # )
    # plot_tc_example(
    #     name=b"HATO",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     # bbox=None,
    #     bbox=(90, 135, 10, 30),
    # )
    # plot_tc_example(
    #     name=b"VICENTE",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=None,
    # )
    # plot_tc_example(
    #     name=b"YORK",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=None,
    # )
    # plot_tc_example(
    #     name=b"ELLEN",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=None,
    # )

    # plot_tc_example(
    #     name=b"BEBINCA",  # Bebinca
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     # bbox=(100, 170, 10, 30),
    #     bbox=None,
    # )

    # plot_tc_example(
    #     name=b"JEBI",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=None,
    # )
    # plot_tc_example(
    #     name=b"MERANTI",
    #     basin=b"WP",
    #     subbasin=b"WPAC",
    #     bbox=None,
    # )
    # plot_tc_example(
    #     name=b"FREDDY",
    #     basin=b"SI",
    #     bbox=None,
    # )
    # save_basin_names()
    # vary_v_cps()
    # plot_normalized_quad(lower_wind_vp=33, lower_wind_obs=33)
    # compare_normalized_potential_size()
    # calculate_potential_size_cat1()
    # add_pi_ps_back_onto_tracks(
    #     variables_to_map=("rmax", "vmax", "r0", "pm", "otl", "t0"),
    #     input_name="IBTrACS.since1980.v04r01.unique.nc",
    #     era5_name="era5_unique_points_ps_cat1.nc",
    #     output_name="IBTrACS.since1980.v04r01.ps_cat1.nc",
    # )
    # vary_limits(num=50)
    # run_all_plots()
    # vary_v_cps()
    # ani_vary_v_cps()
    ani_vary_v_cps(name=b"IRENE")

# add a processing step to exclude cyclone time points where PI is going / has gone down.
