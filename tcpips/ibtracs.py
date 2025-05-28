"""Process IBTrACS data to calculate along-track potential intensity and size from ERA5.

We only want tracks from 1980-2024 (inclusive).

wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc

"""

from typing import Optional, Tuple, Union, List
import os
import numpy as np
import xarray as xr
import ujson
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # Import GridSpec

try:
    from cartopy import crs as ccrs
    from cartopy.feature import NaturalEarthFeature
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
    ax: plt.axis, da: xr.DataArray, label: str, cmap: str, shrink: float = 1
) -> None:
    """Plot a variable on a map.

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
def example_plot_raw():
    # Let's make a 3 panel plot of the era5_unique_points_raw.nc data
    # plot it on global plattee carree projection with coastlines

    era5_unique_data = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_raw.nc")
    )

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
        axs[0], era5_coords_ds["sst"], "Mean SST [$^{\circ}$C]", "viridis", shrink=1
    )
    plot_var_on_map(
        axs[1], era5_coords_ds["msl"], "Mean MSL [hPa]", "viridis_r", shrink=1
    )
    plot_var_on_map(
        axs[2], era5_coords_ds["t2m"], "Mean T2M [$^{\circ}$C]", "viridis", shrink=1
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
def calculate_potential_size():
    """Calculate the size of the cyclone at each timestep using the ERA5 data."""
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
    from tqdm import tqdm

    for i in tqdm(range(0, 20), desc="Calculating potential size in chunks"):
        # get the start and end index of the chunk
        file_name = os.path.join(IBTRACS_DATA_PATH, f"era5_unique_points_ps_{i}.nc")
        if os.path.exists(file_name):
            print(f"File {file_name} already exists, skipping.")
            continue
        start = i * (len(combined_ds.u) // 20)
        end = min(
            len(combined_ds.u),
            (i + 1) * (len(combined_ds.u) // 20),
        )
        # get the chunk of data
        chunk_ds = combined_ds.isel(u=slice(start, end))
        # calculate the potential size for this chunk
        ps_chunk = parallelized_ps(chunk_ds, dryrun=False)
        # save the chunk to a netcdf file
        ps_chunk.to_netcdf(
            os.path.join(IBTRACS_DATA_PATH, f"era5_unique_points_ps_{i}.nc"),
            engine="h5netcdf",
        )

    # # load them all together and save them as one file
    ps_ds = xr.open_mfdataset(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps_*.nc"),
        combine="by_coords",
        parallel=True,
    )
    # save the combined dataset
    ps_ds.to_netcdf(
        os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps.nc"),
        engine="h5netcdf",
    )
    print("Potential size calculated and saved.")


@timeit
def plot_potential_size() -> None:
    """Plot the potential size data."""

    ps_ds = xr.open_dataset(os.path.join(IBTRACS_DATA_PATH, "era5_unique_points_ps.nc"))

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
        "Mean $r_{max}$ [km]",
        "viridis",
        shrink=1,
    )
    plot_var_on_map(
        axs[1],
        grid_avg_ds["r0"] / 1000,
        "Mean $r_a$ [km]",
        "viridis_r",
        shrink=1,
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
    axs[0].set_xlabel(r"Normalized $r_{\mathrm{max}}/r'_{\mathrm{max}}$")
    axs[1].set_xlabel(r"Normalized ${V_{\mathrm{max}}}/{V_{p}}$")
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
        r"Mean Normalized Size, $\frac{r_{\mathrm{max}}}{r'_{\mathrm{max}}}$",
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

    ibtracs_ds = xr.open_dataset(
        os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.cps_inputs.nc")
    )
    ds = ibtracs_ds

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
    axs[0].set_xlabel(r"$r_{\mathrm{max}}$ [km]")
    axs[1].set_xlabel(r"$V_{\mathrm{max}}$ [m/s]")
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
        r"Mean $r_{\mathrm{max}}$ [km]",
        "viridis",
        shrink=1,
    )
    plot_var_on_map(
        axs[2],
        avg_ds["vmax"],
        r"Mean $V_{\mathrm{max}}$ [m/s]",
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
    ax.set_xlabel(r"Normalized $r_{\mathrm{max}}/r''_{\mathrm{max}}$")
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


def select_katrina_from_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    Select Katrina from the dataset.

    Args:
        ds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset containing only Katrina's data.
    """
    return filter_by_labels(
        ds,
        filter=[
            ("name", [b"KATRINA"]),
            ("basin", [b"NA"]),
            ("subbasin", [b"GM"]),
            ("nature", [b"TS"]),
            ("usa_record", [b"L"]),
        ],
    )


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
                wind_speed_ms,
                # + (ymax - ymin) * 0.01,  # y-position (slightly above the line)
                f"{category}",
                color=color,
                fontsize=6,
                verticalalignment="bottom",
                horizontalalignment="left",
                # transform=ax,  # Use Axes coordinates for text placement
            )


def plot_katrina_example():
    """Plot an example of Katrina's track with the potential intensity and potential size, and corresponding potential size."""
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
    # add name, basin, subbasin, nature, and usa_record to the cps_ds

    for var in ["name", "basin", "subbasin", "nature", "usa_record"]:
        cps_ds[var] = ibtracs_orig[var]
    # Select Katrina's data
    katrina_ds = select_katrina_from_ds(ibtracs_ds)
    katrina_cps_ds = select_katrina_from_ds(cps_ds)

    # Plotting
    plot_defaults()
    fig = plt.figure(figsize=get_dim(ratio=1.1))

    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[3, 1, 1], width_ratios=[100, 1]
    )

    # --- Top Subplot: Geographic Map ---
    # The projection was in subplot_kw, now specified directly for this subplot
    ax_map = fig.add_subplot(
        gs[0, :], projection=ccrs.PlateCarree(central_longitude=-80)
    )

    # _, ax = plt.subplots(
    #     figsize=get_dim(),
    #     subplot_kw={"projection": ccrs.PlateCarree(central_longitude=-80)},
    # )
    ax_map.set_extent([-100, -70, 20, 40], crs=ccrs.PlateCarree())

    kat_impact = landings_only(katrina_ds).isel(date_time=2)
    print(katrina_ds["time"].values)
    print(kat_impact["time"])
    times = (
        (katrina_ds["time"].values - kat_impact["time"].values) / 1e9 / 60 / 60
    )  # nanoseconds to hours

    ig = ax_map.scatter(
        katrina_ds["lon"],
        katrina_ds["lat"],
        c=times,
        s=1,
        marker="x",
        # color="blue",
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(ig, ax=ax_map, shrink=0.94, label="Time since impact [hours]")
    # let's add the coastline in
    ax_map.coastlines(resolution="50m", color="grey", linewidth=0.5)
    # add rivers
    ax_map.add_feature(cfeature.RIVERS, linewidth=0.5)
    # add lakes
    ax_map.add_feature(cfeature.LAKES, linewidth=0.5)
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
        katrina_ds["usa_wind"].values.ravel() * 0.514444,  # convert knots to m/s
        label=r"$V_{\mathrm{max}}$ Obs. [m s$^{-1}$]",
        color="blue",
    )
    ax_line1.plot(
        times.ravel(),
        katrina_ds["vmax"].values.ravel()
        * 0.8,  # already in m/s, but at gradient wind not 10m, so need to reduce by v_reduc
        label=r"$V_{\mathrm{p}}$@10m [m s$^{-1}$]",
        color="green",
    )
    ax_line1.set_ylabel(r"$V_{\mathrm{max}}$ [m s$^{-1}$]")
    add_saffir_simpson_boundaries(
        ax_line1,
    )
    # ax_line1.set_xlabel("Time since impact [hours]")
    ax_line1.set_xlim(np.nanmin(times), np.nanmax(times))
    ax_line1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    ax_line2 = fig.add_subplot(gs[2, 0])
    ax_line2.plot(
        times.ravel(),
        katrina_ds["usa_rmw"].values.ravel()
        * 1852
        / 1000,  # convert nautical miles to km
        label=r"$r_{\mathrm{max}}$ Obs. [km]",
        color="blue",
    )
    ax_line2.plot(
        times.ravel(),
        katrina_ds["rmax"].values.ravel() / 1000,  # convert m to km
        label=r"$r_{\mathrm{max}}$ PS [km]"
        + "\n"
        + r"($V_{\mathrm{max}}=V_{\mathrm{p}}$)",
        color="green",
    )
    ax_line2.plot(
        times.ravel(),
        katrina_cps_ds["rmax"].values.ravel() / 1000,  # convert m to km
        label=r"$r_{\mathrm{max}}$ CPS [km]"
        + "\n"
        + r"($V_{\mathrm{max}}=V_{\mathrm{max}} \text{ Obs.}$)",
        color="orange",
    )
    ax_line2.set_ylabel(r"$r_{\mathrm{max}}$ [km]")
    ax_line2.set_xlabel("Time since impact [hours]")
    # expand legend to the right of the plot
    ax_line2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    ax_line2.set_xlim(np.nanmin(times), np.nanmax(times))

    label_subplots([ax_map])

    label_subplots([ax_line1, ax_line2], override="outside", start_from=1)
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_PATH, "katrina_track.pdf"), dpi=300, bbox_inches="tight"
    )

    plt.clf()
    plt.close()


def check_sizes():
    """Loop through all of the IBTrACS datasets used as inputs and outputs and print their sizes."""
    datasets = [
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


if __name__ == "__main__":
    # python -m tcpips.ibtracs
    # download_ibtracs_data()
    # print("IBTrACS data downloaded and ready for processing.")
    # ibtracs_to_era5_map()
    # plot_unique_points()
    # era5_unique_points_raw()
    # example_plot_raw()
    # process_era5_raw()
    # plot_era5_processed()
    # # calculate_potential_intensity()
    # plot_potential_intensity()
    # # calculate_potential_size()
    # plot_potential_size()
    # add_pi_ps_back_onto_tracks()
    # create_normalized_variables()
    # plot_normalized_variables()
    # calculate_cps(v_reduc=0.8, test=True)
    # calculate_cps(v_reduc=0.8, test=False)
    # plot_cps()
    # plot_normalized_cps()
    # check_sizes()
    plot_katrina_example()
