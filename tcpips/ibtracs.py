# wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc
"""Process IBTrACS data to calculate along-track potential intensity and size from ERA5.

We only want tracks from 1980-2024 (inclusive).

"""
import os
import xarray as xr
from .constants import DATA_PATH

IBTRACS_DATA_PATH = os.path.join(DATA_PATH, "ibtracs")
os.makedirs(IBTRACS_DATA_PATH, exist_ok=True)
IBTRACS_DATA_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"
IBTRACS_DATA_FILE = os.path.join(IBTRACS_DATA_PATH, "IBTrACS.since1980.v04r01.nc")


def download_ibtracs_data() -> None:
    """Download IBTrACS data if not already downloaded"""
    if not os.path.exists(IBTRACS_DATA_FILE):
        print(f"Downloading IBTrACS data from {IBTRACS_DATA_URL}...")
        os.system(f"wget {IBTRACS_DATA_URL} -O {IBTRACS_DATA_FILE}")
        print("Download complete.")
    else:
        print("IBTrACS data already downloaded.")


def corresponding_era5():
    # load ibtracs
    ibtracs_ds = xr.open_dataset(IBTRACS_DATA_FILE)
    # next get era5 coordinates (longitude, latitude, time)
    for storm_id in ibtracs_ds.sid:
        storm = ibtracs_ds.sel(sid=storm_id)
        # get the start and end time of the storm
        start_time = storm.time.min().values
        end_time = storm.time.max().values

        # check if the storm is within the desired time range
        if start_time >= "1980-01-01" and end_time <= "2024-12-31":
            # get the latitude and longitude of the storm centre
            lat = storm.lat.values
            lon = storm.lon.values

            # loop through the time steps of the storm
            for i in range(len(storm.time)):
                # get the latitude and longitude of the storm centre at this time step
                lat_i = lat[i]
                lon_i = lon[i]

                # find the closest era5 data point to the cyclone centre at this time step
                # (this part is not implemented yet)


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


if __name__ == "__main__":
    # python -m tcpips.ibtracs
    # download_ibtracs_data()
    # print("IBTrACS data downloaded and ready for processing.")
    corresponding_era5()
