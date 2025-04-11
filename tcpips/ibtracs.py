# wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc
"""Process IBTrACS data to calculate along-track potential intensity and size from ERA5.

We only want tracks from 1980-2024 (inclusive).

"""
import os
from .constants import DATA_PATH

IBTRACS_DATA_PATH = os.path.join(DATA_PATH, "ibtracs")
os.makedirs(IBTRACS_DATA_PATH, exist_ok=True)
IBTRACS_DATA_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc"
IBTRACS_DATA_FILE = os.path.join(IBTRACS_DATA_PATH, "IBTrACS.ALL.v04r01.nc")


def download_ibtracs_data() -> None:
    """Download IBTrACS data if not already downloaded"""
    if not os.path.exists(IBTRACS_DATA_FILE):
        print(f"Downloading IBTrACS data from {IBTRACS_DATA_URL}...")
        os.system(f"wget {IBTRACS_DATA_URL} -O {IBTRACS_DATA_FILE}")
        print("Download complete.")
    else:
        print("IBTrACS data already downloaded.")


# open ibtracs data and era5 data
# loop through ibtracs data (by track, then by time) and find the closest era5 data point to the centre of the cyclone at that timestep.
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
    download_ibtracs_data()
    print("IBTrACS data downloaded and ready for processing.")
