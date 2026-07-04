"""Constants for adforce package."""

import os
import warnings
from pathlib import Path
from pyproj import Geod
from sithom.place import BoundingBox, Point

# Let's assume a simple Geod for all calculations (won't make much difference)
GEOD = Geod(ellps="WGS84")

# regional bounding boxes for ERA5 download.
# Gulf of Mexico box (lons, lats)
GOM_BBOX = BoundingBox([-100, -80], [15, 35], desc="Gulf of Mexico Bounding Box")
# New Orleans box (lons, lats)
NO_BBOX = BoundingBox([-92, -86.5], [28.5, 30.8], desc="New Orleans Area Bounding Box")


# Significant places in North America (lon, lat)
NEW_ORLEANS = Point(-90.0715, 29.9511, desc="New Orleans")  # lon , lat
MIAMI = Point(-80.1918, 25.7617, desc="Miami")
GALVERSTON = Point(-94.7977, 29.3013, desc="Galveston")

# Comparable East Asian cities
HONG_KONG = Point(114.1095, 22.3964, desc="Hong Kong")
HANOI = Point(105.8019, 21.0285, desc="Hanoi")
SHANGHAI = Point(121.4737, 31.2304, desc="Shanghai")


# Paths
SRC_PATH = Path(__file__).parent
SETUP_PATH = os.path.join(SRC_PATH, "setup")


# check if we are in n01 or n02 consortium of ARCHER2
if "n01/n01" in SRC_PATH.as_posix():
    CON = "n01"
elif "n02/n02" in SRC_PATH.as_posix():
    CON = "n02"
else:
    CON = None

# Root directory where the ADCIRC experiment inputs/outputs live
# (e.g. {DATA_ROOT}/exp/..., {DATA_ROOT}/tcpips/exp/...).
# Resolution order:
#   1. WORSTSURGE_DATA_ROOT environment variable (highest priority, any machine),
#   2. ARCHER2 consortium detection (n01/n02) from the source checkout path,
#   3. the historical ARCHER2 n01 path (ultimate fallback, unchanged behaviour).
ARCHER2_N01_DATA_ROOT = "/work/n01/n01/sithom/adcirc-swan"
ARCHER2_N02_DATA_ROOT = "/work/n02/n02/sdat2/adcirc-swan"

_env_data_root = os.environ.get("WORSTSURGE_DATA_ROOT")
if _env_data_root:
    DATA_ROOT = _env_data_root
elif CON == "n02":
    DATA_ROOT = ARCHER2_N02_DATA_ROOT
else:
    DATA_ROOT = ARCHER2_N01_DATA_ROOT
    if CON is None:
        warnings.warn(
            "adforce.constants: unrecognized machine (source path "
            f"'{SRC_PATH.as_posix()}' is not in the n01 or n02 consortium of "
            "ARCHER2). Falling back to the ARCHER2 n01 ADCIRC data root "
            f"'{ARCHER2_N01_DATA_ROOT}', which is probably wrong on this "
            "machine. Set the WORSTSURGE_DATA_ROOT environment variable to "
            "point at the directory containing your ADCIRC experiment "
            "outputs.",
            stacklevel=2,
        )

PROJ_PATH = Path(SRC_PATH).parent
DATA_PATH = os.path.join(PROJ_PATH, "data")
TMP_PATH = os.path.join(DATA_PATH, "tmp")
os.makedirs(TMP_PATH, exist_ok=True)
CONFIG_PATH = os.path.join(SRC_PATH, "config")
FIGURE_PATH = os.path.join(PROJ_PATH, "img", "adforce")
FORT63_EXAMPLE = os.path.join(DATA_PATH, "fort.63.nc")
