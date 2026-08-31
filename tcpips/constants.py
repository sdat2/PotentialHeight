"""This file is used to save all possible project wide constants.

Includes source folder, the project path, etc.

Example:
    Import statement at top of script::

        from tcpips.constants import PROJECT_PATH, FIGURE_PATH

"""

# import os/pathlib to manipulate file names.
from typing import Dict, List, Tuple
import os
import pathlib


PANGEO_CMIP6_URL: str = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(PROJECT_PATH, "img"))
CONFIG_PATH: str = os.path.join(SRC_PATH, "config")
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")


def _ensure_dir(path) -> None:
    """``mkdir -p`` that tolerates dangling symlinks.

    ``data/era5`` is a symlink onto an external volume; when the volume is
    unmounted, ``os.makedirs(..., exist_ok=True)`` on it (or any child) still
    raises FileExistsError, which used to make ``import tcpips`` fail on any
    machine without the drive attached (breaking transitive importers like
    ``adforce.training.driver``). Directory creation is best-effort here --
    the code that actually writes will surface a real error at write time.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

# General data from e.g. paper or cmip etc.
GOM: Tuple[float] = (25.443701, -90.013120)  # Centre of Gulf of Mexico, lat, lon
MONTHS: List[str] = [  # 3 letter month names
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

QUARTERS: List[str] = [  # 3-letter quarter names Q1, Q2, Q3, Q4
    "JFM",
    "AMJ",
    "JAS",
    "OND",
]

SEASONS: List[str] = [  # 3-letter season names DJF, MAM, JJA, SON
    "DJF",
    "MAM",
    "JJA",
    "SON",
]

# Directories for ERA5 data processing steps: make them ahead of time.
ERA5_PATH: str = os.path.join(DATA_PATH, "era5")  # main data folder
_ensure_dir(ERA5_PATH)
ERA5_RAW_PATH: str = os.path.join(ERA5_PATH, "raw")  # download data here
_ensure_dir(ERA5_RAW_PATH)
ERA5_REGRIDDED_PATH: str = os.path.join(ERA5_PATH, "regrid")  # regridded data here
_ensure_dir(ERA5_REGRIDDED_PATH)
ERA5_PI_OG_PATH: str = os.path.join(ERA5_PATH, "pi_og")  # pi no bias correction before
_ensure_dir(ERA5_PI_OG_PATH)  # potential intensity on original grid
ERA5_PI_PATH: str = os.path.join(ERA5_PATH, "pi")  # pi on new grid
_ensure_dir(ERA5_PI_PATH)  # potential intensity on new grid
ERA5_PS_OG_PATH: str = os.path.join(
    ERA5_PATH, "ps_og"
)  # potential size on original grid
_ensure_dir(ERA5_PS_OG_PATH)  # potential size on
ERA5_PS_PATH = os.path.join(ERA5_PATH, "ps")  # potential size
_ensure_dir(ERA5_PS_PATH)  # potential size on new grid
ERA5_PRODUCTS_PATH = os.path.join(ERA5_PATH, "products")  # products from ERA5
_ensure_dir(ERA5_PRODUCTS_PATH)  # products from ERA5
ERA5_FIGURE_PATH: str = os.path.join(FIGURE_PATH, "era5")  # figures from ERA5
_ensure_dir(ERA5_FIGURE_PATH)  # figures from ERA5

# Directories for CMIP6 data processing steps: make them ahead of time.
CMIP6_PATH: str = os.path.join(DATA_PATH, "cmip6")  # main data folder
_ensure_dir(CMIP6_PATH)
RAW_PATH: str = os.path.join(CMIP6_PATH, "raw")  # download data here
_ensure_dir(RAW_PATH)
REGRIDDED_PATH: str = os.path.join(CMIP6_PATH, "regridded")  # regridded data here
_ensure_dir(REGRIDDED_PATH)
CDO_PATH: str = os.path.join(CMIP6_PATH, "regrid")
_ensure_dir(CDO_PATH)
BIAS_CORRECTED_PATH = os.path.join(
    CMIP6_PATH, "bias_corrected"
)  # bias corrected data here
_ensure_dir(BIAS_CORRECTED_PATH)
PI_PATH: str = os.path.join(CMIP6_PATH, "pi")  # pi no bias correction before
_ensure_dir(PI_PATH)
PI2_PATH: str = os.path.join(CMIP6_PATH, "pi2")  # pi after cdo regridding
_ensure_dir(PI2_PATH)
PI3_PATH: str = os.path.join(CMIP6_PATH, "pi3")  # pi after temp profile fix
_ensure_dir(PI3_PATH)
PI4_PATH: str = os.path.join(CMIP6_PATH, "pi4")  # pi with different temp profile fix
_ensure_dir(PI4_PATH)
BC_PI_PATH: str = os.path.join(CMIP6_PATH, "bc_pi")  # pi after bias correction
_ensure_dir(BC_PI_PATH)
PS_PATH = os.path.join(CMIP6_PATH, "ps")  # potential size
_ensure_dir(PS_PATH)  # potential size

BIAS_PATH = os.path.join(CMIP6_PATH, "bias")  # bias data
_ensure_dir(BIAS_PATH)  # bias data

# Constants for converting CMIP6 variables to PI input variables
CONVERSION_NAMES: Dict[str, str] = {
    "tos": "sst",
    "hus": "q",
    "ta": "t",
    "psl": "msl",
    "hurs": "rh",  # relative humidity useful for calulating potential size (but can be derived from q, t, and msl)
}
CONVERSION_MULTIPLES: Dict[str, float] = {
    "hus": 1000,
    "psl": 0.01,  # "plev": 0.01
}
CONVERSION_ADDITIONS: Dict[str, float] = {"ta": -273.15}
CONVERSION_UNITS: Dict[str, str] = {
    "hus": "g/kg",
    "psl": "hPa",
    "tos": "degC",
    "ta": "degC",  # "plev": "hPa"
    "rh": "%",
}
