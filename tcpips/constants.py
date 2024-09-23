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

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(PROJECT_PATH, "img"))
CONFIG_PATH: str = os.path.join(SRC_PATH, "config")
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")

# General data from e.g. paper or cmip etc.
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")
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

# Directories for CMIP6 data processing steps
CMIP6_PATH: str = os.path.join(DATA_PATH, "cmip6")
os.makedirs(CMIP6_PATH, exist_ok=True)
RAW_PATH: str = os.path.join(CMIP6_PATH, "raw")  # download data here
os.makedirs(RAW_PATH, exist_ok=True)
REGRIDDED_PATH: str = os.path.join(CMIP6_PATH, "regridded")  # regridded data here
os.makedirs(REGRIDDED_PATH, exist_ok=True)
BIAS_CORRECTED_PATH = os.path.join(
    CMIP6_PATH, "bias_corrected"
)  # bias corrected data here
os.makedirs(BIAS_CORRECTED_PATH, exist_ok=True)
PI_PATH: str = os.path.join(DATA_PATH, "pi")  # pi no bias correction before
os.makedirs(PI_PATH, exist_ok=True)
BC_PI_PATH: str = os.path.join(DATA_PATH, "bc_pi")
os.makedirs(BC_PI_PATH, exist_ok=True)

# Constants for converting CMIP6 variables to PI input variables
CONVERSION_NAMES: Dict[str, str] = {"tos": "sst", "hus": "q", "ta": "t", "psl": "msl"}
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
}
