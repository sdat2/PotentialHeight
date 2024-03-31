"""This file is used to save all possible project wide constants.

Includes source folder, the project path, etc.

Example:
    Import statement at top of script::

        from tcpips.constants import PROJECT_PATH, FIGURE_PATH

"""

# import os/pathlib to manipulate file names.
import os
import pathlib

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(PROJECT_PATH, "img"))
CONFIG_PATH = os.path.join(SRC_PATH, "config")
DATA_PATH = os.path.join(PROJECT_PATH, "data")

# General data from e.g. paper or cmip etc.
DATA_PATH = os.path.join(PROJECT_PATH, "data")
GOM = (25.443701, -90.013120)  # Centre of Gulf of Mexico, lat, lon
MONTHS = [  # 3 letter month names
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

QUARTERS = [  # 3-letter quarter names
    "JFM",
    "AMJ",
    "JAS",
    "OND",
]

CMIP6_PATH = os.path.join(DATA_PATH, "cmip6")
os.makedirs(CMIP6_PATH, exist_ok=True)
