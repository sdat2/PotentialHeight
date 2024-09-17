"""CLE15 python package constants."""

import pathlib
import os

constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(SRC_PATH, "img"))
DATA_PATH = os.path.join(SRC_PATH, "data")
SUP_PATH = os.path.join(SRC_PATH, "sup")


# could probably move some of these to a config file.
BACKGROUND_PRESSURE = 1015 * 100  # [Pa]
TEMP_0K = 273.15  # [K]
DEFAULT_SURF_TEMP = 299  # [K]
W_COOL_DEFAULT = 0.002  # [K m s-1]
RHO_AIR_DEFAULT = 1.15  # [kg m-3]
F_COR_DEFAULT = 5e-5  # [s-2]
