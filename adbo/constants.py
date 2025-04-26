"""This file is used to save all possible project wide constants.

Includes source folder, the project path, etc.

Example:
    Import statement at top of script::

        from tcpips.constants import PROJECT_PATH, FIGURE_PATH

"""

# import os/pathlib to manipulate file names.
import os
import pathlib
import yaml


# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(PROJECT_PATH, "img"))
CONFIG_PATH = os.path.join(SRC_PATH, "config")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "adbo")
os.makedirs(DATA_PATH, exist_ok=True)
AD_PATH = pathlib.Path(
    os.path.dirname(PROJECT_PATH)
)  # assume adcirc is in the parent directory of the project
EXP_PATH = os.path.join(
    PROJECT_PATH, "exp"
)  # for n02 use AD_PATH, for n01 use PROJECT_PATH
os.makedirs(EXP_PATH, exist_ok=True)
# ROOT: str = "/work/n01/n01/sithom/adcirc-swan/"  # ARCHER2 path, move to constants

DEFAULT_CONSTRAINTS: dict = yaml.safe_load(
    open(os.path.join(CONFIG_PATH, "3d_constraints.yaml"))
)
