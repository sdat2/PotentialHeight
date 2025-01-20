import os
from pathlib import Path

SRC_PATH = Path(__file__).parent
SETUP_PATH = os.path.join(SRC_PATH, "setup")
PROJ_PATH = Path(SRC_PATH).parent
DATA_PATH = os.path.join(PROJ_PATH, "data", "worst")
os.makedirs(DATA_PATH, exist_ok=True)
CONFIG_PATH = os.path.join(SRC_PATH, "config")
FIGURE_PATH = os.path.join(PROJ_PATH, "img")
