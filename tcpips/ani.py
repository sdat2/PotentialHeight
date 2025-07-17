import os
import xarray as xr
from sithom.plot import feature_grid
from tcpips.constants import FIGURE_PATH, PI_PATH


def ani_example(exp="ssp585", model="CESM", member="r4i1f1p1"):
    ds = xr.open_dataset(os.path.join(PI_PATH, exp, model, member) + ".nc")


if __name__ == "__main__":
    ani_example()
