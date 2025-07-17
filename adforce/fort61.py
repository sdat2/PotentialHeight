"""Fort.61"""

import os
import matplotlib.pyplot as plt
from .mesh import xr_loader


def plot_model_tgauges(folder: str = "../../kat.nws13/") -> None:
    """
    Plot the zeta timeseries from the fort.61 file.

    Args:
        folder (str, optional): Path to the fort.61 file. Defaults to "../../kat.nws13/".
    """

    station_nc = xr_loader(os.path.join(folder, "fort.61.nc"))
    plt.plot(station_nc.time.values, station_nc.zeta.values)


if __name__ == "__main__":
    # python -m adforce.fort61
    plot_model_tgauges()
    plt.show()
