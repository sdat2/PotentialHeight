"""Fort.61 (ADCIRC station-elevation output) reading and plotting."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mesh import xr_loader


def read_fort61(path: str) -> pd.DataFrame:
    """Read an ADCIRC ``fort.61.nc`` into a tidy long-format DataFrame.

    Columns: ``station`` (0-based index), ``name`` (trimmed station name, or
    the index as a string when the file has no ``station_name`` variable),
    ``x``/``y`` (station lon/lat), ``time``, ``zeta`` (elevation in metres;
    NaN where the file uses a fill/dry value).

    Args:
        path (str): Path to a ``fort.61.nc`` file (or a run directory
            containing one).

    Returns:
        pd.DataFrame: One row per (station, time step).
    """
    if os.path.isdir(path):
        path = os.path.join(path, "fort.61.nc")
    ds = xr_loader(path)
    zeta = np.asarray(ds.zeta.values, dtype=float)  # (time, station)
    t = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    x = np.asarray(ds.x.values, dtype=float)
    y = np.asarray(ds.y.values, dtype=float)
    n_station = zeta.shape[1]
    if "station_name" in ds:
        raw = ds.station_name.values
        names = [
            (b.tobytes().decode(errors="ignore") if hasattr(b, "tobytes") else str(b))
            .strip("\x00 ")
            .strip()
            for b in raw
        ]
    else:
        names = [str(i) for i in range(n_station)]
    frames = [
        pd.DataFrame(
            dict(station=i, name=names[i], x=x[i], y=y[i], time=t, zeta=zeta[:, i])
        )
        for i in range(n_station)
    ]
    return pd.concat(frames, ignore_index=True)


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
