"""Simulation sources: where a storm's simulated water-level field comes from.

Two granularities:

* :class:`HFArchiveSource` / :class:`Fort63RunSource` provide the full field
  ``(x, y, wet, elev, t)`` that :func:`adforce.eval.validate.validate_storm`
  samples with its nearest-wet-node lookup. ``wet`` is the array the
  wet-criterion runs on (total water depth), ``elev`` the elevation series
  that is scored (surge for the tide-off HF archive, raw zeta -- total water
  level under tidal forcing -- for run dirs).
* :class:`RunDirSource` provides per-gauge series directly from a completed
  adforce run directory, preferring the cheapest artifact present:
  ``gauge_ts.parquet`` (remote-extracted) > ``fort.61.nc`` (station output)
  > ``fort.63.nc`` (full field, nearest-wet sampling).

Geometry note: the HF archive stores element-centroid dual-graph fields with
a gap-free ``WD`` (so ``np.nanmin`` is a sufficient wet criterion), while
``fort.63.nc`` is node-based with NaN when a node dries -- Fort63RunSource
maps dry samples to ``-inf`` so the shared nearest-wet criterion rejects any
node that ever dries, reproducing the published behaviour of
``rerun/adcirc/extract_gauge_series.py``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from . import constants as C

#: per-gauge result: (gauge display name, series [m], node/station distance
#: in degrees (NaN when unknown), artifact the series came from)
GaugeSeries = Tuple[str, pd.Series, float, str]


class FieldSource(Protocol):
    """A storm's full simulated field, in validate_storm's expected shape."""

    def load(
        self, storm: str, fname: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Return ``(x, y, wet, elev, t)`` for one storm."""
        ...


class HFArchiveSource:
    """The published SurgeNet HF archive (element centroids, tide-off).

    A thin delegation around the extraction previously inlined in
    ``validate_storm``: ``ssh = WD + DEM`` is the surge elevation, ``WD`` the
    wet criterion. Behaviour (dtypes, values) is identical by construction --
    the byte-identical ``val_summary.csv`` regression gate pins it.
    """

    def load(self, storm: str, fname: Optional[str] = None):
        import xarray as xr

        from .validate import download_storm

        if fname is None:
            fname = C.STORMS[storm]
        ds = xr.open_dataset(download_storm(fname))
        x, y, DEM, WD = ds.x.values, ds.y.values, ds.DEM.values, ds.WD.values
        ssh = WD + DEM[None, :]
        t = pd.to_datetime(ds.time.values)
        return x, y, WD, ssh, t


class Fort63RunSource:
    """Full-field source for one adforce run directory's ``fort.63.nc``.

    ``elev`` is raw zeta: total water level when the run had tidal forcing,
    surge-only otherwise (read the run's ``config.yaml`` to know which).
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def load(self, storm: str, fname: Optional[str] = None):
        from .extract import load_fort63

        x, y, depth, zeta, t = load_fort63(os.path.join(self.run_dir, "fort.63.nc"))
        total = zeta + depth[None, :]
        # NaN (dry) -> -inf so nanmin-based wet criteria reject drying nodes.
        wet = np.where(np.isfinite(total), total, -np.inf)
        return x, y, wet, zeta, t


class RunDirSource:
    """Per-gauge series from a completed run dir, cheapest artifact first.

    Args:
        run_dir (str): The per-storm run directory.
        prefer (tuple): Artifact preference order among
            ``("gauge_ts", "fort61", "fort63")``. Gauges the preferred
            artifact cannot supply fall through to the next one.
        max_station_deg (float): fort.61 stations are matched to gauges by
            nearest Euclidean distance in degrees within this radius (station
            lists are configured locations, not guaranteed to be the CO-OPS
            panel).
    """

    def __init__(
        self,
        run_dir: str,
        prefer: Tuple[str, ...] = ("gauge_ts", "fort61", "fort63"),
        max_station_deg: float = 0.02,
    ):
        self.run_dir = run_dir
        self.prefer = prefer
        self.max_station_deg = max_station_deg

    # -- one loader per artifact kind; each returns {sid: GaugeSeries} ------ #
    def _from_gauge_ts(self, gauges: pd.DataFrame) -> Dict[str, GaugeSeries]:
        fp = os.path.join(self.run_dir, "gauge_ts.parquet")
        if not os.path.exists(fp):
            return {}
        df = pd.read_parquet(fp)
        out: Dict[str, GaugeSeries] = {}
        wanted = set(gauges.sid.astype(str))
        for sid, grp in df.groupby("sid", sort=False):
            sid = str(sid)
            if sid not in wanted:
                continue
            grp = grp.sort_values("time")
            s = pd.Series(grp.zeta.values, index=pd.DatetimeIndex(grp.time))
            out[sid] = (str(grp.gauge.iloc[0]), s, float("nan"), "gauge_ts")
        return out

    def _from_fort61(self, gauges: pd.DataFrame) -> Dict[str, GaugeSeries]:
        fp = os.path.join(self.run_dir, "fort.61.nc")
        if not os.path.exists(fp):
            return {}
        from adforce.fort61 import read_fort61

        df = read_fort61(fp)
        stations = df.drop_duplicates("station")[["station", "x", "y"]]
        out: Dict[str, GaugeSeries] = {}
        for g in gauges.itertuples():
            d = np.hypot(stations.x - g.lon, stations.y - g.lat)
            i = int(d.idxmin())
            if float(d.loc[i]) > self.max_station_deg:
                continue
            grp = df[df.station == stations.station.loc[i]].sort_values("time")
            s = pd.Series(grp.zeta.values, index=pd.DatetimeIndex(grp.time))
            out[str(g.sid)] = (str(g.name), s, float(d.loc[i]), "fort61")
        return out

    def _from_fort63(self, gauges: pd.DataFrame) -> Dict[str, GaugeSeries]:
        fp = os.path.join(self.run_dir, "fort.63.nc")
        if not os.path.exists(fp):
            return {}
        from scipy.spatial import cKDTree

        from .validate import _nearest_wet

        x, y, wet, elev, t = Fort63RunSource(self.run_dir).load("")
        tree = cKDTree(np.column_stack([x, y]))
        out: Dict[str, GaugeSeries] = {}
        for g in gauges.itertuples():
            idx, dist = _nearest_wet(tree, wet, g.lon, g.lat)
            if idx is None:
                continue
            s = pd.Series(elev[:, idx], index=t)
            out[str(g.sid)] = (str(g.name), s, dist, "fort63")
        return out

    def sim_series(self, gauges: pd.DataFrame) -> Dict[str, GaugeSeries]:
        """Series for each gauge (``sid, name, lat, lon`` DataFrame rows)."""
        loaders = {
            "gauge_ts": self._from_gauge_ts,
            "fort61": self._from_fort61,
            "fort63": self._from_fort63,
        }
        out: Dict[str, GaugeSeries] = {}
        for kind in self.prefer:
            missing = gauges[~gauges.sid.astype(str).isin(out)]
            if missing.empty:
                break
            out.update(loaders[kind](missing))
        return out
