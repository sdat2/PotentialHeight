"""Reduce adforce run directories to per-gauge water-level series.

The remote-side reducer of the eval pipeline: a tidal mid-res ``fort.63.nc``
is 5-8 GB, so each run is reduced to a small long-format parquet (columns
``storm, sid, gauge, time, zeta`` -- the exact schema of the legacy
``data/comp/{lowres,midres}`` sweep extracts) right where it ran; only the
reduced artifact travels to the laptop.

Ported from ``rerun/adcirc/extract_gauge_series.py`` (which produced the
published resolution-comparison numbers); the node selection mirrors
``adforce.eval.validate._nearest_wet`` with the fort.63-specific guard that a
node must never dry (NaN zeta) during the run.

Run (hydra; config root adforce/eval/config/extract_config.yaml)::

    python -m adforce.eval.extract runs_root=/work/lowres_runs
    python -m adforce.eval.extract runs_root=... out=combined.parquet per_run=false
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C


def load_fort63(path: str):
    """Read ``fort.63.nc`` -> ``(x, y, depth, zeta, t)``; zeta NaN when dry."""
    import netCDF4 as nc

    ds = nc.Dataset(path)
    x = np.asarray(ds.variables["x"][:])
    y = np.asarray(ds.variables["y"][:])
    depth = np.asarray(ds.variables["depth"][:])
    zeta = ds.variables["zeta"][:]  # (time, node), masked/fill when dry
    zeta = np.ma.filled(zeta, np.nan) if np.ma.isMaskedArray(zeta) else np.asarray(zeta)
    t = nc.num2date(
        ds.variables["time"][:],
        ds.variables["time"].units,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    ds.close()
    return x, y, depth, zeta, pd.DatetimeIndex(t)


def gauge_frame(boxes: Optional[list] = None) -> pd.DataFrame:
    """CO-OPS gauge panel as a DataFrame (``sid, name, lat, lon``).

    Defaults to the union of the Gulf and Florida boxes -- the panel the
    validation sweep scores against (the remote side ships this as a CSV so
    extraction needs no CO-OPS access).
    """
    from .coops import gulf_gauges

    boxes = boxes if boxes is not None else [C.GAUGE_BOX, C.FLORIDA_BOX]
    rows, seen = [], set()
    for box in boxes:
        for sid, name, lat, lon in gulf_gauges(box):
            if sid not in seen:
                seen.add(sid)
                rows.append(dict(sid=str(sid), name=name, lat=lat, lon=lon))
    return pd.DataFrame(rows)


def extract_run(
    run_dir: str,
    gauges: pd.DataFrame,
    max_deg: float = C.MAX_NODE_DEG,
    wet_min: float = C.WET_MIN_M,
    knn: int = C.KNN,
) -> pd.DataFrame:
    """Extract every matchable gauge's zeta series from one run directory.

    The ``storm`` column is the run-dir basename (legacy ``<i>_<NAME>_<YEAR>``
    convention decodes via :func:`adforce.eval.cells.dir_to_storm`).

    A node qualifies for a gauge when it is within ``max_deg`` of it and its
    total depth ``zeta + depth`` stays finite (never dries) and above
    ``wet_min`` for the whole run -- exactly the published behaviour of
    ``rerun/adcirc/extract_gauge_series.py``.
    """
    from scipy.spatial import cKDTree

    storm = os.path.basename(os.path.normpath(run_dir))
    x, y, depth, zeta, t = load_fort63(os.path.join(run_dir, "fort.63.nc"))
    total = zeta + depth[None, :]
    tree = cKDTree(np.c_[x, y])
    frames = []
    for g in gauges.itertuples():
        d, idx = tree.query([g.lon, g.lat], k=min(knn, len(x)))
        pick = None
        for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            if di > max_deg:
                break
            col = total[:, ii]
            if np.isfinite(col).all() and np.nanmin(col) > wet_min:
                pick = int(ii)
                break
        if pick is None:
            continue
        frames.append(
            pd.DataFrame(
                dict(storm=storm, sid=str(g.sid), gauge=g.name, time=t, zeta=zeta[:, pick])
            )
        )
    if not frames:
        return pd.DataFrame(columns=["storm", "sid", "gauge", "time", "zeta"])
    return pd.concat(frames, ignore_index=True)


def extract_runs(
    runs_root: str,
    gauges: pd.DataFrame,
    out: Optional[str] = None,
    per_run: bool = True,
    max_deg: float = C.MAX_NODE_DEG,
    wet_min: float = C.WET_MIN_M,
    knn: int = C.KNN,
) -> pd.DataFrame:
    """Extract every run dir under ``runs_root`` that has a ``fort.63.nc``.

    Args:
        runs_root (str): Parent of the per-storm run directories.
        gauges (pd.DataFrame): ``sid, name, lat, lon`` rows.
        out (Optional[str]): Combined-parquet path (legacy sweep convention).
        per_run (bool): Also write ``<run>/gauge_ts.parquet`` beside each
            ``fort.63.nc`` (the minimal-artifact convention the laptop
            harvests).

    Returns:
        pd.DataFrame: The combined long-format table.
    """
    frames = []
    for run in sorted(p for p in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(p)):
        if not os.path.exists(os.path.join(run, "fort.63.nc")):
            continue
        df = extract_run(run, gauges, max_deg=max_deg, wet_min=wet_min, knn=knn)
        n_gauges = df.sid.nunique() if not df.empty else 0
        print(f"{os.path.basename(run)}: {n_gauges}/{len(gauges)} gauges matched")
        if per_run and not df.empty:
            df.to_parquet(os.path.join(run, "gauge_ts.parquet"), index=False)
        frames.append(df)
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["storm", "sid", "gauge", "time", "zeta"])
    )
    if out:
        combined.to_parquet(out, index=False)
        print(f"wrote {out} ({len(combined)} rows)")
    return combined


@hydra.main(version_base=None, config_path="config", config_name="extract_config")
def main(cfg: DictConfig) -> None:
    gauges = (
        pd.read_csv(cfg.gauges_csv, dtype={"sid": str})
        if cfg.gauges_csv
        else gauge_frame()
    )
    extract_runs(
        cfg.runs_root,
        gauges,
        out=cfg.out,
        per_run=cfg.per_run,
        max_deg=cfg.scoring.max_node_deg,
        wet_min=cfg.scoring.wet_min_m,
        knn=cfg.scoring.knn,
    )


if __name__ == "__main__":
    main()
