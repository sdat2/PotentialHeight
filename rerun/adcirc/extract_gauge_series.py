"""Extract simulated water-level series at NOAA gauges from ADCIRC runs.

SUPERSEDED by ``adforce.eval.extract`` (hydra-driven port, same node
selection and output schema); kept as the frozen record of the GCP sweeps
that produced ``data/comp/{lowres,midres}`` and ``rerun/results/*``.

For every run directory ``<runs_dir>/<i>_<NAME>_<YEAR>/`` containing a
``fort.63.nc``, find the nearest mesh node to each gauge in ``--gauges``
(CSV: sid,name,lat,lon) that stays wet (min total depth zeta+depth >
``--wet-min``) within ``--max-deg``, and write one long-format parquet
(columns: storm, sid, gauge, time, zeta) to ``--out``. Mirrors the node
selection of comp.validate so low-resolution sweeps score like-for-like
against the same de-tided observations.

Runs inside the worstsurge container (needs netCDF4/xarray/pandas):
    python extract_gauge_series.py --runs-dir /work/lowres_runs \
        --gauges /work/gauges_both_boxes.csv --out /work/lowres_gauge_series.parquet
"""

import argparse
import glob
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_fort63(path: str):
    ds = nc.Dataset(path)
    for v in ("neta", "nvel"):
        if v in ds.variables:
            del ds.variables[v]
    x = np.asarray(ds.variables["x"][:])
    y = np.asarray(ds.variables["y"][:])
    depth = np.asarray(ds.variables["depth"][:])
    zeta = np.asarray(ds.variables["zeta"][:])  # (time, node), masked/fill when dry
    if hasattr(zeta, "filled"):
        zeta = np.ma.filled(zeta, np.nan)
    t = nc.num2date(
        ds.variables["time"][:],
        ds.variables["time"].units,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    ds.close()
    return x, y, depth, zeta, pd.DatetimeIndex(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--gauges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-deg", type=float, default=0.12)
    ap.add_argument("--wet-min", type=float, default=0.3)
    ap.add_argument("--knn", type=int, default=60)
    a = ap.parse_args()

    gauges = pd.read_csv(a.gauges, dtype={"sid": str})
    frames = []
    for run in sorted(glob.glob(os.path.join(a.runs_dir, "*_*_*"))):
        f63 = os.path.join(run, "fort.63.nc")
        if not os.path.exists(f63):
            print(f"(skip {os.path.basename(run)}: no fort.63.nc)")
            continue
        storm = os.path.basename(run)
        x, y, depth, zeta, t = load_fort63(f63)
        total = zeta + depth[None, :]
        tree = cKDTree(np.c_[x, y])
        n_hit = 0
        for g in gauges.itertuples():
            d, idx = tree.query([g.lon, g.lat], k=a.knn)
            pick = None
            for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
                if di > a.max_deg:
                    break
                col = total[:, ii]
                if np.isfinite(col).all() and np.nanmin(col) > a.wet_min:
                    pick = int(ii)
                    break
            if pick is None:
                continue
            frames.append(
                pd.DataFrame(
                    dict(storm=storm, sid=g.sid, gauge=g.name, time=t, zeta=zeta[:, pick])
                )
            )
            n_hit += 1
        print(f"{storm}: {n_hit}/{len(gauges)} gauges matched")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
