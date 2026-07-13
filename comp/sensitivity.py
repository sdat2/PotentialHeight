"""Sensitivity of the headline skill to the analysis' free thresholds.

A score that only looks good at one hand-picked set of cut-offs is not trustworthy.
This module re-derives bias / RMSE / r as each knob is varied, to show the result is
robust rather than tuned. Two groups of knobs:

  * clean-filter cut-offs (MIN_OBS_PEAK_M, MAX_TIMING_HR) only re-label existing
    gauge-storm pairs, so they sweep cheaply straight from ``val_summary.csv``.

  * node-selection cut-offs (WET_MIN_M, MAX_NODE_DEG, KNN) change which mesh node is
    sampled and therefore the simulated peak, so they require the cached storm
    netCDFs. Each storm is loaded ONCE and re-sampled for every combination.

Run::

    python -m comp.sensitivity              # cheap filter sweep + node-selection sweep
    python -m comp.sensitivity --no-node    # cheap filter sweep only (no netCDF)
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from . import constants as C
from .validate import classify_setting, download_storm, metrics, within_storm_r
from .coops import gulf_gauges, observed_residual

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# cheap: vary the clean-filter cut-offs from the summary CSV
# --------------------------------------------------------------------------- #
def filter_sweep() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(C.OUT_PATH, "val_summary.csv"))
    df["sid"] = df["sid"].astype(str)
    base = df[~df.failed]
    rows = []

    # vary the meaningful-surge threshold (default MIN_OBS_PEAK_M)
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        d = base[(base.obs_peak >= thr) & (base.peak_dt_hr.abs() <= C.MAX_TIMING_HR)]
        b, e, r = metrics(d)
        rows.append(
            dict(
                knob="MIN_OBS_PEAK_M",
                value=thr,
                n=len(d),
                bias=b,
                rmse=e,
                r=r,
                within_r=within_storm_r(d),
            )
        )
    # vary the simultaneous-peak window (default MAX_TIMING_HR)
    for thr in (3, 4, 6, 9, 12, 24):
        d = base[(base.obs_peak >= C.MIN_OBS_PEAK_M) & (base.peak_dt_hr.abs() <= thr)]
        b, e, r = metrics(d)
        rows.append(
            dict(
                knob="MAX_TIMING_HR",
                value=thr,
                n=len(d),
                bias=b,
                rmse=e,
                r=r,
                within_r=within_storm_r(d),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# expensive: vary node-selection knobs (re-sample mesh nodes from the netCDFs)
# --------------------------------------------------------------------------- #
def _nearest_wet_param(tree, WD, lon, lat, wet_min, max_deg, knn):
    d, idx = tree.query([lon, lat], k=knn)
    for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if di > max_deg:
            break
        if np.nanmin(WD[:, ii]) > wet_min:
            return int(ii)
    return None


def node_sweep(
    combos: Optional[List[Tuple[float, float, int]]] = None,
    storms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Re-run peak extraction for several (WET_MIN_M, MAX_NODE_DEG, KNN) combos.

    Loads each storm netCDF once; re-uses the cached observed residuals. Reports the
    clean-pair skill for each combo so it can be compared with the default
    (0.30, 0.12, 60)."""
    if combos is None:
        combos = [
            (0.10, 0.12, 60),
            (0.30, 0.12, 60),
            (0.50, 0.12, 60),
            (0.30, 0.06, 60),
            (0.30, 0.20, 60),
            (0.30, 0.12, 30),
        ]
    items = {k: C.STORMS[k] for k in (storms or C.STORMS)}
    gauges = gulf_gauges()
    # accumulate per-combo lists of (sim_peak, obs_peak, storm, timing) over clean-ish pairs
    acc: Dict[Tuple[float, float, int], list] = {cmb: [] for cmb in combos}
    for storm, fname in items.items():
        year = int(storm.split()[-1])
        try:
            ds = xr.open_dataset(download_storm(fname))
        except Exception as e:  # pragma: no cover
            print(f"!! {storm}: {e}")
            continue
        x, y, DEM, WD = ds.x.values, ds.y.values, ds.DEM.values, ds.WD.values
        ssh = WD + DEM[None, :]
        t = pd.to_datetime(ds.time.values)
        s0, s1 = t[0], t[-1]
        tree = cKDTree(np.column_stack([x, y]))
        for sid, name, lat, lon in gauges:
            obs, _ = observed_residual(sid, lat, year, s0, s1)
            if obs.empty or obs.size < 12 or (sid, name) in []:
                continue
            if obs.max() < C.MIN_OBS_PEAK_M:  # only meaningful-surge gauges
                continue
            if (storm, sid) in C.KNOWN_FAILED:
                continue
            obs_peak = float(obs.max())
            obs_tmax = obs.idxmax()
            for cmb in combos:
                idx = _nearest_wet_param(tree, WD, lon, lat, *cmb)
                if idx is None:
                    continue
                sim = pd.Series(ssh[:, idx], index=t)
                timing = (sim.idxmax() - obs_tmax).total_seconds() / 3600.0
                acc[cmb].append((float(sim.max()), obs_peak, storm, timing))
        ds.close()
        print(f"  {storm:14s} done")
    rows = []
    for cmb, recs in acc.items():
        d = pd.DataFrame(recs, columns=["sim_peak", "obs_peak", "storm", "timing"])
        d = d[d.timing.abs() <= C.MAX_TIMING_HR]  # apply the default timing gate
        b, e, r = metrics(d)
        rows.append(
            dict(
                wet_min=cmb[0],
                max_deg=cmb[1],
                knn=cmb[2],
                n=len(d),
                bias=b,
                rmse=e,
                r=r,
                within_r=within_storm_r(d),
                default=(cmb == (C.WET_MIN_M, C.MAX_NODE_DEG, C.KNN)),
            )
        )
    return pd.DataFrame(rows)


def run(do_node: bool = True) -> None:
    fs = filter_sweep()
    print("=== clean-filter sweep (from val_summary.csv) ===")
    print(fs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fs.to_csv(os.path.join(C.OUT_PATH, "sensitivity_filter.csv"), index=False)
    if do_node:
        print("\n=== node-selection sweep (re-sampling netCDFs) ===")
        ns = node_sweep()
        print(ns.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        ns.to_csv(os.path.join(C.OUT_PATH, "sensitivity_node.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-node", action="store_true", help="skip the netCDF node sweep")
    run(do_node=not ap.parse_args().no_node)


if __name__ == "__main__":
    main()
