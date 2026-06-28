"""Validate historical ADCIRC surge against de-tided NOAA gauges.

Pipeline, per storm:
  1. download the storm's netCDF from Hugging Face (``HF_REPO``);
  2. extract the simulated surge (SSH = WD + DEM) at the nearest *wet* mesh node
     to each NOAA CO-OPS gauge in the regional box;
  3. fetch + de-tide the gauge record (:func:`comp.coops.observed_residual`);
  4. compare peak surge and peak timing, tag "clean" pairs, score skill.

ADCIRC here is surge-only (no tides), so we always compare against the de-tided
observed residual rather than total water level.

Run::

    python -m comp.validate                # full sweep, all STORMS
    python -m comp.validate --storms "Ida 2021" "Katrina 2005"
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from . import constants as C
from .coops import Gauge, gulf_gauges, observed_residual

# Gauges whose name implies a semi-enclosed (bay / estuary / river / inlet)
# setting, where a medium-resolution mesh and the no-tide assumption are least
# reliable. Used only to report open-coast skill separately -- a transparent
# heuristic, not a hard exclusion.
SEMI_ENCLOSED_KW = (
    "bay", "river", "bayou", "lake", "canal", "lock", "bridge", "dock",
    "creek", "bank", "channel", "turning basin", "ship", "inner",
)


def classify_setting(name: str) -> str:
    n = name.lower()
    return "semi-enclosed" if any(k in n for k in SEMI_ENCLOSED_KW) else "open-coast"


def download_storm(fname: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=C.HF_REPO, repo_type="dataset", filename=fname,
        local_dir=C.HF_STORM_CACHE,
    )


def _nearest_wet(tree: cKDTree, WD: np.ndarray, lon: float, lat: float
                 ) -> Tuple[Optional[int], Optional[float]]:
    """Nearest node to (lon,lat) that never dries below ``WET_MIN_M``."""
    d, idx = tree.query([lon, lat], k=C.KNN)
    for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if di > C.MAX_NODE_DEG:
            break
        if np.nanmin(WD[:, ii]) > C.WET_MIN_M:
            return int(ii), float(di)
    return None, None


def validate_storm(storm: str, fname: str, gauges: List[Gauge]
                   ) -> Tuple[List[dict], Dict[str, tuple]]:
    """Return (rows, series) for one storm. ``series[name] = (sim, obs)``."""
    year = int(storm.split()[-1])
    ds = xr.open_dataset(download_storm(fname))
    x, y, DEM, WD = ds.x.values, ds.y.values, ds.DEM.values, ds.WD.values
    ssh = WD + DEM[None, :]
    t = pd.to_datetime(ds.time.values)
    s0, s1 = t[0], t[-1]
    tree = cKDTree(np.column_stack([x, y]))

    rows: List[dict] = []
    series: Dict[str, tuple] = {}
    for sid, name, lat, lon in gauges:
        idx, dist = _nearest_wet(tree, WD, lon, lat)
        if idx is None:
            continue
        obs, method = observed_residual(sid, lat, year, s0, s1)
        if obs.empty or obs.size < 12:
            continue
        sim = pd.Series(ssh[:, idx], index=t)
        series[name] = (sim, obs)
        timing = (sim.idxmax() - obs.idxmax()).total_seconds() / 3600.0
        rows.append(dict(
            storm=storm, sid=sid, name=name, setting=classify_setting(name),
            node_deg=round(dist, 3), sim_peak=round(float(sim.max()), 2),
            obs_peak=round(float(obs.max()), 2), peak_dt_hr=round(timing, 1),
            method=method, n_obs=int(obs.size),
        ))
    return rows, series


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sid"] = df["sid"].astype(str)  # robust to CSV round-trip parsing sid as int
    df["failed"] = [(s, i) in C.KNOWN_FAILED for s, i in zip(df.storm, df.sid)]
    df["poor_event"] = df.storm.isin(C.POOR_SURGE_EVENTS)
    df["clean"] = (
        (df.peak_dt_hr.abs() <= C.MAX_TIMING_HR)
        & (df.obs_peak >= C.MIN_OBS_PEAK_M)
        & ~df.failed
    )
    return df


def metrics(d: pd.DataFrame) -> Tuple[float, float, float]:
    """(bias, RMSE, r) of simulated vs observed peak surge."""
    if len(d) < 2:
        return (np.nan, np.nan, np.nan)
    err = d.sim_peak - d.obs_peak
    return (err.mean(), float(np.sqrt((err ** 2).mean())),
            float(np.corrcoef(d.obs_peak, d.sim_peak)[0, 1]))


def report(df: pd.DataFrame) -> None:
    cln = df[df.clean]
    print("\n=== PER-STORM (clean pairs) ===")
    for storm in df.storm.unique():
        d = cln[cln.storm == storm]
        if len(d):
            b, e, r = metrics(d)
            print(f"  {storm:14s} n={len(d):2d}  bias={b:+.2f}  RMSE={e:.2f}  r={r:.2f}")
    b, e, r = metrics(cln)
    print(f"\nOVERALL clean (n={len(cln)}): bias={b:+.2f} m  RMSE={e:.2f} m  r={r:.2f}")
    print(f"clean median |peak timing| = {cln.peak_dt_hr.abs().median():.1f} h")
    for setting in ("open-coast", "semi-enclosed"):
        d = cln[cln.setting == setting]
        b, e, r = metrics(d)
        print(f"  {setting:14s} n={len(d):3d}  bias={b:+.2f}  RMSE={e:.2f}  r={r:.2f}")
    print(f"de-tide methods: {dict(df.method.value_counts())}")


def scatter(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cln = df[df.clean]
    b, e, r = metrics(cln)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    storms = list(df.storm.unique())
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(storms)))
    for s, c in zip(storms, colors):
        sc = df[(df.storm == s) & df.clean]
        sx = df[(df.storm == s) & ~df.clean]
        ax.scatter(sc.obs_peak, sc.sim_peak, color=[c], label=s, s=36, zorder=3)
        ax.scatter(sx.obs_peak, sx.sim_peak, facecolors="none", edgecolors=[c],
                   alpha=0.35, s=28, zorder=2)
    m = max(df.obs_peak.max(), df.sim_peak.max()) * 1.1 + 0.3
    ax.plot([0, m], [0, m], "k--", alpha=0.5)
    if len(cln) > 1:
        sl, ic = np.polyfit(cln.obs_peak, cln.sim_peak, 1)
        ax.plot([0, m], [ic, sl * m + ic], color="0.3", lw=1, label=f"fit slope={sl:.2f}")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("Observed peak residual (m)")
    ax.set_ylabel("Simulated peak surge (m)")
    ax.set_title(f"Peak surge: historical ADCIRC vs NOAA residual\n"
                 f"clean n={len(cln)}: bias={b:+.2f} m  RMSE={e:.2f} m  r={r:.2f}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=135)
    print(f"wrote {path}")


def plot_examples(panels: List[Tuple[str, str]], path: str,
                  ncol: int = 3) -> None:
    """Plot simulated surge vs de-tided observed residual for chosen (storm, gauge).

    ``panels`` is a list of ``(storm, gauge_name)`` tuples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gauges = gulf_gauges()
    cache: Dict[str, Dict[str, tuple]] = {}
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, (storm, gname) in zip(axes.ravel(), panels):
        if storm not in cache:
            _, cache[storm] = validate_storm(storm, C.STORMS[storm], gauges)
        match = [k for k in cache[storm] if gname.lower() in k.lower()]
        if not match:
            ax.set_title(f"{gname}\n(no data)", fontsize=8); continue
        sim, obs = cache[storm][match[0]]
        ax.plot(sim.index, sim.values, color="tab:blue", lw=1.6, label="ADCIRC surge")
        ax.plot(obs.index, obs.values, color="black", lw=1.1, label="NOAA residual")
        ax.set_title(f"{storm}: {match[0][:24]}", fontsize=8, loc="left")
        ax.set_ylabel("surge (m)"); ax.grid(alpha=0.3)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_fontsize(6); lab.set_ha("right")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def run(storms: Optional[List[str]] = None) -> pd.DataFrame:
    items = {k: C.STORMS[k] for k in (storms or C.STORMS)}
    gauges = gulf_gauges()
    print(f"{len(gauges)} candidate gauges in box; {len(items)} storms")
    rows: List[dict] = []
    for storm, fname in items.items():
        try:
            r, _ = validate_storm(storm, fname, gauges)
        except Exception as e:  # pragma: no cover
            print(f"!! {storm}: {e}")
            continue
        rows += r
        print(f"  {storm:14s}: {len(r):2d} gauges with data")
    df = add_flags(pd.DataFrame(rows)).sort_values(
        ["storm", "clean", "obs_peak"], ascending=[True, False, False])
    out_csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    df.to_csv(out_csv, index=False)
    report(df)
    scatter(df, os.path.join(C.FIGURE_PATH, "val_scatter.png"))
    print(f"wrote {out_csv}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storms", nargs="*", default=None,
                    help="subset of storm names (default: all)")
    run(ap.parse_args().storms)


if __name__ == "__main__":
    main()
