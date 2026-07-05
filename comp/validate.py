"""Validate historical ADCIRC surge against de-tided NOAA gauges.

Pipeline, per storm:
  1. download the storm's netCDF from Hugging Face (``HF_REPO``);
  2. extract the simulated surge (SSH = WD + DEM) at the nearest *wet* mesh element
     centroid (the archived dual-graph node) to each NOAA CO-OPS gauge in the box;
  3. fetch + de-tide the gauge record (:func:`comp.coops.observed_residual`);
  4. score peak surge (bias/RMSE/correlation, with bootstrap CIs and a within-storm
     spatial correlation), the full hydrograph (:func:`timeseries_skill`), and peak
     timing; tag "clean" pairs and regenerate the paper figures + LaTeX table.

ADCIRC here is surge-only (no tides), so we always compare against the de-tided
observed residual rather than total water level.

Each storm's de-tided ``(sim, obs)`` series are cached as Parquet under
``data/comp/ts_cache/`` on a full sweep (write-through), keyed by the node-selection +
de-tiding parameters so the cache self-invalidates if those change. ``--examples-only``
then regenerates the example-panel figure from that cache without re-running the (slow)
utide de-tiding.

Run::

    python -m comp.validate                # full sweep, all STORMS (populates the cache)
    python -m comp.validate --storms "Ida 2021" "Katrina 2005"
    python -m comp.validate --examples-only         # just the example figure, from cache (fast)
    python -m comp.validate --examples-only --refresh-cache   # recompute the series first
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


def timeseries_skill(sim: pd.Series, obs: pd.Series
                     ) -> Tuple[float, float, int]:
    """Temporal skill of the simulated surge hydrograph against the observed
    residual: ``(corr, rmse, n_overlap)`` over the gauges' common time window.

    The simulated surge (2-hourly) is linearly interpolated onto the observed
    (hourly) residual times within the overlap, so this scores the whole storm
    time series, not just its peak. Returns NaNs if the overlap is too short or
    either series is flat (correlation undefined).
    """
    if sim.empty or obs.empty:
        return (np.nan, np.nan, 0)
    lo, hi = max(sim.index[0], obs.index[0]), min(sim.index[-1], obs.index[-1])
    o = obs.loc[lo:hi].dropna()
    if o.size < C.TS_MIN_OVERLAP:
        return (np.nan, np.nan, int(o.size))
    st = sim.index.astype("int64").to_numpy()
    si = np.interp(o.index.astype("int64").to_numpy(), st, sim.values)
    err = si - o.values
    rmse = float(np.sqrt((err ** 2).mean()))
    if np.std(si) == 0 or np.std(o.values) == 0:
        return (np.nan, rmse, int(o.size))
    return (float(np.corrcoef(o.values, si)[0, 1]), rmse, int(o.size))


# --------------------------------------------------------------------------- #
# Per-storm time-series cache. The de-tided (sim, obs) series are the slow part
# (utide harmonic fits per gauge), so we store one storm's whole ``series`` dict as
# a tidy long-format Parquet table (columns: gauge, kind in {sim,obs}, time, value;
# plus a ``tag`` column holding the parameter version). Parquet is typed,
# language-agnostic and preserves the datetime index exactly -- unlike a pickle it is
# not tied to the Python/pandas version and is safe to read. The ``tag`` is built
# from the node-selection + de-tiding parameters, so the cache self-invalidates if any
# change; ``refresh=True`` forces a recompute.
# --------------------------------------------------------------------------- #
def _ts_cache_tag() -> str:
    """Version tag from every parameter that affects the cached series.

    The leading ``v1`` is a manual schema/algorithm version: bump it if the de-tiding or
    node-selection *code* (not just these constants) changes, to invalidate stale caches.
    """
    return (f"v1|deg{C.MAX_NODE_DEG}|wet{C.WET_MIN_M}|knn{C.KNN}"
            f"|ut{C.UTIDE_MIN_SAMPLES}|box{C.GAUGE_BOX}")


def _series_cache_path(storm: str) -> str:
    return os.path.join(C.TS_CACHE, storm.replace(" ", "_") + ".parquet")


def _save_series_cache(storm: str, series: Dict[str, tuple]) -> None:
    try:
        frames = []
        for name, (sim, obs) in series.items():
            for kind, s in (("sim", sim), ("obs", obs)):
                frames.append(pd.DataFrame({
                    "gauge": name, "kind": kind,
                    "time": pd.DatetimeIndex(s.index),
                    "value": np.asarray(s.values, dtype=float)}))
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["gauge", "kind", "time", "value"]))
        df["tag"] = _ts_cache_tag()
        df.to_parquet(_series_cache_path(storm), index=False)
    except Exception as e:  # pragma: no cover - caching must never break the pipeline
        print(f"  (warning: could not cache {storm} series: {e})")


def _load_series_cache(storm: str) -> Optional[Dict[str, tuple]]:
    """Return the cached series dict, or ``None`` on miss / stale tag / unreadable."""
    path = _series_cache_path(storm)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty or df["tag"].iloc[0] != _ts_cache_tag():
            return None                       # parameters changed -> recompute
        series: Dict[str, tuple] = {}
        for name, grp in df.groupby("gauge", sort=False):
            grp = grp.sort_values("time")
            sim = grp.loc[grp.kind == "sim"].set_index("time")["value"]
            obs = grp.loc[grp.kind == "obs"].set_index("time")["value"]
            series[name] = (sim, obs)
        return series
    except Exception:
        return None                           # unreadable / schema drift -> recompute


def load_storm_series(storm: str, fname: str, gauges: List[Gauge],
                      refresh: bool = False) -> Dict[str, tuple]:
    """Cached accessor for one storm's ``{gauge: (sim, obs)}`` series.

    Reads the pickle when present and current; otherwise runs the full
    :func:`validate_storm` (which write-through populates the cache).
    """
    if not refresh:
        cached = _load_series_cache(storm)
        if cached is not None:
            return cached
    _, series = validate_storm(storm, fname, gauges)
    return series


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
        ts_r, ts_rmse, ts_n = timeseries_skill(sim, obs)
        rows.append(dict(
            storm=storm, sid=sid, name=name, setting=classify_setting(name),
            node_deg=round(dist, 3), sim_peak=round(float(sim.max()), 2),
            obs_peak=round(float(obs.max()), 2), peak_dt_hr=round(timing, 1),
            ts_r=round(ts_r, 3), ts_rmse=round(ts_rmse, 3), ts_n=ts_n,
            method=method, n_obs=int(obs.size),
        ))
    _save_series_cache(storm, series)         # write-through: a full sweep populates the cache
    return rows, series


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sid"] = df["sid"].astype(str)  # robust to CSV round-trip parsing sid as int
    df["failed"] = [(s, i) in C.KNOWN_FAILED for s, i in zip(df.storm, df.sid)]
    df["poor_event"] = df.storm.isin(C.POOR_SURGE_EVENTS)
    # ``valid``: a real, meaningful-surge gauge record (NOT conditioned on timing),
    # so timing skill can be reported without selecting on the very quantity scored.
    df["valid"] = (df.obs_peak >= C.MIN_OBS_PEAK_M) & ~df.failed
    # ``clean``: ``valid`` and the peak is simultaneous -- used for peak-skill scoring.
    df["clean"] = df.valid & (df.peak_dt_hr.abs() <= C.MAX_TIMING_HR)
    return df


def metrics(d: pd.DataFrame) -> Tuple[float, float, float]:
    """(bias, RMSE, r) of simulated vs observed peak surge."""
    if len(d) < 2:
        return (np.nan, np.nan, np.nan)
    err = d.sim_peak - d.obs_peak
    return (err.mean(), float(np.sqrt((err ** 2).mean())),
            float(np.corrcoef(d.obs_peak, d.sim_peak)[0, 1]))


def within_storm_r(d: pd.DataFrame) -> float:
    """Across-gauge correlation of peak surge after removing each storm's mean,
    i.e. the spatial skill with between-storm magnitude differences taken out
    (a stricter test than the pooled r, which the storm spread inflates)."""
    if d.storm.nunique() < 1 or len(d) < 3:
        return np.nan
    o = d.obs_peak - d.groupby("storm").obs_peak.transform("mean")
    s = d.sim_peak - d.groupby("storm").sim_peak.transform("mean")
    if o.std() == 0 or s.std() == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def bootstrap_ci(d: pd.DataFrame, n: int = C.N_BOOTSTRAP,
                 seed: int = C.BOOTSTRAP_SEED) -> Dict[str, Tuple[float, float]]:
    """5-95% percentile CIs for pooled (bias, RMSE, r) by resampling pairs."""
    rng = np.random.default_rng(seed)
    err = (d.sim_peak - d.obs_peak).to_numpy()
    obs, sim = d.obs_peak.to_numpy(), d.sim_peak.to_numpy()
    bias, rmse, r = [], [], []
    for _ in range(n):
        k = rng.integers(0, len(d), len(d))
        e = err[k]
        bias.append(e.mean())
        rmse.append(np.sqrt((e ** 2).mean()))
        o, s = obs[k], sim[k]
        r.append(np.corrcoef(o, s)[0, 1] if o.std() and s.std() else np.nan)
    pct = lambda a: (float(np.nanpercentile(a, 5)), float(np.nanpercentile(a, 95)))
    return {"bias": pct(bias), "rmse": pct(rmse), "r": pct(r)}


def report(df: pd.DataFrame) -> None:
    cln = df[df.clean]
    print("\n=== PER-STORM (clean pairs) ===")
    for storm in df.storm.unique():
        d = cln[cln.storm == storm]
        if len(d):
            b, e, r = metrics(d)
            tsr = d.ts_r.median()
            print(f"  {storm:14s} n={len(d):2d}  bias={b:+.2f}  RMSE={e:.2f}  "
                  f"r_peak={r:.2f}  r_series(med)={tsr:.2f}")
    b, e, r = metrics(cln)
    ci = bootstrap_ci(cln)
    print(f"\nOVERALL clean (n={len(cln)}):")
    print(f"  peak bias = {b:+.2f} m  [{ci['bias'][0]:+.2f}, {ci['bias'][1]:+.2f}]")
    print(f"  peak RMSE = {e:.2f} m  [{ci['rmse'][0]:.2f}, {ci['rmse'][1]:.2f}]")
    print(f"  peak r    = {r:.2f}    [{ci['r'][0]:.2f}, {ci['r'][1]:.2f}]")
    print(f"  within-storm spatial r = {within_storm_r(cln):.2f} "
          f"(between-storm magnitude removed)")
    print(f"  time-series r   median={cln.ts_r.median():.2f}  mean={cln.ts_r.mean():.2f}")
    print(f"  time-series RMSE median={cln.ts_rmse.median():.2f} m")
    # Timing reported over ``valid`` (meaningful surge, not gauge-failed) WITHOUT
    # the timing gate, so it does not select on the quantity being reported.
    val = df[df.valid]
    within = (val.peak_dt_hr.abs() <= C.MAX_TIMING_HR).mean()
    print(f"  median |peak timing| over valid pairs (n={len(val)}) = "
          f"{val.peak_dt_hr.abs().median():.1f} h; "
          f"{100 * within:.0f}% within {C.MAX_TIMING_HR:.0f} h")
    print("--- by setting (clean) ---")
    for setting in ("open-coast", "semi-enclosed"):
        d = cln[cln.setting == setting]
        b, e, r = metrics(d)
        print(f"  {setting:14s} n={len(d):3d}  bias={b:+.2f}  RMSE={e:.2f}  r={r:.2f}")
    print(f"de-tide methods: {dict(df.method.value_counts())}")


def _setup_plt():
    """Headless matplotlib with the repo's paper style (sithom STIXGeneral serif,
    Computer-Modern mathtext, STD colour cycle) applied, so comp figures match the
    rest of the paper rather than matplotlib defaults."""
    import matplotlib
    matplotlib.use("Agg")
    from sithom.plot import plot_defaults
    plot_defaults()
    import matplotlib.pyplot as plt
    return plt


def _savefig(fig, paths: List[str]) -> None:
    """Save one figure to several paths (e.g. a quick-look PNG and a paper PDF).
    Figures are sized via sithom.get_dim to the LaTeX text width, so they are
    included at width=\\linewidth with no rescaling (and thus no font-size drift)."""
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
        print(f"wrote {p}")


def scatter(df: pd.DataFrame, paths: List[str]) -> None:
    plt = _setup_plt()
    from sithom.plot import get_dim

    cln = df[df.clean]
    # sized for inclusion at 0.7\linewidth (see appendix) so fonts match body text
    fig, ax = plt.subplots(figsize=get_dim(fraction_of_line_width=0.7, ratio=1.0))
    storms = list(df.storm.unique())
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(storms)))
    for s, c in zip(storms, colors):
        sc = df[(df.storm == s) & df.clean]
        sx = df[(df.storm == s) & ~df.clean]
        ax.scatter(sc.obs_peak, sc.sim_peak, color=[c], label=s, s=22, zorder=3)
        ax.scatter(sx.obs_peak, sx.sim_peak, facecolors="none", edgecolors=[c],
                   alpha=0.35, s=18, zorder=2)
    m = max(df.obs_peak.max(), df.sim_peak.max()) * 1.1 + 0.3
    ax.plot([0, m], [0, m], "k--", alpha=0.5, lw=0.8)
    if len(cln) > 1:
        sl, ic = np.polyfit(cln.obs_peak, cln.sim_peak, 1)
        ax.plot([0, m], [ic, sl * m + ic], color="0.3", lw=1, label=f"Fit (slope {sl:.2f})")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed peak residual [m]")
    ax.set_ylabel("Simulated peak surge [m]")
    ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9,
              handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)
    ax.grid(alpha=0.3)
    _savefig(fig, paths)
    plt.close(fig)


def plot_examples(panels: List[Tuple[str, str]], paths: List[str],
                  ncol: int = 2, refresh: bool = False) -> None:
    """Plot simulated surge vs de-tided observed residual for chosen (storm, gauge).

    ``panels`` is a list of ``(storm, gauge_name)`` tuples. The per-storm series are
    loaded from the time-series cache (:func:`load_storm_series`), so regenerating this
    figure is instant once the cache exists; pass ``refresh=True`` to rebuild it.
    """
    plt = _setup_plt()
    import matplotlib.dates as mdates
    from sithom.plot import get_dim, OX_BLUE

    gauges = gulf_gauges()
    cache: Dict[str, Dict[str, tuple]] = {}
    nrow = int(np.ceil(len(panels) / ncol))
    # taller (ratio ~1) for the stacked rows; wider 2-col panels give the date axis room.
    fig, axes = plt.subplots(nrow, ncol, figsize=get_dim(ratio=0.95), squeeze=False)
    for i, (ax, (storm, gname)) in enumerate(zip(axes.ravel(), panels)):
        letter = f"({chr(97 + i)}) "                      # panel id in the (left) title -> no label clash
        if storm not in cache:
            cache[storm] = load_storm_series(storm, C.STORMS[storm], gauges, refresh=refresh)
        match = [k for k in cache[storm] if gname.lower() in k.lower()]
        if not match:
            ax.set_title(f"{letter}{gname} (no data)", fontsize=7, loc="left"); continue
        sim, obs = cache[storm][match[0]]
        tsr, _, _ = timeseries_skill(sim, obs)
        # contrasting pair (dark blue vs black was too similar): model in
        # orange, observations in black — a colour-blind-safe combination
        ax.plot(sim.index, sim.values, color="tab:orange", lw=1.3, label="ADCIRC surge")
        ax.plot(obs.index, obs.values, color="black", lw=0.9, label="NOAA residual")
        rtxt = "" if np.isnan(tsr) else f" ($r={tsr:.2f}$)"
        gauge = match[0][:24].rstrip(", ")            # trim long names without a dangling comma
        ax.set_title(f"{letter}{storm}: {gauge}{rtxt}", fontsize=7, loc="left")
        ax.set_ylabel("Surge [m]"); ax.grid(alpha=0.3)
        # few, short date ticks ("Aug 24") instead of ~10 crowded "2005-08-24" labels
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_fontsize(6); lab.set_ha("right")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=6, loc="upper left")
    _savefig(fig, paths)
    plt.close(fig)


def latex_table(df: pd.DataFrame, path: str) -> None:
    """Write the per-storm skill ``tabular`` that the appendix \\inputs.

    Emits only the ``tabular`` environment (the surrounding ``table`` float,
    caption and label live in ``paper/appendix.tex``), so the prose stays
    hand-edited while every number is generated -- they cannot drift apart.
    Storms in ``POOR_SURGE_EVENTS`` are flagged with a ``$^{\\dagger}$``.
    Columns: storm, n, peak bias, peak RMSE, peak r, median time-series r.
    """
    cln = df[df.clean]
    lines = [
        "% GENERATED by comp.validate.latex_table -- do not edit by hand.",
        r"\begin{tabular}{lrrrrr}",
        r"  \hline \hline",
        r"  \textbf{Storm} & \textbf{$n$} & \textbf{Bias (m)} & "
        r"\textbf{RMSE (m)} & \textbf{$r_{\mathrm{peak}}$} & "
        r"\textbf{$r_{\mathrm{series}}$} \\",
        r"  \hline",
    ]
    for storm in C.STORMS:                      # fixed (chronological) order
        d = cln[cln.storm == storm]
        if not len(d):
            continue
        b, e, r = metrics(d)
        tsr = d.ts_r.median()
        dag = r"$^{\dagger}$" if storm in C.POOR_SURGE_EVENTS else ""
        name = storm.replace(" ", " (", 1) + ")"     # "Katrina 2005" -> "Katrina (2005)"
        lines.append(f"  {name}{dag} & {len(d)} & ${b:+.2f}$ & {e:.2f} & "
                     f"{r:.2f} & {tsr:.2f} \\\\")
    b, e, r = metrics(cln)
    lines += [
        r"  \hline",
        rf"  \textbf{{All storms}} & \textbf{{{len(cln)}}} & "
        rf"$\mathbf{{{b:+.2f}}}$ & \textbf{{{e:.2f}}} & \textbf{{{r:.2f}}} & "
        rf"\textbf{{{cln.ts_r.median():.2f}}} \\",
        r"  \hline \hline",
        r"\end{tabular}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
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

    # Paper artifacts: quick-look PNGs under the module, final PDFs and the
    # generated LaTeX table straight to the thesis tree (see constants).
    scatter(df, [os.path.join(C.FIGURE_PATH, "val_scatter.png"),
                 os.path.join(C.PAPER_IMG_PATH, "comp_val_scatter.pdf")])
    # Only regenerate the example panels on a full sweep (they need the example
    # storms, which a --storms subset may not include).
    if storms is None:
        plot_examples(C.EXAMPLE_PANELS,
                      [os.path.join(C.FIGURE_PATH, "val_examples.png"),
                       os.path.join(C.PAPER_IMG_PATH, "comp_val_examples.pdf")])
        latex_table(df, os.path.join(C.PAPER_TEX_PATH, "comp_val_table.tex"))
    print(f"wrote {out_csv}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storms", nargs="*", default=None,
                    help="subset of storm names (default: all)")
    ap.add_argument("--examples-only", action="store_true",
                    help="regenerate only the example-panel figure from the cached "
                         "time series (no sweep) -- instant once the cache exists")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="force-recompute the cached time series instead of reading it")
    a = ap.parse_args()
    if a.examples_only:
        plot_examples(C.EXAMPLE_PANELS,
                      [os.path.join(C.FIGURE_PATH, "val_examples.png"),
                       os.path.join(C.PAPER_IMG_PATH, "comp_val_examples.pdf")],
                      refresh=a.refresh_cache)
        return
    run(a.storms)


if __name__ == "__main__":
    main()
