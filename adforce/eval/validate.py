"""Validate historical ADCIRC surge against de-tided NOAA gauges.

Pipeline, per storm:
  1. download the storm's netCDF from Hugging Face (``HF_REPO``);
  2. extract the simulated surge (SSH = WD + DEM) at the nearest *wet* mesh element
     centroid (the archived dual-graph node) to each NOAA CO-OPS gauge in the box;
  3. fetch + de-tide the gauge record (:func:`adforce.eval.coops.observed_residual`);
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

Run (hydra overrides; config root adforce/eval/config/eval_config.yaml)::

    python -m adforce.eval.validate                # full sweep, all STORMS (populates the cache)
    python -m adforce.eval.validate 'storms=["Ida 2021","Katrina 2005"]'
    python -m adforce.eval.validate validate.examples_only=true   # example figure, from cache (fast)
    python -m adforce.eval.validate validate.examples_only=true validate.refresh=true
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import constants as C
from .coops import Gauge, gulf_gauges, observed_residual

# Gauges whose name implies a semi-enclosed (bay / estuary / river / inlet)
# setting, where a medium-resolution mesh and the no-tide assumption are least
# reliable. Used only to report open-coast skill separately -- a transparent
# heuristic, not a hard exclusion.
SEMI_ENCLOSED_KW = (
    "bay",
    "river",
    "bayou",
    "lake",
    "canal",
    "lock",
    "bridge",
    "dock",
    "creek",
    "bank",
    "channel",
    "turning basin",
    "ship",
    "inner",
)


def classify_setting(name: str) -> str:
    n = name.lower()
    return "semi-enclosed" if any(k in n for k in SEMI_ENCLOSED_KW) else "open-coast"


def download_storm(fname: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=C.HF_REPO,
        repo_type="dataset",
        filename=fname,
        local_dir=C.HF_STORM_CACHE,
    )


def _nearest_wet(
    tree: cKDTree, WD: np.ndarray, lon: float, lat: float
) -> Tuple[Optional[int], Optional[float]]:
    """Nearest node to (lon,lat) that never dries below ``WET_MIN_M``."""
    d, idx = tree.query([lon, lat], k=C.KNN)
    for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if di > C.MAX_NODE_DEG:
            break
        if np.nanmin(WD[:, ii]) > C.WET_MIN_M:
            return int(ii), float(di)
    return None, None


def timeseries_skill(sim: pd.Series, obs: pd.Series) -> Tuple[float, float, int]:
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
    rmse = float(np.sqrt((err**2).mean()))
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
def _ts_cache_tag(storm: str) -> str:
    """Version tag from every parameter that affects the cached series.

    The leading ``v1`` is a manual schema/algorithm version: bump it if the de-tiding or
    node-selection *code* (not just these constants) changes, to invalidate stale caches.
    The gauge-selection box is per-storm (Gulf vs Florida, ``C.box_for``); for the
    original Gulf storms the tag string is byte-identical to the single-box era, so
    their caches stay valid.
    """
    return (
        f"v1|deg{C.MAX_NODE_DEG}|wet{C.WET_MIN_M}|knn{C.KNN}"
        f"|ut{C.UTIDE_MIN_SAMPLES}|box{C.box_for(storm)}"
    )


def _series_cache_path(storm: str) -> str:
    return os.path.join(C.TS_CACHE, storm.replace(" ", "_") + ".parquet")


def _save_series_cache(storm: str, series: Dict[str, tuple]) -> None:
    try:
        C.ensure_dirs()  # cache dirs are created lazily, not at import
        frames = []
        for name, (sim, obs) in series.items():
            for kind, s in (("sim", sim), ("obs", obs)):
                frames.append(
                    pd.DataFrame(
                        {
                            "gauge": name,
                            "kind": kind,
                            "time": pd.DatetimeIndex(s.index),
                            "value": np.asarray(s.values, dtype=float),
                        }
                    )
                )
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["gauge", "kind", "time", "value"])
        )
        df["tag"] = _ts_cache_tag(storm)
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
        if df.empty or df["tag"].iloc[0] != _ts_cache_tag(storm):
            return None  # parameters changed -> recompute
        series: Dict[str, tuple] = {}
        for name, grp in df.groupby("gauge", sort=False):
            grp = grp.sort_values("time")
            sim = grp.loc[grp.kind == "sim"].set_index("time")["value"]
            obs = grp.loc[grp.kind == "obs"].set_index("time")["value"]
            series[name] = (sim, obs)
        return series
    except Exception:
        return None  # unreadable / schema drift -> recompute


def load_storm_series(
    storm: str, fname: str, gauges: List[Gauge], refresh: bool = False
) -> Dict[str, tuple]:
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


def validate_storm(
    storm: str, fname: str, gauges: List[Gauge], source=None
) -> Tuple[List[dict], Dict[str, tuple]]:
    """Return (rows, series) for one storm. ``series[name] = (sim, obs)``.

    ``source`` is a :class:`adforce.eval.sources.FieldSource` supplying
    ``(x, y, wet, elev, t)``; the default HF-archive source reproduces the
    original inline extraction exactly (``wet = WD``, ``elev = WD + DEM``).
    The nearest-wet sampling and scoring below are source-agnostic.
    """
    year = int(storm.split()[-1])
    if source is None:
        from .sources import HFArchiveSource

        source = HFArchiveSource()
    x, y, wet, elev, t = source.load(storm, fname)
    s0, s1 = t[0], t[-1]
    tree = cKDTree(np.column_stack([x, y]))

    rows: List[dict] = []
    series: Dict[str, tuple] = {}
    for sid, name, lat, lon in gauges:
        idx, dist = _nearest_wet(tree, wet, lon, lat)
        if idx is None:
            continue
        obs, method = observed_residual(sid, lat, year, s0, s1)
        if obs.empty or obs.size < 12:
            continue
        sim = pd.Series(elev[:, idx], index=t)
        series[name] = (sim, obs)
        timing = (sim.idxmax() - obs.idxmax()).total_seconds() / 3600.0
        ts_r, ts_rmse, ts_n = timeseries_skill(sim, obs)
        rows.append(
            dict(
                storm=storm,
                sid=sid,
                name=name,
                setting=classify_setting(name),
                node_deg=round(dist, 3),
                sim_peak=round(float(sim.max()), 2),
                obs_peak=round(float(obs.max()), 2),
                peak_dt_hr=round(timing, 1),
                ts_r=round(ts_r, 3),
                ts_rmse=round(ts_rmse, 3),
                ts_n=ts_n,
                method=method,
                n_obs=int(obs.size),
            )
        )
    _save_series_cache(storm, series)  # write-through: a full sweep populates the cache
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
    return (
        err.mean(),
        float(np.sqrt((err**2).mean())),
        float(np.corrcoef(d.obs_peak, d.sim_peak)[0, 1]),
    )


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


def bootstrap_ci(
    d: pd.DataFrame, n: int = C.N_BOOTSTRAP, seed: int = C.BOOTSTRAP_SEED
) -> Dict[str, Tuple[float, float]]:
    """5-95% percentile CIs for pooled (bias, RMSE, r) by resampling pairs."""
    rng = np.random.default_rng(seed)
    err = (d.sim_peak - d.obs_peak).to_numpy()
    obs, sim = d.obs_peak.to_numpy(), d.sim_peak.to_numpy()
    bias, rmse, r = [], [], []
    for _ in range(n):
        k = rng.integers(0, len(d), len(d))
        e = err[k]
        bias.append(e.mean())
        rmse.append(np.sqrt((e**2).mean()))
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
            print(
                f"  {storm:14s} n={len(d):2d}  bias={b:+.2f}  RMSE={e:.2f}  "
                f"r_peak={r:.2f}  r_series(med)={tsr:.2f}"
            )
    b, e, r = metrics(cln)
    ci = bootstrap_ci(cln)
    print(f"\nOVERALL clean (n={len(cln)}):")
    print(f"  peak bias = {b:+.2f} m  [{ci['bias'][0]:+.2f}, {ci['bias'][1]:+.2f}]")
    print(f"  peak RMSE = {e:.2f} m  [{ci['rmse'][0]:.2f}, {ci['rmse'][1]:.2f}]")
    print(f"  peak r    = {r:.2f}    [{ci['r'][0]:.2f}, {ci['r'][1]:.2f}]")
    print(
        f"  within-storm spatial r = {within_storm_r(cln):.2f} "
        f"(between-storm magnitude removed)"
    )
    print(
        f"  time-series r   median={cln.ts_r.median():.2f}  mean={cln.ts_r.mean():.2f}"
    )
    print(f"  time-series RMSE median={cln.ts_rmse.median():.2f} m")
    # Timing reported over ``valid`` (meaningful surge, not gauge-failed) WITHOUT
    # the timing gate, so it does not select on the quantity being reported.
    val = df[df.valid]
    within = (val.peak_dt_hr.abs() <= C.MAX_TIMING_HR).mean()
    print(
        f"  median |peak timing| over valid pairs (n={len(val)}) = "
        f"{val.peak_dt_hr.abs().median():.1f} h; "
        f"{100 * within:.0f}% within {C.MAX_TIMING_HR:.0f} h"
    )
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
        ax.scatter(
            sx.obs_peak,
            sx.sim_peak,
            facecolors="none",
            edgecolors=[c],
            alpha=0.35,
            s=18,
            zorder=2,
        )
    m = max(df.obs_peak.max(), df.sim_peak.max()) * 1.1 + 0.3
    ax.plot([0, m], [0, m], "k--", alpha=0.5, lw=0.8)
    if len(cln) > 1:
        sl, ic = np.polyfit(cln.obs_peak, cln.sim_peak, 1)
        ax.plot(
            [0, m], [ic, sl * m + ic], color="0.3", lw=1, label=f"Fit (slope {sl:.2f})"
        )
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed peak residual [m]")
    ax.set_ylabel("Simulated peak surge [m]")
    ax.legend(
        fontsize=6,
        ncol=2,
        loc="upper left",
        framealpha=0.9,
        handletextpad=0.3,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    ax.grid(alpha=0.3)
    _savefig(fig, paths)
    plt.close(fig)


def plot_examples(
    panels: List[Tuple[str, str]],
    paths: List[str],
    ncol: int = 2,
    refresh: bool = False,
    extra_titles: Optional[Dict[Tuple[str, str], str]] = None,
) -> None:
    """Plot simulated surge vs de-tided observed residual for chosen (storm, gauge).

    ``panels`` is a list of ``(storm, gauge_name)`` tuples. The per-storm series are
    loaded from the time-series cache (:func:`load_storm_series`), so regenerating this
    figure is instant once the cache exists; pass ``refresh=True`` to rebuild it.
    """
    plt = _setup_plt()
    import matplotlib.dates as mdates
    from sithom.plot import get_dim, OX_BLUE

    cache: Dict[str, Dict[str, tuple]] = {}
    nrow = int(np.ceil(len(panels) / ncol))
    # taller (ratio ~1) for the stacked rows; wider 2-col panels give the date axis room.
    fig, axes = plt.subplots(nrow, ncol, figsize=get_dim(ratio=0.95), squeeze=False)
    extra_titles = extra_titles or {}
    for i, (ax, (storm, gname)) in enumerate(zip(axes.ravel(), panels)):
        letter = f"({chr(97 + i)}) "  # panel id in the (left) title -> no label clash
        if storm not in cache:
            cache[storm] = load_storm_series(
                storm,
                C.STORMS[storm],
                gulf_gauges(C.box_for(storm)),  # per-storm region (Gulf/Florida)
                refresh=refresh,
            )
        match = [k for k in cache[storm] if gname.lower() in k.lower()]
        if not match:
            ax.set_title(f"{letter}{gname} (no data)", fontsize=7, loc="left")
            continue
        sim, obs = cache[storm][match[0]]
        tsr, _, _ = timeseries_skill(sim, obs)
        # contrasting pair (dark blue vs black was too similar): model in
        # orange, observations in black — a colour-blind-safe combination
        ax.plot(sim.index, sim.values, color="tab:orange", lw=1.3, label="ADCIRC surge")
        ax.plot(obs.index, obs.values, color="black", lw=0.9, label="NOAA residual")
        rtxt = "" if np.isnan(tsr) else f" ($r={tsr:.2f}$)"
        rtxt += extra_titles.get((storm, gname), "")
        gauge = match[0][:24].rstrip(", ")  # trim long names without a dangling comma
        ax.set_title(f"{letter}{storm}: {gauge}{rtxt}", fontsize=7, loc="left")
        ax.set_ylabel("Surge [m]")
        ax.grid(alpha=0.3)
        # few, short date ticks ("Aug 24") instead of ~10 crowded "2005-08-24" labels
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_fontsize(6)
            lab.set_ha("right")
    for ax in axes.ravel()[len(panels) :]:
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
    for storm in C.STORMS:  # fixed (chronological) order
        d = cln[cln.storm == storm]
        if not len(d):
            continue
        b, e, r = metrics(d)
        tsr = d.ts_r.median()
        dag = r"$^{\dagger}$" if storm in C.POOR_SURGE_EVENTS else ""
        name = storm.replace(" ", " (", 1) + ")"  # "Katrina 2005" -> "Katrina (2005)"
        lines.append(
            f"  {name}{dag} & {len(d)} & ${b:+.2f}$ & {e:.2f} & "
            f"{r:.2f} & {tsr:.2f} \\\\"
        )
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
    C.ensure_dirs()
    items = {k: C.STORMS[k] for k in (storms or C.STORMS)}
    print(
        f"{len(gulf_gauges())} candidate gauges in Gulf box, "
        f"{len(gulf_gauges(C.FLORIDA_BOX))} in Florida box; {len(items)} storms"
    )
    rows: List[dict] = []
    for storm, fname in items.items():
        try:
            # per-storm gauge region: Gulf storms keep the original box (and
            # their cached series); Florida storms score the Miami-region coast
            r, _ = validate_storm(storm, fname, gulf_gauges(C.box_for(storm)))
        except Exception as e:  # pragma: no cover
            print(f"!! {storm}: {e}")
            continue
        rows += r
        print(f"  {storm:14s}: {len(r):2d} gauges with data")
    df = add_flags(pd.DataFrame(rows)).sort_values(
        ["storm", "clean", "obs_peak"], ascending=[True, False, False]
    )
    out_csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    df.to_csv(out_csv, index=False)
    report(df)

    # Paper artifacts: quick-look PNGs under the module, final PDFs and the
    # generated LaTeX table straight to the thesis tree (see constants).
    scatter(
        df,
        [
            os.path.join(C.FIGURE_PATH, "val_scatter.png"),
            os.path.join(C.PAPER_IMG_PATH, "comp_val_scatter.pdf"),
        ],
    )
    # Only regenerate the example panels on a full sweep (they need the example
    # storms, which a --storms subset may not include).
    if storms is None:
        plot_examples(
            C.EXAMPLE_PANELS,
            [
                os.path.join(C.FIGURE_PATH, "val_examples.png"),
                os.path.join(C.PAPER_IMG_PATH, "comp_val_examples.pdf"),
            ],
        )
        latex_table(df, os.path.join(C.PAPER_TEX_PATH, "comp_val_table.tex"))
    print(f"wrote {out_csv}")
    return df


CITY_POINTS = {
    # (lon, lat) of the three study cities (adforce.constants Points); each
    # gauge-storm pair is assigned to the nearest city for the failure panels.
    "new_orleans": (-90.0715, 29.9511),
    "galveston": (-94.7977, 29.3013),
    "miami": (-80.1918, 25.7617),
}

_MESH_CACHE: Dict[str, tuple] = {}


def _read_fort14(resolution: str = "mid"):
    """EC95d mesh from ``adforce/setup/fort.14.<res>`` as
    ``(lon, lat, depth, triangles)`` (depth positive down; 0-based tris)."""
    if resolution in _MESH_CACHE:
        return _MESH_CACHE[resolution]
    from adforce.constants import SETUP_PATH

    path = os.path.join(SETUP_PATH, f"fort.14.{resolution}")
    with open(path) as f:
        f.readline()  # description
        ne, npt = (int(v) for v in f.readline().split()[:2])
        nodes = np.loadtxt((next(f) for _ in range(npt)), usecols=(1, 2, 3))
        tris = (
            np.loadtxt((next(f) for _ in range(ne)), dtype=int, usecols=(2, 3, 4)) - 1
        )
    _MESH_CACHE[resolution] = (nodes[:, 0], nodes[:, 1], nodes[:, 2], tris)
    return _MESH_CACHE[resolution]


def _draw_mesh_bathymetry(ax, plt, transform=None, edges: bool = True):
    """Model bathymetry as the map base: tricontourf of fort.14 depth (Blues,
    shallow light -> deep dark), model land (depth <= 0) in gray, and faint
    element edges so mesh resolution is visible. Returns the contour set for
    a colorbar, or None when the mesh is unavailable."""
    try:
        lon, lat, depth, tris = _read_fort14()
    except Exception as e:  # pragma: no cover - deck availability
        print(f"(no fort.14 mesh layer: {e})")
        return None
    import matplotlib.tri as mtri

    kw = {"transform": transform} if transform is not None else {}
    tri = mtri.Triangulation(lon, lat, tris)
    ax.tricontourf(  # model land
        tri, depth, levels=[depth.min() - 1.0, 0.0], colors=["0.92"], zorder=0, **kw
    )
    levels = [0, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 4500]
    cs = ax.tricontourf(
        tri, depth, levels=levels, cmap="Blues", extend="max", zorder=0.4, alpha=0.85, **kw
    )
    if edges:
        ax.triplot(tri, color="0.5", lw=0.08, alpha=0.25, zorder=0.6, **kw)
    return cs


def plot_failures(n_panels: int = 6, refresh: bool = False) -> None:
    """Per-city worst-case example panels (robustness view).

    For each study city (New Orleans, Galveston, Miami) select the ``n_panels``
    *valid* gauge-storm pairs (meaningful observed surge, not a documented
    gauge failure -- but NOT restricted to the simultaneous-peak "clean"
    subset, so timing misses count) with the largest absolute difference
    between simulated and observed peak surge, and render their hydrographs
    with the standard example-panel plotter. Requires a completed sweep
    (``val_summary.csv`` + populated time-series cache).
    """
    csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"{csv} not found: run `python -m adforce.eval.validate` first")
    df = pd.read_csv(csv)
    df["sid"] = df["sid"].astype(str)
    # gauge coordinates over both region boxes
    coord = {
        str(sid): (lon, lat)
        for box in (C.GAUGE_BOX, C.FLORIDA_BOX)
        for sid, name, lat, lon in gulf_gauges(box)
    }
    df = df[df.sid.isin(coord)].copy()
    lons = df.sid.map(lambda s: coord[s][0])
    lats = df.sid.map(lambda s: coord[s][1])
    df["city"] = [
        min(
            CITY_POINTS,
            key=lambda c: (CITY_POINTS[c][0] - lo) ** 2 + (CITY_POINTS[c][1] - la) ** 2,
        )
        for lo, la in zip(lons, lats)
    ]
    df["peak_diff"] = (df.sim_peak - df.obs_peak).abs()
    for city in CITY_POINTS:
        # |peak_dt| guard: rank only same-event misses -- overlapping simulation
        # windows can otherwise pair one storm's simulated peak with ANOTHER
        # storm's observed peak (e.g. Gustav's surge inside Ike's window),
        # which is an artifact of the window, not a model failure. 48 h is
        # generous enough to keep genuine timing misses in.
        sub = (
            df[
                df.valid.astype(bool)
                & (df.city == city)
                & (df.peak_dt_hr.abs() <= 48.0)
            ]
            .sort_values("peak_diff", ascending=False)
            .head(n_panels)
        )
        if sub.empty:
            print(f"(no valid pairs for {city})")
            continue
        panels = list(zip(sub.storm, sub.name))
        extra = {
            (r.storm, r.name): f" $\\Delta$peak {r.sim_peak - r.obs_peak:+.2f} m"
            for r in sub.itertuples()
        }
        print(
            f"{city}: "
            + "; ".join(f"{s}/{g} ({e.strip()})" for (s, g), e in extra.items())
        )
        plot_examples(
            panels,
            [
                os.path.join(C.FIGURE_PATH, f"val_failures_{city}.png"),
                os.path.join(C.PAPER_IMG_PATH, f"comp_val_failures_{city}.pdf"),
            ],
            refresh=refresh,
            extra_titles=extra,
        )


def plot_city_key(
    n_panels: int = 4, refresh: bool = False, max_deg: float = 2.0
) -> None:
    """Per-city KEY-event panels: history vs the model at each study city.

    For New Orleans, Galveston and Miami, select the ``n_panels`` *valid*
    gauge-storm pairs with the largest OBSERVED peak surge within
    ``max_deg`` degrees of that city (one panel per distinct storm -- the
    region's defining historical events), and render observed de-tided
    residual vs simulated surge with the standard example-panel plotter.
    The headline "did the model capture this city's storm history" figure;
    the complement of :func:`plot_failures` (which ranks by mismatch).

    The radius matters: without it the nearest-city split assigns the whole
    Atlantic seaboard to Miami, so "Miami" panels showed Fernandina Beach
    (~500 km away). Note the panels are gauge-record-limited, not
    meteorology-limited: Katrina's extreme-surge gauges failed or predate
    the network, and Ida's two nearest gauges are documented KNOWN_FAILED
    instrument losses, so those storms cannot headline their own city.
    """
    csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"{csv} not found: run `python -m adforce.eval.validate` first")
    df = pd.read_csv(csv)
    df["sid"] = df["sid"].astype(str)
    coord = {
        str(sid): (lon, lat)
        for box in (C.GAUGE_BOX, C.FLORIDA_BOX)
        for sid, name, lat, lon in gulf_gauges(box)
    }
    df = df[df.sid.isin(coord)].copy()
    lons = df.sid.map(lambda s: coord[s][0])
    lats = df.sid.map(lambda s: coord[s][1])
    for city, (clo, cla) in CITY_POINTS.items():
        # gauges within max_deg of THIS city (cities may share none: the
        # nearest-city split of plot_failures would hand the whole Atlantic
        # coast to Miami); same |peak_dt| window-artifact guard as
        # plot_failures; one panel per storm (the max-surge gauge of each)
        # so the figure spans the region's distinct key events rather than
        # four gauges of one landfall
        dist = np.hypot(lons - clo, lats - cla)
        sub = (
            df[
                df.valid.astype(bool)
                & (dist <= max_deg)
                & (df.peak_dt_hr.abs() <= 48.0)
            ]
            .sort_values("obs_peak", ascending=False)
            .drop_duplicates("storm")
            .head(n_panels)
        )
        if sub.empty:
            print(f"(no valid pairs for {city})")
            continue
        panels = list(zip(sub.storm, sub.name))
        extra = {
            (r.storm, r.name): f" $\\Delta${r.sim_peak - r.obs_peak:+.2f} m"
            for r in sub.itertuples()
        }
        print(
            f"{city}: "
            + "; ".join(
                f"{s}/{g} (obs {o:.2f} m)"
                for (s, g), o in zip(panels, sub.obs_peak)
            )
        )
        plot_examples(
            panels,
            [
                os.path.join(C.FIGURE_PATH, f"val_city_key_{city}.png"),
                os.path.join(C.PAPER_IMG_PATH, f"comp_val_city_key_{city}.pdf"),
            ],
            refresh=refresh,
            extra_titles=extra,
        )


def plot_gauge_map(max_deg: float = 2.0) -> None:
    """Geographic overview of the validation gauge panel.

    One map: coastline, every CO-OPS gauge in the two selection boxes (open
    = in the panel; filled = contributes >= 1 *valid* pair to
    ``val_summary.csv``; red cross = documented KNOWN_FAILED instrument
    loss), the Gulf/Florida selection boxes, and the three study cities with
    the ``max_deg`` catchment circles used by :func:`plot_city_key`.
    Distances are Euclidean in degrees, matching the selection metric.
    """
    csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"{csv} not found: run `python -m adforce.eval.validate` first")
    df = pd.read_csv(csv)
    df["sid"] = df["sid"].astype(str)
    valid_sids = set(df[df.valid.astype(bool)].sid)
    failed_sids = {sid for _, sid in C.KNOWN_FAILED}

    plt = _setup_plt()
    from sithom.plot import get_dim

    try:  # coastline via cartopy when available; plain axes otherwise
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=get_dim(ratio=0.45))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="0.35", zorder=1
        )
        gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6}
        cs = _draw_mesh_bathymetry(ax, plt, transform=ccrs.PlateCarree(), edges=False)
    except Exception as e:  # pragma: no cover - cartopy/data availability
        print(f"(no cartopy coastline: {e})")
        fig, ax = plt.subplots(figsize=get_dim(ratio=0.45))
        ax.set_xlabel("Longitude [$^\\circ$E]")
        ax.set_ylabel("Latitude [$^\\circ$N]")
        ax.grid(alpha=0.3)
        cs = _draw_mesh_bathymetry(ax, plt, edges=False)

    for box, label, color in (
        (C.GAUGE_BOX, "Gulf box", "tab:blue"),
        (C.FLORIDA_BOX, "Florida box", "tab:green"),
    ):
        (lo0, lo1), (la0, la1) = box["lon"], box["lat"]
        ax.plot(
            [lo0, lo1, lo1, lo0, lo0],
            [la0, la0, la1, la1, la0],
            color=color,
            lw=0.8,
            ls=":",
            label=label,
            zorder=2,
        )

    seen = set()
    for box in (C.GAUGE_BOX, C.FLORIDA_BOX):
        for sid, name, lat, lon in gulf_gauges(box):
            sid = str(sid)
            if sid in seen:
                continue
            seen.add(sid)
            if sid in valid_sids:
                ax.plot(lon, lat, "o", ms=3.5, color="tab:orange", mec="k", mew=0.3, alpha=0.7, zorder=4)
            else:
                ax.plot(lon, lat, "o", ms=3, mfc="none", mec="0.5", mew=0.6, alpha=0.7, zorder=3)
            if sid in failed_sids:
                ax.plot(lon, lat, "x", ms=5, color="tab:red", mew=1.0, zorder=5)

    theta = np.linspace(0, 2 * np.pi, 100)
    for city, (clo, cla) in CITY_POINTS.items():
        ax.plot(clo, cla, "*", ms=11, color="k", mec="w", mew=0.5, zorder=6)
        ax.plot(
            clo + max_deg * np.cos(theta),
            cla + max_deg * np.sin(theta),
            color="k",
            lw=0.6,
            ls="--",
            alpha=0.6,
            zorder=2,
        )
        # per-city offsets keep labels off the gauge dots (over open water /
        # Lake Pontchartrain); New Orleans anchors right-aligned to its star
        xytext, ha = {
            "galveston": ((7, -12), "left"),
            "new_orleans": ((-7, 9), "right"),
            "miami": ((7, 6), "left"),
        }[city]
        ax.annotate(
            city.replace("_", " ").title(),
            (clo, cla),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=7,
        )

    # legend proxies (marker styles used above)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", ms=4, color="tab:orange", mec="k", mew=0.3,
               label="Gauge with valid pairs"),
        Line2D([], [], marker="o", ls="", ms=3.5, mfc="none", mec="0.5", label="Panel gauge (no valid pair)"),
        Line2D([], [], marker="x", ls="", ms=5, color="tab:red", label="Known instrument failure"),
        Line2D([], [], marker="*", ls="", ms=9, color="k", mec="w",
               label=f"Study city (r={max_deg:g}$^\\circ$)"),
    ]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=5.5, loc="lower left", framealpha=0.9)
    if hasattr(ax, "set_extent"):  # cartopy GeoAxes
        ax.set_extent([-98.5, -78.5, 23.5, 31.8])
    else:
        ax.set_xlim(-98.5, -78.5)
        ax.set_ylim(23.5, 31.8)
    if cs is not None:
        fig.colorbar(cs, ax=ax, shrink=0.75, pad=0.02, label="Model depth [m]")
    _savefig(
        fig,
        [
            os.path.join(C.FIGURE_PATH, "val_gauge_map.png"),
            os.path.join(C.PAPER_IMG_PATH, "comp_val_gauge_map.pdf"),
        ],
    )
    plt.close(fig)


def plot_city_gauge_maps(max_deg: float = 2.0) -> None:
    """Per-city zoomed gauge maps with every gauge NAMED (reference figures).

    One map per study city, extent = catchment circle + margin, each gauge
    labelled with its CO-OPS name (orange = contributes valid pairs, open =
    panel gauge without one, red cross = KNOWN_FAILED). Labels use a simple
    greedy vertical stagger so dense clusters (e.g. the Mississippi coast)
    stay legible.
    """
    csv = os.path.join(C.OUT_PATH, "val_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"{csv} not found: run `python -m adforce.eval.validate` first")
    df = pd.read_csv(csv)
    df["sid"] = df["sid"].astype(str)
    valid_sids = set(df[df.valid.astype(bool)].sid)
    failed_sids = {sid for _, sid in C.KNOWN_FAILED}
    gauges = {
        str(sid): (name, lat, lon)
        for box in (C.GAUGE_BOX, C.FLORIDA_BOX)
        for sid, name, lat, lon in gulf_gauges(box)
    }

    plt = _setup_plt()
    from sithom.plot import get_dim

    for city, (clo, cla) in CITY_POINTS.items():
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            fig = plt.figure(figsize=get_dim(ratio=0.85))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(
                cfeature.COASTLINE.with_scale("10m"), lw=0.5, edgecolor="0.35", zorder=1
            )
            gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.4)
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 6}
            cs = _draw_mesh_bathymetry(ax, plt, transform=ccrs.PlateCarree(), edges=True)
        except Exception as e:  # pragma: no cover
            print(f"(no cartopy coastline: {e})")
            fig, ax = plt.subplots(figsize=get_dim(ratio=0.85))
            ax.grid(alpha=0.3)
            cs = _draw_mesh_bathymetry(ax, plt, edges=True)

        m = max_deg + 0.45
        ax.plot(clo, cla, "*", ms=13, color="k", mec="w", mew=0.5, zorder=6)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            clo + max_deg * np.cos(theta),
            cla + max_deg * np.sin(theta),
            "k--",
            lw=0.6,
            alpha=0.6,
            zorder=2,
        )

        # gauges in extent, sorted by latitude for the label stagger
        local = sorted(
            (
                (sid, n, la, lo)
                for sid, (n, la, lo) in gauges.items()
                if abs(lo - clo) <= m and abs(la - cla) <= m
            ),
            key=lambda t: -t[2],
        )
        # cluster-indexed offset ladder: the k-th label inside a congested
        # patch gets the k-th vertical offset, so e.g. the six Mobile-Bay
        # gauges fan out instead of overprinting
        DY = (3, 12, -11, 21, -20, 30, -29, 39)
        placed = []
        for sid, name, la, lo in local:
            if sid in valid_sids:
                ax.plot(lo, la, "o", ms=4, color="tab:orange", mec="k", mew=0.3, alpha=0.7, zorder=4)
            else:
                ax.plot(lo, la, "o", ms=3.5, mfc="none", mec="0.5", mew=0.6, alpha=0.7, zorder=3)
            if sid in failed_sids:
                ax.plot(lo, la, "x", ms=6, color="tab:red", mew=1.0, zorder=5)
            near = sum(
                1 for pla, plo in placed if abs(pla - la) < 0.16 and abs(plo - lo) < 1.6
            )
            dy = DY[min(near, len(DY) - 1)]
            ha = "right" if lo > clo + m - 0.75 else "left"  # keep inside the frame
            placed.append((la, lo))
            ax.annotate(
                name[:28].rstrip(", "),
                (lo, la),
                textcoords="offset points",
                xytext=(-5 if ha == "right" else 5, dy),
                ha=ha,
                fontsize=5,
                zorder=7,
            )
        ax.set_title(
            f"{city.replace('_', ' ').title()} gauge panel "
            f"(r={max_deg:g}$^\\circ$; filled = valid pairs)",
            fontsize=8,
        )
        if hasattr(ax, "set_extent"):
            ax.set_extent([clo - m, clo + m, cla - m, cla + m])
        else:
            ax.set_xlim(clo - m, clo + m)
            ax.set_ylim(cla - m, cla + m)
        if cs is not None:
            fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02, label="Model depth [m]")
        _savefig(
            fig,
            [
                os.path.join(C.FIGURE_PATH, f"val_gauge_map_{city}.png"),
                os.path.join(C.PAPER_IMG_PATH, f"comp_val_gauge_map_{city}.pdf"),
            ],
        )
        plt.close(fig)


_LEGACY_FLAGS = {
    "--storms": "'storms=[\"Ida 2021\"]'",
    "--examples-only": "validate.examples_only=true",
    "--refresh-cache": "validate.refresh=true",
    "--failures": "validate.failures=true",
    "--n-failures": "validate.n_failures=6",
}


@hydra.main(version_base=None, config_path="config", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()
    v = cfg.validate
    if v.gauge_map:
        plot_gauge_map(max_deg=v.city_radius_deg)
        return
    if v.city_maps:
        plot_city_gauge_maps(max_deg=v.city_radius_deg)
        return
    if v.city_key:
        plot_city_key(
            n_panels=v.n_city, refresh=v.refresh, max_deg=v.city_radius_deg
        )
        return
    if v.failures:
        plot_failures(n_panels=v.n_failures, refresh=v.refresh)
        return
    if v.examples_only:
        plot_examples(
            C.EXAMPLE_PANELS,
            [
                os.path.join(C.FIGURE_PATH, "val_examples.png"),
                os.path.join(C.PAPER_IMG_PATH, "comp_val_examples.pdf"),
            ],
            refresh=v.refresh,
        )
        return
    run(list(cfg.storms) if cfg.storms else None)


if __name__ == "__main__":
    from ._cli import reject_legacy_flags

    reject_legacy_flags(_LEGACY_FLAGS, "adforce.eval.validate")
    main()
