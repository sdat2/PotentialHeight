"""Robustness of the surge-validation result to the *de-tiding* choice.

Re-detides each clean gauge-storm pair's raw CO-OPS water level with several alternative
methods and recomputes the pooled peak-surge skill (r, bias, RMSE) on the SAME 152 clean
pairs, to show whether the headline (peak r ~ 0.89, bias ~ -0.30 m) depends on how we remove
the tide. The simulated peak does not depend on de-tiding, so it is taken from the cached
sweep (``val_summary.csv``); only the observed peak is recomputed per method. The pair set is
held fixed at the baseline ``clean`` pairs so the comparison isolates the de-tiding effect.

Reuses the cached raw water level (CO-OPS cache) and storm netCDFs (HF cache), so it is fast
after a sweep -- except the utide variants, which re-solve a year-long harmonic fit per pair.

A full run also writes the appendix table ``<thesis>/paper/comp_detide_table.tex`` (the
``tabular`` that ``tab:detide-robustness`` ``\\input``s), so the paper table regenerates with
the data and cannot drift -- like the per-storm table from :func:`comp.validate.latex_table`.

Run::

    python -m comp.detide_sensitivity                 # all methods + skew, all pairs, writes the table
    python -m comp.detide_sensitivity --methods godin_lowpass noaa_predictions
    python -m comp.detide_sensitivity --limit 40      # quick subset (no table written)
    python -m comp.detide_sensitivity --skew          # instantaneous vs skew-surge metric only
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from . import constants as C
from .coops import _coops, fetch_year, gulf_gauges
from .validate import download_storm


# --------------------------------------------------------------------------- #
# Alternative de-tiding methods. Each maps the raw year-long water level (plus the
# station / latitude / storm window it may need) to a residual Series; the storm
# peak is then ``residual.loc[s0:s1].max()``. The first is the production baseline.
# --------------------------------------------------------------------------- #
def _utide_resid(wl: pd.Series, lat: float, method: str = "robust",
                 trend: bool = True) -> pd.Series:
    import utide
    # DatetimeIndex passed directly: date2num floats break utide 0.3.1's
    # astronomical arguments (zero constituents selected) — see comp/coops.py
    coef = utide.solve(wl.index, wl.values, lat=lat, method=method, trend=trend,
                       conf_int="none", verbose=False)
    tide = utide.reconstruct(wl.index, coef, verbose=False).h
    return pd.Series(wl.values - tide, index=wl.index)


def _lowpass_resid(wl: pd.Series, kind: str = "godin") -> pd.Series:
    """Residual = water level minus a low-pass (tide) estimate.

    ``godin``: the classic A24*A24*A25 cascade of moving averages (~removes diurnal +
    semidiurnal tides, ~no phase shift). ``doodson``: the 39-point Doodson X0 filter.
    A low-pass tide estimate leaves the full-band surge in the residual, but a wide filter
    can attenuate a sharp surge peak -- which is exactly what we test.
    """
    x = wl.interpolate(limit=3)
    if kind == "godin":
        a = x
        for w in (24, 24, 25):
            a = a.rolling(w, center=True, min_periods=w - 6).mean()
        return (wl - a).dropna()
    if kind == "doodson":
        # Doodson X0 symmetric 39-term hourly filter (weights/30, zero where unlisted).
        w = np.zeros(39)
        for i, v in [(0, 2), (1, 1), (2, 1), (3, 2), (5, 2), (6, 1), (7, 1), (8, 2),
                     (11, 1), (12, 1), (13, 2), (15, 2), (16, 1), (17, 1), (18, 2),
                     (19, 0), (20, 1), (21, 1), (22, 2), (24, 2), (25, 1), (26, 1),
                     (28, 2), (29, 0), (30, 1), (31, 1), (32, 1), (34, 2), (35, 1),
                     (37, 1)]:
            w[i] = v
        w = np.concatenate([w[:0:-1], w]) / 30.0     # symmetric, sum -> 1
        a = np.convolve(x.values, w, mode="same")
        return pd.Series(wl.values - a, index=wl.index).iloc[len(w)//2:-(len(w)//2)].dropna()
    raise ValueError(kind)


def _noaa_pred_resid(station: str, window: Tuple) -> pd.Series:
    """Verified water level minus the CO-OPS published astronomical prediction over the
    storm window (the residual NOAA itself reports as 'observed surge')."""
    b = pd.Timestamp(window[0]).strftime("%Y%m%d")
    e = pd.Timestamp(window[1]).strftime("%Y%m%d")
    wl = _coops(station, b, e, "hourly_height")
    if wl.empty:
        wl = _coops(station, b, e, "water_level")
    pred = _coops(station, b, e, "predictions")
    if wl.empty or pred.empty:
        return pd.Series(dtype=float)
    j = pd.concat({"wl": wl, "pred": pred}, axis=1).dropna()
    return j["wl"] - j["pred"]


# method-id -> callable(wl, lat, station, window) -> residual Series
METHODS: Dict[str, callable] = {
    "utide_robust_year": lambda wl, lat, st, win: _utide_resid(wl, lat, "robust", True),   # baseline
    "utide_ols_year":    lambda wl, lat, st, win: _utide_resid(wl, lat, "ols", True),
    "utide_notrend":     lambda wl, lat, st, win: _utide_resid(wl, lat, "robust", False),
    "godin_lowpass":     lambda wl, lat, st, win: _lowpass_resid(wl, "godin"),
    "doodson_lowpass":   lambda wl, lat, st, win: _lowpass_resid(wl, "doodson"),
    "noaa_predictions":  lambda wl, lat, st, win: _noaa_pred_resid(st, win),
}
BASELINE = "utide_robust_year"


def _utide_tide(wl: pd.Series, lat: float) -> pd.Series:
    """Predicted astronomical tide (utide robust year fit) on the water-level index."""
    import utide
    # DatetimeIndex passed directly: date2num floats break utide 0.3.1's
    # astronomical arguments (zero constituents selected) — see comp/coops.py
    coef = utide.solve(wl.index, wl.values, lat=lat, method="robust", trend=True,
                       conf_int="none", verbose=False)
    return pd.Series(utide.reconstruct(wl.index, coef, verbose=False).h, index=wl.index)


def _skew_surge_peak(wl: pd.Series, tide: pd.Series, win: Tuple) -> float:
    """Storm skew-surge peak = max over tidal cycles of (max observed - max predicted tide).

    The skew surge (Horsburgh & Wilson 2007) compares the peak observed level in each tidal
    cycle to the peak predicted tide in the same cycle, so it is insensitive to tidal-phase
    error -- the defensible peak target for a tide-excluding model. Cycles are split at the
    predicted-tide low waters.
    """
    from scipy.signal import find_peaks
    o = wl.loc[win[0]:win[1]]
    td = tide.loc[win[0]:win[1]]
    if len(o) < 12:
        return np.nan
    troughs, _ = find_peaks(-td.values, distance=8)        # tide low waters, >= 8 h apart
    bounds = np.r_[0, troughs, len(td) - 1]
    skews = [float(o.values[a:b].max() - td.values[a:b].max())
             for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 4]
    return max(skews) if skews else np.nan


def _storm_windows() -> Dict[str, Tuple]:
    """storm -> (start, end) timestamps from the storm netCDF time axis (cached)."""
    win = {}
    for storm, fname in C.STORMS.items():
        try:
            ds = xr.open_dataset(download_storm(fname))
            t = pd.to_datetime(ds.time.values)
            win[storm] = (t[0], t[-1])
        except Exception as e:
            print(f"  (window load failed for {storm}: {e})")
    return win


def _pooled(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    err = sim - obs
    return dict(n=int(m.sum()), bias=float(err.mean()),
                rmse=float(np.sqrt((err ** 2).mean())),
                r=float(np.corrcoef(obs, sim)[0, 1]) if m.sum() > 1 else np.nan)


# display labels + row order for the generated appendix table (tab:detide-robustness)
TABLE_LABELS = {
    "utide_robust_year": "UTide robust, annual fit (baseline)",
    "utide_ols_year": "UTide ordinary least squares",
    "utide_notrend": "UTide, no linear trend",
    "noaa_predictions": "NOAA published predictions",
    "skew_surge": "skew surge (UTide tide)",
    "godin_lowpass": "Godin $24$--$24$--$25$ low-pass",
    "doodson_lowpass": "Doodson low-pass",
}
TABLE_ORDER = ["utide_robust_year", "utide_ols_year", "utide_notrend",
               "noaa_predictions", "skew_surge", "godin_lowpass", "doodson_lowpass"]


def _latex_table(pooled: Dict[str, dict], path: str) -> None:
    """Emit the de-tiding-robustness ``tabular`` the appendix \\inputs.

    Only the ``tabular`` (the ``table`` float, caption and label live in
    ``paper/appendix.tex``), matching :func:`comp.validate.latex_table`, so the prose stays
    hand-edited while every number is generated -- the table cannot drift from the data.
    """
    lines = [
        "% GENERATED by comp.detide_sensitivity -- do not edit by hand.",
        r"\begin{tabular}{lcc}",
        r"  \hline",
        r"  de-tiding method & $r$ & bias [m] \\",
        r"  \hline",
    ]
    for mid in TABLE_ORDER:
        p = pooled.get(mid)
        if p:                                    # r to 3 dp: the harmonic deltas are sub-0.01
            lines.append(f"  {TABLE_LABELS[mid]} & ${p['r']:.3f}$ & ${p['bias']:+.2f}$ \\\\")
    lines += [r"  \hline", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def run(methods=None, limit: Optional[int] = None, write_table: bool = True) -> pd.DataFrame:
    methods = methods or list(METHODS)
    if BASELINE not in methods:                  # always present, for the delta columns
        methods = [BASELINE] + list(methods)
    df = pd.read_csv(os.path.join(C.OUT_PATH, "val_summary.csv"))
    df["sid"] = df["sid"].astype(str)
    clean = df[df["clean"].astype(bool)].copy()
    if limit:
        clean = clean.head(limit)
    lat_of = {str(sid): lat for sid, name, lat, lon in gulf_gauges()}
    windows = _storm_windows()

    # recompute the observed peak per (pair, method); the skew surge reuses the robust fit
    recs = []
    for _, row in clean.iterrows():
        storm, sid = row["storm"], str(row["sid"])
        win = windows.get(storm)
        if win is None or sid not in lat_of:
            continue
        wl = fetch_year(sid, int(storm.split()[-1]))
        rec = {"storm": storm, "sid": sid, "sim_peak": row["sim_peak"]}
        try:                                     # one robust UTide fit -> baseline peak + skew
            tide = _utide_tide(wl, lat_of[sid])
            rec[BASELINE] = float((wl - tide).loc[win[0]:win[1]].max())
            rec["skew_surge"] = _skew_surge_peak(wl, tide, win)
        except Exception as e:
            print(f"  robust UTide failed for {storm}/{sid}: {e}")
            rec[BASELINE] = rec["skew_surge"] = np.nan
        for mname in methods:
            if mname == BASELINE:
                continue                         # already computed from the shared fit
            try:
                resid = METHODS[mname](wl, lat_of[sid], sid, win)
                seg = resid.loc[win[0]:win[1]] if not resid.empty else resid
                rec[mname] = float(seg.max()) if seg.size else np.nan
            except Exception as e:
                print(f"  {mname} failed for {storm}/{sid}: {e}")
                rec[mname] = np.nan
        recs.append(rec)
    out = pd.DataFrame(recs)

    cols = list(dict.fromkeys(methods + ["skew_surge"]))   # methods then skew, no duplicates
    print(f"\nDe-tiding robustness on {len(out)} clean pairs "
          f"(simulated peak fixed; observed peak re-detided):\n")
    print(f"{'method':22s} {'n':>4} {'r':>6} {'bias':>7} {'rmse':>6}  {'dr':>6} {'dbias':>7}")
    base = _pooled(out[BASELINE].to_numpy(), out["sim_peak"].to_numpy())
    pooled: Dict[str, dict] = {}
    for mname in cols:
        if mname not in out:
            continue
        p = _pooled(out[mname].to_numpy(), out["sim_peak"].to_numpy())
        pooled[mname] = p
        tag = "  <- baseline" if mname == BASELINE else ""
        print(f"{mname:22s} {p['n']:>4} {p['r']:>6.3f} {p['bias']:>+7.3f} {p['rmse']:>6.3f}  "
              f"{p['r']-base['r']:>+6.3f} {p['bias']-base['bias']:>+7.3f}{tag}")
    pd.DataFrame([{"method": m, **pooled[m]} for m in pooled]).to_csv(
        os.path.join(C.OUT_PATH, "detide_sensitivity.csv"), index=False)
    if write_table and not limit:                # write the paper table only from a full run
        _latex_table(pooled, os.path.join(C.PAPER_TEX_PATH, "comp_detide_table.tex"))
    return pd.DataFrame([{"method": m, **pooled[m]} for m in pooled])


def run_skew(limit: Optional[int] = None) -> pd.DataFrame:
    """Compare the instantaneous-residual peak (baseline metric) with the phase-insensitive
    skew-surge peak, both against the simulated peak, on the same clean pairs."""
    df = pd.read_csv(os.path.join(C.OUT_PATH, "val_summary.csv"))
    df["sid"] = df["sid"].astype(str)
    clean = df[df["clean"].astype(bool)].copy()
    if limit:
        clean = clean.head(limit)
    lat_of = {str(sid): lat for sid, name, lat, lon in gulf_gauges()}
    windows = _storm_windows()
    rows = []
    for _, row in clean.iterrows():
        storm, sid = row["storm"], str(row["sid"])
        win = windows.get(storm)
        if win is None or sid not in lat_of:
            continue
        wl = fetch_year(sid, int(storm.split()[-1]))
        try:
            tide = _utide_tide(wl, lat_of[sid])
            inst = float((wl - tide).loc[win[0]:win[1]].max())
            skew = _skew_surge_peak(wl, tide, win)
        except Exception as e:
            print(f"  skew failed for {storm}/{sid}: {e}")
            continue
        rows.append({"sim_peak": row["sim_peak"], "instantaneous": inst, "skew_surge": skew})
    out = pd.DataFrame(rows)
    print(f"\nSkew-surge vs instantaneous-residual peak on {len(out)} clean pairs:\n")
    print(f"{'metric':22s} {'n':>4} {'r':>6} {'bias':>7} {'rmse':>6}")
    for col in ("instantaneous", "skew_surge"):
        p = _pooled(out[col].to_numpy(), out["sim_peak"].to_numpy())
        print(f"{col:22s} {p['n']:>4} {p['r']:>6.3f} {p['bias']:>+7.3f} {p['rmse']:>6.3f}")
    return out


def write_table_from_csv(path: Optional[str] = None) -> None:
    """Regenerate the appendix tabular from the cached ``detide_sensitivity.csv`` -- instant,
    no re-detiding -- so the table can be restyled/reformatted without the 15-min recompute."""
    res = pd.read_csv(os.path.join(C.OUT_PATH, "detide_sensitivity.csv"))
    pooled = {row["method"]: {"r": float(row["r"]), "bias": float(row["bias"])}
              for _, row in res.iterrows()}
    _latex_table(pooled, path or os.path.join(C.PAPER_TEX_PATH, "comp_detide_table.tex"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of method ids")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clean pairs (quick)")
    ap.add_argument("--skew", action="store_true",
                    help="instantaneous-residual vs skew-surge peak metric (phase-insensitive)")
    ap.add_argument("--table-only", action="store_true",
                    help="regenerate the appendix table from the cached CSV (no re-detiding)")
    a = ap.parse_args()
    if a.table_only:
        write_table_from_csv()
    elif a.skew:
        run_skew(a.limit)
    else:
        run(a.methods, a.limit)


if __name__ == "__main__":
    main()
