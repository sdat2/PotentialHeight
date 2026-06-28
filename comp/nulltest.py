"""Negative-control / falsification tests for the tide-gauge validation.

The headline skill (clean-pair peak r = 0.89) is only meaningful if it is *specific*
to the right storm, the right gauge, and the right time. This module runs the
placebo tests that try to destroy the signal; if the real score sits far outside the
null distributions, the skill is a real physical match and not an artefact of
autocorrelation, both-signals-being-positive, or threshold tuning.

Tests (peak-level ones need only ``val_summary.csv``; the lag test needs the cached
storm netCDFs):

  1. global label permutation   -- break the sim<->obs pairing entirely; null r ~ 0.
  2. within-storm permutation    -- shuffle only *which gauge* matches which, keeping
                                    each storm's magnitude; isolates spatial skill.
  3. cross-storm same-gauge      -- sim of storm A vs obs of storm B at the SAME gauge;
                                    measures leftover gauge climatology (some gauges are
                                    just surgier), which the true score must beat.
  4. temporal-lag curve          -- shift the simulated hydrograph by +/- days and
                                    recompute the time-series correlation; real skill
                                    peaks sharply at lag 0.

Run::

    python -m comp.nulltest                 # peak-level nulls + lag curve (+ figure)
    python -m comp.nulltest --no-lag        # skip the netCDF-heavy lag test
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import constants as C
from .validate import metrics, timeseries_skill, validate_storm, within_storm_r
from .coops import gulf_gauges

warnings.filterwarnings("ignore")
RNG_SEED = 0


def _r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(C.OUT_PATH, "val_summary.csv"))
    df["sid"] = df["sid"].astype(str)
    return df[df.clean].copy().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. global label permutation
# --------------------------------------------------------------------------- #
def perm_global(c: pd.DataFrame, n: int = 5000, seed: int = RNG_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    obs, sim = c.obs_peak.values, c.sim_peak.values
    obs_r = _r(obs, sim)
    null = np.array([_r(obs, rng.permutation(sim)) for _ in range(n)])
    return dict(observed=obs_r, null_mean=float(null.mean()), null_sd=float(null.std()),
                null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                p=float((null >= obs_r).mean()), null=null)


# --------------------------------------------------------------------------- #
# 2. within-storm permutation (preserve per-storm magnitude, shuffle gauge id)
# --------------------------------------------------------------------------- #
def perm_within_storm(c: pd.DataFrame, n: int = 5000, seed: int = RNG_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    obs, sim = c.obs_peak.values, c.sim_peak.values
    obs_r = _r(obs, sim)                       # pooled, compared to within-shuffled null
    codes = c.storm.astype("category").cat.codes.values
    groups = [np.where(codes == g)[0] for g in np.unique(codes)]
    null = np.empty(n)
    for i in range(n):
        sp = sim.copy()
        for idx in groups:
            sp[idx] = rng.permutation(sp[idx])
        null[i] = _r(obs, sp)
    return dict(observed_pooled=obs_r, observed_spatial=within_storm_r(c),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                p=float((null >= obs_r).mean()), null=null)


# --------------------------------------------------------------------------- #
# 3. cross-storm same-gauge pairing
# --------------------------------------------------------------------------- #
def cross_storm(c: pd.DataFrame, seed: int = RNG_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    by_sid = {k: v for k, v in c.groupby("sid")}
    sim, obs = [], []
    for _, row in c.iterrows():
        alt = by_sid[row.sid]
        alt = alt[alt.storm != row.storm]
        if len(alt):
            sim.append(row.sim_peak)
            obs.append(alt.obs_peak.values[rng.integers(len(alt))])
    return dict(n=len(sim), r=_r(sim, obs), observed=_r(c.obs_peak, c.sim_peak))


# --------------------------------------------------------------------------- #
# 4. temporal-lag curve (needs cached storm netCDFs)
# --------------------------------------------------------------------------- #
def _clean_keys(c: pd.DataFrame) -> Dict[str, List[str]]:
    """storm -> list of clean gauge names, from the summary."""
    out: Dict[str, List[str]] = {}
    for storm, d in c.groupby("storm"):
        out[storm] = list(d.name)
    return out


def lag_curve(storms: Optional[List[str]] = None,
              lags_days: Tuple[float, ...] = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3),
              ) -> pd.DataFrame:
    """Median time-series correlation over clean pairs as the simulated hydrograph
    is shifted by each lag (days). Real surge skill peaks sharply at lag 0."""
    c = load_clean()
    keys = _clean_keys(c)
    items = {k: C.STORMS[k] for k in (storms or C.STORMS) if k in keys}
    gauges = gulf_gauges()
    per_lag: Dict[float, List[float]] = {L: [] for L in lags_days}
    for storm, fname in items.items():
        try:
            _, series = validate_storm(storm, fname, gauges)
        except Exception as e:  # pragma: no cover
            print(f"!! {storm}: {e}")
            continue
        wanted = set(keys[storm])
        n_used = 0
        for name, (sim, obs) in series.items():
            if name not in wanted:
                continue
            n_used += 1
            for L in lags_days:
                shifted = pd.Series(sim.values, index=sim.index + pd.Timedelta(days=L))
                r, _, _ = timeseries_skill(shifted, obs)
                if np.isfinite(r):
                    per_lag[L].append(r)
        print(f"  {storm:14s}: {n_used} clean gauges")
    rows = [dict(lag_days=L, median_ts_r=float(np.nanmedian(per_lag[L])),
                 mean_ts_r=float(np.nanmean(per_lag[L])), n=len(per_lag[L]))
            for L in lags_days]
    return pd.DataFrame(rows).sort_values("lag_days").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# report + figure
# --------------------------------------------------------------------------- #
def plot(g: Dict, w: Dict, lag: Optional[pd.DataFrame], paths: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from sithom.plot import plot_defaults, get_dim, label_subplots, BRICK_RED, OX_BLUE
    plot_defaults()
    import matplotlib.pyplot as plt

    ncol = 3 if lag is not None else 2
    fig, ax = plt.subplots(1, ncol, figsize=get_dim(ratio=0.33))
    for a, res, title in [(ax[0], g, "Global permutation"),
                          (ax[1], w, "Within-storm permutation")]:
        a.hist(res["null"], bins=40, color="0.75", edgecolor="none")
        obs = res.get("observed", res.get("observed_pooled"))
        a.axvline(obs, color=BRICK_RED, lw=1.8, label=f"observed $r={obs:.2f}$")
        a.set_title(f"{title}\n(null max ${res['null_max']:.2f}$, $p<10^{{-3}}$)", fontsize=7)
        a.set_xlabel("correlation $r$"); a.legend(fontsize=6)
    ax[0].set_ylabel("count")
    if lag is not None:
        a = ax[2]
        a.plot(lag.lag_days, lag.median_ts_r, "o-", color=OX_BLUE, ms=4)
        a.axvline(0, color="0.5", ls=":")
        a.set_title("Temporal-lag null", fontsize=7)
        a.set_xlabel("simulated-surge lag [days]"); a.set_ylabel("median time-series $r$")
        a.grid(alpha=0.3)
    label_subplots(ax.ravel().tolist()[:ncol], override="outside", fontsize=8)
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
        print(f"wrote {p}")


def run(do_lag: bool = True) -> None:
    c = load_clean()
    print(f"clean pairs: n={len(c)}\n")
    g = perm_global(c)
    print(f"[1] GLOBAL permutation     : observed r={g['observed']:.3f}  "
          f"null mean={g['null_mean']:+.3f} sd={g['null_sd']:.3f} max={g['null_max']:.3f}  "
          f"p={g['p']:.2e}")
    w = perm_within_storm(c)
    print(f"[2] WITHIN-STORM permutation: pooled r={w['observed_pooled']:.3f} "
          f"spatial r={w['observed_spatial']:.3f}  null max={w['null_max']:.3f}  p={w['p']:.2e}")
    x = cross_storm(c)
    print(f"[3] CROSS-STORM same-gauge  : r={x['r']:+.3f}  (true {x['observed']:.3f}, n={x['n']})")
    lag = None
    if do_lag:
        print("[4] TEMPORAL-LAG curve (loading storm netCDFs)...")
        lag = lag_curve()
        print(lag.to_string(index=False))
        lag.to_csv(os.path.join(C.OUT_PATH, "nulltest_lag.csv"), index=False)
    plot(g, w, lag, [os.path.join(C.FIGURE_PATH, "val_nulltests.png"),
                     os.path.join(C.PAPER_IMG_PATH, "comp_val_nulltests.pdf")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-lag", action="store_true", help="skip the netCDF-heavy lag test")
    run(do_lag=not ap.parse_args().no_lag)


if __name__ == "__main__":
    main()
