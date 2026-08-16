"""Validate tide-only model runs against CO-OPS astronomical predictions.

The gate before trusting any tide-on surge skill (see the resolution/tides
verdict): score the model's tides *independently of the storm* by comparing
each tide-only run's gauge series against NOAA's published predictions over
the same window. Reports, per (storm, gauge):

* ``datum_offset_m`` -- mean(model) - mean(prediction) over the overlap
  (model geoid vs NTDE-epoch MSL + steric; a result, not noise),
* ``amp_ratio`` -- std(model)/std(prediction) after de-meaning
  (tidal-amplitude fidelity),
* ``lag_min`` -- phase lag from the cross-correlation peak on a sub-hourly
  grid (positive = model lags the prediction),
* ``r`` / ``rmse_m`` -- de-meaned correlation / RMSE over the overlap.

Run (hydra; config root adforce/eval/config/tidecheck_config.yaml)::

    python -m adforce.eval.tidecheck series=data/comp/lowres/low_tide_runs_gauge_series.parquet \
        label=low out=data/comp/out/tidecheck_low.csv
    python -m adforce.eval.tidecheck 'series=data/comp/midres/gs/mid_tide_runs_gs_*.parquet' label=mid
"""

from __future__ import annotations

import glob
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C
from .detide import noaa_predictions
from .pairs import storm_key


def tide_skill(
    sim: pd.Series,
    pred: pd.Series,
    grid_min: int = 6,
    max_lag_hr: float = 3.0,
) -> dict:
    """Score one model tide series against the prediction over their overlap.

    Both series are linearly interpolated onto a common ``grid_min``-minute
    grid (the model output is sub-hourly, predictions hourly), de-meaned, and
    cross-correlated within ``+/- max_lag_hr`` to find the phase lag.

    Returns:
        dict: ``n_hr, datum_offset_m, amp_ratio, lag_min, r, rmse_m``
        (NaNs when the overlap is under 2 days -- too short for a meaningful
        amplitude/phase estimate).
    """
    nan = dict(
        n_hr=0,
        datum_offset_m=np.nan,
        amp_ratio=np.nan,
        lag_min=np.nan,
        r=np.nan,
        rmse_m=np.nan,
    )
    if sim.empty or pred.empty:
        return nan
    lo, hi = max(sim.index[0], pred.index[0]), min(sim.index[-1], pred.index[-1])
    n_hr = (hi - lo).total_seconds() / 3600.0
    if n_hr < 48:
        nan["n_hr"] = max(0, int(n_hr))
        return nan

    grid = pd.date_range(lo, hi, freq=f"{grid_min}min")
    gi = grid.astype("int64").to_numpy()
    s = np.interp(gi, sim.index.astype("int64").to_numpy(), sim.values)
    p = np.interp(gi, pred.index.astype("int64").to_numpy(), pred.values)
    offset = float(s.mean() - p.mean())
    s = s - s.mean()
    p = p - p.mean()
    p_std, s_std = float(p.std()), float(s.std())
    if p_std == 0 or s_std == 0:
        nan.update(n_hr=int(n_hr), datum_offset_m=offset)
        return nan

    max_k = int(max_lag_hr * 60 / grid_min)
    best_k, best_r = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        a, b = (s[k:], p[: len(p) - k]) if k > 0 else (s[: len(s) + k], p[-k:])
        rk = float(np.corrcoef(a, b)[0, 1])
        if rk > best_r:
            best_k, best_r = k, rk
    return dict(
        n_hr=int(n_hr),
        datum_offset_m=round(offset, 3),
        amp_ratio=round(s_std / p_std, 3),
        lag_min=best_k * grid_min,  # positive = model lags the prediction
        r=round(float(np.corrcoef(s, p)[0, 1]), 3),  # lag-0, what a user sees
        rmse_m=round(float(np.sqrt(((s - p) ** 2).mean())), 3),
    )


def check_series(series: pd.DataFrame, out: Optional[str] = None) -> pd.DataFrame:
    """Tide-check every (storm, gauge) of a tide-only gauge-series table.

    Args:
        series: Long-format ``storm, sid, gauge, time, zeta`` table
            (``adforce.eval.extract`` output / legacy sweep parquet).
        out: Optional CSV path.

    Returns:
        pd.DataFrame: one row per (storm, gauge) with the
        :func:`tide_skill` columns plus ``key, sid, gauge``.
    """
    rows = []
    for (storm, sid), grp in series.groupby(["storm", "sid"], sort=True):
        grp = grp.sort_values("time")
        sim = pd.Series(grp.zeta.values, index=pd.DatetimeIndex(grp.time))
        pred = noaa_predictions(str(sid), sim.index[0], sim.index[-1])
        rows.append(
            dict(
                key=storm_key(storm),
                sid=str(sid),
                gauge=grp.gauge.iloc[0],
                **tide_skill(sim, pred),
            )
        )
    df = pd.DataFrame(rows).sort_values(["key", "sid"], ignore_index=True)
    ok = df.dropna(subset=["amp_ratio"])
    if len(ok):
        print(
            f"{len(ok)}/{len(df)} pairs scored: "
            f"median amp_ratio={ok.amp_ratio.median():.2f}  "
            f"median lag={ok.lag_min.median():+.0f} min  "
            f"median r={ok.r.median():.3f}  "
            f"median rmse={ok.rmse_m.median():.2f} m  "
            f"median |datum offset|={ok.datum_offset_m.abs().median():.2f} m"
        )
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


@hydra.main(version_base=None, config_path="config", config_name="tidecheck_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()
    files = sorted(glob.glob(str(cfg.series)))
    if not files:
        raise SystemExit(f"no files match series={cfg.series!r}")
    series = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = cfg.out or f"{C.OUT_PATH}/tidecheck_{cfg.label}.csv"
    check_series(series, out=out)


if __name__ == "__main__":
    main()
