"""Tide-on scoring: model total water level vs raw gauge observations.

The tide-on counterpart of ``validate`` (which scores surge-only runs against
de-tided residuals). For each (storm, gauge) of a storm+tide run:

* obs = raw CO-OPS water level over the run window (year-cache slice);
* the model-geoid vs gauge-datum offset is removed by pre-forcing-window
  alignment (:func:`adforce.eval.detide.align_pair`) and REPORTED as
  ``datum_offset_m``;
* peaks and hydrograph skill are scored on the aligned pair;
* the **skew surge** (per-tidal-cycle max difference; phase-insensitive) is
  the headline metric -- instantaneous peak comparison is ill-posed under
  model tidal-phase error (see ``tidecheck``): model skew uses the matching
  tide-only run as its astronomical reference, observed skew uses the CO-OPS
  prediction.

Run (hydra; config root adforce/eval/config/twl_config.yaml)::

    python -m adforce.eval.twl both_series=data/comp/lowres/low_both_full.parquet \
        tide_series=data/comp/lowres/low_tide_runs_gauge_series.parquet label=low
"""

from __future__ import annotations

import glob
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C
from .coops import fetch_year
from .detide import align_pair, noaa_predictions, skew_surge_peak
from .pairs import storm_key
from .validate import timeseries_skill


def _common_index(a: pd.Series, b: pd.Series):
    """Inner-join two series onto their shared timestamps (skew_surge_peak
    slices positionally, so both sides must share one index)."""
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return j["a"], j["b"]


def _series_by_pair(df: pd.DataFrame) -> dict:
    out = {}
    for (storm, sid), grp in df.groupby(["storm", "sid"], sort=True):
        grp = grp.sort_values("time")
        out[(storm_key(storm), str(sid))] = (
            grp.gauge.iloc[0],
            pd.Series(grp.zeta.values, index=pd.DatetimeIndex(grp.time)),
        )
    return out


def twl_table(
    both_series: pd.DataFrame,
    tide_series: Optional[pd.DataFrame] = None,
    align_window_hr: float = 48.0,
    out: Optional[str] = None,
) -> pd.DataFrame:
    """Score storm+tide runs against raw gauge water level.

    Args:
        both_series: Long-format ``storm, sid, gauge, time, zeta`` table of
            storm+tide runs (total water level).
        tide_series: Matching tide-only runs (the model's astronomical
            reference for the skew surge; skew columns are NaN without it).
        align_window_hr: Pre-forcing alignment window (the spinup period --
            storm still far away).
        out: Optional CSV path.

    Returns:
        pd.DataFrame: per (storm, gauge): ``key, sid, gauge, n_hr,
        datum_offset_m, peak_sim, peak_obs, peak_bias, ts_r, ts_rmse,
        skew_sim, skew_obs, skew_bias``.
    """
    tides = _series_by_pair(tide_series) if tide_series is not None else {}
    rows = []
    for (key, sid), (gauge, sim) in _series_by_pair(both_series).items():
        t0, t1 = sim.index[0], sim.index[-1]
        year = int(key.rsplit("_", 1)[-1])
        obs = fetch_year(sid, year).loc[t0:t1]
        n_hr = (
            (min(t1, obs.index[-1]) - max(t0, obs.index[0])).total_seconds() / 3600.0
            if len(obs)
            else 0.0
        )
        row = dict(
            key=key,
            sid=sid,
            gauge=gauge,
            n_hr=int(max(0, n_hr)),
            datum_offset_m=np.nan,
            peak_sim=np.nan,
            peak_obs=np.nan,
            peak_bias=np.nan,
            ts_r=np.nan,
            ts_rmse=np.nan,
            skew_sim=np.nan,
            skew_obs=np.nan,
            skew_bias=np.nan,
        )
        if n_hr < 2 * align_window_hr:  # need alignment window + storm
            rows.append(row)
            continue
        sim_al, obs_al, offset = align_pair(
            sim, obs, forcing_start=t0, window_hr=align_window_hr
        )
        ts_r, ts_rmse, _ = timeseries_skill(sim_al, obs_al)
        row.update(
            datum_offset_m=round(offset, 3),
            peak_sim=round(float(sim_al.max()), 2),
            peak_obs=round(float(obs_al.max()), 2),
            peak_bias=round(float(sim_al.max() - obs_al.max()), 2),
            ts_r=round(ts_r, 3),
            ts_rmse=round(ts_rmse, 3),
        )
        # skew surge: model vs its own tide-only run, obs vs the prediction
        mtide = tides.get((key, sid))
        pred = noaa_predictions(sid, t0, t1)
        if mtide is not None and len(pred):
            mt_al, _, _ = align_pair(mtide[1], mtide[1], forcing_start=t0, window_hr=align_window_hr)
            pr_al, _, _ = align_pair(pred, pred, forcing_start=t0, window_hr=align_window_hr)
            skew_sim = skew_surge_peak(*_common_index(sim_al, mt_al), (t0, t1))
            skew_obs = skew_surge_peak(*_common_index(obs_al, pr_al), (t0, t1))
            if np.isfinite(skew_sim) and np.isfinite(skew_obs):
                row.update(
                    skew_sim=round(float(skew_sim), 2),
                    skew_obs=round(float(skew_obs), 2),
                    skew_bias=round(float(skew_sim - skew_obs), 2),
                )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["key", "sid"], ignore_index=True)
    ok = df.dropna(subset=["peak_bias"])
    if len(ok):
        print(
            f"{len(ok)}/{len(df)} pairs scored: "
            f"median peak bias={ok.peak_bias.median():+.2f} m  "
            f"median ts_r={ok.ts_r.median():.3f}  "
            f"median ts_rmse={ok.ts_rmse.median():.2f} m  "
            f"median |datum offset|={ok.datum_offset_m.abs().median():.2f} m"
        )
        sk = df.dropna(subset=["skew_bias"])
        if len(sk):
            print(
                f"skew surge ({len(sk)} pairs): "
                f"median bias={sk.skew_bias.median():+.2f} m  "
                f"(sim {sk.skew_sim.median():.2f} vs obs {sk.skew_obs.median():.2f})"
            )
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


@hydra.main(version_base=None, config_path="config", config_name="twl_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()

    def load(pattern):
        files = sorted(glob.glob(str(pattern))) if pattern else []
        return (
            pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if files
            else None
        )

    both = load(cfg.both_series)
    if both is None:
        raise SystemExit(f"no files match both_series={cfg.both_series!r}")
    out = cfg.out or f"{C.OUT_PATH}/twl_{cfg.label}.csv"
    twl_table(
        both,
        tide_series=load(cfg.tide_series),
        align_window_hr=cfg.align_window_hr,
        out=out,
    )


if __name__ == "__main__":
    main()
