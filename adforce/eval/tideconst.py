"""Constituent-level tidal diagnosis: model vs NOAA, per constituent.

``tidecheck`` says the model over-amplifies coastal tides ~1.4x overall;
this module says WHICH constituents, which separates candidate causes:

* all semidiurnal constituents inflated together, diurnals fine -> boundary
  forcing / shelf response (resonance, friction) of the semidiurnal band;
* uniform inflation across bands -> friction / global calibration;
* one constituent wrong -> forcing-database or constituent-list bug.

Method: fit the SAME constituent set (utide, fixed list, DatetimeIndex --
never date2num, see coops.py) to the model tide-only series and to the CO-OPS
prediction over the identical storm window, and compare amplitudes/phases.
Caveat: storm windows are ~13-15 days, at the edge of the M2/S2 Rayleigh
separation -- per-window leakage is mitigated by pooling the per-(gauge,
constituent) ratios across storms and reporting medians.

Run (hydra; config root adforce/eval/config/tideconst_config.yaml)::

    python -m adforce.eval.tideconst series=data/comp/lowres/low_tide_runs_gauge_series.parquet \
        label=low limit_storms=5
"""

from __future__ import annotations

import glob
import warnings
from typing import Optional, Sequence

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C
from .detide import noaa_predictions
from .extract import gauge_frame
from .pairs import storm_key

#: major Gulf constituents: semidiurnal band then diurnal band
CONSTITUENTS = ("M2", "S2", "N2", "K1", "O1")


def _fit_amps(
    series: pd.Series, lat: float, constituents: Sequence[str]
) -> Optional[pd.DataFrame]:
    """utide amplitudes/phases for a fixed constituent list; None on failure."""
    import utide

    if series.size < 24 * 8:  # need >= ~8 days for even a forced fit
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            coef = utide.solve(
                series.index,  # DatetimeIndex direct (date2num breaks utide)
                series.values,
                lat=lat,
                constit=list(constituents),
                method="ols",
                trend=False,
                nodal=True,
                conf_int="none",
                verbose=False,
            )
        except Exception:
            return None
    return pd.DataFrame(
        dict(constituent=list(coef.name), amp=coef.A, phase=coef.g)
    ).set_index("constituent")


def constituent_table(
    series: pd.DataFrame,
    constituents: Sequence[str] = CONSTITUENTS,
    limit_storms: Optional[int] = None,
    min_pred_std: float = 0.05,
    out: Optional[str] = None,
) -> pd.DataFrame:
    """Per (storm, gauge, constituent) model-vs-prediction amplitude/phase.

    Args:
        series: Tide-only long-format ``storm, sid, gauge, time, zeta`` table.
        constituents: Fixed constituent list fitted to BOTH sides.
        limit_storms: Diagnose only the first N storms (fits are slow).
        min_pred_std: Skip near-tideless gauges (prediction std below this).
        out: Optional CSV path.

    Returns:
        pd.DataFrame: ``key, sid, gauge, constituent, amp_model, amp_pred,
        ratio, dphase_deg`` (positive dphase = model lags).
    """
    lats = {str(r.sid): float(r.lat) for r in gauge_frame().itertuples()}
    storms = sorted(series.storm.unique())
    if limit_storms:
        storms = storms[: int(limit_storms)]
    rows = []
    for storm in storms:
        sdf = series[series.storm == storm]
        key = storm_key(storm)
        for sid, grp in sdf.groupby("sid", sort=True):
            sid = str(sid)
            lat = lats.get(sid)
            if lat is None:
                continue
            grp = grp.sort_values("time")
            sim = pd.Series(grp.zeta.values, index=pd.DatetimeIndex(grp.time))
            sim = sim.resample("1h").mean().dropna()  # hourly is plenty for tides
            pred = noaa_predictions(sid, sim.index[0], sim.index[-1])
            if pred.empty or float(pred.std()) < min_pred_std:
                continue
            fit_m = _fit_amps(sim, lat, constituents)
            fit_p = _fit_amps(pred, lat, constituents)
            if fit_m is None or fit_p is None:
                continue
            for name in constituents:
                if name not in fit_m.index or name not in fit_p.index:
                    continue
                am, ap = float(fit_m.amp[name]), float(fit_p.amp[name])
                if ap < 0.01:  # constituent absent at this gauge
                    continue
                dph = (float(fit_m.phase[name]) - float(fit_p.phase[name]) + 180) % 360 - 180
                rows.append(
                    dict(
                        key=key,
                        sid=sid,
                        gauge=grp.gauge.iloc[0],
                        constituent=name,
                        amp_model=round(am, 3),
                        amp_pred=round(ap, 3),
                        ratio=round(am / ap, 3),
                        dphase_deg=round(dph, 1),
                    )
                )
        print(f"  {key}: fitted")
    df = pd.DataFrame(rows)
    if len(df):
        pooled = (
            df.groupby("constituent")
            .agg(
                n=("ratio", "size"),
                amp_pred_med=("amp_pred", "median"),
                ratio_med=("ratio", "median"),
                ratio_q25=("ratio", lambda s: s.quantile(0.25)),
                ratio_q75=("ratio", lambda s: s.quantile(0.75)),
                dphase_med=("dphase_deg", "median"),
            )
            .reindex([c for c in constituents if c in set(df.constituent)])
        )
        print(pooled.to_string(float_format=lambda v: f"{v:.2f}"))
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


@hydra.main(version_base=None, config_path="config", config_name="tideconst_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()
    files = sorted(glob.glob(str(cfg.series)))
    if not files:
        raise SystemExit(f"no files match series={cfg.series!r}")
    series = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = cfg.out or f"{C.OUT_PATH}/tideconst_{cfg.label}.csv"
    constituent_table(
        series,
        constituents=tuple(cfg.constituents),
        limit_storms=cfg.limit_storms,
        min_pred_std=cfg.min_pred_std,
        out=out,
    )


if __name__ == "__main__":
    main()
