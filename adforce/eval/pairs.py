"""Model-vs-model comparison: resolution bias and tide-surge interaction.

Consolidates and generalizes ``rerun/adcirc/score_resolution.py`` and the
(uncommitted) script behind ``rerun/results/tide_surge_interaction.csv``:

* :func:`resolution_bias_table` -- join a candidate cell's gauge-series peaks
  onto ``val_summary.csv`` (baseline peaks + de-tided observed peaks + clean
  flags), keyed by (storm, sid).
* :func:`tide_surge_interaction` -- the forcing triple: per (storm, gauge)
  ``interaction = peak(storm+tide) - peak(storm) - peak(tide)``. Storm-only
  peaks fall back to the HF-archive ``val_summary`` peaks where no storm-only
  run exists (``storm_src`` records ``series`` vs ``archive``).
* :func:`compare_cells` -- general pairwise peaks of any two cells' series.

Inputs are the long-format gauge-series tables (``storm, sid, gauge, time,
zeta``) that :mod:`adforce.eval.extract` writes (and the legacy
``data/comp/{lowres,midres}`` sweep parquets already hold).

Run (hydra; config root adforce/eval/config/pairs_config.yaml)::

    python -m adforce.eval.pairs action=resolution_bias series=data/comp/lowres/low_storm_gauge_series.parquet
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C


def storm_key(storm: str) -> str:
    """``'140_LAURA_2020'`` or ``'Laura 2020'`` -> ``'LAURA_2020'``."""
    s = re.sub(r"^\d+_", "", str(storm))
    return s.replace(" ", "_").upper()


def peak_table(series: pd.DataFrame) -> pd.DataFrame:
    """Per-(storm, gauge) peak of a long-format series table.

    Returns columns ``key, sid, gauge, peak`` (key = ``NAME_YEAR``).
    """
    df = series.copy()
    df["key"] = df.storm.map(storm_key)
    df["sid"] = df.sid.astype(str)
    return df.groupby(["key", "sid"], as_index=False).agg(
        gauge=("gauge", "first"), peak=("zeta", "max")
    )


def archive_peak_table(val_summary: Optional[str] = None) -> pd.DataFrame:
    """HF-archive storm-only peaks from ``val_summary.csv`` (columns as
    :func:`peak_table`); the fallback for storms without a storm-only run."""
    path = val_summary or os.path.join(C.OUT_PATH, "val_summary.csv")
    vs = pd.read_csv(path)
    return pd.DataFrame(
        dict(
            key=vs.storm.map(storm_key),
            sid=vs.sid.astype(str),
            gauge=vs.name,
            peak=vs.sim_peak,
        )
    )


def resolution_bias_table(
    series_parquet: str,
    val_summary: Optional[str] = None,
    out: Optional[str] = None,
    min_obs_peak: float = 1.0,
) -> pd.DataFrame:
    """Join candidate-cell peaks onto the validation summary (exact port of
    ``rerun/adcirc/score_resolution.py``; writes the same CSV shape).

    Args:
        series_parquet (str): Long-format gauge series of the candidate cell
            (e.g. the low-res sweep extract).
        val_summary (Optional[str]): Baseline ``val_summary.csv`` (default:
            the standard output path).
        out (Optional[str]): CSV to write (default: none).
        min_obs_peak (float): Pooled-ratio report threshold on observed peak.

    Returns:
        pd.DataFrame: ``val_summary`` columns + ``low_peak`` per (storm, sid).
    """
    from .cells import dir_to_storm

    ser = pd.read_parquet(series_parquet)
    ser["storm_name"] = ser.storm.map(dir_to_storm)
    low = (
        ser.groupby(["storm_name", "sid"], as_index=False)
        .zeta.max()
        .rename(columns={"zeta": "low_peak", "storm_name": "storm"})
    )
    low["sid"] = low.sid.astype(str)

    val = pd.read_csv(val_summary or os.path.join(C.OUT_PATH, "val_summary.csv"))
    val["sid"] = val.sid.astype(str)
    df = val.merge(low, on=["storm", "sid"], how="left")
    clean = df[
        df.clean.astype(bool) & df.low_peak.notna() & (df.obs_peak >= min_obs_peak)
    ].copy()
    clean["mid_ratio"] = clean.sim_peak / clean.obs_peak
    clean["low_ratio"] = clean.low_peak / clean.obs_peak

    print(f"clean pairs with obs>={min_obs_peak} m and low coverage: {len(clean)}")
    for name, r in (("mid/obs", clean.mid_ratio), ("low/obs", clean.low_ratio)):
        print(f"  {name}: mean {r.mean():.2f}  median {np.median(r):.2f}")
    print(f"  low/mid peak ratio: mean {(clean.low_peak / clean.sim_peak).mean():.2f}")

    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


def tide_surge_interaction(
    storm_series: Optional[pd.DataFrame],
    tide_series: pd.DataFrame,
    both_series: pd.DataFrame,
    res: str,
    archive_peaks: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Peak-based nonlinear tide-surge interaction for one resolution.

    ``interaction = peak(storm+tide) - peak(storm) - peak(tide)`` per
    (storm, gauge); a (tide, both) pair is required, the storm-only peak
    comes from ``storm_series`` where available (``storm_src='series'``) and
    otherwise from ``archive_peaks`` (``storm_src='archive'``, the HF-archive
    ``val_summary`` convention of the published mid-res rows).

    Returns:
        pd.DataFrame: ``res, key, sid, gauge, zeta_storm, zeta_tide,
        zeta_both, interaction, storm_src`` (the exact schema of
        ``rerun/results/tide_surge_interaction.csv``).
    """
    tide = peak_table(tide_series).rename(columns={"peak": "zeta_tide"})
    both = peak_table(both_series).rename(columns={"peak": "zeta_both"})
    df = tide.merge(both.drop(columns="gauge"), on=["key", "sid"], how="inner")

    parts = []
    if storm_series is not None and len(storm_series):
        s = peak_table(storm_series).drop(columns="gauge")
        s = s.rename(columns={"peak": "zeta_storm"})
        s["storm_src"] = "series"
        parts.append(s)
    if archive_peaks is not None and len(archive_peaks):
        a = archive_peaks.drop(columns="gauge").rename(columns={"peak": "zeta_storm"})
        a = a.copy()
        a["storm_src"] = "archive"
        parts.append(a)
    if not parts:
        raise ValueError("need storm_series and/or archive_peaks for storm-only peaks")
    storm = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["key", "sid"], keep="first"  # series wins over archive
    )

    df = df.merge(storm, on=["key", "sid"], how="inner")
    df["interaction"] = df.zeta_both - df.zeta_storm - df.zeta_tide
    df["res"] = res
    return df[
        [
            "res",
            "key",
            "sid",
            "gauge",
            "zeta_storm",
            "zeta_tide",
            "zeta_both",
            "interaction",
            "storm_src",
        ]
    ].sort_values(["key", "sid"], ignore_index=True)


def compare_cells(
    frames: Dict[str, pd.DataFrame], baseline: str, out: Optional[str] = None
) -> pd.DataFrame:
    """Pairwise peak comparison of every cell against a baseline cell.

    Args:
        frames: ``{cell_id: long-format gauge-series table}``.
        baseline: Key in ``frames`` to compare everything against.
        out: Optional CSV path.

    Returns:
        pd.DataFrame: one row per (cell, storm, gauge) with
        ``cell, key, sid, gauge, peak, base_peak, peak_diff``.
    """
    if baseline not in frames:
        raise ValueError(f"baseline {baseline!r} not in frames {sorted(frames)}")
    base = peak_table(frames[baseline]).rename(columns={"peak": "base_peak"})
    rows = []
    for cell, series in frames.items():
        if cell == baseline:
            continue
        p = peak_table(series).drop(columns="gauge")
        m = p.merge(base, on=["key", "sid"], how="inner")
        m["cell"] = cell
        m["peak_diff"] = m.peak - m.base_peak
        rows.append(m)
    df = (
        pd.concat(rows, ignore_index=True)[
            ["cell", "key", "sid", "gauge", "peak", "base_peak", "peak_diff"]
        ]
        if rows
        else pd.DataFrame(
            columns=["cell", "key", "sid", "gauge", "peak", "base_peak", "peak_diff"]
        )
    )
    if out:
        df.to_csv(out, index=False)
    return df


@hydra.main(version_base=None, config_path="config", config_name="pairs_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()
    if cfg.action == "resolution_bias":
        resolution_bias_table(
            cfg.series,
            val_summary=cfg.val_summary,
            out=cfg.out,
            min_obs_peak=cfg.min_obs_peak,
        )
    elif cfg.action == "interaction":
        storm = pd.read_parquet(cfg.storm_series) if cfg.storm_series else None
        archive = archive_peak_table(cfg.val_summary) if cfg.use_archive else None
        df = tide_surge_interaction(
            storm,
            pd.read_parquet(cfg.tide_series),
            pd.read_parquet(cfg.both_series),
            res=cfg.res,
            archive_peaks=archive,
        )
        if cfg.out:
            df.to_csv(cfg.out, index=False)
            print(f"wrote {cfg.out} ({len(df)} rows)")
    else:
        raise SystemExit(f"unknown action {cfg.action!r} (resolution_bias|interaction)")


if __name__ == "__main__":
    main()
