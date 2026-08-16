"""Shared obs-side utilities for tide-on evaluation (Phase 2).

Windowed CO-OPS fetches (raw water level, astronomical predictions), the
pre-storm-mean alignment that removes the model-geoid vs gauge-datum offset
from a (sim, obs) pair while keeping it *reported*, and the skew-surge peak
(promoted from ``detide_sensitivity``; Horsburgh & Wilson 2007) -- the
phase-insensitive peak metric that becomes the headline for tide-on cells.

Datum note: CO-OPS series are requested at MSL (1983-2001 NTDE epoch); the
EC95d model geoid carries no seasonal steric cycle or SLR since the mesh
epoch. The O(0.1-0.3 m) mismatch is the same order as the biases being
measured, so tide-on comparisons must either align on a pre-forcing window
(:func:`align_pair`) or treat the offset as a result, never ignore it.
"""

from __future__ import annotations

import os
import time as _time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import constants as C
from .coops import _coops

#: polite pause after a live (non-cached) CO-OPS request
POLITE_SLEEP_S = 0.3


def _window_strings(t0, t1) -> Tuple[str, str]:
    return pd.Timestamp(t0).strftime("%Y%m%d"), pd.Timestamp(t1).strftime("%Y%m%d")


def _polite(sid: str, product: str, b: str, e: str, datum: str = "MSL") -> None:
    """Sleep briefly when the request is about to go to the live API."""
    key = f"{sid}_{product}_{b}_{e}_{datum}.csv"
    if not os.path.exists(os.path.join(C.COOPS_CACHE, key)):
        _time.sleep(POLITE_SLEEP_S)


def fetch_window_wl(sid: str, t0, t1) -> pd.Series:
    """Raw observed water level over a window (verified ``hourly_height``,
    falling back to ``water_level``), hourly, metres MSL."""
    b, e = _window_strings(t0, t1)
    _polite(sid, "hourly_height", b, e)
    wl = _coops(sid, b, e, "hourly_height")
    if wl.empty:
        _polite(sid, "water_level", b, e)
        wl = _coops(sid, b, e, "water_level")
    return wl


def noaa_predictions(sid: str, t0, t1) -> pd.Series:
    """CO-OPS published astronomical tide prediction over a window (hourly,
    metres MSL) -- the reference for validating a tide-only model run."""
    b, e = _window_strings(t0, t1)
    _polite(sid, "predictions", b, e)
    return _coops(sid, b, e, "predictions")


def align_pair(
    sim: pd.Series,
    obs: pd.Series,
    forcing_start=None,
    window_hr: float = 48.0,
) -> Tuple[pd.Series, pd.Series, float]:
    """Remove the common datum/steric offset from a (sim, obs) pair.

    Subtracts each series' mean over the ``window_hr`` hours before
    ``forcing_start`` (default: the start of the overlap) from that series.

    Returns:
        (sim_aligned, obs_aligned, datum_offset_m): the offset is
        ``mean(sim) - mean(obs)`` over the alignment window -- report it, it
        is scientifically interesting (geoid vs NTDE-epoch MSL + steric).
    """
    lo = max(sim.index[0], obs.index[0])
    start = pd.Timestamp(forcing_start) if forcing_start is not None else lo
    w0, w1 = start, start + pd.Timedelta(hours=window_hr)
    sim_m = float(sim.loc[w0:w1].mean())
    obs_m = float(obs.loc[w0:w1].mean())
    if not np.isfinite(sim_m) or not np.isfinite(obs_m):
        return sim, obs, float("nan")
    return sim - sim_m, obs - obs_m, sim_m - obs_m


def skew_surge_peak(wl: pd.Series, tide: pd.Series, win: Tuple) -> float:
    """Storm skew-surge peak = max over tidal cycles of (max observed - max predicted tide).

    The skew surge (Horsburgh & Wilson 2007) compares the peak observed level in each tidal
    cycle to the peak predicted tide in the same cycle, so it is insensitive to tidal-phase
    error -- the defensible peak target for a tide-excluding model. Cycles are split at the
    predicted-tide low waters. (Promoted verbatim from ``detide_sensitivity``.)
    """
    from scipy.signal import find_peaks

    o = wl.loc[win[0] : win[1]]
    td = tide.loc[win[0] : win[1]]
    if len(o) < 12:
        return np.nan
    troughs, _ = find_peaks(-td.values, distance=8)  # tide low waters, >= 8 h apart
    bounds = np.r_[0, troughs, len(td) - 1]
    skews = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # positional slicing assumes o and td share one index (their original
        # detide_sensitivity contract); guard the ragged case (callers should
        # inner-join first -- see twl._common_index)
        ov, tv = o.values[a:b], td.values[a:b]
        if b - a >= 4 and ov.size and tv.size:
            skews.append(float(ov.max() - tv.max()))
    return max(skews) if skews else np.nan
