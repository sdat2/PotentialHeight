"""NOAA CO-OPS tide-gauge access and de-tiding for the ``comp`` module.

Fetches water-level observations from the CO-OPS data-getter API (cached on disk)
and produces a de-tided storm-surge residual. The residual is computed by a robust
``utide`` harmonic analysis of the storm's full calendar year (so it does not depend
on CO-OPS having published harmonic predictions for the station); if the record is
too sparse to fit, it falls back to the CO-OPS ``predictions`` product.
"""

from __future__ import annotations

import io
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

from .constants import (
    COOPS_API,
    COOPS_CACHE,
    COOPS_MDAPI,
    GAUGE_BOX,
    UTIDE_MIN_SAMPLES,
)

Gauge = Tuple[str, str, float, float]  # (station_id, name, lat, lon)


def gulf_gauges(box: dict = GAUGE_BOX) -> List[Gauge]:
    """Return CO-OPS water-level stations within ``box`` (cached metadata)."""
    cache = os.path.join(COOPS_CACHE, "stations.json")
    if os.path.exists(cache):
        stations = json.load(open(cache))
    else:
        r = requests.get(COOPS_MDAPI, params={"type": "waterlevels"}, timeout=60)
        r.raise_for_status()
        stations = r.json()["stations"]
        json.dump(stations, open(cache, "w"))
    (lon0, lon1), (lat0, lat1) = box["lon"], box["lat"]
    return [
        (s["id"], s["name"], float(s["lat"]), float(s["lng"]))
        for s in stations
        if lon0 <= s["lng"] <= lon1 and lat0 <= s["lat"] <= lat1
    ]


def _coops(
    station: str, begin: str, end: str, product: str, datum: str = "MSL"
) -> pd.Series:
    """Fetch one hourly CO-OPS series (GMT, metric), cached on disk by request."""
    key = f"{station}_{product}_{begin}_{end}_{datum}.csv"
    fp = os.path.join(COOPS_CACHE, key)
    if os.path.exists(fp):
        text = open(fp).read()
    else:
        params = dict(
            begin_date=begin,
            end_date=end,
            station=station,
            product=product,
            datum=datum,
            units="metric",
            time_zone="gmt",
            interval="h",
            application="worstsurge_comp",
            format="csv",
        )
        try:
            r = requests.get(COOPS_API, params=params, timeout=90)
            r.raise_for_status()
            text = r.text
        except Exception:
            text = ""
        open(fp, "w").write(text)
    if (
        not text
        or "Error" in text
        or "No " in text
        or len(text.strip().splitlines()) < 2
    ):
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    vcol = [c for c in df.columns if c != "Date Time"][0]
    return pd.Series(
        pd.to_numeric(df[vcol], errors="coerce").values,
        index=pd.to_datetime(df["Date Time"]),
    ).dropna()


def fetch_year(station: str, year: int) -> pd.Series:
    """Hourly water level (MSL) for a calendar year; verified then preliminary."""
    wl = _coops(station, f"{year}0101", f"{year}1231", "hourly_height")
    if wl.empty:
        wl = _coops(station, f"{year}0101", f"{year}1231", "water_level")
    return wl


def observed_residual(
    station: str, lat: float, year: int, start, end
) -> Tuple[pd.Series, str]:
    """De-tided surge residual over ``[start, end]``.

    Returns ``(residual_series, method)`` where method is ``"utide"``,
    ``"pred"`` (CO-OPS predictions fallback) or ``"none"``.
    """
    import utide

    wl = fetch_year(station, year)
    if wl.size >= UTIDE_MIN_SAMPLES:
        try:
            # Pass the DatetimeIndex directly: feeding matplotlib date2num
            # floats to utide 0.3.1 breaks its astronomical arguments under
            # matplotlib's modern (1970) epoch, and the solve silently
            # selects ZERO constituents (residual keeps the full tide).
            # Verified on Grand Isle 2020: date2num path -> empty constituent
            # list, tide std 0.03 m; DatetimeIndex -> SA/K1/O1 amplitudes
            # 0.16/0.12/0.12 m, tide std 0.17 m. Same fix as comp.annual_max.
            coef = utide.solve(
                wl.index,
                wl.values,
                lat=lat,
                method="robust",
                trend=True,
                conf_int="none",
                verbose=False,
            )
            tide = utide.reconstruct(wl.index, coef, verbose=False).h
            resid = pd.Series(wl.values - tide, index=wl.index)
            return resid.loc[start:end], "utide"
        except Exception as e:  # pragma: no cover - sparse/odd records
            print(f"    utide failed for {station} ({year}): {e}")

    # fallback: CO-OPS harmonic predictions over the storm window only
    b = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    wlw = _coops(station, b, e, "hourly_height")
    if wlw.empty:
        wlw = _coops(station, b, e, "water_level")
    pred = _coops(station, b, e, "predictions")
    if wlw.empty or pred.empty:
        return pd.Series(dtype=float), "none"
    j = pd.concat({"wl": wlw, "pred": pred}, axis=1).dropna()
    return (j["wl"] - j["pred"]), "pred"
