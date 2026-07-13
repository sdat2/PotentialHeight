"""Annual-maximum de-tided surge residuals from NOAA CO-OPS tide gauges.

For a station and a year range this module downloads hourly water levels
(one calendar-year request at a time -- the CO-OPS ``hourly_height`` product
accepts up to a year per request), de-tides each calendar year separately with
a robust ``utide`` harmonic fit (mean + trend included, following
:func:`comp.coops.observed_residual`), and records the ANNUAL MAXIMUM of the
surge residual for every year with adequate data.

Missing-data policy: a year is skipped unless (a) it has at least
``UTIDE_MIN_SAMPLES`` hourly samples (stable harmonic fit, same threshold as
``comp.coops``) and (b) at least ``MIN_YEAR_COVERAGE`` (80%) of the year's
hours are present, since with large gaps the true annual maximum may fall in a
gap and the recorded maximum would be biased low. Even in accepted years the
maximum can be truncated if the gauge failed *during* the peak (e.g. Grand
Isle in Ida 2021, see ``comp.constants.KNOWN_FAILED``); such years are flagged
with ``max_at_gap_edge`` rather than dropped.

Everything is cached under ``data/comp/``: the raw CO-OPS responses land in
``COOPS_CACHE`` (via :mod:`comp.coops`), the per-year de-tided residuals and
the annual-maxima table are Parquet files under ``ANNUAL_MAX_CACHE``, keyed by
the de-tiding parameters so the cache self-invalidates if those change.
Reruns are therefore free.

Datum note: water levels are requested relative to MSL, but the residual is
insensitive to the datum because the harmonic fit absorbs the mean (and a
linear trend, which also removes most local sea-level rise within a year).

Run::

    python -m comp.annual_max --station 8761724                 # Grand Isle, LA
    python -m comp.annual_max --station 8735180 --start 1980    # Dauphin Island, AL
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from . import constants as C
from . import coops


def _hours_in_year(year: int) -> int:
    """Number of hours in a calendar year.

    Args:
        year (int): Calendar year.

    Returns:
        int: 8760 for common years, 8784 for leap years.

    Examples:
        >>> _hours_in_year(2021)
        8760
        >>> _hours_in_year(2020)
        8784
    """
    return int((pd.Timestamp(year + 1, 1, 1) - pd.Timestamp(year, 1, 1)).days * 24)


def station_meta(station: str) -> Optional[dict]:
    """Name/lat/lon for a CO-OPS station from the cached station list.

    Reuses the same ``stations.json`` cache as :func:`comp.coops.gulf_gauges`
    (all water-level stations, no bounding box), fetching it once if absent.

    Args:
        station (str): CO-OPS station id, e.g. ``"8761724"``.

    Returns:
        Optional[dict]: ``{"id", "name", "lat", "lon"}`` or ``None`` if unknown.
    """
    import requests

    cache = os.path.join(C.COOPS_CACHE, "stations.json")
    if os.path.exists(cache):
        stations = json.load(open(cache))
    else:
        r = requests.get(C.COOPS_MDAPI, params={"type": "waterlevels"}, timeout=60)
        r.raise_for_status()
        stations = r.json()["stations"]
        json.dump(stations, open(cache, "w"))
    for s in stations:
        if s["id"] == station:
            return dict(
                id=s["id"], name=s["name"], lat=float(s["lat"]), lon=float(s["lng"])
            )
    return None


# --------------------------------------------------------------------------- #
# Fetching. comp.coops._coops caches every response, including transport
# failures (as zero-byte files). A genuine "no data" year is a NON-empty API
# error message and is accepted at once; a zero-byte cache entry is deleted and
# the year re-requested a few times, so a one-off network blip cannot
# permanently poison the record.
# --------------------------------------------------------------------------- #
def _clear_failed_cache(station: str, year: int) -> bool:
    """Delete zero-byte (cached transport failure) responses for a year.

    Returns:
        bool: True if anything was deleted (i.e. a retry is worthwhile).
    """
    cleared = False
    for product in ("hourly_height", "water_level"):
        fp = os.path.join(
            C.COOPS_CACHE, f"{station}_{product}_{year}0101_{year}1231_MSL.csv"
        )
        if os.path.exists(fp) and os.path.getsize(fp) == 0:
            os.remove(fp)
            cleared = True
    return cleared


def fetch_year_wl(
    station: str,
    year: int,
    retries: int = C.COOPS_RETRIES,
    sleep_s: float = C.COOPS_SLEEP_S,
) -> pd.Series:
    """Hourly water level (MSL, GMT) for one calendar year, cached + polite.

    Delegates to :func:`comp.coops.fetch_year` (verified ``hourly_height``,
    falling back to preliminary ``water_level``); on an empty result caused by
    a cached transport failure the bad cache entry is cleared and the request
    retried with a growing sleep. Tests monkeypatch this function.

    Args:
        station (str): CO-OPS station id.
        year (int): Calendar year.
        retries (int, optional): Max re-requests after transport failures.
        sleep_s (float, optional): Base polite pause after live requests.

    Returns:
        pd.Series: Hourly water level indexed by GMT timestamp (may be empty).
    """
    live = not os.path.exists(
        os.path.join(
            C.COOPS_CACHE, f"{station}_hourly_height_{year}0101_{year}1231_MSL.csv"
        )
    )
    wl = coops.fetch_year(station, year)
    attempt = 0
    while wl.empty and attempt < retries and _clear_failed_cache(station, year):
        attempt += 1
        time.sleep(sleep_s * attempt)
        live = True
        wl = coops.fetch_year(station, year)
    if live:
        time.sleep(sleep_s)  # be polite between live year-long requests
    return wl


# --------------------------------------------------------------------------- #
# De-tiding and the annual maximum
# --------------------------------------------------------------------------- #
def detide_year(wl: pd.Series, lat: float, method: str = "robust") -> pd.Series:
    """De-tide one calendar year of hourly water levels with ``utide``.

    Mirrors :func:`comp.coops.observed_residual` (harmonic fit with mean and
    linear trend over the full year, residual = observed - reconstruction),
    with one deliberate difference: the DatetimeIndex is passed to ``utide``
    directly instead of via ``matplotlib.dates.date2num``. With matplotlib's
    modern 1970 date epoch, ``date2num`` times break utide 0.3.1's
    astronomical arguments and the solve silently selects ZERO constituents
    (verified on Grand Isle 2020: no constituents via date2num; SA/K1/O1
    amplitudes 0.16/0.12/0.12 m via the index). Datetimes are unambiguous.

    Args:
        wl (pd.Series): Hourly water level indexed by timestamp.
        lat (float): Station latitude (degrees) for the nodal corrections.
        method (str, optional): ``utide.solve`` method; ``"robust"`` (default,
            IRLS -- storm peaks barely leak into the fit) or ``"ols"`` (faster).

    Returns:
        pd.Series: Surge residual on the same index.
    """
    import utide

    coef = utide.solve(
        wl.index,
        wl.values,
        lat=lat,
        method=method,
        trend=True,
        conf_int="none",
        verbose=False,
    )
    tide = utide.reconstruct(wl.index, coef, verbose=False).h
    return pd.Series(wl.values - tide, index=wl.index)


def max_at_gap_edge(
    series: pd.Series, gap_hr: float = 3.0, window_hr: float = 6.0
) -> bool:
    """Flag whether the series maximum sits next to a data gap (or record edge).

    If the gauge failed as the surge peaked (a common failure mode in major
    hurricanes) the recorded annual maximum is a lower bound. The flag is True
    when a gap longer than ``gap_hr`` starts within ``window_hr`` of the
    maximum, or the maximum is within ``window_hr`` of the record ends.

    Args:
        series (pd.Series): Hourly residual series.
        gap_hr (float, optional): Minimum gap length to count. Defaults to 3 h.
        window_hr (float, optional): Proximity window. Defaults to 6 h.

    Returns:
        bool: True if the maximum is suspiciously close to missing data.
    """
    tmax = series.idxmax()
    win = pd.Timedelta(hours=window_hr)
    if tmax - series.index[0] <= win or series.index[-1] - tmax <= win:
        return True
    dt_hr = np.diff(series.index.astype("int64").to_numpy()) / 3.6e12
    for i in np.nonzero(dt_hr > gap_hr)[0]:
        # gap spans (index[i], index[i+1]); flag if it overlaps [tmax +/- win]
        if series.index[i] <= tmax + win and series.index[i + 1] >= tmax - win:
            return True
    return False


def _table_tag(method: str) -> str:
    """Cache version tag; bump the leading v1 if the algorithm changes."""
    return f"v1|cov{C.MIN_YEAR_COVERAGE}|ut{C.UTIDE_MIN_SAMPLES}|{method}"


def _resid_path(station: str, year: int, method: str) -> str:
    return os.path.join(C.ANNUAL_MAX_CACHE, f"resid_{station}_{year}_{method}.parquet")


def _table_path(station: str, start: int, end: int, method: str) -> str:
    return os.path.join(
        C.ANNUAL_MAX_CACHE, f"annmax_{station}_{start}_{end}_{method}.parquet"
    )


def year_residual(
    station: str, year: int, lat: float, method: str = "robust", refresh: bool = False
) -> Optional[pd.Series]:
    """De-tided hourly residual for one station-year (Parquet write-through).

    Applies the missing-data policy documented in the module docstring and
    returns ``None`` for skipped years.

    Args:
        station (str): CO-OPS station id.
        year (int): Calendar year.
        lat (float): Station latitude.
        method (str, optional): ``utide.solve`` method. Defaults to "robust".
        refresh (bool, optional): Recompute even if cached. Defaults to False.

    Returns:
        Optional[pd.Series]: Residual series, or None if the year is skipped.
    """
    fp = _resid_path(station, year, method)
    if not refresh and os.path.exists(fp):
        df = pd.read_parquet(fp)
        if not df.empty and df["tag"].iloc[0] == _table_tag(method):
            return pd.Series(df["resid"].values, index=pd.DatetimeIndex(df["time"]))
    wl = fetch_year_wl(station, year)
    coverage = wl.size / _hours_in_year(year)
    if wl.size < C.UTIDE_MIN_SAMPLES or coverage < C.MIN_YEAR_COVERAGE:
        print(f"  {station} {year}: skipped (n={wl.size}, coverage={coverage:.2f})")
        return None
    resid = detide_year(wl, lat, method=method)
    try:
        pd.DataFrame(
            {"time": resid.index, "resid": resid.values, "tag": _table_tag(method)}
        ).to_parquet(fp, index=False)
    except Exception as e:  # pragma: no cover - caching must never break the run
        print(f"  (warning: could not cache {station} {year} residual: {e})")
    return resid


def annual_maxima(
    station: str,
    start: int = C.AM_START_YEAR,
    end: int = C.AM_END_YEAR,
    method: str = "robust",
    refresh: bool = False,
) -> pd.DataFrame:
    """Annual maxima of the de-tided surge residual for ``start..end``.

    The completed table is cached as Parquet under ``ANNUAL_MAX_CACHE`` so
    reruns are free; per-year residuals are cached independently, so extending
    the year range only computes the new years.

    Args:
        station (str): CO-OPS station id, e.g. "8761724" (Grand Isle, LA).
        start (int, optional): First year (inclusive). Defaults to 1980.
        end (int, optional): Last year (inclusive). Defaults to 2025.
        method (str, optional): ``utide.solve`` method. Defaults to "robust".
        refresh (bool, optional): Recompute everything. Defaults to False.

    Returns:
        pd.DataFrame: One row per accepted year with columns
            ``station, name, year, ann_max_m, t_max, n_obs, coverage,
            max_at_gap_edge``.
    """
    tp = _table_path(station, start, end, method)
    if not refresh and os.path.exists(tp):
        df = pd.read_parquet(tp)
        if not df.empty and df["tag"].iloc[0] == _table_tag(method):
            return df.drop(columns="tag")
    meta = station_meta(station)
    if meta is None:
        raise ValueError(f"unknown CO-OPS station id {station!r}")
    print(
        f"annual maxima for {station} ({meta['name']}), {start}-{end}, "
        f"method={method}"
    )
    rows = []
    for year in range(start, end + 1):
        resid = year_residual(
            station, year, meta["lat"], method=method, refresh=refresh
        )
        if resid is None or resid.empty:
            continue
        rows.append(
            dict(
                station=station,
                name=meta["name"],
                year=year,
                ann_max_m=float(resid.max()),
                t_max=resid.idxmax(),
                n_obs=int(resid.size),
                coverage=round(resid.size / _hours_in_year(year), 3),
                max_at_gap_edge=max_at_gap_edge(resid),
            )
        )
        print(
            f"  {year}: max={rows[-1]['ann_max_m']:+.3f} m at "
            f"{rows[-1]['t_max']:%Y-%m-%d %H:%M} "
            f"(coverage={rows[-1]['coverage']:.2f}"
            f"{', NEAR GAP' if rows[-1]['max_at_gap_edge'] else ''})"
        )
    df = pd.DataFrame(rows)
    try:
        df.assign(tag=_table_tag(method)).to_parquet(tp, index=False)
        print(f"wrote {tp}")
    except Exception as e:  # pragma: no cover
        print(f"  (warning: could not cache annual maxima: {e})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--station",
        default="8761724",
        help="CO-OPS station id (default 8761724, Grand Isle LA)",
    )
    ap.add_argument("--start", type=int, default=C.AM_START_YEAR)
    ap.add_argument("--end", type=int, default=C.AM_END_YEAR)
    ap.add_argument(
        "--method",
        default="robust",
        choices=["robust", "ols"],
        help="utide.solve method (default robust)",
    )
    ap.add_argument(
        "--refresh", action="store_true", help="recompute instead of reading the caches"
    )
    a = ap.parse_args()
    df = annual_maxima(a.station, a.start, a.end, method=a.method, refresh=a.refresh)
    if df.empty:
        print("no usable years")
        return
    print(
        f"\n{len(df)} usable years of {a.end - a.start + 1}; "
        f"max residual {df.ann_max_m.max():.2f} m in "
        f"{int(df.loc[df.ann_max_m.idxmax(), 'year'])}; "
        f"{int(df.max_at_gap_edge.sum())} year(s) flagged max-near-gap"
    )


if __name__ == "__main__":
    main()
