"""Network-free tests for the gauge annual-maximum EVT pipeline.

Two layers are covered without touching the CO-OPS API:

1. ``comp.annual_max``: a synthetic hourly water-level series (known harmonic
   tide + trend + one dominant storm-surge bump per year) is pushed through
   the de-tide + annual-maximum pipeline with the fetch monkeypatched. The
   recovered annual maxima must match the injected storm amplitudes, the
   coverage gate must skip broken years, and the Parquet caches must make
   reruns fetch-free.
2. ``worst.gauge_fit``: synthetic bounded-GEV annual maxima with a KNOWN upper
   bound are fitted with case I (bound known) and case II (unbounded); case I
   must reproduce the bound exactly and both must recover the truth within
   sampling tolerance. Bootstrap envelopes for case I must respect the bound.

A true-network smoke test at Grand Isle is guarded behind the
``WORSTSURGE_NETWORK_TESTS=1`` environment variable.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genextreme

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import comp.annual_max as am
from comp import constants as C
from worst.gauge_fit import (
    POTENTIAL_HEIGHT_M,
    bootstrap_bands,
    fit_cases,
    gringorten_rp,
    return_level,
)

# --------------------------------------------------------------------------- #
# Synthetic gauge: harmonic tide + trend + noise + one storm bump per year
# --------------------------------------------------------------------------- #
M2_HR = 12.4206012  # M2 period [h]
K1_HR = 23.9344697  # K1 period [h]
STATION = "0000000"
LAT = 29.263  # Grand Isle latitude (utide nodal corrections)
STORM_PEAKS = {2001: 0.9, 2002: 1.6, 2003: 0.7}  # injected surge maxima [m]


def _synth_year(year: int, storm_peak: float, seed: int) -> pd.Series:
    """One calendar year of hourly synthetic water level."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    t = np.arange(idx.size, dtype=float)
    tide = 0.25 * np.cos(2 * np.pi * t / M2_HR + 0.3) + 0.15 * np.cos(
        2 * np.pi * t / K1_HR + 1.1
    )
    trend = 2e-6 * t  # small within-year drift
    # storm bump centred mid-September, 12 h e-folding width
    peak_hr = float((pd.Timestamp(f"{year}-09-15") - idx[0]).total_seconds() / 3600)
    surge = storm_peak * np.exp(-((t - peak_hr) ** 2) / (2 * 12.0**2))
    noise = rng.normal(0.0, 0.015, idx.size)
    return pd.Series(tide + trend + surge + noise, index=idx)


@pytest.fixture()
def synth_pipeline(monkeypatch, tmp_path):
    """Route comp.annual_max at the synthetic gauge and an isolated cache."""
    series = {y: _synth_year(y, p, seed=y) for y, p in STORM_PEAKS.items()}
    calls = {"fetch": 0}

    def fake_fetch(station, year, **kw):
        calls["fetch"] += 1
        return series.get(year, pd.Series(dtype=float)).copy()

    monkeypatch.setattr(am, "fetch_year_wl", fake_fetch)
    monkeypatch.setattr(
        am,
        "station_meta",
        lambda s: dict(id=s, name="Synthetic Isle", lat=LAT, lon=-90.0),
    )
    monkeypatch.setattr(C, "ANNUAL_MAX_CACHE", str(tmp_path))
    return series, calls


def test_pipeline_recovers_storm_annual_maxima(synth_pipeline):
    series, calls = synth_pipeline
    df = am.annual_maxima(STATION, 2001, 2003, method="ols")
    assert list(df.year) == sorted(STORM_PEAKS)
    for _, row in df.iterrows():
        truth = STORM_PEAKS[int(row.year)]
        # the de-tided residual max must recover the injected storm amplitude
        assert row.ann_max_m == pytest.approx(truth, abs=0.12), int(row.year)
        # and it must find it at the right time (mid-September bump)
        assert (
            abs((row.t_max - pd.Timestamp(f"{int(row.year)}-09-15")).total_seconds())
            < 36 * 3600
        )
        assert not row.max_at_gap_edge
    # the harmonic fit really removed the tide: residual variance collapses
    # to noise + storm bump, far below the raw (tidal) variance
    resid = am.year_residual(STATION, 2003, LAT, method="ols")  # from cache
    assert series[2003].std() > 0.15  # tide dominates the raw std
    assert resid.std() < 0.06  # residual is noise + one bump

    # second call is served from the Parquet cache: no further fetches
    n_fetch = calls["fetch"]
    df2 = am.annual_maxima(STATION, 2001, 2003, method="ols")
    assert calls["fetch"] == n_fetch
    assert list(df2.year) == list(df.year)
    np.testing.assert_allclose(df2.ann_max_m.values, df.ann_max_m.values)


def test_pipeline_skips_low_coverage_year(synth_pipeline, monkeypatch):
    series, _ = synth_pipeline
    # break 2002: keep only the first 40% of the year (< MIN_YEAR_COVERAGE)
    broken = series[2002].iloc[: int(0.4 * len(series[2002]))]

    def fetch_with_gap(station, year, **kw):
        if year == 2002:
            return broken.copy()
        return series.get(year, pd.Series(dtype=float)).copy()

    monkeypatch.setattr(am, "fetch_year_wl", fetch_with_gap)
    df = am.annual_maxima(STATION, 2001, 2003, method="ols")
    assert list(df.year) == [2001, 2003]  # 2002 dropped by coverage gate


def test_max_at_gap_edge_flags_truncated_peak():
    idx = pd.date_range("2001-06-01", periods=2000, freq="h")
    t = np.arange(idx.size, dtype=float)
    s = pd.Series(np.exp(-((t - 1000) ** 2) / (2 * 12.0**2)), index=idx)
    assert not am.max_at_gap_edge(s)
    # gauge dies 2 h after the peak for a day: max now borders a gap
    dead = s.drop(s.index[1002:1026])
    assert am.max_at_gap_edge(dead)


# --------------------------------------------------------------------------- #
# GEV fits: case I respects the known bound, both recover the truth
# --------------------------------------------------------------------------- #
BETA_TRUE, GAMMA_TRUE, Z_STAR = 0.5, -0.25, 3.0
ALPHA_TRUE = Z_STAR + BETA_TRUE / GAMMA_TRUE  # Coles: z* = alpha - beta/gamma
N_YEARS = 60


@pytest.fixture(scope="module")
def gev_sample() -> np.ndarray:
    rng = np.random.default_rng(7)
    data = genextreme.rvs(
        c=-GAMMA_TRUE, loc=ALPHA_TRUE, scale=BETA_TRUE, size=N_YEARS, random_state=rng
    )
    assert data.max() < Z_STAR
    return data


def test_fit_cases_recover_truth_and_bound(gev_sample):
    fits = fit_cases(gev_sample, Z_STAR)
    a1, b1, g1 = fits["I"]
    a2, b2, g2 = fits["II"]
    # case I: the fitted endpoint IS the imposed bound (by construction)
    assert a1 - b1 / g1 == pytest.approx(Z_STAR, abs=1e-6)
    # both recover the truth within sampling tolerance (n=60)
    for name, (b, g) in {"I": (b1, g1), "II": (b2, g2)}.items():
        assert b == pytest.approx(BETA_TRUE, abs=0.2), name
        assert g == pytest.approx(GAMMA_TRUE, abs=0.2), name
    # case I return levels can never exceed the bound, even at 1-in-1e6 yr
    assert float(return_level(a1, b1, g1, 1e6)) <= Z_STAR + 1e-9
    # both 100-yr RVs are near the true one
    rv100_true = float(return_level(ALPHA_TRUE, BETA_TRUE, GAMMA_TRUE, 100.0))
    assert float(return_level(a1, b1, g1, 100.0)) == pytest.approx(rv100_true, abs=0.25)
    assert float(return_level(a2, b2, g2, 100.0)) == pytest.approx(rv100_true, abs=0.4)


def test_degenerate_bound_gives_nan(gev_sample):
    fits = fit_cases(gev_sample, float(gev_sample.max()) - 0.1)
    assert all(np.isnan(v) for v in fits["I"])  # bound below sample max
    assert all(np.isfinite(v) for v in fits["II"])  # unbounded fit unaffected


def test_bootstrap_bands_finite_and_bounded(gev_sample):
    fits = fit_cases(gev_sample, Z_STAR)
    rps = np.array([10.0, 100.0, 500.0])
    b1 = bootstrap_bands(
        fits["I"], "I", len(gev_sample), z_star=Z_STAR, n_boot=40, rps=rps, seed=1
    )
    b2 = bootstrap_bands(fits["II"], "II", len(gev_sample), n_boot=40, rps=rps, seed=1)
    for b in (b1, b2):
        assert np.isfinite(b["lo"]).all() and np.isfinite(b["hi"]).all()
        assert (b["hi"] >= b["lo"]).all()
    # case I: even the 95th-percentile envelope respects the bound
    assert (b1["hi"] <= Z_STAR + 1e-9).all()
    # case II envelope is wider at long return periods (the paper's point)
    assert (b2["hi"] - b2["lo"])[-1] > (b1["hi"] - b1["lo"])[-1]


def test_gringorten_positions_monotone_and_span():
    rp = gringorten_rp(45)
    assert rp.shape == (45,)
    assert (np.diff(rp) > 0).all()
    assert rp[0] > 1.0  # smallest maximum: ~1 yr
    assert 50 < rp[-1] < 120  # largest of 45 years: ~80 yr


def test_potential_heights_present_for_target_gauges():
    # the two demo gauges must have bounds, sourced from the paper's BO table
    assert POTENTIAL_HEIGHT_M["8761724"] == pytest.approx(5.886)
    assert POTENTIAL_HEIGHT_M["8735180"] == pytest.approx(3.567)
    assert all(v > 0 for v in POTENTIAL_HEIGHT_M.values())


# --------------------------------------------------------------------------- #
# Guarded true-network smoke test (off by default; hits the CO-OPS API)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("WORSTSURGE_NETWORK_TESTS") != "1",
    reason="set WORSTSURGE_NETWORK_TESTS=1 to hit the CO-OPS API",
)
def test_network_fetch_grand_isle_year():
    wl = am.fetch_year_wl("8761724", 2020)
    assert wl.size > 8000  # near-complete hourly year
    assert wl.index.is_monotonic_increasing
