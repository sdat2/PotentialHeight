"""Unit + regression tests for the comp (tide-gauge validation) module.

Covers the pure, network-free functions in comp.validate (skill metrics, time-series
alignment, the valid/clean gating, the LaTeX-table generator) and pins the headline
numbers from a completed sweep so they cannot silently drift. The regression test is
skipped when the summary CSV is absent (e.g. a fresh checkout with no cached data), so
the unit tests still run in CI without network or the multi-GB storm archive.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp import constants as C
from comp.validate import (
    add_flags,
    bootstrap_ci,
    classify_setting,
    latex_table,
    metrics,
    timeseries_skill,
    within_storm_r,
    _nearest_wet,
)
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# timeseries_skill: interpolation, overlap gate, flat-series NaN, lag collapse
# --------------------------------------------------------------------------- #
def _hours(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def test_timeseries_skill_identical_series_is_perfect():
    idx = _hours(48)
    s = pd.Series(np.sin(np.arange(48) / 3.0), index=idx)
    r, rmse, n = timeseries_skill(s, s)
    assert r == pytest.approx(1.0, abs=1e-9)
    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert n == 48


def test_timeseries_skill_interpolates_coarse_sim_onto_hourly_obs():
    # obs hourly on a straight line; sim sampled every 2 h on the SAME line.
    obs = pd.Series(np.arange(48, dtype=float), index=_hours(48))
    sim = pd.Series(np.arange(0, 48, 2, dtype=float), index=_hours(48)[::2])
    r, rmse, n = timeseries_skill(sim, obs)
    # linear interpolation of a linear sim recovers obs exactly
    assert r == pytest.approx(1.0, abs=1e-9)
    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_timeseries_skill_short_overlap_returns_nan():
    obs = pd.Series(np.arange(48.0), index=_hours(48, "2020-01-01"))
    # sim window barely overlaps obs (fewer than TS_MIN_OVERLAP hours)
    sim = pd.Series(np.arange(5.0), index=_hours(5, "2020-01-02 23:00"))
    r, rmse, n = timeseries_skill(sim, obs)
    assert np.isnan(r)
    assert n < C.TS_MIN_OVERLAP


def test_timeseries_skill_flat_series_gives_nan_r_finite_rmse():
    idx = _hours(48)
    obs = pd.Series(np.sin(np.arange(48) / 3.0), index=idx)
    flat = pd.Series(np.zeros(48), index=idx)
    r, rmse, n = timeseries_skill(flat, obs)
    assert np.isnan(r)            # correlation undefined for a constant series
    assert np.isfinite(rmse)      # RMSE still well defined
    assert rmse > 0


def test_timeseries_skill_degrades_under_temporal_lag():
    # A peaked hydrograph: lag-0 should score higher than a 1-day shift.
    idx = _hours(120)
    t = np.arange(120)
    obs = pd.Series(np.exp(-((t - 60) ** 2) / (2 * 8 ** 2)), index=idx)
    r0, _, _ = timeseries_skill(obs, obs)
    shifted = pd.Series(obs.values, index=idx + pd.Timedelta("24h"))
    rlag, _, _ = timeseries_skill(shifted, obs)
    assert r0 > rlag


# --------------------------------------------------------------------------- #
# within_storm_r: removes between-storm magnitude offset
# --------------------------------------------------------------------------- #
def test_within_storm_r_removes_between_storm_offset():
    # Two storms, each with a perfect within-storm sim==obs relation but very
    # different magnitudes. Pooled r and within-storm r should both be ~1.
    a = pd.DataFrame({"storm": "A", "obs_peak": [0.4, 0.6, 0.8], "sim_peak": [0.4, 0.6, 0.8]})
    b = pd.DataFrame({"storm": "B", "obs_peak": [3.0, 3.2, 3.4], "sim_peak": [3.0, 3.2, 3.4]})
    df = pd.concat([a, b], ignore_index=True)
    assert within_storm_r(df) == pytest.approx(1.0, abs=1e-9)


def test_within_storm_r_kills_pure_between_storm_signal():
    # Within each storm there is NO spatial signal (sim constant), only a
    # between-storm magnitude difference. Pooled r would be high; within-storm ~0/NaN.
    a = pd.DataFrame({"storm": "A", "obs_peak": [0.5, 0.5, 0.5], "sim_peak": [0.5, 0.5, 0.5]})
    b = pd.DataFrame({"storm": "B", "obs_peak": [3.0, 3.0, 3.0], "sim_peak": [3.0, 3.0, 3.0]})
    df = pd.concat([a, b], ignore_index=True)
    # both within-storm series are flat -> correlation undefined
    assert np.isnan(within_storm_r(df))


# --------------------------------------------------------------------------- #
# bootstrap_ci: determinism + brackets the point estimate
# --------------------------------------------------------------------------- #
def _toy_clean(n=60, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(0.5, 3.0, n)
    sim = obs - 0.3 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"obs_peak": obs, "sim_peak": sim})


def test_bootstrap_ci_is_deterministic_under_seed():
    d = _toy_clean()
    a = bootstrap_ci(d, n=500, seed=0)
    b = bootstrap_ci(d, n=500, seed=0)
    assert a == b


def test_bootstrap_ci_brackets_point_estimate():
    d = _toy_clean()
    bias, rmse, r = metrics(d)
    ci = bootstrap_ci(d, n=2000, seed=0)
    assert ci["bias"][0] <= bias <= ci["bias"][1]
    assert ci["rmse"][0] <= rmse <= ci["rmse"][1]
    assert ci["r"][0] <= r <= ci["r"][1]


# --------------------------------------------------------------------------- #
# metrics, classify_setting
# --------------------------------------------------------------------------- #
def test_metrics_known_values():
    d = pd.DataFrame({"obs_peak": [1.0, 2.0, 3.0], "sim_peak": [1.5, 2.5, 3.5]})
    bias, rmse, r = metrics(d)
    assert bias == pytest.approx(0.5)
    assert rmse == pytest.approx(0.5)
    assert r == pytest.approx(1.0)


def test_metrics_too_few_points_is_nan():
    d = pd.DataFrame({"obs_peak": [1.0], "sim_peak": [1.0]})
    assert all(np.isnan(x) for x in metrics(d))


@pytest.mark.parametrize("name,expected", [
    ("Galveston Bay", "semi-enclosed"),
    ("Calcasieu River", "semi-enclosed"),
    ("Dauphin Island", "open-coast"),
    ("Pilots Station East, S.W. Pass", "open-coast"),
    ("Shell Beach", "open-coast"),
])
def test_classify_setting(name, expected):
    assert classify_setting(name) == expected


# --------------------------------------------------------------------------- #
# add_flags: valid vs clean gating, KNOWN_FAILED, timing gate
# --------------------------------------------------------------------------- #
def _row(storm, sid, obs_peak, peak_dt_hr):
    return dict(storm=storm, sid=sid, obs_peak=obs_peak, peak_dt_hr=peak_dt_hr)


def test_add_flags_valid_and_clean_gating():
    rows = [
        _row("Laura 2020", "111", 1.0, 1.0),    # valid + clean
        _row("Laura 2020", "222", 0.2, 1.0),    # below MIN_OBS_PEAK -> not valid/clean
        _row("Laura 2020", "333", 1.0, 9.0),    # valid but timing > gate -> not clean
    ]
    df = add_flags(pd.DataFrame(rows))
    by = {r.sid: r for _, r in df.iterrows()}
    assert by["111"].valid and by["111"].clean
    assert (not by["222"].valid) and (not by["222"].clean)
    assert by["333"].valid and (not by["333"].clean)


def test_add_flags_known_failed_excluded():
    (storm, sid), = list(C.KNOWN_FAILED)[:1]
    df = add_flags(pd.DataFrame([_row(storm, sid, 3.0, 0.0)]))
    assert bool(df.iloc[0].failed) is True
    assert bool(df.iloc[0].valid) is False   # failed gauges are never valid/clean
    assert bool(df.iloc[0].clean) is False


# --------------------------------------------------------------------------- #
# latex_table: column count, dagger on poor-surge events, totals row
# --------------------------------------------------------------------------- #
def test_latex_table_structure(tmp_path):
    poor = sorted(C.POOR_SURGE_EVENTS)[0]          # e.g. "Harvey 2017"
    other = next(s for s in C.STORMS if s not in C.POOR_SURGE_EVENTS)
    rows = []
    for storm in (other, poor):
        for k in range(4):
            rows.append(dict(storm=storm, sid=str(k), obs_peak=1.0 + 0.1 * k,
                             sim_peak=1.0 + 0.1 * k, peak_dt_hr=0.0,
                             ts_r=0.7, setting="open-coast"))
    df = add_flags(pd.DataFrame(rows))
    out = tmp_path / "tab.tex"
    latex_table(df, str(out))
    text = out.read_text()
    body = [ln for ln in text.splitlines() if ln.strip().endswith(r"\\")]
    assert all(ln.count("&") == 5 for ln in body)             # 6 columns
    assert r"$^{\dagger}$" in text                              # dagger on poor event
    assert any("All storms" in ln for ln in body)              # totals row present
    assert r"\begin{tabular}{lrrrrr}" in text


# --------------------------------------------------------------------------- #
# _nearest_wet: wet-min logic + max-distance cutoff
# --------------------------------------------------------------------------- #
def test_nearest_wet_skips_drying_node_for_deeper_wet_node():
    # node 0 is closest but dries (min WD below WET_MIN_M); node 1 is slightly
    # farther but always wet -> should be selected.
    pts = np.array([[0.00, 0.0], [0.01, 0.0]])
    tree = cKDTree(pts)
    WD = np.array([[C.WET_MIN_M - 0.1, C.WET_MIN_M + 1.0],   # t0
                   [C.WET_MIN_M - 0.2, C.WET_MIN_M + 1.0]])  # t1
    idx, dist = _nearest_wet(tree, WD, lon=0.0, lat=0.0)
    assert idx == 1


def test_nearest_wet_returns_none_when_all_too_far():
    pts = np.array([[10.0, 10.0]])                # far outside MAX_NODE_DEG
    tree = cKDTree(pts)
    WD = np.array([[5.0], [5.0]])
    idx, dist = _nearest_wet(tree, WD, lon=0.0, lat=0.0)
    assert idx is None


# --------------------------------------------------------------------------- #
# REGRESSION: pin the headline numbers from the committed sweep
# --------------------------------------------------------------------------- #
SUMMARY_CSV = os.path.join(C.OUT_PATH, "val_summary.csv")


@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_regression_headline_numbers():
    df = pd.read_csv(SUMMARY_CSV)
    df["sid"] = df["sid"].astype(str)
    clean = df[df.clean]
    # Pins regenerated 2026-07-05 after fixing the utide date2num bug in
    # comp/coops.py (the old sweep's "de-tided" residuals still contained the
    # tide; the fix improved bias -0.30 -> -0.20, RMSE 0.54 -> 0.50, and
    # median time-series r 0.65 -> 0.80).
    bias, rmse, r = metrics(clean)
    assert len(clean) == 126
    assert bias == pytest.approx(-0.20, abs=0.02)
    assert rmse == pytest.approx(0.50, abs=0.02)
    assert r == pytest.approx(0.885, abs=0.01)
    assert within_storm_r(clean) == pytest.approx(0.885, abs=0.02)
    assert clean.ts_r.median() == pytest.approx(0.80, abs=0.03)


@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_regression_per_storm_counts():
    df = pd.read_csv(SUMMARY_CSV)
    counts = df[df.clean].groupby("storm").size().to_dict()
    # a few anchor storms; full set is 14 storms / 126 clean pairs
    # (regenerated 2026-07-05 with the fixed utide de-tiding)
    assert counts["Delta 2020"] == 21
    assert counts["Nicholas 2021"] == 10
    assert counts["Katrina 2005"] == 4


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS: encode "the signal is real" as guarded regression tests.
# (peak-level only -> needs the summary CSV but not the multi-GB netCDFs.)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_global_permutation_separates_signal_from_null():
    from comp.nulltest import perm_global, load_clean
    res = perm_global(load_clean(), n=1000, seed=0)
    # the real correlation must sit far above every shuffled (null) correlation
    assert res["observed"] > 0.85
    assert res["null_max"] < 0.5
    assert res["observed"] - res["null_max"] > 0.4
    assert res["p"] < 1e-3


@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_within_storm_permutation_shows_real_spatial_skill():
    from comp.nulltest import perm_within_storm, load_clean
    res = perm_within_storm(load_clean(), n=1000, seed=0)
    # spatial skill (storm means removed) beats the within-storm-shuffled null
    assert res["observed_pooled"] > res["null_max"]
    assert res["p"] < 1e-3


@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_cross_storm_pairing_collapses_skill():
    from comp.nulltest import cross_storm, load_clean
    res = cross_storm(load_clean(), seed=0)
    # pairing a gauge's sim with a DIFFERENT storm's obs must lose most of the skill
    assert res["r"] < 0.4
    assert res["observed"] - res["r"] > 0.45


# --------------------------------------------------------------------------- #
# Additional edge-case coverage (boundaries, empty/flat, ordering, invariance)
# --------------------------------------------------------------------------- #
def test_timeseries_skill_empty_inputs():
    empty = pd.Series(dtype=float)
    s = pd.Series(np.arange(48.0), index=_hours(48))
    r, rmse, n = timeseries_skill(empty, s)
    assert np.isnan(r) and np.isnan(rmse) and n == 0
    r, rmse, n = timeseries_skill(s, empty)
    assert np.isnan(r) and np.isnan(rmse) and n == 0


def test_timeseries_skill_overlap_boundary_is_24():
    # exactly TS_MIN_OVERLAP hourly obs inside the sim window -> finite;
    # one fewer -> NaN. Pins the >=/< boundary.
    sim = pd.Series(np.linspace(0, 1, 48), index=_hours(48))
    obs_ok = pd.Series(np.linspace(0, 1, C.TS_MIN_OVERLAP), index=_hours(C.TS_MIN_OVERLAP))
    obs_lo = pd.Series(np.linspace(0, 1, C.TS_MIN_OVERLAP - 1),
                       index=_hours(C.TS_MIN_OVERLAP - 1))
    assert np.isfinite(timeseries_skill(sim, obs_ok)[0])
    assert np.isnan(timeseries_skill(sim, obs_lo)[0])


def test_timeseries_skill_flat_obs_gives_nan_r():
    idx = _hours(48)
    sim = pd.Series(np.sin(np.arange(48) / 3.0), index=idx)
    flat_obs = pd.Series(np.ones(48), index=idx)
    r, rmse, n = timeseries_skill(sim, flat_obs)
    assert np.isnan(r) and np.isfinite(rmse)


def test_bootstrap_ci_seed_sensitivity():
    d = _toy_clean()
    assert bootstrap_ci(d, n=500, seed=0)["bias"] != bootstrap_ci(d, n=500, seed=1)["bias"]


def test_within_storm_r_single_storm_equals_plain_corr():
    d = pd.DataFrame({"storm": "A", "obs_peak": [0.4, 0.9, 1.4, 2.0],
                      "sim_peak": [0.3, 1.0, 1.2, 2.1]})
    plain = np.corrcoef(d.obs_peak, d.sim_peak)[0, 1]
    assert within_storm_r(d) == pytest.approx(plain, abs=1e-9)


def test_within_storm_r_invariant_to_between_storm_offset():
    a = pd.DataFrame({"storm": "A", "obs_peak": [0.4, 0.6, 0.9], "sim_peak": [0.5, 0.55, 1.0]})
    b = a.assign(storm="B", obs_peak=a.obs_peak + 5.0, sim_peak=a.sim_peak + 5.0)
    base = within_storm_r(pd.concat([a, a.assign(storm="B")], ignore_index=True))
    offset = within_storm_r(pd.concat([a, b], ignore_index=True))
    assert base == pytest.approx(offset, abs=1e-9)   # storm-mean removal kills the offset


def test_add_flags_boundary_thresholds():
    rows = [
        _row("Laura 2020", "1", C.MIN_OBS_PEAK_M, 0.0),         # obs == 0.4 -> valid
        _row("Laura 2020", "2", C.MIN_OBS_PEAK_M - 0.01, 0.0),  # 0.39 -> not valid
        _row("Laura 2020", "3", 1.0, C.MAX_TIMING_HR),          # |dt| == 6 -> clean
        _row("Laura 2020", "4", 1.0, -C.MAX_TIMING_HR),         # -6 -> clean (abs)
        _row("Laura 2020", "5", 1.0, C.MAX_TIMING_HR + 0.1),    # 6.1 -> not clean
    ]
    df = add_flags(pd.DataFrame(rows)).set_index("sid")
    assert df.loc["1"].valid and not df.loc["2"].valid
    assert df.loc["3"].clean and df.loc["4"].clean and not df.loc["5"].clean


def test_nearest_wet_strict_threshold():
    pts = np.array([[0.0, 0.0]])
    tree = cKDTree(pts)
    at = np.array([[C.WET_MIN_M], [C.WET_MIN_M]])          # min == 0.3 -> rejected (strict >)
    above = np.array([[C.WET_MIN_M + 0.01], [C.WET_MIN_M + 0.01]])
    assert _nearest_wet(tree, at, 0.0, 0.0)[0] is None
    assert _nearest_wet(tree, above, 0.0, 0.0)[0] == 0


def test_latex_table_row_order_and_name_reformat(tmp_path):
    # two storms out of chronological order in the df; table must follow C.STORMS order.
    late = list(C.STORMS)[-1]      # Idalia 2023
    early = list(C.STORMS)[0]      # Katrina 2005
    rows = []
    for storm in (late, early):    # deliberately reversed
        for k in range(3):
            rows.append(dict(storm=storm, sid=str(k), obs_peak=1.0 + 0.1 * k,
                             sim_peak=1.0 + 0.1 * k, peak_dt_hr=0.0, ts_r=0.7,
                             setting="open-coast"))
    df = add_flags(pd.DataFrame(rows))
    out = tmp_path / "t.tex"
    latex_table(df, str(out))
    text = out.read_text()
    i_early = text.index(early.replace(" ", " (") + ")")
    i_late = text.index(late.replace(" ", " (") + ")")
    assert i_early < i_late                                # chronological, not df order
    assert "Katrina (2005)" in text                        # "Name YYYY" -> "Name (YYYY)"


@pytest.mark.skipif(not os.path.exists(SUMMARY_CSV),
                    reason="val_summary.csv not present (no cached sweep)")
def test_regression_population_and_methods():
    df = pd.read_csv(SUMMARY_CSV)
    df["sid"] = df["sid"].astype(str)
    assert len(df) == 432
    assert df.storm.nunique() == 14
    # regenerated 2026-07-05 with the fixed utide de-tiding: real de-tiding
    # shifts which (storm, gauge) pairs pass the validity/clean gates
    assert df.valid.sum() == 279
    assert df.clean.sum() == 126
    assert df.failed.sum() == 2
    methods = dict(df.method.value_counts())
    assert methods.get("utide") == 410 and methods.get("pred") == 22


# --------------------------------------------------------------------------- #
# Per-storm time-series cache (Parquet): exact round-trip + parameter staleness.
# Network-free -- uses synthetic ragged (sim, obs) series in a tmp cache dir.
# --------------------------------------------------------------------------- #
def test_series_cache_roundtrip_and_staleness(tmp_path, monkeypatch):
    import comp.validate as cv

    monkeypatch.setattr(C, "TS_CACHE", str(tmp_path))   # isolate from the real cache
    h = pd.date_range("2021-08-27", periods=48, freq="h")
    series = {                                          # sim 2-hourly, obs hourly (ragged)
        "Gauge A": (pd.Series(np.arange(24.0), index=h[::2]),
                    pd.Series(np.sin(np.arange(48) / 3.0), index=h)),
        "Gauge B, S.W": (pd.Series(np.cos(np.arange(24.0)), index=h[::2]),
                         pd.Series(np.arange(48.0), index=h)),
    }
    cv._save_series_cache("Test 2021", series)
    assert cv._series_cache_path("Test 2021").endswith(".parquet")

    loaded = cv._load_series_cache("Test 2021")
    assert loaded is not None and set(loaded) == set(series)
    for name in series:
        for orig, got in zip(series[name], loaded[name]):
            assert str(got.index.dtype).startswith("datetime64")  # datetime preserved
            assert got.index.equals(orig.index)
            np.testing.assert_allclose(got.values, orig.values)   # values exact

    monkeypatch.setattr(cv, "_ts_cache_tag", lambda: "DIFFERENT-PARAMS")
    assert cv._load_series_cache("Test 2021") is None             # stale tag -> recompute
    assert cv._load_series_cache("Never Cached 1999") is None     # missing -> recompute
