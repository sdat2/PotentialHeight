"""Unit + regression tests for the adforce.eval (tide-gauge validation) module.

Covers the pure, network-free functions in adforce.eval.validate (skill metrics, time-series
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

from adforce.eval import constants as C
from adforce.eval.validate import (
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
    assert np.isnan(r)  # correlation undefined for a constant series
    assert np.isfinite(rmse)  # RMSE still well defined
    assert rmse > 0


def test_timeseries_skill_degrades_under_temporal_lag():
    # A peaked hydrograph: lag-0 should score higher than a 1-day shift.
    idx = _hours(120)
    t = np.arange(120)
    obs = pd.Series(np.exp(-((t - 60) ** 2) / (2 * 8**2)), index=idx)
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
    a = pd.DataFrame(
        {"storm": "A", "obs_peak": [0.4, 0.6, 0.8], "sim_peak": [0.4, 0.6, 0.8]}
    )
    b = pd.DataFrame(
        {"storm": "B", "obs_peak": [3.0, 3.2, 3.4], "sim_peak": [3.0, 3.2, 3.4]}
    )
    df = pd.concat([a, b], ignore_index=True)
    assert within_storm_r(df) == pytest.approx(1.0, abs=1e-9)


def test_within_storm_r_kills_pure_between_storm_signal():
    # Within each storm there is NO spatial signal (sim constant), only a
    # between-storm magnitude difference. Pooled r would be high; within-storm ~0/NaN.
    a = pd.DataFrame(
        {"storm": "A", "obs_peak": [0.5, 0.5, 0.5], "sim_peak": [0.5, 0.5, 0.5]}
    )
    b = pd.DataFrame(
        {"storm": "B", "obs_peak": [3.0, 3.0, 3.0], "sim_peak": [3.0, 3.0, 3.0]}
    )
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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Galveston Bay", "semi-enclosed"),
        ("Calcasieu River", "semi-enclosed"),
        ("Dauphin Island", "open-coast"),
        ("Pilots Station East, S.W. Pass", "open-coast"),
        ("Shell Beach", "open-coast"),
    ],
)
def test_classify_setting(name, expected):
    assert classify_setting(name) == expected


# --------------------------------------------------------------------------- #
# add_flags: valid vs clean gating, KNOWN_FAILED, timing gate
# --------------------------------------------------------------------------- #
def _row(storm, sid, obs_peak, peak_dt_hr):
    return dict(storm=storm, sid=sid, obs_peak=obs_peak, peak_dt_hr=peak_dt_hr)


def test_add_flags_valid_and_clean_gating():
    rows = [
        _row("Laura 2020", "111", 1.0, 1.0),  # valid + clean
        _row("Laura 2020", "222", 0.2, 1.0),  # below MIN_OBS_PEAK -> not valid/clean
        _row("Laura 2020", "333", 1.0, 9.0),  # valid but timing > gate -> not clean
    ]
    df = add_flags(pd.DataFrame(rows))
    by = {r.sid: r for _, r in df.iterrows()}
    assert by["111"].valid and by["111"].clean
    assert (not by["222"].valid) and (not by["222"].clean)
    assert by["333"].valid and (not by["333"].clean)


def test_add_flags_known_failed_excluded():
    ((storm, sid),) = list(C.KNOWN_FAILED)[:1]
    df = add_flags(pd.DataFrame([_row(storm, sid, 3.0, 0.0)]))
    assert bool(df.iloc[0].failed) is True
    assert bool(df.iloc[0].valid) is False  # failed gauges are never valid/clean
    assert bool(df.iloc[0].clean) is False


# --------------------------------------------------------------------------- #
# latex_table: column count, dagger on poor-surge events, totals row
# --------------------------------------------------------------------------- #
def test_latex_table_structure(tmp_path):
    poor = sorted(C.POOR_SURGE_EVENTS)[0]  # e.g. "Harvey 2017"
    other = next(s for s in C.STORMS if s not in C.POOR_SURGE_EVENTS)
    rows = []
    for storm in (other, poor):
        for k in range(4):
            rows.append(
                dict(
                    storm=storm,
                    sid=str(k),
                    obs_peak=1.0 + 0.1 * k,
                    sim_peak=1.0 + 0.1 * k,
                    peak_dt_hr=0.0,
                    ts_r=0.7,
                    setting="open-coast",
                )
            )
    df = add_flags(pd.DataFrame(rows))
    out = tmp_path / "tab.tex"
    latex_table(df, str(out))
    text = out.read_text()
    body = [ln for ln in text.splitlines() if ln.strip().endswith(r"\\")]
    assert all(ln.count("&") == 5 for ln in body)  # 6 columns
    assert r"$^{\dagger}$" in text  # dagger on poor event
    assert any("All storms" in ln for ln in body)  # totals row present
    assert r"\begin{tabular}{lrrrrr}" in text


# --------------------------------------------------------------------------- #
# _nearest_wet: wet-min logic + max-distance cutoff
# --------------------------------------------------------------------------- #
def test_nearest_wet_skips_drying_node_for_deeper_wet_node():
    # node 0 is closest but dries (min WD below WET_MIN_M); node 1 is slightly
    # farther but always wet -> should be selected.
    pts = np.array([[0.00, 0.0], [0.01, 0.0]])
    tree = cKDTree(pts)
    WD = np.array(
        [
            [C.WET_MIN_M - 0.1, C.WET_MIN_M + 1.0],  # t0
            [C.WET_MIN_M - 0.2, C.WET_MIN_M + 1.0],
        ]
    )  # t1
    idx, dist = _nearest_wet(tree, WD, lon=0.0, lat=0.0)
    assert idx == 1


def test_nearest_wet_returns_none_when_all_too_far():
    pts = np.array([[10.0, 10.0]])  # far outside MAX_NODE_DEG
    tree = cKDTree(pts)
    WD = np.array([[5.0], [5.0]])
    idx, dist = _nearest_wet(tree, WD, lon=0.0, lat=0.0)
    assert idx is None


# --------------------------------------------------------------------------- #
# REGRESSION: pin the headline numbers from the committed sweep
# --------------------------------------------------------------------------- #
SUMMARY_CSV = os.path.join(C.OUT_PATH, "val_summary.csv")


@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
def test_regression_headline_numbers():
    df = pd.read_csv(SUMMARY_CSV)
    df["sid"] = df["sid"].astype(str)
    clean = df[df.clean]
    # Pins regenerated 2026-07-26 for the three-city extension: 5 Florida
    # storms added (Frances/Jeanne/Matthew/Irma/Nicole) for the Miami study
    # region. The original 14 Gulf storms' pairs are BYTE-IDENTICAL to the
    # 2026-07-05 sweep (126 clean; per-storm anchors below unchanged); the
    # +20 Florida clean pairs run cold on the wave-exposed Atlantic coast
    # (regional bias -0.36 m), moving the pooled numbers from
    # bias -0.20 -> -0.22, r 0.885 -> 0.87.
    bias, rmse, r = metrics(clean)
    assert len(clean) == 146
    assert bias == pytest.approx(-0.22, abs=0.02)
    assert rmse == pytest.approx(0.50, abs=0.02)
    assert r == pytest.approx(0.871, abs=0.01)
    assert within_storm_r(clean) == pytest.approx(0.871, abs=0.02)
    assert clean.ts_r.median() == pytest.approx(0.80, abs=0.03)


@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
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
@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
def test_global_permutation_separates_signal_from_null():
    from adforce.eval.nulltest import perm_global, load_clean

    res = perm_global(load_clean(), n=1000, seed=0)
    # the real correlation must sit far above every shuffled (null) correlation
    assert res["observed"] > 0.85
    assert res["null_max"] < 0.5
    assert res["observed"] - res["null_max"] > 0.4
    assert res["p"] < 1e-3


@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
def test_within_storm_permutation_shows_real_spatial_skill():
    from adforce.eval.nulltest import perm_within_storm, load_clean

    res = perm_within_storm(load_clean(), n=1000, seed=0)
    # spatial skill (storm means removed) beats the within-storm-shuffled null
    assert res["observed_pooled"] > res["null_max"]
    assert res["p"] < 1e-3


@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
def test_cross_storm_pairing_collapses_skill():
    from adforce.eval.nulltest import cross_storm, load_clean

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
    obs_ok = pd.Series(
        np.linspace(0, 1, C.TS_MIN_OVERLAP), index=_hours(C.TS_MIN_OVERLAP)
    )
    obs_lo = pd.Series(
        np.linspace(0, 1, C.TS_MIN_OVERLAP - 1), index=_hours(C.TS_MIN_OVERLAP - 1)
    )
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
    assert (
        bootstrap_ci(d, n=500, seed=0)["bias"] != bootstrap_ci(d, n=500, seed=1)["bias"]
    )


def test_within_storm_r_single_storm_equals_plain_corr():
    d = pd.DataFrame(
        {
            "storm": "A",
            "obs_peak": [0.4, 0.9, 1.4, 2.0],
            "sim_peak": [0.3, 1.0, 1.2, 2.1],
        }
    )
    plain = np.corrcoef(d.obs_peak, d.sim_peak)[0, 1]
    assert within_storm_r(d) == pytest.approx(plain, abs=1e-9)


def test_within_storm_r_invariant_to_between_storm_offset():
    a = pd.DataFrame(
        {"storm": "A", "obs_peak": [0.4, 0.6, 0.9], "sim_peak": [0.5, 0.55, 1.0]}
    )
    b = a.assign(storm="B", obs_peak=a.obs_peak + 5.0, sim_peak=a.sim_peak + 5.0)
    base = within_storm_r(pd.concat([a, a.assign(storm="B")], ignore_index=True))
    offset = within_storm_r(pd.concat([a, b], ignore_index=True))
    assert base == pytest.approx(
        offset, abs=1e-9
    )  # storm-mean removal kills the offset


def test_add_flags_boundary_thresholds():
    rows = [
        _row("Laura 2020", "1", C.MIN_OBS_PEAK_M, 0.0),  # obs == 0.4 -> valid
        _row("Laura 2020", "2", C.MIN_OBS_PEAK_M - 0.01, 0.0),  # 0.39 -> not valid
        _row("Laura 2020", "3", 1.0, C.MAX_TIMING_HR),  # |dt| == 6 -> clean
        _row("Laura 2020", "4", 1.0, -C.MAX_TIMING_HR),  # -6 -> clean (abs)
        _row("Laura 2020", "5", 1.0, C.MAX_TIMING_HR + 0.1),  # 6.1 -> not clean
    ]
    df = add_flags(pd.DataFrame(rows)).set_index("sid")
    assert df.loc["1"].valid and not df.loc["2"].valid
    assert df.loc["3"].clean and df.loc["4"].clean and not df.loc["5"].clean


def test_nearest_wet_strict_threshold():
    pts = np.array([[0.0, 0.0]])
    tree = cKDTree(pts)
    at = np.array([[C.WET_MIN_M], [C.WET_MIN_M]])  # min == 0.3 -> rejected (strict >)
    above = np.array([[C.WET_MIN_M + 0.01], [C.WET_MIN_M + 0.01]])
    assert _nearest_wet(tree, at, 0.0, 0.0)[0] is None
    assert _nearest_wet(tree, above, 0.0, 0.0)[0] == 0


def test_latex_table_row_order_and_name_reformat(tmp_path):
    # two storms out of chronological order in the df; table must follow C.STORMS order.
    late = list(C.STORMS)[-1]  # last storm chronologically (Idalia 2023)
    early = list(C.STORMS)[0]  # first storm chronologically (Frances 2004)
    rows = []
    for storm in (late, early):  # deliberately reversed
        for k in range(3):
            rows.append(
                dict(
                    storm=storm,
                    sid=str(k),
                    obs_peak=1.0 + 0.1 * k,
                    sim_peak=1.0 + 0.1 * k,
                    peak_dt_hr=0.0,
                    ts_r=0.7,
                    setting="open-coast",
                )
            )
    df = add_flags(pd.DataFrame(rows))
    out = tmp_path / "t.tex"
    latex_table(df, str(out))
    text = out.read_text()
    i_early = text.index(early.replace(" ", " (") + ")")
    i_late = text.index(late.replace(" ", " (") + ")")
    assert i_early < i_late  # chronological, not df order
    name, year = early.rsplit(" ", 1)
    assert f"{name} ({year})" in text  # "Name YYYY" -> "Name (YYYY)"


@pytest.mark.skipif(
    not os.path.exists(SUMMARY_CSV),
    reason="val_summary.csv not present (no cached sweep)",
)
def test_regression_population_and_methods():
    df = pd.read_csv(SUMMARY_CSV)
    df["sid"] = df["sid"].astype(str)
    # 2026-07-26 three-city extension: 432 Gulf pairs (unchanged) + 35
    # Florida pairs over the 5 added storms
    assert len(df) == 467
    assert df.storm.nunique() == 19
    assert df.valid.sum() == 302
    assert df.clean.sum() == 146
    assert df.failed.sum() == 2
    methods = dict(df.method.value_counts())
    assert methods.get("utide") == 445 and methods.get("pred") == 22


# --------------------------------------------------------------------------- #
# Per-storm time-series cache (Parquet): exact round-trip + parameter staleness.
# Network-free -- uses synthetic ragged (sim, obs) series in a tmp cache dir.
# --------------------------------------------------------------------------- #
def test_series_cache_roundtrip_and_staleness(tmp_path, monkeypatch):
    import adforce.eval.validate as cv

    monkeypatch.setattr(C, "TS_CACHE", str(tmp_path))  # isolate from the real cache
    h = pd.date_range("2021-08-27", periods=48, freq="h")
    series = {  # sim 2-hourly, obs hourly (ragged)
        "Gauge A": (
            pd.Series(np.arange(24.0), index=h[::2]),
            pd.Series(np.sin(np.arange(48) / 3.0), index=h),
        ),
        "Gauge B, S.W": (
            pd.Series(np.cos(np.arange(24.0)), index=h[::2]),
            pd.Series(np.arange(48.0), index=h),
        ),
    }
    cv._save_series_cache("Test 2021", series)
    assert cv._series_cache_path("Test 2021").endswith(".parquet")

    loaded = cv._load_series_cache("Test 2021")
    assert loaded is not None and set(loaded) == set(series)
    for name in series:
        for orig, got in zip(series[name], loaded[name]):
            assert str(got.index.dtype).startswith("datetime64")  # datetime preserved
            assert got.index.equals(orig.index)
            np.testing.assert_allclose(got.values, orig.values)  # values exact

    monkeypatch.setattr(cv, "_ts_cache_tag", lambda: "DIFFERENT-PARAMS")
    assert cv._load_series_cache("Test 2021") is None  # stale tag -> recompute
    assert cv._load_series_cache("Never Cached 1999") is None  # missing -> recompute


# --------------------------------------------------------------------------- #
# Hydra config vs constants anti-drift, and cache-tag byte-stability.
# The scoring YAML mirrors constants.py; the ts-cache tag string keys the
# utide-expensive Parquet caches, so its byte-stability IS the cache validity.
# --------------------------------------------------------------------------- #
def test_scoring_config_matches_constants():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../adforce/eval/config"):
        cfg = compose(config_name="eval_config")
        am = compose(config_name="annual_max_config")
    s = cfg.scoring
    assert s.max_node_deg == C.MAX_NODE_DEG
    assert s.wet_min_m == C.WET_MIN_M
    assert s.knn == C.KNN
    assert s.max_timing_hr == C.MAX_TIMING_HR
    assert s.min_obs_peak_m == C.MIN_OBS_PEAK_M
    assert s.ts_min_overlap == C.TS_MIN_OVERLAP
    assert s.n_bootstrap == C.N_BOOTSTRAP
    assert s.bootstrap_seed == C.BOOTSTRAP_SEED
    assert s.utide_min_samples == C.UTIDE_MIN_SAMPLES
    assert am.start == C.AM_START_YEAR
    assert am.end == C.AM_END_YEAR


def test_cache_tag_matches_legacy_literal():
    """The tag must stay byte-identical to the comp/-era string: a formatting
    slip would silently invalidate every cached utide fit and re-hammer CO-OPS."""
    from adforce.eval.validate import _ts_cache_tag

    assert _ts_cache_tag("Katrina 2005") == (
        "v1|deg0.12|wet0.3|knn60|ut2000"
        "|box{'lon': (-97.6, -84.0), 'lat': (27.3, 30.9)}"
    )
    assert _ts_cache_tag("Irma 2017") == (  # Florida-box storm
        "v1|deg0.12|wet0.3|knn60|ut2000"
        "|box{'lon': (-82.3, -79.7), 'lat': (24.4, 30.8)}"
    )


# --------------------------------------------------------------------------- #
# Cell identity, run-dir status, sources, fort.61/63 readers (Commit C).
# All synthetic / tmp_path -- no network, no real run dirs.
# --------------------------------------------------------------------------- #
def test_cell_id_and_dir_to_storm():
    from adforce.eval.cells import ConfigCell, cell_id, dir_to_storm, storm_to_slug

    assert cell_id(ConfigCell()) == "res-mid_tide-off_swan-off_f-storm"
    assert (
        cell_id(ConfigCell(resolution="low", tide=True, forcing="both"))
        == "res-low_tide-on_swan-off_f-both"
    )
    assert cell_id(ConfigCell(physics_tag="rh08")).endswith("_p-rh08")
    assert dir_to_storm("22_MICHAEL_2018") == "Michael 2018"
    assert dir_to_storm("152_KATRINA_2005") == "Katrina 2005"
    assert dir_to_storm("not-a-run-dir") is None
    assert storm_to_slug("Katrina 2005", "152_KATRINA_2005.nc") == "152_KATRINA_2005"


def test_run_status_lifecycle(tmp_path):
    from omegaconf import OmegaConf

    from adforce.eval.cells import ConfigCell
    from adforce.eval.status import SUCCESS_MARKER, RunStatus, run_status

    cell = ConfigCell(resolution="mid", tide=False, swan=False)
    run = tmp_path / "152_KATRINA_2005"
    assert run_status(str(run), cell) is RunStatus.MISSING  # no dir
    run.mkdir()
    assert run_status(str(run), cell) is RunStatus.MISSING  # no config.yaml
    cfg = OmegaConf.create(
        {"adcirc": {"resolution": {"value": "mid"}, "tide": {"value": False}, "swan": {"value": False}}}
    )
    OmegaConf.save(cfg, str(run / "config.yaml"))
    assert run_status(str(run), cell) is RunStatus.FAILED  # no receipt yet
    (run / "slurm.out").write_text(f"stuff\n{SUCCESS_MARKER}\n")
    assert run_status(str(run), cell) is RunStatus.SUCCESS
    (run / "gauge_ts.parquet").write_bytes(b"")
    assert run_status(str(run), cell) is RunStatus.EXTRACTED
    foreign = ConfigCell(resolution="low")
    assert run_status(str(run), foreign) is RunStatus.FOREIGN
    assert run_status(str(run), None) is RunStatus.EXTRACTED  # no provenance check


def _write_fort63(path, x, y, depth, zeta, times):
    """Synthetic node-based fort.63.nc via xarray."""
    import xarray as xr

    ds = xr.Dataset(
        dict(
            x=("node", np.asarray(x, dtype=float)),
            y=("node", np.asarray(y, dtype=float)),
            depth=("node", np.asarray(depth, dtype=float)),
            zeta=(("time", "node"), np.asarray(zeta, dtype=float)),
        ),
        coords=dict(time=times),
    )
    ds.to_netcdf(path)


def test_extract_run_node_selection(tmp_path):
    """Nearest node that never dries wins; drying (NaN) nodes are rejected."""
    from adforce.eval.extract import extract_run

    run = tmp_path / "22_MICHAEL_2018"
    run.mkdir()
    t = pd.date_range("2018-10-09", periods=6, freq="h")
    # node 0: nearest to the gauge but dries (NaN); node 1: wet throughout;
    # node 2: far away (> max_deg).
    zeta = np.array(
        [
            [np.nan, 0.5, 0.1],
            [0.2, 0.6, 0.1],
            [0.3, 0.9, 0.1],
            [0.2, 0.7, 0.1],
            [np.nan, 0.5, 0.1],
            [0.1, 0.4, 0.1],
        ]
    )
    _write_fort63(
        run / "fort.63.nc",
        x=[-90.00, -90.02, -91.5],
        y=[29.00, 29.02, 29.5],
        depth=[5.0, 5.0, 5.0],
        zeta=zeta,
        times=t,
    )
    gauges = pd.DataFrame(
        [dict(sid="8761724", name="Grand Isle", lat=29.0, lon=-90.0)]
    )
    df = extract_run(str(run), gauges, max_deg=0.12, wet_min=0.3, knn=3)
    assert list(df.columns) == ["storm", "sid", "gauge", "time", "zeta"]
    assert df.storm.unique().tolist() == ["22_MICHAEL_2018"]
    assert df.sid.unique().tolist() == ["8761724"]
    np.testing.assert_allclose(df.zeta.values, zeta[:, 1])  # picked the wet node

    # An all-drying mesh yields no rows.
    run2 = tmp_path / "all_dry"
    run2.mkdir()
    _write_fort63(
        run2 / "fort.63.nc",
        x=[-90.0],
        y=[29.0],
        depth=[5.0],
        zeta=np.full((6, 1), np.nan),
        times=t,
    )
    assert extract_run(str(run2), gauges).empty


def test_read_fort61_roundtrip(tmp_path):
    import xarray as xr

    from adforce.fort61 import read_fort61

    t = pd.date_range("2018-10-09", periods=4, freq="h")
    zeta = np.array([[0.1, 1.0], [0.2, 1.1], [0.3, 1.2], [0.2, 1.3]])
    ds = xr.Dataset(
        dict(
            x=("station", [-90.0, -89.5]),
            y=("station", [29.0, 29.5]),
            zeta=(("time", "station"), zeta),
        ),
        coords=dict(time=t),
    )
    ds.to_netcdf(tmp_path / "fort.61.nc")
    df = read_fort61(str(tmp_path / "fort.61.nc"))
    assert set(df.columns) == {"station", "name", "x", "y", "time", "zeta"}
    assert df.station.nunique() == 2
    np.testing.assert_allclose(
        df[df.station == 1].sort_values("time").zeta.values, zeta[:, 1]
    )
    # directory form resolves fort.61.nc inside
    assert len(read_fort61(str(tmp_path))) == len(df)


def test_rundir_source_prefers_cheapest_artifact(tmp_path):
    from adforce.eval.sources import RunDirSource

    run = tmp_path / "run"
    run.mkdir()
    t = pd.date_range("2018-10-09", periods=4, freq="h")
    gauges = pd.DataFrame(
        [dict(sid="1", name="A", lat=29.0, lon=-90.0)]
    )
    # only fort.63 present -> fort63 sampling
    _write_fort63(
        run / "fort.63.nc",
        x=[-90.0],
        y=[29.0],
        depth=[5.0],
        zeta=np.array([[0.5], [0.6], [0.7], [0.6]]),
        times=t,
    )
    out = RunDirSource(str(run)).sim_series(gauges)
    assert out["1"][3] == "fort63"
    # gauge_ts.parquet appears -> preferred over fort63
    pd.DataFrame(
        dict(storm="run", sid="1", gauge="A", time=t, zeta=[1.0, 2.0, 3.0, 2.0])
    ).to_parquet(run / "gauge_ts.parquet", index=False)
    out = RunDirSource(str(run)).sim_series(gauges)
    name, series, deg, src = out["1"]
    assert src == "gauge_ts" and series.max() == 3.0


# --------------------------------------------------------------------------- #
# Tide-on Phase 2a: obs-side utilities (detide.py) + tide-only validation
# (tidecheck.py). All synthetic -- no network.
# --------------------------------------------------------------------------- #
M2_HR = 12.42  # principal lunar semidiurnal period


def test_tide_skill_recovers_amp_lag_offset():
    from adforce.eval.tidecheck import tide_skill

    t_pred = pd.date_range("2020-08-01", periods=24 * 6, freq="h")  # 6 days hourly
    hrs = np.arange(len(t_pred))
    pred = pd.Series(0.5 * np.sin(2 * np.pi * hrs / M2_HR), index=t_pred)
    t_sim = pd.date_range("2020-08-01", periods=24 * 6 * 18, freq="200s")  # model cadence
    hs = (t_sim - t_sim[0]).total_seconds() / 3600.0
    sim = pd.Series(
        0.4 * np.sin(2 * np.pi * (hs - 0.5) / M2_HR) + 0.15, index=t_sim
    )  # 80% amplitude, 30-min lag, +15 cm datum offset
    out = tide_skill(sim, pred)
    assert out["n_hr"] >= 24 * 5
    assert abs(out["amp_ratio"] - 0.8) < 0.05
    assert abs(out["lag_min"] - 30) <= 6  # one grid step
    assert abs(out["datum_offset_m"] - 0.15) < 0.02
    assert out["r"] > 0.9  # lag-0 r under a 30-min M2 shift ~ cos(14.5 deg)

    # too-short overlap -> NaNs, not garbage
    short = tide_skill(sim.iloc[: 18 * 24], pred.iloc[:24])
    assert np.isnan(short["amp_ratio"])


def test_align_pair_reports_offset():
    from adforce.eval.detide import align_pair

    t = pd.date_range("2021-08-25", periods=24 * 6, freq="h")
    obs = pd.Series(np.sin(np.arange(len(t)) / 5.0), index=t)
    sim = obs + 0.3  # pure datum shift
    s2, o2, off = align_pair(sim, obs, forcing_start=t[0], window_hr=48)
    assert abs(off - 0.3) < 1e-9
    np.testing.assert_allclose(s2.values, o2.values, atol=1e-12)


def test_skew_surge_promoted_and_phase_insensitive():
    import adforce.eval.detide_sensitivity as ds
    from adforce.eval.detide import skew_surge_peak

    assert ds._skew_surge_peak is skew_surge_peak  # re-export, not a copy
    t = pd.date_range("2020-08-01", periods=24 * 4, freq="h")
    hrs = np.arange(len(t))
    tide = pd.Series(0.5 * np.sin(2 * np.pi * hrs / M2_HR), index=t)
    # observed = phase-shifted tide + 0.8 m: instantaneous residual is phase-
    # contaminated, the skew surge is not.
    wl = pd.Series(0.5 * np.sin(2 * np.pi * (hrs - 2) / M2_HR) + 0.8, index=t)
    skew = skew_surge_peak(wl, tide, (t[0], t[-1]))
    assert abs(skew - 0.8) < 0.05


def test_twl_table_synthetic(monkeypatch):
    """Tide-on scoring end-to-end on synthetic series: recovers the datum
    offset, the peak bias, and a phase-insensitive skew-surge bias."""
    import adforce.eval.twl as twl

    t = pd.date_range("2020-08-01", periods=24 * 8, freq="h")  # 8-day run
    hrs = np.arange(len(t))
    tide = 0.5 * np.sin(2 * np.pi * hrs / M2_HR)
    bump = 1.5 * np.exp(-0.5 * ((hrs - 24 * 6) / 6.0) ** 2)  # landfall day 6
    sim = pd.Series(tide + bump + 0.2, index=t)  # +20 cm model datum offset
    obs = pd.Series(tide + 0.9 * bump, index=t)  # model over-predicts 10%
    obs = obs.drop(obs.index[50:60]).drop(obs.index[100:103])  # gauge gaps
    # (ragged obs vs full-length prediction is what crashed the first real
    # run: skew_surge_peak slices positionally -> inner-join at the call site)

    def frame(vals):
        return pd.DataFrame(
            dict(storm="154_TEST_2020", sid="42", gauge="G", time=t, zeta=vals)
        )

    monkeypatch.setattr(twl, "fetch_year", lambda sid, year: obs)
    monkeypatch.setattr(
        twl, "noaa_predictions", lambda sid, t0, t1: pd.Series(tide, index=t)
    )
    df = twl.twl_table(frame(sim.values), tide_series=frame(tide + 0.2))
    assert len(df) == 1
    r = df.iloc[0]
    assert r.key == "TEST_2020" and r.n_hr >= 24 * 7
    assert abs(r.datum_offset_m - 0.2) < 0.02
    assert abs(r.peak_bias - 0.15) < 0.1
    assert r.ts_r > 0.95
    assert abs(r.skew_bias - 0.15) < 0.15 and r.skew_sim > 1.0


def test_tideconst_recovers_planted_ratio(monkeypatch):
    """Constituent fits recover a planted M2-only 1.4x amplification."""
    import adforce.eval.tideconst as tc

    t = pd.date_range("2020-08-01", periods=24 * 15, freq="h")  # 15 days
    hrs = np.arange(len(t))
    m2 = np.sin(2 * np.pi * hrs / M2_HR)
    k1 = np.sin(2 * np.pi * hrs / 23.93)
    pred = pd.Series(0.30 * m2 + 0.10 * k1, index=t)
    sim = pd.Series(0.42 * m2 + 0.10 * k1, index=t)  # M2 x1.4, K1 x1.0

    monkeypatch.setattr(tc, "noaa_predictions", lambda sid, t0, t1: pred)
    monkeypatch.setattr(
        tc,
        "gauge_frame",
        lambda: pd.DataFrame([dict(sid="42", name="G", lat=29.0, lon=-90.0)]),
    )
    series = pd.DataFrame(
        dict(storm="1_TEST_2020", sid="42", gauge="G", time=t, zeta=sim.values)
    )
    df = tc.constituent_table(series, constituents=("M2", "K1"))
    r = df.set_index("constituent")
    assert abs(r.ratio["M2"] - 1.4) < 0.05
    assert abs(r.ratio["K1"] - 1.0) < 0.05
    assert abs(r.dphase_deg["M2"]) < 5


def test_noaa_predictions_uses_cache(tmp_path, monkeypatch):
    import adforce.eval.coops as coops_mod
    from adforce.eval import detide

    monkeypatch.setattr(coops_mod, "COOPS_CACHE", str(tmp_path))
    monkeypatch.setattr(detide.C, "COOPS_CACHE", str(tmp_path))
    csv = "Date Time, Prediction\n" + "\n".join(
        f"2020-08-{d:02d} 00:00,{0.1 * d:.2f}" for d in range(1, 6)
    )
    (tmp_path / "1234567_predictions_20200801_20200805_MSL.csv").write_text(csv)
    s = detide.noaa_predictions("1234567", "2020-08-01", "2020-08-05")
    assert len(s) == 5 and abs(s.iloc[-1] - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# Launch planning + harvest (Commit E). plan() is pure -- no ADCIRC, no
# network; everything runs against tmp_path run dirs.
# --------------------------------------------------------------------------- #
def _launch_cfg(tmp_path, **over):
    from hydra import compose, initialize

    overrides = [f"runs_root={tmp_path}", "study=t", 'storms=["Katrina 2005","Ida 2021"]']
    overrides += [f"{k}={v}" for k, v in over.items()]
    with initialize(version_base=None, config_path="../adforce/eval/config"):
        return compose(config_name="launch_config", overrides=overrides)


def test_launch_plan_statuses_and_actions(tmp_path):
    from omegaconf import OmegaConf

    from adforce.eval.launch import plan
    from adforce.eval.status import SUCCESS_MARKER

    cfg = _launch_cfg(tmp_path)
    table = plan(cfg)
    # res_x_tide: 4 cells x 2 storms; low/mid decks ship in adforce/setup
    assert len(table) == 8
    assert set(table.cell) == {
        "res-low_tide-off",
        "res-low_tide-on",
        "res-mid_tide-off",
        "res-mid_tide-on",
    }
    assert (table.status == "missing").all() and (table.action == "run").all()

    # fabricate a successful run -> skipped on the next plan
    run = tmp_path / "t" / "res-mid_tide-off" / "152_KATRINA_2005"
    run.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {"adcirc": {"resolution": {"value": "mid"}, "tide": {"value": False}, "swan": {"value": False}}}
        ),
        str(run / "config.yaml"),
    )
    (run / "slurm.out").write_text(SUCCESS_MARKER + "\n")
    table = plan(cfg)
    row = table[(table.cell == "res-mid_tide-off") & (table.slug == "152_KATRINA_2005")]
    assert row.action.item() == "skip" and row.status.item() == "success"

    # a FOREIGN dir (wrong resolution inside) blocks unless overwrite
    OmegaConf.save(
        OmegaConf.create(
            {"adcirc": {"resolution": {"value": "low"}, "tide": {"value": False}, "swan": {"value": False}}}
        ),
        str(run / "config.yaml"),
    )
    table = plan(cfg)
    row = table[(table.cell == "res-mid_tide-off") & (table.slug == "152_KATRINA_2005")]
    assert row.action.item() == "blocked" and row.status.item() == "foreign"


def test_launch_plan_controls_and_blocked_cells(tmp_path):
    from adforce.eval.launch import plan

    # controls add one tide-only cell per resolution with a tide-on cell
    table = plan(_launch_cfg(tmp_path, controls=True))
    assert "res-low_tide-on_swan-off_f-tide" in set(table.cell)
    assert "res-mid_tide-on_swan-off_f-tide" in set(table.cell)
    assert len(table) == (4 + 2) * 2

    # high resolution (no local deck) and swan cells are blocked, loudly
    cfg = _launch_cfg(tmp_path, matrix="archive_default")
    cfg.matrix.cells = [
        dict(name="hi", overrides={"adcirc.resolution.value": "high"}),
        dict(name="sw", overrides={"adcirc.swan.value": True}),
    ]
    cfg.matrix.baseline = "hi"
    table = plan(cfg)
    assert (table.action == "blocked").all()
    reasons = " ".join(table.reason)
    assert "fort.14.high" in reasons and "SWAN" in reasons


def test_fort13_mannings_edit(tmp_path):
    """Default-line rewrite: only that line changes; overrides untouched."""
    from adforce.fort13 import read_mannings_default, write_mannings_default

    src = tmp_path / "fort.13"
    src.write_text(
        "test mesh\n"
        "5\n"
        "2\n"
        "sea_surface_height_above_geoid\n m\n 1\n 0.000000\n"
        "mannings_n_at_sea_floor\n m\n 1\n 0.022000\n"
        "sea_surface_height_above_geoid\n0\n"
        "mannings_n_at_sea_floor\n2\n3 0.050000\n4 0.030000\n"
    )
    assert read_mannings_default(str(src)) == 0.022
    dst = tmp_path / "fort.13.n0035"
    write_mannings_default(str(src), str(dst), 0.035)
    assert read_mannings_default(str(dst)) == 0.035
    a, b = src.read_text().splitlines(), dst.read_text().splitlines()
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [10]
    assert "3 0.050000" in b  # per-node overrides preserved

    # the shipped decks read as the documented control value
    mid = os.path.join(str(REPO_ROOT), "adforce", "setup", "fort.13.mid")
    if os.path.exists(mid):
        assert read_mannings_default(mid) == 0.022


def test_mannings_matrix_plan_and_provenance(tmp_path):
    from omegaconf import OmegaConf

    from adforce.eval.cells import ConfigCell, cell_id
    from adforce.eval.launch import plan
    from adforce.eval.status import SUCCESS_MARKER, RunStatus, run_status

    assert cell_id(ConfigCell(tide=True, forcing="tide", mannings_n=0.028)) == (
        "res-mid_tide-on_swan-off_f-tide_n0.028"
    )

    cfg = _launch_cfg(tmp_path, matrix="mannings_tide")
    table = plan(cfg)
    assert set(table.cell) == {"tide-n0.022", "tide-n0.028", "tide-n0.035"}
    assert (table.action == "run").all()

    # provenance: a run with matching eval_axes is SUCCESS for its own cell,
    # FOREIGN for a different-n cell; a LEGACY dir (no eval_axes) can never
    # satisfy a mannings/tide-only cell
    cell = ConfigCell(resolution="mid", tide=True, forcing="tide", mannings_n=0.028)
    run = tmp_path / "t" / "tide-n0.028" / "152_KATRINA_2005"
    run.mkdir(parents=True)
    base = {
        "adcirc": {
            "resolution": {"value": "mid"},
            "tide": {"value": True},
            "swan": {"value": False},
        }
    }
    OmegaConf.save(
        OmegaConf.create({**base, "eval_axes": {"forcing": "tide", "mannings_n": 0.028}}),
        str(run / "config.yaml"),
    )
    (run / "slurm.out").write_text(SUCCESS_MARKER + "\n")
    assert run_status(str(run), cell) is RunStatus.SUCCESS
    other = ConfigCell(resolution="mid", tide=True, forcing="tide", mannings_n=0.035)
    assert run_status(str(run), other) is RunStatus.FOREIGN
    OmegaConf.save(OmegaConf.create(base), str(run / "config.yaml"))  # legacy dir
    assert run_status(str(run), cell) is RunStatus.FOREIGN


def test_wrap_compose_inside_hydra_app_needs_clear():
    """eval.launch (a @hydra.main app) calls wrap.get_default_config(), which
    re-initializes hydra and threw "GlobalHydra is already initialized" on
    the first real GCP launch (the dry-run path never composes the wrap
    config, so local gates missed it). Reproduce the app state and pin the
    clear-then-compose fix."""
    from hydra import initialize
    from hydra.core.global_hydra import GlobalHydra

    from adforce.wrap import get_default_config

    ctx = initialize(version_base=None, config_path="../adforce/eval/config")
    ctx.__enter__()  # emulate being inside a running hydra app
    try:
        with pytest.raises(ValueError, match="GlobalHydra"):
            get_default_config()
        GlobalHydra.instance().clear()  # the launch() fix
        cfg = get_default_config()
        assert str(cfg.adcirc.resolution.value) in ("low", "mid", "high")
    finally:
        GlobalHydra.instance().clear()


def test_config_hash_and_harvest_command(tmp_path):
    from omegaconf import OmegaConf

    from adforce.eval.harvest import rsync_command
    from adforce.eval.launch import config_hash

    a = OmegaConf.create({"x": 1, "y": {"z": "s"}})
    assert config_hash(a) == config_hash(OmegaConf.create({"x": 1, "y": {"z": "s"}}))
    assert config_hash(a) != config_hash(OmegaConf.create({"x": 2, "y": {"z": "s"}}))

    cmd = rsync_command("host:/work/exp/eval", "study1", str(tmp_path))
    assert cmd[0] == "rsync" and cmd[-2] == "host:/work/exp/eval/study1/"
    assert "--include=gauge_ts.parquet" in cmd and "--exclude=*" in cmd


def test_driver_seam_importable():
    """The extracted per-storm seam exists with the expected signature.

    Skipped where the training driver's HPC-only deps (adcircpy/stormevents)
    are not installed -- the same reason pytest.ini --ignores the module."""
    import inspect

    try:
        from adforce.training.driver import drive_storm, is_run_successful
    except ModuleNotFoundError as e:
        pytest.skip(f"training-driver dependency absent locally: {e.name}")

    params = list(inspect.signature(drive_storm).parameters)
    assert params[:4] == ["storm", "storm_ds", "run_directory", "cfg"]
    assert "mode" in params and "spinup_days" in params
    assert is_run_successful("/nonexistent") is False


# --------------------------------------------------------------------------- #
# Matrix grammar + model-vs-model (Commit D). The reproduction tests pin the
# ported pairs.py against the published rerun/results artifacts and are
# skipped when the cached sweep data is absent.
# --------------------------------------------------------------------------- #
def test_expand_matrix_grammar():
    from adforce.eval.cells import expand_matrix

    cells = expand_matrix(
        dict(
            axes={
                "adcirc.resolution.value": ["low", "mid"],
                "adcirc.tide.value": [False, True],
            },
            name_keys={
                "adcirc.resolution.value": "res",
                "adcirc.tide.value": "tide",
            },
            exclude=[{"adcirc.resolution.value": "low", "adcirc.tide.value": True}],
            include=[
                {"name": "swan", "overrides": {"adcirc.swan.value": True}}
            ],
            baseline="res-mid_tide-off",
        )
    )
    names = [n for n, _ in cells]
    assert names == ["res-low_tide-off", "res-mid_tide-off", "res-mid_tide-on", "swan"]
    assert dict(cells)["res-mid_tide-on"] == {
        "adcirc.resolution.value": "mid",
        "adcirc.tide.value": True,
    }
    with pytest.raises(ValueError, match="baseline"):
        expand_matrix(dict(axes={}, cells=[], baseline="nope"))
    with pytest.raises(ValueError, match="duplicate"):
        expand_matrix(
            dict(
                cells=[
                    {"name": "a", "overrides": {}},
                    {"name": "a", "overrides": {}},
                ]
            )
        )


def test_compare_cells_synthetic():
    from adforce.eval.pairs import compare_cells

    t = pd.date_range("2020-01-01", periods=3, freq="h")

    def frame(peak):
        return pd.DataFrame(
            dict(storm="1_TEST_2020", sid="42", gauge="G", time=t, zeta=[0.0, peak, 0.1])
        )

    df = compare_cells({"base": frame(1.0), "cand": frame(1.5)}, baseline="base")
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.cell, r.key, r.peak, r.base_peak, r.peak_diff) == (
        "cand",
        "TEST_2020",
        1.5,
        1.0,
        0.5,
    )
    with pytest.raises(ValueError, match="baseline"):
        compare_cells({"a": frame(1.0)}, baseline="missing")


_LOWRES = os.path.join(
    str(REPO_ROOT), "data", "comp", "lowres", "low_storm_gauge_series.parquet"
)
_RERUN_RB = os.path.join(str(REPO_ROOT), "rerun", "results", "resolution_bias.csv")
_RERUN_TSI = os.path.join(
    str(REPO_ROOT), "rerun", "results", "tide_surge_interaction.csv"
)


@pytest.mark.skipif(
    not (os.path.exists(_LOWRES) and os.path.exists(_RERUN_RB) and os.path.exists(SUMMARY_CSV)),
    reason="cached low-res sweep / published artifact not present",
)
def test_resolution_bias_reproduces_published():
    """pairs.resolution_bias_table must reproduce rerun/results/resolution_bias.csv
    (the published low-vs-mid resolution comparison) from the cached extract."""
    import io

    from adforce.eval.pairs import resolution_bias_table

    new = resolution_bias_table(_LOWRES)
    buf = io.StringIO()
    new.to_csv(buf, index=False)
    buf.seek(0)
    new_rt = pd.read_csv(buf)
    old = pd.read_csv(_RERUN_RB)
    assert list(new_rt.columns) == list(old.columns)
    assert new_rt.shape == old.shape
    np.testing.assert_allclose(
        new_rt.select_dtypes("number").fillna(-999).values,
        old.select_dtypes("number").fillna(-999).values,
    )


@pytest.mark.skipif(
    not (os.path.exists(_LOWRES) and os.path.exists(_RERUN_TSI)),
    reason="cached low-res sweep / published artifact not present",
)
def test_tide_surge_interaction_low_reproduces_published():
    """The low-res forcing triple must reproduce every published row exactly
    (storm/tide/both peaks, interaction, and series-vs-archive provenance)."""
    from adforce.eval.pairs import tide_surge_interaction

    lowres = os.path.dirname(_LOWRES)
    new = tide_surge_interaction(
        pd.read_parquet(_LOWRES),
        pd.read_parquet(os.path.join(lowres, "low_tide_runs_gauge_series.parquet")),
        pd.read_parquet(os.path.join(lowres, "low_both_full.parquet")),
        "low",
    )
    old = pd.read_csv(_RERUN_TSI, dtype={"sid": str})
    old = old[old.res == "low"]
    m = new.merge(old, on=["res", "key", "sid"], suffixes=("_n", "_o"))
    assert len(m) == len(old) == len(new) == 683
    for col in ("zeta_storm", "zeta_tide", "zeta_both", "interaction"):
        np.testing.assert_allclose(m[f"{col}_n"], m[f"{col}_o"])
    assert (m.storm_src_n == m.storm_src_o).all()


def test_validate_storm_source_seam(tmp_path, monkeypatch):
    """validate_storm scores an injected FieldSource without touching HF/CO-OPS."""
    import adforce.eval.validate as cv

    monkeypatch.setattr(C, "TS_CACHE", str(tmp_path))
    t = pd.date_range("2021-08-28", periods=48, freq="h")
    surge = np.concatenate([np.linspace(0, 2.0, 24), np.linspace(2.0, 0, 24)])

    class FakeSource:
        def load(self, storm, fname=None):
            x = np.array([-90.0])
            y = np.array([29.0])
            wet = np.full((48, 1), 5.0)  # always wet
            elev = surge[:, None]
            return x, y, wet, elev, pd.DatetimeIndex(t)

    obs = pd.Series(surge * 0.9, index=t)
    monkeypatch.setattr(cv, "observed_residual", lambda *a, **k: (obs, "utide"))
    rows, series = cv.validate_storm(
        "Fake 2021", "fake.nc", [("42", "Fake Gauge", 29.0, -90.0)], source=FakeSource()
    )
    assert len(rows) == 1 and rows[0]["sid"] == "42"
    assert rows[0]["sim_peak"] == 2.0
    assert abs(rows[0]["peak_dt_hr"]) < 1e-9  # aligned peaks
    assert "Fake Gauge" in series
