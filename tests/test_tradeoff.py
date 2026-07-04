"""Tests for the size--intensity tradeoff curve (w22.tradeoff) and its
integration with the Bayesian-optimization loop (adbo.exp).

The curve tests use a synthetic warm-pool environment (Wang 2022-like) so
they need no external data. The integration test runs run_bayesopt_exp in
wrap_test mode (no ADCIRC), checking that each BO sample gets its own CLE15
profile generated on the curve.
"""

import os

import numpy as np
import pytest

from w22.tradeoff import (
    CAT1_10M,
    V_REDUC_DEFAULT,
    TradeoffCurve,
    tradeoff_ds,
)

# synthetic warm-pool environment (roughly the Wang 2022 canonical case)
ENV = {
    "vmax": 83.0,  # potential intensity, gradient level [m/s]
    "msl": 1015.0,  # [mbar]
    "sst": 26.85,  # [degC]
    "t0": 200.0,  # outflow temperature [K]
    "lat": 29.5,  # [degN]
    "rh": 0.9,
}


@pytest.fixture(scope="module")
def curve_dataset():
    """Small tradeoff curve on the synthetic environment (7 points)."""
    return tradeoff_ds(ENV, num=7)


@pytest.fixture(scope="module")
def curve(curve_dataset):
    return TradeoffCurve(curve_dataset)


def test_curve_values_physical(curve_dataset) -> None:
    """Curve values are finite and physically sensible."""
    ds = curve_dataset
    assert np.all(np.isfinite(ds["r0"].values)), "r0 contains NaN"
    assert np.all(np.isfinite(ds["rmax"].values)), "rmax contains NaN"
    assert np.all(ds["r0"].values > ds["rmax"].values), "r0 must exceed rmax"
    assert np.all(ds["rmax"].values > 1e3), "rmax should be > 1 km"
    assert np.all(ds["r0"].values < 5e6), "r0 should be < 5000 km"
    # pressures: pc <= pm < ambient
    ambient_pa = ENV["msl"] * 100.0
    assert np.all(ds["pm"].values < ambient_pa)
    assert np.all(ds["pc"].values <= ds["pm"].values + 1e-6)
    # grid spans [Cat1 gradient, V_p]
    v = ds["vmax"].values
    assert np.isclose(v[0], CAT1_10M / V_REDUC_DEFAULT)
    assert np.isclose(v[-1], ENV["vmax"])


def test_tradeoff_is_monotonic(curve_dataset) -> None:
    """The size-intensity tradeoff: more intense storms have smaller cores."""
    rmax = curve_dataset["rmax"].values
    assert np.all(np.diff(rmax) < 0), (
        f"rmax(V) should be strictly decreasing, got {rmax}"
    )
    # central pressure drops as intensity rises
    pc = curve_dataset["pc"].values
    assert np.all(np.diff(pc) < 0), f"pc(V) should decrease, got {pc}"


def test_interpolator_roundtrip(curve_dataset, curve) -> None:
    """PCHIP interpolators reproduce the grid values exactly."""
    v = curve_dataset["vmax"].values
    for i in (0, len(v) // 2, -1):
        assert np.isclose(curve.r0(float(v[i])), curve_dataset["r0"].values[i])
        assert np.isclose(curve.rmax(float(v[i])), curve_dataset["rmax"].values[i])
        assert np.isclose(curve.pm(float(v[i])), curve_dataset["pm"].values[i])


def test_out_of_range_raises(curve) -> None:
    with pytest.raises(ValueError):
        curve.r0(curve.v_min - 5.0)
    with pytest.raises(ValueError):
        curve.profile(curve.v_max + 5.0)


def test_profile_on_curve(curve, tmp_path) -> None:
    """Profile generation: max wind matches the sampled intensity, and the
    JSON written is in the adforce-readable format."""
    v_mid = 0.5 * (curve.v_min + curve.v_max)
    out_json = os.path.join(tmp_path, "profile.json")
    profile = curve.profile(v_mid, out_path=out_json)
    for key in ("rr", "VV", "p"):
        assert key in profile, f"profile missing {key}"
    vv = np.asarray(profile["VV"], dtype=float)
    rr = np.asarray(profile["rr"], dtype=float)
    pp = np.asarray(profile["p"], dtype=float)
    assert abs(float(np.nanmax(vv)) - v_mid) < 1.5, (
        f"profile max wind {np.nanmax(vv):.2f} != sampled vmax {v_mid:.2f}"
    )
    assert rr[0] >= 0 and np.all(np.diff(rr) > 0), "rr must be increasing"
    # pressure profile: minimum at centre, rising outward to ~ambient [hPa]
    assert pp[0] == np.min(pp)
    assert abs(pp[-1] - ENV["msl"]) < 20.0
    # written file loads and matches
    import json

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert np.allclose(np.asarray(loaded["VV"], dtype=float), vv, equal_nan=True)


def test_curve_file_roundtrip(curve_dataset, tmp_path) -> None:
    """Curve survives a save/load cycle with attrs intact."""
    path = os.path.join(tmp_path, "curve.nc")
    curve_dataset.to_netcdf(path)
    loaded = TradeoffCurve.from_file(path)
    assert np.isclose(loaded.v_max, ENV["vmax"])
    assert np.isclose(loaded.ds.attrs["env_lat"], ENV["lat"])
    v_mid = 0.5 * (loaded.v_min + loaded.v_max)
    assert loaded.r0(v_mid) > loaded.rmax(v_mid) > 0


def test_4d_bo_wrap_test_smoke(curve_dataset, tmp_path) -> None:
    """End-to-end 4D BO in wrap_test mode: each sample gets an on-curve
    profile.json in its run folder and vmax lands in the ledger."""
    pytest.importorskip("trieste")
    import json
    import yaml
    from adbo.constants import CONFIG_PATH
    from adbo.exp import run_bayesopt_exp

    curve_nc = os.path.join(tmp_path, "curve.nc")
    curve_dataset.to_netcdf(curve_nc)
    constraints = yaml.safe_load(
        open(os.path.join(CONFIG_PATH, "4d_constraints.yaml"))
    )
    exp_name = "tradeoff_smoke"
    run_bayesopt_exp(
        constraints=constraints,
        seed=11,
        exp_name=exp_name,
        root_exp_direc=str(tmp_path),
        init_steps=3,
        daf_steps=1,
        wrap_test=True,
        resume=True,
        curve_path=curve_nc,
    )
    exp_dir = os.path.join(tmp_path, exp_name)
    with open(os.path.join(exp_dir, "experiments.json"), "r") as f:
        ledger = json.load(f)
    assert len(ledger) == 4, f"expected 4 evaluations, got {len(ledger)}"
    v_lo = CAT1_10M / V_REDUC_DEFAULT
    for call, rec in ledger.items():
        assert "vmax" in rec, f"ledger entry {call} missing vmax"
        assert v_lo - 1e-6 <= rec["vmax"] <= ENV["vmax"] + 1e-6
        profile_json = os.path.join(exp_dir, f"exp_{int(call):04}", "profile.json")
        assert os.path.exists(profile_json), f"missing {profile_json}"
        with open(profile_json, "r", encoding="utf-8") as f:
            profile = json.load(f)
        vv_max = float(np.nanmax(np.asarray(profile["VV"], dtype=float)))
        assert abs(vv_max - rec["vmax"]) < 1.5, (
            f"profile max wind {vv_max:.2f} inconsistent with sampled "
            f"vmax {rec['vmax']:.2f} for call {call}"
        )
    # bo-config records the curve for provenance/resume
    with open(os.path.join(exp_dir, "bo-config.json"), "r") as f:
        bo_cfg = json.load(f)
    assert bo_cfg["curve_path"] == curve_nc
    assert bo_cfg["constraints"]["vmax"]["min"] is not None
