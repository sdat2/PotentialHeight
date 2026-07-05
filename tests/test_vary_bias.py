"""Tests for worst.vary_bias: systematic bias in the assumed GEV upper bound.

A tiny configuration (Nr=20 resamples, ns=50 samples, 3 bias points, one sigma,
one gamma) is run once (module-scoped fixture) and the assertions check:

* all recorded outputs are finite (after skipna where degenerate fits give NaN);
* at b = 0 with sigma = 0 the bounded fit (I) has a 5-95% range no wider than the
  unbounded fit (II) for both return periods -- the paper's established result;
* at strongly negative b the empirical-maximum safeguard engages in nearly all
  resamples (clip fraction rises toward 1), and far more often than at b >= 0;
* the whole experiment runs in under 60 s.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worst.vary_bias import get_fit_ds, report_crossings

LP, UP = 0.05, 0.95  # 5-95% envelope, as in the paper figures


def _tiny_config(data_dir: str):
    """Small config: Nr=20, ns=50, 3 bias points, 1 sigma, 1 gamma.

    Args:
        data_dir (str): Temporary directory for the cached netCDF.

    Returns:
        DictConfig: Configuration for a fast run.
    """
    return OmegaConf.create(
        {
            "z_star": 7.0,
            "beta": 1.0,
            "gammas": [-0.2],
            "bias": {"min": -3.0, "max": 3.0, "steps": 3},
            "sigmas": [0.0],
            "ns": 50,
            "seed_steps_Nr": 20,
            "seed_offset": 0,
            "upper_bound_padding": 0.005,
            "quantiles": [0.01, 0.002],
            "fit": {
                "alpha_guess": None,  # start unbounded fit at mean(data)
                "beta_guess": 1.0,
                "gamma_guess": -0.1,
                "steps": 1000,
                "lr": 0.01,
                "force_weibull": False,
            },
            "regenerate": True,
            "data_dir": data_dir,
            "figure": {"lp": LP, "up": UP},
            "verbose": False,
            "color": {
                "true_gev": "black",
                "max_known": "#18520c",
                "max_unknown": "#d95f02",
            },
        }
    )


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """Run the tiny experiment once, returning (config, dataset, runtime seconds)."""
    config = _tiny_config(str(tmp_path_factory.mktemp("vary_bias")))
    t0 = time.monotonic()
    ds = get_fit_ds(config)
    runtime = time.monotonic() - t0
    return config, ds, runtime


def test_runtime_under_60_s(experiment) -> None:
    """The tiny configuration completes in under a minute."""
    _, _, runtime = experiment
    print(f"vary_bias tiny run took {runtime:.1f} s")
    assert runtime < 60.0


def test_outputs_finite(experiment) -> None:
    """True/unbounded RVs, cell means, clip fractions and fitted params are finite."""
    _, ds, _ = experiment
    assert np.isfinite(ds["rv_true"].values).all()
    # unbounded fits: all resamples succeed on this easy sample size
    assert np.isfinite(ds["rv_ubu"].values).all()
    # bounded fits: NaNs only from degenerate optima, dropped by skipna; every cell
    # must retain a finite mean over seeds
    cell_mean = ds["rv_ubk"].mean("seed", skipna=True)
    assert np.isfinite(cell_mean.values).all()
    frac = ds["clip_engaged"].mean("seed")
    assert np.isfinite(frac.values).all()
    assert ((frac.values >= 0.0) & (frac.values <= 1.0)).all()
    assert np.isfinite(ds["beta_hat"].mean("seed", skipna=True).values).all()
    assert np.isfinite(ds["gamma_hat"].mean("seed", skipna=True).values).all()
    assert np.isfinite(ds["z_star_assumed"].values).all()


def test_unbiased_bound_has_narrower_range_than_unbounded(experiment) -> None:
    """At b=0 (sigma=0) method I's 5-95% range <= method II's, both return periods."""
    _, ds, _ = experiment
    ubk = ds["rv_ubk"].sel(bias=0.0, sigma=0.0).isel(gamma=0)
    ubu = ds["rv_ubu"].isel(gamma=0)
    for rp in ds.rp.values.tolist():
        r1 = float(
            ubk.sel(rp=rp).quantile(UP, dim="seed", skipna=True)
            - ubk.sel(rp=rp).quantile(LP, dim="seed", skipna=True)
        )
        r2 = float(
            ubu.sel(rp=rp).quantile(UP, dim="seed", skipna=True)
            - ubu.sel(rp=rp).quantile(LP, dim="seed", skipna=True)
        )
        print(f"RV{int(rp)}: range I = {r1:.2f} m, range II = {r2:.2f} m")
        assert r1 <= r2


def test_clip_fraction_rises_at_strongly_negative_bias(experiment) -> None:
    """At b=-3 m the empirical-max safeguard engages in nearly all resamples."""
    _, ds, _ = experiment
    frac = ds["clip_engaged"].mean("seed").isel(gamma=0, sigma=0)
    frac_neg = float(frac.sel(bias=-3.0))
    frac_zero = float(frac.sel(bias=0.0))
    frac_pos = float(frac.sel(bias=3.0))
    print(f"clip fraction: b=-3: {frac_neg:.2f}, b=0: {frac_zero:.2f}, b=+3: {frac_pos:.2f}")
    assert frac_neg > 0.8  # rises toward 1 when the bound sits below typical maxima
    assert frac_neg > frac_zero
    assert frac_neg > frac_pos
    assert frac_pos == 0.0  # a bound 3 m above truth is never below the sample max


def test_report_crossings_runs(experiment) -> None:
    """The b* crossing report runs and returns an entry per (gamma, sigma, rp)."""
    config, ds, _ = experiment
    out = report_crossings(config, ds)
    assert len(out) == ds.gamma.size * ds.sigma.size * ds.rp.size
    for summary in out.values():
        assert np.isfinite(summary["rmse_ii"])
