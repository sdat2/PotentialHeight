"""Cross-implementation consistency tests for the bounded-GEV fits.

The bounded-EVT claim rests on several deliberately decoupled GEV
negative-log-likelihood implementations agreeing:

* ``worst.tens`` scipy L-BFGS path (``fit_gev_upper_bound_known`` /
  ``_fit_lbfgs_known``): ``genextreme.logpdf`` with ``c = -gamma``.
* ``worst.tens`` TensorFlow/Adam path (``method="adam"``):
  ``tfd.GeneralizedExtremeValue`` with ``concentration = gamma``.
* ``worst.evt_theory._nll_bnd``: hand-written Coles NLL, Nelder-Mead,
  parameters ``(log sigma, log(-xi))``.
* ``worst.vary_nonstationary._ns_lbfgs_bounded`` with a constant bound:
  ``genextreme.logpdf`` with ``c = -gamma``, parameters ``(log beta, log(-gamma))``.

All use the Coles convention internally (shape ``gamma = xi``, negative for the
Weibull class, upper bound ``z* = alpha - beta/gamma``); scipy's ``c`` is
``-gamma``. One bounded-GEV sample is generated, each implementation is fitted
with the KNOWN bound, and the recovered (beta, gamma) are asserted to agree
pairwise. Observed empirical agreement (n=800, seed 42): max pairwise
|dbeta| = 1.3e-06, |dgamma| = 2.2e-07, so the 1e-4 tolerance is comfortable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.optimize import minimize
from scipy.stats import genextreme

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worst.evt_theory import _nll_bnd
from worst.tens import fit_gev_upper_bound_known
from worst.vary_nonstationary import _ns_lbfgs_bounded

BETA_TRUE = 1.0
GAMMA_TRUE = -0.2  # Coles shape xi < 0, Weibull class
Z_STAR = 5.0  # known upper bound = alpha - beta/gamma
ALPHA_TRUE = Z_STAR + BETA_TRUE / GAMMA_TRUE  # = 0.0
N_SAMPLES = 800
SEED = 42
PAIRWISE_TOL = 1e-4  # observed max pairwise diff ~1.3e-6; see module docstring


def _bounded_gev_sample() -> np.ndarray:
    """Generate one fixed-seed bounded-GEV sample (scipy ``c = -gamma``).

    Returns:
        np.ndarray: Sample of annual maxima with upper endpoint ``Z_STAR``.
    """
    rng = np.random.default_rng(SEED)
    return genextreme.rvs(
        c=-GAMMA_TRUE,
        loc=ALPHA_TRUE,
        scale=BETA_TRUE,
        size=N_SAMPLES,
        random_state=rng,
    )


@pytest.fixture(scope="module")
def fits() -> dict:
    """Fit the sample with each implementation given the known bound.

    Returns:
        dict: Implementation name -> (beta_hat, gamma_hat), Coles convention.
    """
    data = _bounded_gev_sample()
    assert data.max() < Z_STAR  # sample must respect the known bound

    out = {}

    # 1. worst.tens scipy L-BFGS (default method).
    _, beta, gamma = fit_gev_upper_bound_known(data, Z_STAR, method="lbfgs")
    out["tens_lbfgs"] = (float(beta), float(gamma))

    # 2. worst.tens TensorFlow/Adam.
    _, beta, gamma = fit_gev_upper_bound_known(
        data, Z_STAR, method="adam", opt_steps=1000, lr=0.01
    )
    out["tens_adam"] = (float(beta), float(gamma))

    # 3. worst.evt_theory hand-written Coles NLL, Nelder-Mead (same call
    #    pattern as evt_theory._fit_bnd, which only returns a return level).
    res = minimize(
        _nll_bnd,
        [0.0, np.log(0.2)],
        args=(data, Z_STAR),
        method="Nelder-Mead",
        options=dict(xatol=1e-7, fatol=1e-7, maxiter=6000),
    )
    out["evt_theory_nm"] = (float(np.exp(res.x[0])), float(-np.exp(res.x[1])))

    # 4. worst.vary_nonstationary non-stationary L-BFGS with a constant bound.
    cfg = OmegaConf.create({"beta_guess": 1.0, "gamma_guess": -0.1})
    beta, gamma = _ns_lbfgs_bounded(data, np.full(N_SAMPLES, Z_STAR), cfg)
    out["vary_ns_lbfgs"] = (float(beta), float(gamma))

    for name, (b, g) in out.items():
        print(f"{name:16s} beta={b:.8f} gamma={g:.8f}")
    return out


def test_all_fits_finite_weibull(fits: dict) -> None:
    """Every implementation returns finite beta > 0 and gamma < 0."""
    for name, (beta, gamma) in fits.items():
        assert np.isfinite(beta) and beta > 0.0, name
        assert np.isfinite(gamma) and gamma < 0.0, name


def test_pairwise_parameter_agreement(fits: dict) -> None:
    """All implementations agree pairwise on (beta, gamma) within PAIRWISE_TOL."""
    names = sorted(fits)
    for i, ni in enumerate(names):
        for nj in names[i + 1 :]:
            d_beta = abs(fits[ni][0] - fits[nj][0])
            d_gamma = abs(fits[ni][1] - fits[nj][1])
            print(f"{ni} vs {nj}: dbeta={d_beta:.2e} dgamma={d_gamma:.2e}")
            assert d_beta < PAIRWISE_TOL, (ni, nj, d_beta)
            assert d_gamma < PAIRWISE_TOL, (ni, nj, d_gamma)


def test_fits_recover_truth(fits: dict) -> None:
    """Recovered parameters lie within sampling error of the truth (~3 SE)."""
    for name, (beta, gamma) in fits.items():
        assert abs(beta - BETA_TRUE) < 0.15, (name, beta)
        assert abs(gamma - GAMMA_TRUE) < 0.10, (name, gamma)


def test_degenerate_bound_returns_nan() -> None:
    """A bound below the sample maximum yields NaNs, not garbage parameters."""
    data = _bounded_gev_sample()
    alpha, beta, gamma = fit_gev_upper_bound_known(
        data, float(data.max()) - 0.5, method="lbfgs"
    )
    assert np.isnan(alpha) and np.isnan(beta) and np.isnan(gamma)
