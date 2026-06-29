import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tensorflow_probability import distributions as tfd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worst.tens import (
    fit_gev_upper_bound_known,
    fit_gev_upper_bound_not_known,
    seed_all,
)
from worst.vary_nonstationary import (
    fit_nonstationary_bounded,
    fit_nonstationary_unbounded,
)


def _sample_gev(alpha, beta, gamma, n=80, seed=0):
    seed_all(seed)
    dist = tfd.GeneralizedExtremeValue(loc=alpha, scale=beta, concentration=gamma)
    return dist.sample(n).numpy()


def test_stationary_unbounded_fit_returns_finite_weibull_like_params():
    data = _sample_gev(alpha=1.0, beta=0.35, gamma=-0.15, n=120, seed=1)

    alpha_hat, beta_hat, gamma_hat = fit_gev_upper_bound_not_known(
        data,
        opt_steps=120,
        lr=0.01,
        alpha_guess=1.0,
        beta_guess=0.5,
        gamma_guess=-0.1,
        force_weibull=True,
        verbose=False,
    )

    assert np.isfinite(alpha_hat)
    assert np.isfinite(beta_hat) and beta_hat > 0.0
    assert np.isfinite(gamma_hat) and gamma_hat < 0.0


def test_stationary_bounded_fit_respects_known_upper_bound():
    z_star = 2.2
    data = _sample_gev(alpha=z_star + 0.5 / (-0.12), beta=0.25, gamma=-0.12, n=140, seed=2)

    alpha_hat, beta_hat, gamma_hat = fit_gev_upper_bound_known(
        data,
        z_star,
        opt_steps=120,
        lr=0.01,
        beta_guess=0.3,
        gamma_guess=-0.1,
        verbose=False,
    )

    assert np.isfinite(alpha_hat)
    assert np.isfinite(beta_hat) and beta_hat > 0.0
    assert np.isfinite(gamma_hat) and gamma_hat < 0.0


def test_nonstationary_bounded_fit_returns_finite_weibull_params():
    t = np.linspace(0.0, 1.0, 60)
    z_star_t = 2.0 + 0.75 * t
    beta_true = 0.25
    gamma_true = -0.1
    alpha_t = z_star_t + beta_true / gamma_true

    data = np.array(
        [
            tfd.GeneralizedExtremeValue(
                loc=alpha_t_i,
                scale=beta_true,
                concentration=gamma_true,
            ).sample(1).numpy()[0]
            for alpha_t_i in alpha_t
        ]
    )

    cfg = OmegaConf.create({"steps": 100, "lr": 0.01, "beta_guess": 0.5, "gamma_guess": -0.1})
    beta_hat, gamma_hat = fit_nonstationary_bounded(data, z_star_t, cfg)

    assert np.isfinite(beta_hat) and beta_hat > 0.0
    assert np.isfinite(gamma_hat) and gamma_hat < 0.0


def test_nonstationary_unbounded_fit_recovers_positive_trend():
    t = np.linspace(0.0, 1.0, 80)
    alpha0 = 1.0
    alpha1 = 0.4
    beta = 0.2
    gamma = -0.08

    mu_t = alpha0 + alpha1 * t
    data = np.array(
        [
            tfd.GeneralizedExtremeValue(loc=mu_i, scale=beta, concentration=gamma).sample(1).numpy()[0]
            for mu_i in mu_t
        ]
    )

    cfg = OmegaConf.create(
        {
            "steps": 150,
            "lr": 0.01,
            "alpha_guess": 1.0,
            "beta_guess": 0.3,
            "gamma_guess": -0.1,
            "force_weibull": True,
        }
    )
    alpha0_hat, alpha1_hat, beta_hat, gamma_hat = fit_nonstationary_unbounded(data, t, cfg)

    assert np.isfinite(alpha0_hat)
    assert np.isfinite(alpha1_hat)
    assert np.isfinite(beta_hat) and beta_hat > 0.0
    assert np.isfinite(gamma_hat) and gamma_hat < 0.0
    assert alpha1_hat > 0.0
