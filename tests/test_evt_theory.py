"""Unit + consistency tests for worst.evt_theory (bounded-EVT crossover theory).

Covers the pure analytic pieces (return levels, the closed-form crossover window),
the seeded Monte-Carlo population quantities (Gumbel plateau, saturating bias,
Fisher information), and a small direct GEV-fit simulation cross-checked against
the closed form. All Monte-Carlo sizes are shrunk from the paper defaults
(3M -> 300k samples, 700 -> 60 replicates) so the whole file runs in ~15 s;
tolerances are set accordingly. The expensive pieces are computed once in
module-scoped fixtures. Note the paper itself documents the closed form as
"~1.6x the simulation", so the closed-form-vs-simulation check is deliberately
an order-of-magnitude (factor) test, not a tight one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import genextreme

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worst.evt_theory import (
    B_INF_REF,
    C_P_REF,
    MU,
    RPS,
    SIG,
    W,
    XI,
    ZP_TRUE,
    ZSTAR,
    _fit_bnd,
    _fit_unb,
    _population_sample,
    _rl_from_endpoint,
    asymptotic_se_zstar,
    closed_form_crossover,
    cornish_fisher_RII,
    fisher_information,
    gumbel_plateau,
    return_level,
    saturating_bias_curve,
    simulate_crossover,
)


# --------------------------------------------------------------------------- #
# Module-scoped fixtures: compute the (seeded) Monte-Carlo quantities once.
# Sizes are cut well below the paper defaults to keep the file fast; the seeds
# are fixed inside the module (population sample) so results are deterministic.
# --------------------------------------------------------------------------- #
US = np.array([0.5, 2.0, 8.0, 50.0])  # bound margins for the saturating-bias curve


@pytest.fixture(scope="module")
def I3():
    """Per-observation 3x3 Fisher information at the reference GEV (400k sample)."""
    return fisher_information(size=400_000, seed=1)


@pytest.fixture(scope="module")
def plateau():
    """Gumbel-limit plateau ``(b_inf, (mu_g, sig_g))`` from a 300k sample."""
    return gumbel_plateau(size=300_000, seed=0)


@pytest.fixture(scope="module")
def sat():
    """Saturating pseudo-true bias curve ``(bias, cp)`` over :data:`US` (300k sample)."""
    return saturating_bias_curve(US, size=300_000, seed=0)


@pytest.fixture(scope="module")
def cross():
    """Small direct GEV-fit crossover simulation at n=50 (60 replicates)."""
    return simulate_crossover(50, n_rep=60, sigmas=(0.5, 1, 2, 3, 4, 6, 9, 12))


# --------------------------------------------------------------------------- #
# return_level: pin the exact GEV quantile convention against scipy.
# The module uses Coles' xi (scipy c = -xi) and p = annual exceedance
# probability, i.e. z_p = F^{-1}(1 - p) with y = -log(1 - p).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mu,sig,xi", [(2.0, 1.0, -0.2), (0.0, 0.5, -0.1), (5.0, 2.0, -0.35)]
)
@pytest.mark.parametrize("T", [10, 100, 500])
def test_return_level_matches_scipy_genextreme(mu, sig, xi, T):
    p = 1.0 / T
    ours = return_level(mu, sig, xi, p)
    scipy_rl = genextreme.ppf(1.0 - p, c=-xi, loc=mu, scale=sig)
    assert ours == pytest.approx(
        scipy_rl, abs=1e-10
    ), f"return_level({mu},{sig},{xi},p=1/{T}) = {ours} != scipy ppf(1-1/T) = {scipy_rl}"


def test_return_level_monotone_and_bounded_by_endpoint():
    # For xi<0 the return level rises with T and approaches (never exceeds) the
    # upper endpoint z* = mu - sig/xi, but only slowly (~p^{-xi}).
    ps = 1.0 / np.array([2, 10, 100, 500, 1e4, 1e9])
    zs = np.array([return_level(MU, SIG, XI, p) for p in ps])
    assert np.all(np.diff(zs) > 0), f"return level not increasing in T: {zs}"
    assert np.all(zs < ZSTAR), f"return level exceeds the endpoint {ZSTAR}: {zs}"
    assert (
        ZSTAR - zs[-1] < 0.1
    ), f"z_p at p=1e-9 should be within 0.1 of z*={ZSTAR}, got {zs[-1]}"


def test_rl_from_endpoint_consistent_with_return_level():
    # _rl_from_endpoint(psi, ...) is return_level with mu = psi + sig/xi (exact algebra).
    for psi, sig, xi, p in [
        (7.0, 1.0, -0.2, 0.002),
        (3.0, 0.4, -0.1, 0.01),
        (10.0, 2.0, -0.3, 0.05),
    ]:
        a = _rl_from_endpoint(psi, sig, xi, p)
        b = return_level(psi + sig / xi, sig, xi, p)
        assert a == pytest.approx(
            b, abs=1e-12
        ), f"endpoint parameterisation mismatch: {a} vs {b}"


def test_module_reference_constants():
    # z* = mu - sig/xi = 7 for the reference (2, 1, -0.2) GEV.
    assert ZSTAR == pytest.approx(7.0, abs=1e-12)
    # ZP_TRUE built from RPS via return_level -- pin against an independent scipy call.
    for T, p in RPS.items():
        expect = genextreme.ppf(1.0 - p, c=-XI, loc=MU, scale=SIG)
        assert ZP_TRUE[T] == pytest.approx(expect, abs=1e-10), f"ZP_TRUE[{T}] drifted"
    # Gumbel reduced variate w_p = -log(-log(1-p)).
    assert W[100] == pytest.approx(-np.log(-np.log(0.99)), abs=1e-12)
    assert W[500] > W[100] > 0


# --------------------------------------------------------------------------- #
# Fisher information + asymptotic endpoint SE
# --------------------------------------------------------------------------- #
def test_fisher_information_symmetric_positive_definite(I3):
    assert I3.shape == (3, 3)
    assert np.all(np.isfinite(I3)), f"non-finite Fisher information: {I3}"
    np.testing.assert_allclose(
        I3, I3.T, atol=1e-8, err_msg="Fisher information not symmetric"
    )
    eig = np.linalg.eigvalsh(0.5 * (I3 + I3.T))
    assert np.all(eig > 0), f"Fisher information not positive definite: eig={eig}"


def test_asymptotic_se_zstar_positive_and_scales_inverse_sqrt_n(I3):
    for n in (25, 50, 100, 400):
        se = asymptotic_se_zstar(I3, n)
        assert np.isfinite(se) and se > 0, f"SE(z*) at n={n} not positive finite: {se}"
    # sqrt(1/n) scaling is exact in the leading-order formula: se(4n) = se(n)/2.
    for n in (50, 200):
        assert asymptotic_se_zstar(I3, 4 * n) == pytest.approx(
            asymptotic_se_zstar(I3, n) / 2.0, rel=1e-9
        ), f"SE(z*) does not scale as 1/sqrt(n) at n={n}"


def test_asymptotic_se_zstar_matches_plot_reference(I3):
    # make_closedform_fig hardcodes se_asymp(n) = 1.8 sqrt(50/n); the Fisher-based
    # value at n=50 must reproduce that 1.8 m anchor.
    se50 = asymptotic_se_zstar(I3, 50)
    assert se50 == pytest.approx(
        1.8, abs=0.15
    ), f"SE(z*, n=50) = {se50}, expected ~1.8 m"


def test_cornish_fisher_RII_positive_and_shrinks_like_sqrt_n(I3):
    r50, r100, r400 = (cornish_fisher_RII(I3, n) for n in (50, 100, 400))
    for r in (r50, r100, r400):
        assert np.isfinite(r) and r > 0, f"Cornish-Fisher R_II not positive finite: {r}"
    assert r50 > r100 > r400, f"R_II not decreasing in n: {r50}, {r100}, {r400}"
    # leading order R_II ~ n^{-1/2}: quadrupling n should roughly halve the range.
    assert r100 / r400 == pytest.approx(
        2.0, rel=0.1
    ), f"R_II(100)/R_II(400) = {r100 / r400}, expected ~2 (n^-1/2 scaling)"


# --------------------------------------------------------------------------- #
# Gumbel plateau + saturating bias curve (population / pseudo-true quantities)
# --------------------------------------------------------------------------- #
def test_gumbel_plateau_matches_cached_reference(plateau):
    b_inf, (mu_g, sig_g) = plateau
    assert np.isfinite(mu_g) and np.isfinite(sig_g) and sig_g > 0
    for T in (100, 500):
        assert b_inf[T] == pytest.approx(
            B_INF_REF[T], abs=0.06
        ), f"Gumbel plateau b_inf[{T}] = {b_inf[T]} drifted from reference {B_INF_REF[T]}"
    assert (
        b_inf[500] > b_inf[100] > 0
    ), "plateau should be positive and grow with return period"


def test_saturating_bias_monotone_and_below_plateau(sat, plateau):
    bias, _ = sat
    b_inf, _ = plateau
    for T in (100, 500):
        b = bias[T]
        assert np.all(np.isfinite(b)), f"non-finite bias curve for rp={T}: {b}"
        assert np.all(b > 0), f"pseudo-true bias should be positive for rp={T}: {b}"
        assert np.all(
            np.diff(b) > 0
        ), f"bias not increasing in the margin u for rp={T}: {b}"
        # saturation: never overshoots the Gumbel plateau, and by u=50 has covered
        # most of the way there (b(50)/b_inf ~ 0.87 at the reference GEV).
        assert np.all(
            b < b_inf[T] * 1.02
        ), f"bias exceeds the plateau for rp={T}: {b} vs {b_inf[T]}"
        assert (
            b[-1] > 0.6 * b_inf[T]
        ), f"bias at u={US[-1]} should approach the plateau: {b[-1]} vs b_inf={b_inf[T]}"


def test_fisher_slope_matches_cached_reference(sat):
    _, cp = sat
    for T in (100, 500):
        assert cp[T] == pytest.approx(
            C_P_REF[T], rel=0.1
        ), f"Fisher slope c_p[{T}] = {cp[T]} drifted from reference {C_P_REF[T]}"
    assert cp[500] > cp[100] > 0


# --------------------------------------------------------------------------- #
# Direct GEV-fit crossover simulation + the saturating closed form
# --------------------------------------------------------------------------- #
def test_simulate_crossover_outputs_sane(cross, I3):
    rII, se_robust, cx, sigmas, ranges_I = cross
    assert (
        np.isfinite(rII) and rII > 0
    ), f"unbounded range R_II not positive finite: {rII}"
    assert np.isfinite(se_robust) and se_robust > 0
    assert len(ranges_I) == len(sigmas)
    assert np.all(np.isfinite(ranges_I)) and np.all(ranges_I > 0)
    # bounded range must widen as the bound gets noisier (vacuous -> unbounded-like).
    assert (
        ranges_I[-1] > ranges_I[0]
    ), f"bounded 5-95% range should grow with sigma_z*: {ranges_I}"
    # crossover lies inside the sigma grid (or is nan when never crossed).
    assert (
        np.isnan(cx) or sigmas[0] <= cx <= sigmas[-1]
    ), f"crossover {cx} outside grid {sigmas}"
    # the simulated unbounded range should sit near the Cornish-Fisher asymptote.
    r_cf = cornish_fisher_RII(I3, 50)
    assert (
        0.7 < rII / r_cf < 1.8
    ), f"simulated R_II={rII} vs Cornish-Fisher {r_cf}: ratio outside [0.7, 1.8]"


def test_closed_form_crossover_agrees_with_simulation(cross):
    # The paper's own annotation says "closed form ~ 1.6x sim", so this is a
    # factor-level agreement check (plus Monte-Carlo noise at n_rep=60).
    rII, _, cx_sim, _, _ = cross
    cx_cf = closed_form_crossover(rII, 50, B_INF_REF[500], C_P_REF[500])
    assert (
        np.isfinite(cx_sim) and cx_sim > 0
    ), f"simulated crossover not finite: {cx_sim}"
    assert (
        np.isfinite(cx_cf) and cx_cf > 0
    ), f"closed-form crossover not finite: {cx_cf}"
    ratio = cx_cf / cx_sim
    assert (
        0.7 < ratio < 3.5
    ), f"closed form {cx_cf} vs simulation {cx_sim}: ratio {ratio} outside [0.7, 3.5]"


def test_closed_form_crossover_window_edges():
    b_inf, c_p, n = B_INF_REF[500], C_P_REF[500], 50
    safeguard = c_p * 2.1 * (50.0 / n) ** 0.2  # c_p * Gbar(n)
    # below the lower edge (R_II <= c_p Gbar): no crossover.
    assert np.isnan(closed_form_crossover(safeguard * 0.5, n, b_inf, c_p))
    # above the upper edge (R_II - c_p Gbar >= b_inf): log argument invalid -> nan.
    assert np.isnan(closed_form_crossover(safeguard + b_inf + 0.1, n, b_inf, c_p))
    # inside the window: positive, finite, and increasing in R_II.
    lo = closed_form_crossover(safeguard + 0.3 * b_inf, n, b_inf, c_p)
    hi = closed_form_crossover(safeguard + 0.8 * b_inf, n, b_inf, c_p)
    assert np.isfinite(lo) and lo > 0
    assert np.isfinite(hi) and hi > lo, f"crossover should grow with R_II: {lo} vs {hi}"


def test_simulate_crossover_deterministic():
    # The RNG is seeded internally (default_rng(7 + n)), so a repeat call with the
    # same arguments must reproduce every output bit-for-bit.
    kw = dict(n_rep=12, sigmas=(1.0, 4.0))
    a = simulate_crossover(40, **kw)
    b = simulate_crossover(40, **kw)
    assert (
        a[0] == b[0] and a[1] == b[1]
    ), "R_II / SE not deterministic under the internal seed"
    assert (np.isnan(a[2]) and np.isnan(b[2])) or a[2] == b[
        2
    ], "crossover not deterministic"
    np.testing.assert_array_equal(
        a[4], b[4], err_msg="bounded ranges not deterministic"
    )


def test_bounded_fit_clips_bound_to_data_max():
    # _fit_bnd must clip a supplied bound lying below the sample maximum up to the
    # empirical max, and the fitted return level must stay below that endpoint.
    d = _population_sample(60, seed=3)
    p = RPS[500]
    z_bnd = _fit_bnd(d, d.max() - 0.5, p)  # bound below the data -> clipped
    assert np.isfinite(
        z_bnd
    ), "bounded fit with an infeasible bound should still be finite"
    assert (
        z_bnd < d.max() + 1e-5
    ), f"bounded return level {z_bnd} above the clipped endpoint"
    z_unb = _fit_unb(d, p)
    assert np.isfinite(z_unb), "unbounded fit should be finite"
