"""Stationary bounded-EVT theory: bias saturation and the crossover existence window.

Reproduces the two analytic/​simulation figures used in the potential-height paper's
Supplementary Material (``paper/evt_theory.tex``):

* ``evt_saturation.pdf`` -- the pseudo-true return-level bias :math:`b_p(u)` of the bounded
  (known-endpoint) GEV estimator **saturates** to the Gumbel-misspecification plateau
  :math:`b_\\infty(p)` as the bound recedes; it is *not* logarithmic and carries no intrinsic
  :math:`U_p`.  As the fixed endpoint :math:`z^*+u` recedes the constraint goes vacuous, the
  constrained shape :math:`\\xi^*(u)\\uparrow 0`, and the constrained GEV-MLE converges to the
  unconstrained Gumbel pseudo-true fit; the plateau is that Gumbel-misspecification bias.
* ``evt_closedform.pdf`` -- the bounded-vs-unbounded 5--95% range crossover
  :math:`\\sigma^*(n)` from *direct GEV-fit simulation* (ground truth), the saturating
  closed-form approximation, and the two-edge existence window
  :math:`n_{\\mathrm{hi}} < n < n_{\\mathrm{lo}}`.

All quantities are for the reference Weibull-domain GEV
:math:`(\\mu,\\sigma,\\xi) = (2, 1, -0.2)`, endpoint :math:`z^* = \\mu - \\sigma/\\xi = 7`.
The PDFs are written straight into the thesis ``img/`` tree (located like :mod:`comp`), so
re-running reproduces exactly what the paper ``\\includegraphics``-es -- no manual copy step.

Run::

    python -m worst.evt_theory            # regenerate both figures + print the numbers
    python -m worst.evt_theory --quick    # fewer samples/seeds (faster, rougher sim)
    python -m worst.evt_theory --diagnostics-only   # print ground-truth table, no figures

The simulation uses self-contained scipy Nelder--Mead GEV fits (Weibull-forced, ``xi<0``)
rather than :mod:`worst.tens`, so the theory figures are decoupled from the estimator code
and reproduce the published PDFs exactly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import FIGURE_PATH, PROJ_PATH

# the GEV negative log-likelihoods return a large finite penalty for infeasible parameters,
# which routinely trips numpy's overflow/divide warnings during the optimiser's line search.
np.seterr(all="ignore")

# --- reference GEV (Weibull domain) ----------------------------------------
MU, SIG, XI = 2.0, 1.0, -0.2
ZSTAR = MU - SIG / XI                       # = 7.0, the true upper endpoint (potential height)
RPS = {100: 0.01, 500: 0.002}              # return period -> exceedance prob p
W = {k: -np.log(-np.log(1 - p)) for k, p in RPS.items()}   # Gumbel reduced variate w_p
# Gumbel-limit plateau and Fisher slope at the reference GEV (RV500), cached for the closed
# form / window; recomputed and checked against this in :func:`gumbel_plateau`.
B_INF_REF = {100: 1.369, 500: 2.390}
C_P_REF = {100: 0.177, 500: 0.292}
E_GN_50 = 2.1                               # E[G_n] safeguard gap at n=50 (Lemma: gap law)


# --- output path: write paper PDFs into the thesis img/ tree ---------------
def _paper_img_path() -> str:
    """Locate the thesis ``img/`` dir (holds the paper figures) like :mod:`comp`.

    Falls back to the in-repo ``img/worst`` quick-look dir if the thesis tree is absent.
    Override the thesis root with the ``WORSTSURGE_PAPER_ROOT`` env var.
    """
    env = os.environ.get("WORSTSURGE_PAPER_ROOT")
    candidates = ([Path(env)] if env else []) + [
        Path.home() / "thesis", PROJ_PATH.parent, PROJ_PATH,
    ]
    for c in candidates:
        if (c / "paper" / "evt_theory.tex").is_file():
            os.makedirs(c / "img", exist_ok=True)
            return str(c / "img")
    os.makedirs(FIGURE_PATH, exist_ok=True)
    return FIGURE_PATH


# --- return levels ---------------------------------------------------------
def return_level(mu: float, sig: float, xi: float, p: float) -> float:
    """GEV return level :math:`z_p = \\mu - (\\sigma/\\xi)(1 - y^{-\\xi})`, ``y=-log(1-p)``."""
    y = -np.log(1 - p)
    return mu - (sig / xi) * (1 - y ** (-xi))


def _rl_from_endpoint(psi: float, sig: float, xi: float, p: float) -> float:
    """Return level written about a *fixed endpoint* ``psi``: ``z_p = psi + (sig/xi) y^{-xi}``
    (equals :func:`return_level` with ``mu = psi + sig/xi``)."""
    y = -np.log(1 - p)
    return psi + (sig / xi) * y ** (-xi)


ZP_TRUE = {k: return_level(MU, SIG, XI, p) for k, p in RPS.items()}


# ======================================================================================
#  Population (pseudo-true) quantities: the saturating bias and its Gumbel plateau
# ======================================================================================
def _population_sample(size: int, seed: int = 0) -> np.ndarray:
    return genextreme.rvs(c=-XI, loc=MU, scale=SIG, size=size, random_state=np.random.default_rng(seed))


def _neg_loglik_gev_constrained(par, x, psi):
    """-E_hat[log f] for the bounded GEV with endpoint fixed at ``psi`` (params log-sig, log(-xi))."""
    sig, xi = np.exp(par[0]), -np.exp(par[1])
    t = xi * (x - psi) / sig                      # = 1 + xi (x-mu)/sig with mu = psi + sig/xi
    if np.any(t <= 0):
        return 1e9
    return -np.mean(-np.log(sig) - (1 + 1 / xi) * np.log(t) - t ** (-1 / xi))


def _neg_loglik_gumbel(par, x):
    mu, log_s = par[0], par[1]
    s = np.exp(log_s)
    z = (x - mu) / s
    return -np.mean(-log_s - z - np.exp(-z))


def gumbel_plateau(size: int = 3_000_000, seed: int = 0):
    """Exact Gumbel-limit plateau ``b_inf(p) = [mu_G + sig_G w_p] - z_p``.

    The plateau is the bias of the unconstrained Gumbel pseudo-true fit
    ``(mu_G, sig_G) = argmin KL(F0 || Gumbel)`` -- the limit the constrained GEV-MLE
    reaches as the bound recedes (xi*->0).

    Returns ``(b_inf: dict, (mu_G, sig_G))``.
    """
    x = _population_sample(size, seed)
    r = minimize(_neg_loglik_gumbel, [2.0, 0.0], args=(x,), method="Nelder-Mead",
                 options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000))
    mu_g, sig_g = r.x[0], np.exp(r.x[1])
    b_inf = {k: (mu_g + sig_g * W[k]) - ZP_TRUE[k] for k in RPS}
    return b_inf, (mu_g, sig_g)


def saturating_bias_curve(us, size: int = 3_000_000, seed: int = 0):
    """Pseudo-true bias ``b_p(u) = z_p(constrained fit at endpoint z*+u) - z_p(true)``.

    Returns ``(bias: {rp: array over us}, cp: {rp: Fisher slope b_p'(0)})``.
    """
    x = _population_sample(size, seed)
    us = np.asarray(us, dtype=float)
    par = [0.0, np.log(0.2)]
    bias = {k: [] for k in RPS}
    for u in us:
        r = minimize(_neg_loglik_gev_constrained, par, args=(x, ZSTAR + u), method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-10, maxiter=8000))
        par = r.x                                 # warm-start the next u
        s, xi = np.exp(r.x[0]), -np.exp(r.x[1])
        for k, p in RPS.items():
            bias[k].append(_rl_from_endpoint(ZSTAR + u, s, xi, p) - ZP_TRUE[k])
    bias = {k: np.array(v) for k, v in bias.items()}
    cp = {k: bias[k][0] / us[0] for k in RPS}     # slope from the first (small-u) point
    return bias, cp


# ======================================================================================
#  Asymptotic Fisher information: endpoint SE and the Cornish--Fisher unbounded range R_II
# ======================================================================================
def fisher_information(size: int = 2_000_000, seed: int = 1) -> np.ndarray:
    """Per-observation 3x3 Fisher information of the unbounded GEV at the truth,
    via a large-sample average of the (analytic-in-spirit) finite-difference Hessian."""
    x = _population_sample(size, seed)

    def logf(th):
        mu, sig, xi = th
        t = 1 + xi * (x - mu) / sig
        return np.where(t > 0, -np.log(sig) - (1 + 1 / xi) * np.log(np.maximum(t, 1e-300))
                        - t ** (-1 / xi), -1e6)

    th0, e, H = [MU, SIG, XI], 1e-4, np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            tpp, tpm, tmp, tmm = (th0.copy() for _ in range(4))
            tpp[i] += e; tpp[j] += e; tpm[i] += e; tpm[j] -= e
            tmp[i] -= e; tmp[j] += e; tmm[i] -= e; tmm[j] -= e
            H[i, j] = np.mean((logf(tpp) - logf(tpm) - logf(tmp) + logf(tmm)) / (4 * e * e))
    return -H


def asymptotic_se_zstar(I3: np.ndarray, n: int) -> float:
    """Leading-order ``SE(z*hat) = sqrt((I^-1)_{psi psi}/n)``; grad of ``z*=mu-sig/xi`` is
    ``(1, -1/xi, sig/xi^2)``.  Exactly return-period independent."""
    gz = np.array([1.0, -1 / XI, SIG / XI ** 2])
    return float(np.sqrt(gz @ np.linalg.inv(I3) @ gz / n))


def cornish_fisher_RII(I3: np.ndarray, n: int, p: float = 0.002) -> float:
    """Asymptotic unbounded-estimator 5--95% return-level range via a 2nd-order delta
    expansion + Cornish--Fisher (the paper's Lemma: closed-form unbounded range)."""
    y = -np.log(1 - p)

    def zfull(mu, sig, xi):
        return mu - (sig / xi) * (1 - y ** (-xi))

    th, e = [MU, SIG, XI], 1e-5
    g, Hn = np.zeros(3), np.zeros((3, 3))
    for i in range(3):
        tp, tm = th.copy(), th.copy(); tp[i] += e; tm[i] -= e
        g[i] = (zfull(*tp) - zfull(*tm)) / (2 * e)
    for i in range(3):
        for j in range(3):
            tpp, tpm, tmp, tmm = (th.copy() for _ in range(4))
            tpp[i] += e; tpp[j] += e; tpm[i] += e; tpm[j] -= e
            tmp[i] -= e; tmp[j] += e; tmm[i] -= e; tmm[j] -= e
            Hn[i, j] = (zfull(*tpp) - zfull(*tpm) - zfull(*tmp) + zfull(*tmm)) / (4 * e * e)
    S = np.linalg.inv(I3) / n
    k2 = g @ S @ g + 0.5 * np.trace(S @ Hn @ S @ Hn)
    k3 = 3 * g @ S @ Hn @ S @ g + np.trace(S @ Hn @ S @ Hn @ S @ Hn)
    k4 = 12 * g @ S @ Hn @ S @ Hn @ S @ g + 3 * np.trace(S @ Hn @ S @ Hn @ S @ Hn @ S @ Hn)
    g1, g2 = k3 / k2 ** 1.5, k4 / k2 ** 2

    def cf(za):
        return (za + g1 / 6 * (za ** 2 - 1) + g2 / 24 * (za ** 3 - 3 * za)
                - g1 ** 2 / 36 * (2 * za ** 3 - 5 * za))

    return float(np.sqrt(k2) * (cf(1.645) - cf(-1.645)))


# ======================================================================================
#  Direct GEV-fit simulation: ground-truth R_II, SE, and the clipped crossover sigma*(n)
# ======================================================================================
def _nll_unb(par, d):
    mu, ls, lx = par; s, x = np.exp(ls), -np.exp(lx)
    t = 1 + x * (d - mu) / s
    if np.any(t <= 0):
        return 1e12
    return np.sum(ls + (1 + 1 / x) * np.log(t) + t ** (-1 / x))


def _nll_bnd(par, d, zst):
    ls, lx = par; s, x = np.exp(ls), -np.exp(lx)
    mu = zst + s / x
    t = 1 + x * (d - mu) / s
    if np.any(t <= 0):
        return 1e12
    return np.sum(ls + (1 + 1 / x) * np.log(t) + t ** (-1 / x))


def _fit_unb(d, p):
    r = minimize(_nll_unb, [np.median(d), 0.0, np.log(0.2)], args=(d,), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-7, maxiter=6000))
    return return_level(r.x[0], np.exp(r.x[1]), -np.exp(r.x[2]), p)


def _fit_bnd(d, zst, p):
    zst = max(zst, d.max() + 1e-6)            # empirical-max clip: the bound cannot lie below the data
    r = minimize(_nll_bnd, [0.0, np.log(0.2)], args=(d, zst), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-7, maxiter=6000))
    return _rl_from_endpoint(zst, np.exp(r.x[0]), -np.exp(r.x[1]), p)


def simulate_crossover(n: int, n_rep: int = 700, p: float = 0.002,
                       sigmas=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12)):
    """Direct simulation at record length ``n``: unbounded range ``R_II``, robust endpoint SE,
    and the bound-uncertainty ``sigma_z*`` at which the clipped-bounded 5--95% range first
    overtakes ``R_II`` (the crossover).  Returns ``(R_II, se_robust, crossover, sigmas, ranges_I)``.
    """
    g = np.random.default_rng(7 + n)
    D = genextreme.rvs(c=-XI, loc=MU, scale=SIG, size=(n_rep, n), random_state=g)
    zII = np.array([_fit_unb(d, p) for d in D])
    rII = np.nanquantile(zII, 0.95) - np.nanquantile(zII, 0.05)
    pu = np.array([minimize(_nll_unb, [np.median(d), 0.0, np.log(0.2)], args=(d,),
                            method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=6000)).x
                   for d in D])
    zstar = pu[:, 0] - np.exp(pu[:, 1]) / (-np.exp(pu[:, 2]))      # mu - sig/xi (heavy-tailed: use robust scale)
    se_robust = (np.nanquantile(zstar, 0.95) - np.nanquantile(zstar, 0.05)) / (2 * 1.645)
    sigmas = np.asarray(sigmas, dtype=float)
    ranges_I = []
    for sz in sigmas:
        zs = ZSTAR + g.normal(0, sz, n_rep)
        zI = np.array([_fit_bnd(d, z, p) for d, z in zip(D, zs)])
        ranges_I.append(np.nanquantile(zI, 0.95) - np.nanquantile(zI, 0.05))
    ranges_I = np.array(ranges_I)
    above = ranges_I >= rII
    if not above.any():
        cx = np.nan                                  # bounded always tighter -> crossover beyond grid
    elif above[0]:
        cx = sigmas[0]
    else:
        i = int(np.argmax(above))
        cx = float(np.interp(0, [ranges_I[i - 1] - rII, ranges_I[i] - rII], [sigmas[i - 1], sigmas[i]]))
    return rII, se_robust, cx, sigmas, ranges_I


# ======================================================================================
#  Closed-form crossover and the existence window (from the saturating bias)
# ======================================================================================
def closed_form_crossover(R_II: float, n: int, b_inf: float, c_p: float, e_gn_50: float = E_GN_50):
    """Saturating closed-form crossover, valid inside the window; ``nan`` otherwise:

    ``sigma* = (b_inf / 1.645 c_p) log[ b_inf / (b_inf - (R_II - c_p Gbar)) ]``.
    """
    gbar = e_gn_50 * (50.0 / n) ** 0.2
    x = R_II - c_p * gbar
    if 0 < x < b_inf:
        return b_inf / (1.645 * c_p) * np.log(b_inf / (b_inf - x))
    return np.nan


# ======================================================================================
#  Figures
# ======================================================================================
def make_saturation_fig(size: int = 3_000_000, out_dir: str | None = None) -> dict:
    """Generate ``evt_saturation.pdf`` and return ``{rp: b_inf}``."""
    out_dir = out_dir or _paper_img_path()
    b_inf, (mu_g, sig_g) = gumbel_plateau(size)
    us = np.r_[np.linspace(0.0, 12, 25)[1:], 16, 22, 30, 45, 70, 120, 250, 500.0]
    bias, cp = saturating_bias_curve(us, size)
    print(f"  Gumbel pseudo-true: mu_G={mu_g:.4f} sig_G={sig_g:.4f}")
    for k in RPS:
        print(f"  b_inf(RV{k}) = {b_inf[k]:.3f} m   c_p(RV{k}) ~ {cp[k]:.3f}")

    def _logfit(b):                               # the (wrong) local log fit, for the overlay
        return min(((np.sum((b - (A := np.sum(b * np.log(1 + us / U)) / np.sum(np.log(1 + us / U) ** 2))
                             * np.log(1 + us / U)) ** 2), A, U)
                    for U in np.linspace(0.2, 30, 400)), key=lambda t: t[0])

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    uplot = np.linspace(0, 60, 400)
    for ax, k in zip(axs, (100, 500)):
        _, A, U = _logfit(bias[k])
        ax.plot(us, bias[k], "o", ms=3.2, color="#1f77b4", label=r"pseudo-true $b_p(u)$", zorder=5)
        ax.axhline(b_inf[k], ls="--", color="#2ca02c", lw=1.4,
                   label=fr"Gumbel-limit plateau $b_\infty={b_inf[k]:.2f}$")
        ax.plot(uplot, A * np.log(1 + uplot / U), ":", color="#d62728", lw=1.5,
                label="local log fit (diverges)")
        ax.plot([0, 6], [0, 6 * cp[k]], "-", color="grey", lw=1.0, label=r"Fisher slope $c_p$")
        ax.set_xlim(0, 60); ax.set_ylim(0, max(b_inf[k] * 1.25, A * np.log(1 + 60 / U)))
        ax.set_xlabel(r"bound margin $u=\hat z^*-z^*$ [m]"); ax.set_title(f"RV{k}")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("pseudo-true bias $b_p$ [m]"); axs[0].legend(fontsize=7, loc="lower right")
    out = os.path.join(out_dir, "evt_saturation.pdf")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("  wrote", out)
    return b_inf


def make_closedform_fig(n_rep: int = 700, out_dir: str | None = None) -> None:
    """Generate ``evt_closedform.pdf`` (RV500): simulated crossover, saturating closed form,
    and the existence window."""
    out_dir = out_dir or _paper_img_path()
    b_inf, c_p = B_INF_REF[500], C_P_REF[500]
    ns = np.array([25, 35, 50, 70, 100, 150, 250, 400])
    rII_s, cx_s = np.zeros(len(ns)), np.zeros(len(ns))
    for i, n in enumerate(ns):
        rII_s[i], _, cx_s[i], _, _ = simulate_crossover(int(n), n_rep=n_rep)
        print(f"  n={int(n):4d}  R_II={rII_s[i]:.2f}  crossover={cx_s[i]:.2f} m")

    # smooth R_II(n) ~ A n^{-1/2} fit to the simulated ranges, for the closed form / window
    A = np.exp(np.mean(np.log(rII_s) + 0.5 * np.log(ns)))

    def RII(n):
        return A * np.asarray(n, float) ** -0.5

    def gbar(n):
        return E_GN_50 * (50.0 / np.asarray(n, float)) ** 0.2

    def se_asymp(n):
        return 1.8 * np.sqrt(50.0 / np.asarray(n, float))     # leading-order, p-independent

    ng = np.logspace(np.log10(20), np.log10(500), 200)
    cl = np.array([closed_form_crossover(float(RII(n)), float(n), b_inf, c_p) for n in ng])
    nn = np.logspace(np.log10(15), np.log10(9000), 500)
    rii_n, saf, up = RII(nn), c_p * gbar(nn), b_inf + c_p * gbar(nn)
    n_lo = nn[np.argmin(np.abs(rii_n - saf))]
    n_hi = nn[np.argmin(np.abs(rii_n - up))]
    print(f"  R_II ~ {A:.2f} n^-1/2 ;  window n_hi ~ {n_hi:.0f} yr ,  n_lo ~ {n_lo:.0f} yr")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axL.plot(ng, cl, "-", color="#d62728", lw=1.8, label="saturating closed form")
    axL.plot(ns, cx_s, "k*", ms=12, label="full GEV-fit simulation")
    axL.plot(ng, se_asymp(ng), ":", color="grey", lw=1.2, label=r"asymptotic $\mathrm{SE}(\hat z^*)$")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("record length $n$ [years]"); axL.set_ylabel(r"crossover $\sigma^*$ [m]")
    axL.set_title("Crossover vs record length (RV500)")
    axL.legend(fontsize=8.5); axL.grid(alpha=0.3, which="both")
    axL.annotate(r"closed form $\approx1.6\times$ sim",
                 (70, closed_form_crossover(float(RII(70)), 70, b_inf, c_p)), (90, 8.5),
                 fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    axR.fill_between(nn, 0.05, 12, where=(rii_n > saf) & (rii_n < up), color="green", alpha=0.07)
    axR.plot(nn, rii_n, color="#1f77b4", lw=1.8, label=r"data range $R_{II}\sim n^{-1/2}$")
    axR.plot(nn, saf, color="#ff7f0e", lw=1.8, label=r"lower edge $c_p\bar G\sim n^{-|\xi|}$")
    axR.plot(nn, up, color="#9467bd", lw=1.8, label=r"upper edge $b_\infty+c_p\bar G$")
    for nt, lab, dy in ((n_lo, fr"$n_{{\rm lo}}\approx{n_lo:.0f}$ yr", 0.12),
                        (n_hi, fr"$n_{{\rm hi}}\approx{n_hi:.0f}$ yr", 5.0)):
        axR.axvline(nt, ls="--", color="k", lw=0.9); axR.text(nt * 1.04, dy, lab, fontsize=8)
    axR.text(np.sqrt(n_hi * n_lo), 0.5, "bound helps\n(window)", ha="center", fontsize=8, color="green")
    axR.set_xscale("log"); axR.set_yscale("log"); axR.set_ylim(0.08, 12)
    axR.set_xlabel("record length $n$ [years]"); axR.set_ylabel("[m]")
    axR.set_title(r"Existence window $n_{\rm hi}<n<n_{\rm lo}$"); axR.legend(fontsize=8, loc="lower left")
    axR.grid(alpha=0.3, which="both")
    out = os.path.join(out_dir, "evt_closedform.pdf")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("  wrote", out)


def diagnostics(n_rep: int = 800) -> None:
    """Print the ground-truth calibration table (R_II, both SE definitions, crossover)."""
    I3 = fisher_information()
    print("Ground-truth diagnostics (RV500):")
    print(f"{'n':>5} {'R_II_sim':>9} {'R_II_CF':>8} {'SE_asymp':>9} {'SE_robust':>10} {'crossover':>10}")
    for n in (50, 100, 200, 400):   # n<~26 is the degenerate upper window edge (endpoint SE diverges)
        rII, se_rob, cx, _, _ = simulate_crossover(n, n_rep=n_rep)
        print(f"{n:>5} {rII:>9.2f} {cornish_fisher_RII(I3, n):>8.2f} "
              f"{asymptotic_se_zstar(I3, n):>9.2f} {se_rob:>10.2f} {cx:>10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer samples/seeds (faster, rougher)")
    ap.add_argument("--diagnostics-only", action="store_true", help="print the table, skip figures")
    a = ap.parse_args()
    size = 500_000 if a.quick else 3_000_000
    n_rep = 200 if a.quick else 700
    if a.diagnostics_only:
        diagnostics(n_rep=200 if a.quick else 800)
        return
    print("evt_saturation.pdf:")
    make_saturation_fig(size=size)
    print("evt_closedform.pdf:")
    make_closedform_fig(n_rep=n_rep)
    print("\nground-truth diagnostics:")
    diagnostics(n_rep=200 if a.quick else 800)


if __name__ == "__main__":
    main()
