"""Non-stationary GEV fit to real annual-maximum surge residuals.

Extends :mod:`worst.gauge_fit` (stationary bounded/unbounded fits) with the
non-stationary case Reviewer 1 asked about: an unbounded GEV whose location
parameter drifts linearly in time,

    mu(t) = mu0 + mu1 * (t - t_bar),    sigma, xi constant,

fitted by maximum likelihood to the same de-tided annual maxima
(:mod:`comp.annual_max`). The stationary model is nested (mu1 = 0), so a
likelihood-ratio test gives the significance of the drift, alongside AIC.
Reported per station:

* mu1 (m/decade) with a parametric-bootstrap 5-95% interval,
* LRT p-value and delta-AIC vs the stationary fit,
* the 100-year effective return level evaluated at the first and last year.

This gives the observed-data anchor for the synthetic non-stationary EVT
analysis in Appendix B: the fitted mu1 can be compared with the upper-bound
drift implied by the 2015 -> 2100 potential-height increase.

Outputs ``img/worst/gauge_fit_ns_{station}.pdf`` and
``data/worst/gauge_fit_ns_{station}.json``.

Run::

    python -m worst.gauge_fit_ns --station 8761724      # Grand Isle, LA
    python -m worst.gauge_fit_ns --station 8735180      # Dauphin Island, AL
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, genextreme

from .constants import DATA_PATH, FIGURE_PATH
from .gauge_fit import POTENTIAL_HEIGHT_M, _setup_plt

RNG = np.random.default_rng(42)
N_BOOT = 500


def _nll_stationary(theta: np.ndarray, z: np.ndarray) -> float:
    mu, log_sigma, xi = theta
    return -np.sum(genextreme.logpdf(z, c=-xi, loc=mu, scale=np.exp(log_sigma)))


def _nll_ns(theta: np.ndarray, z: np.ndarray, tc: np.ndarray) -> float:
    mu0, mu1, log_sigma, xi = theta
    return -np.sum(
        genextreme.logpdf(z, c=-xi, loc=mu0 + mu1 * tc, scale=np.exp(log_sigma))
    )


def fit_ns(z: np.ndarray, years: np.ndarray) -> Dict:
    """Fit stationary and mu(t)-linear GEVs; return params + tests."""
    tc = years - years.mean()
    # stationary start values from scipy's own fit
    c0, mu_s, sig_s = genextreme.fit(z)
    th_s = minimize(
        _nll_stationary,
        x0=[mu_s, np.log(sig_s), -c0],
        args=(z,),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
    )
    th_n = minimize(
        _nll_ns,
        x0=[th_s.x[0], 0.0, th_s.x[1], th_s.x[2]],
        args=(z, tc),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 8000},
    )
    lrt = 2.0 * (th_s.fun - th_n.fun)
    p = float(chi2.sf(lrt, df=1))
    return {
        "stationary": {
            "mu": th_s.x[0],
            "log_sigma": th_s.x[1],
            "xi": th_s.x[2],
            "nll": th_s.fun,
            "aic": 2 * 3 + 2 * th_s.fun,
        },
        "nonstationary": {
            "mu0": th_n.x[0],
            "mu1_per_yr": th_n.x[1],
            "log_sigma": th_n.x[2],
            "xi": th_n.x[3],
            "nll": th_n.fun,
            "aic": 2 * 4 + 2 * th_n.fun,
        },
        "lrt": float(lrt),
        "lrt_p": p,
        "delta_aic": float((2 * 3 + 2 * th_s.fun) - (2 * 4 + 2 * th_n.fun)),
    }


def rl_at(theta: Dict, tc: float, rp: float = 100.0) -> float:
    """Effective return level of the NS fit at centred time tc."""
    mu = theta["mu0"] + theta["mu1_per_yr"] * tc
    return float(
        genextreme.isf(
            1.0 / rp, c=-theta["xi"], loc=mu, scale=np.exp(theta["log_sigma"])
        )
    )


def bootstrap_mu1(
    theta: Dict, years: np.ndarray, n_boot: int = N_BOOT
) -> Tuple[float, float]:
    """Parametric bootstrap 5-95% interval for mu1."""
    tc = years - years.mean()
    mu1s = []
    for _ in range(n_boot):
        mu = theta["mu0"] + theta["mu1_per_yr"] * tc
        z = genextreme.rvs(
            c=-theta["xi"],
            loc=mu,
            scale=np.exp(theta["log_sigma"]),
            size=len(years),
            random_state=RNG,
        )
        try:
            f = fit_ns(z, years)
            mu1s.append(f["nonstationary"]["mu1_per_yr"])
        except Exception:
            continue
    lo, hi = np.percentile(mu1s, [5, 95])
    return float(lo), float(hi)


def analyze_station_ns(
    station: str,
    start: int = 1980,
    end: int = 2025,
    method: str = "robust",
    make_figure: bool = True,
) -> dict:
    from comp.annual_max import annual_maxima

    df = annual_maxima(station, start, end, method=method)
    if df.empty:
        raise RuntimeError(f"no usable years for station {station}")
    z = df.ann_max_m.to_numpy(dtype=float)
    years = df.index.to_numpy(dtype=float)
    fit = fit_ns(z, years)
    ns = fit["nonstationary"]
    lo, hi = bootstrap_mu1(ns, years)
    tc0, tc1 = years[0] - years.mean(), years[-1] - years.mean()
    out = {
        "station": station,
        "n_years": int(len(z)),
        "years": [int(y) for y in years],
        "mu1_m_per_decade": 10.0 * ns["mu1_per_yr"],
        "mu1_boot_5_95_m_per_decade": [10.0 * lo, 10.0 * hi],
        "lrt_p": fit["lrt_p"],
        "delta_aic_ns_minus_stat": -fit["delta_aic"],
        "rl100_first_year_m": rl_at(ns, tc0),
        "rl100_last_year_m": rl_at(ns, tc1),
        "stationary": {k: float(v) for k, v in fit["stationary"].items()},
        "nonstationary": {k: float(v) for k, v in ns.items()},
        "z_star_m": POTENTIAL_HEIGHT_M.get(station),
    }
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, f"gauge_fit_ns_{station}.json"), "w") as f:
        json.dump(out, f, indent=1)

    if make_figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _setup_plt()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.scatter(years, z, s=14, color="k", zorder=3, label="annual maxima")
        mu_line = ns["mu0"] + ns["mu1_per_yr"] * (years - years.mean())
        ax.plot(
            years,
            mu_line,
            color="tab:red",
            label=rf"$\mu(t)$: {10*ns['mu1_per_yr']:+.3f} m decade$^{{-1}}$ "
            rf"(p={fit['lrt_p']:.2f})",
        )
        rl = [rl_at(ns, t - years.mean()) for t in years]
        ax.plot(
            years, rl, color="tab:blue", ls="--", label="100-yr effective return level"
        )
        ax.set_xlabel("year")
        ax.set_ylabel("annual max surge residual [m]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_PATH, f"gauge_fit_ns_{station}.pdf"))
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=str, default="8761724")
    parser.add_argument("--start", type=int, default=1980)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    out = analyze_station_ns(args.station, args.start, args.end)
    print(json.dumps({k: v for k, v in out.items() if k != "years"}, indent=1))


if __name__ == "__main__":
    main()
