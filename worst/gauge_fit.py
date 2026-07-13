"""Bounded vs unbounded GEV fits to real annual-maximum surge residuals.

Closes the loop between the paper's Bayesian-optimization potential-height
results and its EVT theory: the annual maxima of de-tided NOAA tide-gauge
surge residuals (from :mod:`comp.annual_max`) are fitted with

* case I  -- Weibull-class GEV with the upper bound FIXED at the site's
  potential height ``z*`` (``worst.tens.fit_gev_upper_bound_known``), and
* case II -- ordinary unbounded maximum-likelihood GEV
  (``worst.tens.fit_gev_upper_bound_not_known``),

and the two return-level curves are compared against Gringorten empirical
plotting positions, with 5-95% parametric-bootstrap envelopes and the bound
drawn as a horizontal line. Outputs: ``img/worst/gauge_fit_{station}.pdf``
and ``data/worst/gauge_fit_{station}.json``.

Caveats: the potential height is a TC-only wind+pressure surge ceiling, while
the observed annual maxima include non-TC events (cold fronts etc.) -- those
are far below the bound, so the bound still applies to the annual maximum.
Gauge failure at the peak (e.g. Grand Isle in Ida 2021) truncates that year's
maximum; such years are flagged upstream (``max_at_gap_edge``) and shown as
open markers.

Run::

    python -m worst.gauge_fit --station 8761724      # Grand Isle, LA
    python -m worst.gauge_fit --station 8735180      # Dauphin Island, AL
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import genextreme

from .constants import DATA_PATH, FIGURE_PATH
from .utils import z_star_from_alpha_beta_gamma

# --------------------------------------------------------------------------- #
# Potential heights z* [m] for the 2025 climate at the seven along-coast NOAA
# stations of the multi-site experiment (paper worst_new.tex, section
# "Spatial variation across the New Orleans coastline").
#
# PROVENANCE: values are the "max1" (August-2025 climate, CLE15 profile,
# potential size/intensity from CESM2-r4i1p1f1 SSP5-8.5) maximum SSH found by
# Bayesian optimization, listed per site in the commented results block of
# /Users/simon/thesis/paper/worst_new.tex (lines ~550-556, "%Pensacola, max1:
# 4.191 m, ..." ) and rounded in the commented table worst:tab:results_summary
# (same values in chapters/worst_thesis.tex ~450-456). Station-id <-> name
# mapping from adbo/plot.py:39-45.
#
# TODO(verify): the per-site numbers come from a commented-out block of the
# paper source, not a machine-readable artifact; regenerate from the BO
# experiment outputs (exp/ or data/bo_exp/) before quoting in the thesis.
# --------------------------------------------------------------------------- #
POTENTIAL_HEIGHT_M: Dict[str, float] = {
    "8729840": 4.191,  # Pensacola, FL
    "8735180": 3.567,  # Dauphin Island, AL
    "8760922": 1.879,  # Pilots Station East, S.W. Pass, LA
    "8761724": 5.886,  # Grand Isle, LA
    "8762075": 4.903,  # Port Fourchon, Belle Pass, LA
    "8762482": 11.694,  # West Bank 1, Bayou Gauche, LA
    "8764044": 9.911,  # Berwick, Atchafalaya River, LA
}

# 2097-climate ("max2") counterparts from the same commented block, kept for
# reference/plots only: 8729840: 4.772, 8735180: 3.946, 8760922: 2.420,
# 8761724: 7.141, 8762075: 5.274, 8762482: 12.714, 8764044: 11.102.

RETURN_PERIODS_KEY = (100.0, 500.0)  # RVs reported in the results json
N_BOOT = 200  # parametric-bootstrap resamples
RP_GRID = np.unique(
    np.concatenate([np.logspace(np.log10(1.05), 3.0, 120), list(RETURN_PERIODS_KEY)])
)


def gringorten_rp(n: int) -> np.ndarray:
    """Gringorten plotting-position return periods for ``n`` sorted maxima.

    The i-th smallest of ``n`` annual maxima gets non-exceedance probability
    ``p_i = (i - 0.44) / (n + 0.12)`` and return period ``1 / (1 - p_i)``.

    Args:
        n (int): Sample size.

    Returns:
        np.ndarray: Return periods (years), ascending with the sorted data.

    Examples:
        >>> [round(float(t), 2) for t in gringorten_rp(5)]
        [1.12, 1.44, 2.0, 3.28, 9.14]
    """
    i = np.arange(1, n + 1)
    p = (i - 0.44) / (n + 0.12)
    return 1.0 / (1.0 - p)


def return_level(alpha: float, beta: float, gamma: float, rp: np.ndarray) -> np.ndarray:
    """GEV return level(s) for return period(s) ``rp`` (annual maxima).

    Args:
        alpha (float): Location parameter.
        beta (float): Scale parameter.
        gamma (float): Shape parameter (Coles convention; scipy c = -gamma).
        rp (np.ndarray): Return period(s) in years.

    Returns:
        np.ndarray: Return level(s) [m].

    Examples:
        >>> round(float(return_level(0.0, 1.0, -1.0, 100.0)), 3)
        0.99
    """
    return genextreme.isf(
        1.0 / np.asarray(rp, dtype=float), c=-gamma, loc=alpha, scale=beta
    )


def fit_cases(data: np.ndarray, z_star: float) -> Dict[str, Tuple[float, float, float]]:
    """Fit case I (bound known at ``z_star``) and case II (unbounded).

    Uses the ``worst.tens`` L-BFGS fitters (imported lazily -- ``worst.tens``
    pulls in TensorFlow). Case I returns NaNs if ``z_star`` is below the
    sample maximum (degenerate likelihood, see ``worst.tens._fit_lbfgs_known``).

    Args:
        data (np.ndarray): Annual maxima [m].
        z_star (float): Known upper bound [m] (site potential height).

    Returns:
        Dict[str, Tuple[float, float, float]]: {"I": (alpha, beta, gamma),
            "II": (alpha, beta, gamma)} in the Coles convention.
    """
    from .tens import fit_gev_upper_bound_known, fit_gev_upper_bound_not_known

    data = np.asarray(data, dtype=float)
    beta_guess = float(np.std(data)) or 1.0
    fit_i = fit_gev_upper_bound_known(
        data, z_star, beta_guess=beta_guess, gamma_guess=-0.1, method="lbfgs"
    )
    fit_ii = fit_gev_upper_bound_not_known(
        data,
        alpha_guess=float(np.mean(data)),
        beta_guess=beta_guess,
        gamma_guess=-0.1,
        method="lbfgs",
    )
    return {"I": tuple(map(float, fit_i)), "II": tuple(map(float, fit_ii))}


def bootstrap_bands(
    params: Tuple[float, float, float],
    case: str,
    n: int,
    z_star: Optional[float] = None,
    n_boot: int = N_BOOT,
    rps: np.ndarray = RP_GRID,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """5-95% parametric-bootstrap envelope of the return-level curve.

    Resamples ``n`` annual maxima from the fitted GEV, refits with the same
    method (case I keeps the bound fixed at ``z_star``), and takes pointwise
    percentiles of the refitted curves. NaN refits (possible only if a refit
    degenerates) are dropped by ``nanpercentile``.

    Args:
        params (Tuple[float, float, float]): Fitted (alpha, beta, gamma).
        case (str): "I" (refit with known bound) or "II" (unbounded refit).
        n (int): Sample size per resample (= number of observed years).
        z_star (Optional[float]): Bound for case I refits.
        n_boot (int, optional): Resamples. Defaults to N_BOOT.
        rps (np.ndarray, optional): Return-period grid. Defaults to RP_GRID.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        Dict[str, np.ndarray]: {"rp": grid, "lo": 5th pct, "hi": 95th pct}.
    """
    from .tens import fit_gev_upper_bound_known, fit_gev_upper_bound_not_known

    alpha, beta, gamma = params
    rng = np.random.default_rng(seed)
    curves = np.full((n_boot, rps.size), np.nan)
    for k in range(n_boot):
        sample = genextreme.rvs(
            c=-gamma, loc=alpha, scale=beta, size=n, random_state=rng
        )
        if case == "I":
            a, b, g = fit_gev_upper_bound_known(
                sample,
                z_star,
                beta_guess=beta,
                gamma_guess=min(gamma, -1e-3),
                method="lbfgs",
            )
        else:
            a, b, g = fit_gev_upper_bound_not_known(
                sample,
                alpha_guess=alpha,
                beta_guess=beta,
                gamma_guess=gamma,
                method="lbfgs",
            )
        if np.isfinite([a, b, g]).all():
            curves[k] = return_level(a, b, g, rps)
    return {
        "rp": rps,
        "lo": np.nanpercentile(curves, 5, axis=0),
        "hi": np.nanpercentile(curves, 95, axis=0),
    }


def _setup_plt():
    """Headless matplotlib with the repo's paper style (as comp.validate)."""
    import matplotlib

    matplotlib.use("Agg")
    from sithom.plot import plot_defaults

    plot_defaults()
    import matplotlib.pyplot as plt

    return plt


def plot_gauge_fit(
    station: str,
    name: str,
    data: np.ndarray,
    gap_flags: np.ndarray,
    fits: dict,
    bands: dict,
    z_star: float,
    out_pdf: str,
) -> None:
    """Return-period figure: empirical points, case I/II curves + envelopes.

    Args:
        station (str): CO-OPS station id (used in the title).
        name (str): Station name.
        data (np.ndarray): Annual maxima [m] (unsorted).
        gap_flags (np.ndarray): Bool per year; True = max near a data gap
            (plotted as an open marker -- possibly truncated by gauge failure).
        fits (dict): Output of :func:`fit_cases`.
        bands (dict): {"I": ..., "II": ...} from :func:`bootstrap_bands`.
        z_star (float): Potential height [m], drawn as a horizontal line.
        out_pdf (str): Output path.
    """
    plt = _setup_plt()
    from sithom.plot import get_dim

    fig, ax = plt.subplots(figsize=get_dim(fraction_of_line_width=0.75, ratio=0.75))
    order = np.argsort(data)
    rp_emp = gringorten_rp(len(data))
    z_sorted, gap_sorted = data[order], np.asarray(gap_flags, bool)[order]
    ax.scatter(
        rp_emp[~gap_sorted],
        z_sorted[~gap_sorted],
        s=14,
        color="black",
        zorder=5,
        label="Observed annual maxima",
    )
    if gap_sorted.any():
        ax.scatter(
            rp_emp[gap_sorted],
            z_sorted[gap_sorted],
            s=16,
            facecolors="none",
            edgecolors="black",
            zorder=5,
            label="Maximum near data gap",
        )
    for case, color in (("I", "green"), ("II", "orange")):
        alpha, beta, gamma = fits[case]
        if not np.isfinite([alpha, beta, gamma]).all():
            continue
        lab = (
            rf"I: bounded at $z^*$ ($\beta$={beta:.2f}, $\gamma$={gamma:.2f})"
            if case == "I"
            else rf"II: unbounded ($\beta$={beta:.2f}, $\gamma$={gamma:.2f})"
        )
        ax.semilogx(
            RP_GRID,
            return_level(alpha, beta, gamma, RP_GRID),
            color=color,
            lw=1.4,
            label=lab,
        )
        b = bands.get(case)
        if b is not None:
            ax.fill_between(
                b["rp"],
                b["lo"],
                b["hi"],
                color=color,
                alpha=0.18,
                lw=0,
                label=f"{case}: 5-95% bootstrap",
            )
    ax.axhline(
        z_star,
        color="purple",
        ls="--",
        lw=1.0,
        label=rf"Potential height $z^*$ = {z_star:.2f} m",
    )
    ax.set_xscale("log")
    ax.set_xlim(1, RP_GRID[-1])
    ax.set_ylim(0, max(z_star * 1.15, float(np.max(data)) * 1.3))
    ax.set_xlabel("Return period [years]")
    ax.set_ylabel("Annual-maximum surge residual [m]")
    ax.set_title(f"{name} ({station}), $n$={len(data)} years", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    plt.close(fig)


def analyze_station(
    station: str,
    start: int = 1980,
    end: int = 2025,
    n_boot: int = N_BOOT,
    method: str = "robust",
    refresh: bool = False,
    make_figure: bool = True,
) -> dict:
    """Full case-I/case-II analysis of one gauge's annual maxima.

    Loads (or computes) the annual maxima via :mod:`comp.annual_max`, fits
    both cases, bootstraps the envelopes, saves the figure and a results json.

    Args:
        station (str): CO-OPS station id (must be in POTENTIAL_HEIGHT_M).
        start (int, optional): First year. Defaults to 1980.
        end (int, optional): Last year. Defaults to 2025.
        n_boot (int, optional): Bootstrap resamples. Defaults to 200.
        method (str, optional): De-tiding method. Defaults to "robust".
        refresh (bool, optional): Recompute the annual maxima. Defaults False.
        make_figure (bool, optional): Save the pdf. Defaults to True.

    Returns:
        dict: The results dictionary that is also written to json.
    """
    from comp.annual_max import annual_maxima

    if station not in POTENTIAL_HEIGHT_M:
        raise ValueError(
            f"no potential height recorded for station {station}; "
            f"known: {sorted(POTENTIAL_HEIGHT_M)}"
        )
    z_star = POTENTIAL_HEIGHT_M[station]
    df = annual_maxima(station, start, end, method=method, refresh=refresh)
    if df.empty:
        raise RuntimeError(f"no usable years for station {station}")
    data = df.ann_max_m.to_numpy(dtype=float)
    name = str(df["name"].iloc[0])
    print(
        f"\n{name} ({station}): n={len(data)} annual maxima, "
        f"sample max={data.max():.2f} m, z*={z_star:.3f} m"
    )
    if data.max() >= z_star:
        print(
            "WARNING: sample maximum exceeds the potential height -- the "
            "bounded fit will be degenerate (NaN)."
        )
    fits = fit_cases(data, z_star)
    bands = {
        case: bootstrap_bands(
            fits[case], case, len(data), z_star=z_star, n_boot=n_boot, seed=0
        )
        for case in ("I", "II")
        if np.isfinite(fits[case]).all()
    }
    results = dict(
        station=station,
        name=name,
        start=start,
        end=end,
        n_years=int(len(data)),
        years=[int(y) for y in df.year],
        detide_method=method,
        sample_max_m=float(data.max()),
        z_star_m=float(z_star),
        z_star_provenance=(
            "BO potential height, Aug-2025 climate, CLE15/"
            "CESM2-r4i1p1f1 SSP5-8.5; worst_new.tex commented "
            "results block (TODO(verify): regenerate from BO "
            "outputs)"
        ),
        n_boot=int(n_boot),
    )
    for case in ("I", "II"):
        alpha, beta, gamma = fits[case]
        entry = dict(alpha=alpha, beta=beta, gamma=gamma)
        if np.isfinite([alpha, beta, gamma]).all():
            entry["upper_bound_m"] = (
                float(z_star_from_alpha_beta_gamma(alpha, beta, gamma))
                if gamma < 0
                else None
            )
            for rp in RETURN_PERIODS_KEY:
                rv = float(return_level(alpha, beta, gamma, rp))
                entry[f"rv{int(rp)}_m"] = rv
                if case in bands:
                    j = int(np.argmin(np.abs(bands[case]["rp"] - rp)))
                    entry[f"rv{int(rp)}_ci90_m"] = [
                        float(bands[case]["lo"][j]),
                        float(bands[case]["hi"][j]),
                    ]
        results[f"case_{case}"] = entry
        print(
            f"  case {case}: alpha={alpha:.3f} beta={beta:.3f} "
            f"gamma={gamma:.3f} "
            + " ".join(
                f"RV{int(rp)}={entry.get(f'rv{int(rp)}_m', float('nan')):.2f}m"
                for rp in RETURN_PERIODS_KEY
            )
        )
    if make_figure:
        plot_gauge_fit(
            station,
            name,
            data,
            df.max_at_gap_edge.to_numpy(),
            fits,
            bands,
            z_star,
            os.path.join(FIGURE_PATH, f"gauge_fit_{station}.pdf"),
        )
    out_json = os.path.join(DATA_PATH, f"gauge_fit_{station}.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"wrote {out_json}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--station",
        default="8761724",
        help="CO-OPS station id (default 8761724, Grand Isle LA)",
    )
    ap.add_argument("--start", type=int, default=1980)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--method", default="robust", choices=["robust", "ols"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    analyze_station(
        a.station, a.start, a.end, n_boot=a.n_boot, method=a.method, refresh=a.refresh
    )


if __name__ == "__main__":
    main()
