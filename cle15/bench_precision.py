"""
Accuracy vs. cost trade-off for the numba CLE15 solver.

Sweeps each resolution / iteration knob independently while holding the
others at their current default values.  Reports timing (ms/call) and
accuracy (% rmax error relative to the pure-Python reference) for the
192-case benchmark grid (Vmax 20-90 m/s, r0 300-2000 km,
f in {3, 5, 7} x 1e-5 s^-1).

Note: r0 = 200 km is excluded because the high-Rossby degenerate regime
produces large bisection sensitivity that inflates the apparent error
without reflecting normal solver behaviour (see cle15.md).

Usage::

    python -m cle15.bench_precision          # print tables only
    python -m cle15.bench_precision --plot   # also save figures

The four knobs investigated are:

1. ``Nr_e04``         – E04 Euler grid points  (default 200 000)
2. ``num_pts_er11``   – ER11 points inside the bisection kernel  (default 5 000)
3. ``nx_intersect``   – intersection-check grid points  (default 4 000)
4. ``max_iter``       – bisection iterations  (default 50)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ── Reference implementation ─────────────────────────────────────────────────
from .cle15 import chavas_et_al_2015_profile as _py_profile

# ── Numba kernels / internals we need to call directly ───────────────────────
import cle15.cle15n as _nb_mod
from .cle15n import (
    _e04_outerwind_r0input_nondim_mm0,
    _bisect_rmaxr0_nb,
    _er11_radprof_with_convergence,
    _radprof_eyeadj,
    warmup as _warmup,
)
from .constants import (
    W_COOL_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    F_COR_DEFAULT,
    CDVARY_DEFAULT,
    CKCDVARY_DEFAULT,
    EYE_ADJ_DEFAULT,
    ALPHA_EYE_DEFAULT,
)

# ── Parameter grid ───────────────────────────────────────────────────────────
# r0 = 200 km is excluded from the benchmark grid because the high-Rossby
# degenerate regime (very small r0, very high Vmax, very low f) produces
# large sensitivity to bisection termination side and inflates the apparent
# error without reflecting normal solver behaviour.  The instability is
# documented separately (see cle15.md, MATLAB cross-validation section).
VMAX_VALS = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
R0_VALS = [300e3, 400e3, 600e3, 800e3, 1000e3, 1200e3, 1600e3, 2000e3]
FCOR_VALS = [3e-5, 5e-5, 7e-5]

ALL_CASES: List[Tuple[float, float, float]] = [
    (v, r, f) for v in VMAX_VALS for r in R0_VALS for f in FCOR_VALS
]
N = len(ALL_CASES)

# Fixed params
_DEFAULTS = dict(
    Cdvary=CDVARY_DEFAULT,
    C_d=CD_DEFAULT,
    w_cool=W_COOL_DEFAULT,
    CkCdvary=CKCDVARY_DEFAULT,
    CkCd_input=CK_CD_DEFAULT,
    eye_adj=EYE_ADJ_DEFAULT,
    alpha_eye=ALPHA_EYE_DEFAULT,
)


# ── Knob defaults ─────────────────────────────────────────────────────────────
DEFAULT_NR_E04 = 200_000
DEFAULT_NUM_PTS_ER11 = 5_000
DEFAULT_NX_INTERSECT = 4_000
DEFAULT_MAX_ITER = 50


# ── Low-level runner that accepts explicit knob values ────────────────────────


def _run_one(
    Vmax: float,
    r0: float,
    fcor: float,
    Nr_e04: int,
    num_pts_er11: int,
    nx_intersect: int,
    max_iter: int,
) -> float:
    """
    Run the numba CLE15 pipeline with explicit knob values.
    Returns rmax [m], or NaN on failure.
    """
    fcor = abs(fcor)
    CkCd = CK_CD_DEFAULT

    # E04 outer profile
    try:
        rrfracr0_E04, MMfracM0_E04 = _e04_outerwind_r0input_nondim_mm0(
            r0, fcor, CDVARY_DEFAULT, CD_DEFAULT, W_COOL_DEFAULT, Nr=Nr_e04
        )
        if len(rrfracr0_E04) < 2:
            return np.nan
    except Exception:
        return np.nan

    M0 = 0.5 * fcor * r0**2

    # Bisection
    rfracrm_max = 50.0
    rmaxr0_final, rmerger0, MmergeM0, _, _ = _bisect_rmaxr0_nb(
        r0,
        fcor,
        Vmax,
        CkCd,
        np.ascontiguousarray(rrfracr0_E04, dtype=np.float64),
        np.ascontiguousarray(MMfracM0_E04, dtype=np.float64),
        M0,
        0.001,
        0.75,
        max_iter,
        1e-6,
        rfracrm_max,
        num_pts_er11,
        nx_intersect,
    )

    if np.isnan(rmaxr0_final):
        return np.nan

    rmax = rmaxr0_final * r0

    # Final high-res ER11 profile (always use fine grid for fair rmax estimate)
    drfracrm = 0.01
    if rmax > 100e3:
        drfracrm /= 10.0
    rfracrm_max_fine = 50.0
    num_fine = max(500, int(rfracrm_max_fine / drfracrm) + 1)
    rr_fine = np.linspace(0, rfracrm_max_fine, num_fine) * rmax
    VV_fine = _er11_radprof_with_convergence(Vmax, rmax, fcor, CkCd, rr_fine)

    if np.all(np.isnan(VV_fine)):
        return np.nan

    valid = ~np.isnan(VV_fine) & (VV_fine >= 0)
    if not np.any(valid):
        return np.nan

    rmax_out = float(rr_fine[valid][np.argmax(VV_fine[valid])])
    return rmax_out


# ── Reference rmax values (computed once from pure-Python) ────────────────────


def _get_reference_rmaxes() -> np.ndarray:
    ref = np.full(N, np.nan)
    for i, (Vmax, r0, fcor) in enumerate(ALL_CASES):
        res = _py_profile(
            Vmax,
            r0,
            fcor,
            _DEFAULTS["Cdvary"],
            _DEFAULTS["C_d"],
            _DEFAULTS["w_cool"],
            _DEFAULTS["CkCdvary"],
            _DEFAULTS["CkCd_input"],
            _DEFAULTS["eye_adj"],
            _DEFAULTS["alpha_eye"],
        )
        ref[i] = res[2]  # rmax
    return ref


# ── Sweep function ────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    name: str
    values: List
    ms_per_call: List[float] = field(default_factory=list)
    mean_err: List[float] = field(default_factory=list)
    max_err: List[float] = field(default_factory=list)
    n_fail: List[int] = field(default_factory=list)


def _sweep(
    name: str,
    values: list,
    ref_rmax: np.ndarray,
    Nr_e04_fn,
    num_pts_fn,
    nx_fn,
    max_iter_fn,
    n_repeats: int = 3,
) -> SweepResult:
    result = SweepResult(name=name, values=values)

    for v in values:
        Nr_e04 = Nr_e04_fn(v)
        num_pts_er11 = num_pts_fn(v)
        nx_intersect = nx_fn(v)
        max_iter = max_iter_fn(v)

        rmaxes = np.full(N, np.nan)
        # Timed runs
        t0 = time.perf_counter()
        for _rep in range(n_repeats):
            for i, (Vmax, r0, fcor) in enumerate(ALL_CASES):
                rmaxes[i] = _run_one(
                    Vmax, r0, fcor, Nr_e04, num_pts_er11, nx_intersect, max_iter
                )
        elapsed = (time.perf_counter() - t0) / n_repeats

        ms = elapsed / N * 1000.0
        valid = np.isfinite(rmaxes) & np.isfinite(ref_rmax)
        n_fail = int(np.sum(~np.isfinite(rmaxes)))
        if np.any(valid):
            errs = np.abs(rmaxes[valid] - ref_rmax[valid]) / ref_rmax[valid] * 100.0
            mean_e = float(np.mean(errs))
            max_e = float(np.max(errs))
        else:
            mean_e = np.nan
            max_e = np.nan

        result.ms_per_call.append(ms)
        result.mean_err.append(mean_e)
        result.max_err.append(max_e)
        result.n_fail.append(n_fail)

    return result


def _print_result(r: SweepResult) -> None:
    col = max(len(str(v)) for v in r.values)
    col = max(col, 8)
    hdr = f"  {'value':>{col}}  {'ms/call':>8}  {'mean err%':>9}  {'max err%':>9}  {'fails':>5}"
    print(f"\n{'─'*60}")
    print(f"  Sweep: {r.name}")
    print(hdr)
    print(f"  {'─'*len(hdr.strip())}")
    for i, v in enumerate(r.values):
        marker = " ◀ default" if v == _get_default(r.name) else ""
        print(
            f"  {str(v):>{col}}  {r.ms_per_call[i]:>8.2f}  "
            f"{r.mean_err[i]:>9.3f}  {r.max_err[i]:>9.3f}  "
            f"{r.n_fail[i]:>5}{marker}"
        )


def _get_default(name: str) -> int:
    return {
        "Nr_e04": DEFAULT_NR_E04,
        "num_pts_er11": DEFAULT_NUM_PTS_ER11,
        "nx_intersect": DEFAULT_NX_INTERSECT,
        "max_iter": DEFAULT_MAX_ITER,
    }[name]


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_results(
    sweeps: list,
    ref_rmax: np.ndarray,
    rmaxes_fast: np.ndarray,
    rmaxes_default: np.ndarray,
) -> None:
    """
    Produce and save three figures that show where precision matters most.

    Parameters
    ----------
    sweeps:
        List of four ``SweepResult`` objects (Nr_e04, num_pts_er11,
        nx_intersect, max_iter) as returned by the four ``_sweep`` calls.
    ref_rmax:
        Array of reference rmax values (km) for all 75 cases.
    rmaxes_fast:
        Numba rmax estimates at the *fast* preset for each case.
    rmaxes_default:
        Numba rmax estimates at the *default* preset for each case.
    """
    import os

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from sithom.plot import plot_defaults, label_subplots, get_dim

    from .constants import FIGURE_PATH

    plot_defaults()
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # ── Colours ───────────────────────────────────────────────────────────────
    C_TIME = "#2166ac"
    C_MEAN = "#d6604d"
    C_MAX = "#f4a582"

    # ── Knob metadata ─────────────────────────────────────────────────────────
    LOG_KNOBS = {"Nr_e04", "num_pts_er11", "nx_intersect"}
    DEFAULTS = {
        "Nr_e04": DEFAULT_NR_E04,
        "num_pts_er11": DEFAULT_NUM_PTS_ER11,
        "nx_intersect": DEFAULT_NX_INTERSECT,
        "max_iter": DEFAULT_MAX_ITER,
    }
    LABELS = {
        "Nr_e04": r"$N_{r,\mathrm{E04}}$",
        "num_pts_er11": r"$N_{\mathrm{ER11}}$",
        "nx_intersect": r"$N_{x,\mathrm{intersect}}$",
        "max_iter": "Max iterations",
    }

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 – knob sweep accuracy + timing
    # ═════════════════════════════════════════════════════════════════════════
    ncols = len(sweeps)
    fig1, axes = plt.subplots(
        1, ncols, figsize=get_dim(fraction_of_line_width=1.5, ratio=0.4)
    )

    for ax, sweep in zip(axes, sweeps):
        xs = sweep.values
        log_x = sweep.name in LOG_KNOBS

        ax2 = ax.twinx()

        # Timing (left axis, blue)
        lns1 = ax.plot(
            xs,
            sweep.ms_per_call,
            color=C_TIME,
            marker="o",
            ms=4,
            lw=1.5,
            label="ms / call",
            zorder=3,
        )
        ax.set_ylabel("ms / call", color=C_TIME)
        ax.tick_params(axis="y", labelcolor=C_TIME)
        if log_x:
            ax.set_xscale("log")

        # Mean error (right axis, red)
        lns2 = ax2.plot(
            xs,
            sweep.mean_err,
            color=C_MEAN,
            marker="s",
            ms=4,
            lw=1.5,
            label="mean err %",
            zorder=3,
        )
        # Max error (right axis, orange)
        lns3 = ax2.plot(
            xs,
            sweep.max_err,
            color=C_MAX,
            marker="^",
            ms=4,
            lw=1.5,
            ls="--",
            label="max err %",
            zorder=3,
        )
        ax2.set_ylabel("rmax error (%)", color=C_MEAN)
        ax2.tick_params(axis="y", labelcolor=C_MEAN)
        ax2.set_ylim(bottom=0)

        # Default marker
        ax.axvline(
            DEFAULTS[sweep.name],
            color="0.4",
            ls=":",
            lw=1,
            label="default",
        )

        ax.set_xlabel(LABELS[sweep.name])
        ax.set_title(sweep.name.replace("_", " "), fontsize=8)

        # Unified legend (only on first panel)
        if ax is axes[0]:
            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=7, loc="upper right")

    label_subplots(axes)
    fig1.suptitle("Solver knob sweep: accuracy vs. cost", y=1.02)
    fig1.tight_layout()
    out1 = os.path.join(FIGURE_PATH, "bench_precision_knobs.pdf")
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {out1}")

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 2 – per-case rmax error heatmaps (fast vs default, by fcor)
    # ═════════════════════════════════════════════════════════════════════════
    nf = len(FCOR_VALS)
    # Use GridSpec with a dedicated narrow colorbar column so the cbar never
    # overlaps the heatmap panels.
    from matplotlib.gridspec import GridSpec

    fig2_w, fig2_h = get_dim(fraction_of_line_width=1.0, ratio=1.2)
    fig2 = plt.figure(figsize=(fig2_w, fig2_h))
    gs = GridSpec(
        nf,
        3,
        figure=fig2,
        width_ratios=[1, 1, 0.07],
        hspace=0.35,
        wspace=0.45,
    )
    axes2 = np.array(
        [[fig2.add_subplot(gs[fi, ci]) for ci in range(2)] for fi in range(nf)]
    )
    cax = fig2.add_subplot(gs[:, 2])  # spans all rows, rightmost column

    # Build (Vmax, r0) error matrices for each fcor
    Vm = np.array(VMAX_VALS)
    R0m = np.array(R0_VALS) / 1e3  # km

    all_errs_fast = np.full((len(FCOR_VALS), len(VMAX_VALS), len(R0_VALS)), np.nan)
    all_errs_def = np.full((len(FCOR_VALS), len(VMAX_VALS), len(R0_VALS)), np.nan)

    for i, (vmax, r0, fcor) in enumerate(ALL_CASES):
        fi = FCOR_VALS.index(fcor)
        vi = VMAX_VALS.index(vmax)
        ri = R0_VALS.index(r0)
        ref = ref_rmax[i]
        if np.isfinite(ref) and ref > 0:
            if np.isfinite(rmaxes_fast[i]):
                all_errs_fast[fi, vi, ri] = abs(rmaxes_fast[i] - ref) / ref * 100.0
            if np.isfinite(rmaxes_default[i]):
                all_errs_def[fi, vi, ri] = abs(rmaxes_default[i] - ref) / ref * 100.0

    vmax_err = np.nanmax([np.nanmax(all_errs_fast), np.nanmax(all_errs_def)])
    vmax_err = max(vmax_err, 0.01)  # guard against all-zero
    norm = Normalize(vmin=0, vmax=vmax_err)
    cmap = "YlOrRd"

    for fi, fcor in enumerate(FCOR_VALS):
        for ci, (errs, title_suffix) in enumerate(
            [(all_errs_fast[fi], "fast"), (all_errs_def[fi], "default")]
        ):
            ax = axes2[fi, ci]
            im = ax.imshow(
                errs.T,  # rows=r0, cols=Vmax
                origin="lower",
                aspect="auto",
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(VMAX_VALS)))
            ax.set_xticklabels([f"{int(v)}" for v in VMAX_VALS], fontsize=7)
            ax.set_yticks(range(len(R0_VALS)))
            ax.set_yticklabels([f"{int(r)}" for r in R0m], fontsize=7)
            if ci == 0:
                ax.set_ylabel(r"$r_0$ (km)", fontsize=8)
                ax.text(
                    -0.45,
                    0.5,
                    rf"$f={fcor*1e5:.0f}\times10^{{-5}}$ s$^{{-1}}$",
                    transform=ax.transAxes,
                    va="center",
                    rotation=90,
                    fontsize=7,
                )
            if fi == 0:
                ax.set_title(f"preset: {title_suffix}", fontsize=8)
            if fi == nf - 1:
                ax.set_xlabel(r"$V_{\max}$ (m s$^{-1}$)", fontsize=8)

    # Shared colourbar in its own dedicated axis
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, cax=cax)
    cbar.set_label(r"$r_{\mathrm{max}}$ error (%)", fontsize=8)

    label_subplots(axes2.ravel().tolist())
    fig2.suptitle(
        "Per-case rmax error: fast vs. default preset\n"
        r"rows = $f_{\mathrm{cor}}$, columns = solver preset",
        y=1.01,
    )
    out2 = os.path.join(FIGURE_PATH, "bench_precision_regime.pdf")
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {out2}")

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 – fast vs. default rmax scatter, coloured by fcor
    # ═════════════════════════════════════════════════════════════════════════
    FCOR_COLORS = ["#1b7837", "#762a83", "#e08214"]
    fig3, ax3 = plt.subplots(figsize=get_dim(ratio=1.0))

    all_rmax_fast_km = rmaxes_fast / 1e3
    all_rmax_def_km = rmaxes_default / 1e3
    ref_km = ref_rmax / 1e3

    for fi, fcor in enumerate(FCOR_VALS):
        mask = np.array([ALL_CASES[i][2] == fcor for i in range(N)], dtype=bool)
        mask &= np.isfinite(all_rmax_fast_km) & np.isfinite(all_rmax_def_km)
        ax3.scatter(
            all_rmax_def_km[mask],
            all_rmax_fast_km[mask],
            color=FCOR_COLORS[fi],
            s=30,
            alpha=0.75,
            label=rf"$f={fcor*1e5:.0f}\times10^{{-5}}$ s$^{{-1}}$",
            zorder=3,
        )

    # 1:1 reference line
    lim_lo = float(np.nanmin([all_rmax_def_km, all_rmax_fast_km])) * 0.95
    lim_hi = float(np.nanmax([all_rmax_def_km, all_rmax_fast_km])) * 1.05
    ax3.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, zorder=1)
    ax3.set_xlim(lim_lo, lim_hi)
    ax3.set_ylim(lim_lo, lim_hi)

    ax3.set_xlabel(r"$r_{\max}$ default preset (km)")
    ax3.set_ylabel(r"$r_{\max}$ fast preset (km)")
    ax3.set_title("Fast vs. default preset: rmax comparison")

    # Bias annotation
    valid_both = np.isfinite(all_rmax_fast_km) & np.isfinite(all_rmax_def_km)
    bias = np.mean(all_rmax_fast_km[valid_both] - all_rmax_def_km[valid_both])
    rmse = np.sqrt(
        np.mean((all_rmax_fast_km[valid_both] - all_rmax_def_km[valid_both]) ** 2)
    )
    ax3.text(
        0.04,
        0.96,
        f"bias = {bias:+.2f} km\nRMSE = {rmse:.2f} km",
        transform=ax3.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax3.legend(fontsize=8, loc="lower right")
    fig3.tight_layout()
    out3 = os.path.join(FIGURE_PATH, "bench_precision_scatter.pdf")
    fig3.savefig(out3, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {out3}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CLE15-numba accuracy vs. cost sweep")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save diagnostic figures to FIGURE_PATH after the sweeps.",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("  CLE15-numba: accuracy vs. cost sweep")
    print(f"  {N} test cases,  {3} timing repeats each")
    print("=" * 60)

    print("\nComputing pure-Python reference rmax values …", flush=True)
    ref_rmax = _get_reference_rmaxes()
    n_ref_fail = int(np.sum(~np.isfinite(ref_rmax)))
    print(f"  Done. Reference failures: {n_ref_fail}/{N}")

    print("\nWarming up numba JIT …", flush=True)
    _warmup()
    # Also warm up with a call using the smallest knob values so all
    # compiled specialisations are cached before we start timing.
    _run_one(50.0, 800e3, 5e-5, 1000, 100, 100, 5)
    print("  Done.", flush=True)

    # ── 1. E04 grid resolution ────────────────────────────────────────────────
    nr_vals = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    r1 = _sweep(
        "Nr_e04",
        nr_vals,
        ref_rmax,
        Nr_e04_fn=lambda v: v,
        num_pts_fn=lambda _: DEFAULT_NUM_PTS_ER11,
        nx_fn=lambda _: DEFAULT_NX_INTERSECT,
        max_iter_fn=lambda _: DEFAULT_MAX_ITER,
    )
    _print_result(r1)

    # ── 2. ER11 points inside bisection ──────────────────────────────────────
    er11_vals = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    r2 = _sweep(
        "num_pts_er11",
        er11_vals,
        ref_rmax,
        Nr_e04_fn=lambda _: DEFAULT_NR_E04,
        num_pts_fn=lambda v: v,
        nx_fn=lambda _: DEFAULT_NX_INTERSECT,
        max_iter_fn=lambda _: DEFAULT_MAX_ITER,
    )
    _print_result(r2)

    # ── 3. Intersection grid resolution ──────────────────────────────────────
    nx_vals = [50, 100, 200, 500, 1_000, 2_000, 4_000, 8_000]
    r3 = _sweep(
        "nx_intersect",
        nx_vals,
        ref_rmax,
        Nr_e04_fn=lambda _: DEFAULT_NR_E04,
        num_pts_fn=lambda _: DEFAULT_NUM_PTS_ER11,
        nx_fn=lambda v: v,
        max_iter_fn=lambda _: DEFAULT_MAX_ITER,
    )
    _print_result(r3)

    # ── 4. Bisection iterations ───────────────────────────────────────────────
    iter_vals = [5, 10, 15, 20, 30, 50, 75, 100]
    r4 = _sweep(
        "max_iter",
        iter_vals,
        ref_rmax,
        Nr_e04_fn=lambda _: DEFAULT_NR_E04,
        num_pts_fn=lambda _: DEFAULT_NUM_PTS_ER11,
        nx_fn=lambda _: DEFAULT_NX_INTERSECT,
        max_iter_fn=lambda v: v,
    )
    _print_result(r4)

    # ── Summary: optimised cheap preset ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Optimised 'fast' preset (from sweep results):")
    FAST = dict(Nr_e04=10_000, num_pts_er11=500, nx_intersect=500, max_iter=20)
    print(
        f"    Nr_e04={FAST['Nr_e04']:,}  num_pts_er11={FAST['num_pts_er11']:,}"
        f"  nx_intersect={FAST['nx_intersect']:,}  max_iter={FAST['max_iter']}"
    )

    rmaxes_fast = np.full(N, np.nan)
    t0 = time.perf_counter()
    for _rep in range(5):
        for i, (Vmax, r0, fcor) in enumerate(ALL_CASES):
            rmaxes_fast[i] = _run_one(
                Vmax,
                r0,
                fcor,
                FAST["Nr_e04"],
                FAST["num_pts_er11"],
                FAST["nx_intersect"],
                FAST["max_iter"],
            )
    elapsed_fast = (time.perf_counter() - t0) / 5

    valid = np.isfinite(rmaxes_fast) & np.isfinite(ref_rmax)
    errs_fast = np.abs(rmaxes_fast[valid] - ref_rmax[valid]) / ref_rmax[valid] * 100.0
    ms_fast = elapsed_fast / N * 1000.0

    rmaxes_default = np.full(N, np.nan)
    t0 = time.perf_counter()
    for _rep in range(5):
        for i, (Vmax, r0, fcor) in enumerate(ALL_CASES):
            rmaxes_default[i] = _run_one(
                Vmax,
                r0,
                fcor,
                DEFAULT_NR_E04,
                DEFAULT_NUM_PTS_ER11,
                DEFAULT_NX_INTERSECT,
                DEFAULT_MAX_ITER,
            )
    elapsed_default = (time.perf_counter() - t0) / 5
    ms_default = elapsed_default / N * 1000.0

    print(f"    ms/call  (fast):    {ms_fast:.2f}")
    print(f"    ms/call  (default): {ms_default:.2f}")
    print(f"    Speedup:            {ms_default/ms_fast:.1f}×")
    print(f"    rmax mean err%:     {np.mean(errs_fast):.3f}")
    print(f"    rmax max  err%:     {np.max(errs_fast):.3f}")
    print(f"    Failures:           {int(np.sum(~np.isfinite(rmaxes_fast)))}/{N}")
    print()

    if args.plot:
        print("Saving figures …", flush=True)
        plot_results([r1, r2, r3, r4], ref_rmax, rmaxes_fast, rmaxes_default)
        print("Done.")


if __name__ == "__main__":
    main()
