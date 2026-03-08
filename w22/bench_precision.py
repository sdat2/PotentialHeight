"""
Accuracy vs. cost trade-off for the numba CLE15 solver.

Sweeps each resolution / iteration knob independently while holding the
others at their current default values.  Reports timing (ms/call) and
accuracy (% rmax error relative to the pure-Python reference) for the
75-case benchmark grid.

Usage::

    python -m w22.bench_precision

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
import w22.cle15n as _nb_mod
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

# ── Parameter grid (identical to bench_cle15.py) ─────────────────────────────
VMAX_VALS = [30.0, 40.0, 50.0, 60.0, 70.0]
R0_VALS = [400e3, 600e3, 800e3, 1000e3, 1200e3]
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
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


if __name__ == "__main__":
    main()
