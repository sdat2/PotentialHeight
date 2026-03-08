"""
Numba-accelerated version of the Chavas et al. (2015) TC wind profile.

This module is a drop-in replacement for :mod:`w22.cle15` that uses
``numba`` JIT compilation to speed up the three main hot-spots:

1. The E04 Euler integration loop  (~200 000 iterations, formerly pure-Python)
2. The ER11 profile formula evaluation  (vectorised, now inside numba)
3. The outer bisection over ``rmaxr0``  (~50 iterations × ER11 cost)
   including the curve-intersection check, all executed inside a single
   ``@njit`` kernel so there is no Python overhead between iterations.

The public API is identical to :mod:`w22.cle15`:

* :func:`chavas_et_al_2015_profile`
* :func:`process_inputs`
* :func:`run_cle15`
* :func:`profile_from_stats`

References
----------
- Chavas et al. (2015) JAS 72, 3403-3428.
- Emanuel & Rotunno (2011) JAS 68, 2236-2249.
- Emanuel (2004) in *Atmospheric Turbulence and Mesoscale Meteorology*.
"""

from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import os
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

import numba
from numba import njit, float64, int64

from .utils import pressure_from_wind
from .constants import (
    FIGURE_PATH,
    BACKGROUND_PRESSURE,
    W_COOL_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    RHO_AIR_DEFAULT,
    CDVARY_DEFAULT,
    CKCDVARY_DEFAULT,
    VMAX_DEFAULT,
    RA_DEFAULT,
    F_COR_DEFAULT,
    EYE_ADJ_DEFAULT,
    ALPHA_EYE_DEFAULT,
)

# ---------------------------------------------------------------------------
# Module-level constants (same as cle15.py)
# ---------------------------------------------------------------------------
CKCD_COEFQUAD = 5.5041e-04
CKCD_COEFLIN = -0.0259
CKCD_COEFCNST = 0.7627

CD_LOWV = 6.2e-4
V_THRESH1 = 6.0
V_THRESH2 = 35.4
CD_HIGHV = 2.35e-3
LINEAR_SLOPE = (CD_HIGHV - CD_LOWV) / (V_THRESH2 - V_THRESH1)

# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass
class SolverConfig:
    """
    Resolution / iteration knobs for the numba CLE15 solver.

    All four parameters control accuracy vs. run-time.  The table below
    summarises the sweep results from ``bench_precision.py`` (75-case
    benchmark, rmax error relative to the pure-Python reference):

    +----------------+----------+-----------+------------+-----------+
    | Preset         | ms/call  | mean err% | max err%   | fails     |
    +================+==========+===========+============+===========+
    | ``"fast"``     |  ~0.8    | ~0.82     | ~3.0       | 0/75      |
    | ``"default"``  |  ~6–7    | ~0.27     | ~1.2       | 0/75      |
    | ``"precise"``  | ~13      | ~0.13     | ~1.0       | 0/75      |
    +----------------+----------+-----------+------------+-----------+

    **Knob summary** (from individual sweeps, all others held at default):

    ``Nr_e04`` — E04 Euler grid points
        Accuracy is *insensitive* to Nr_e04 across the full range tested
        (1 000 – 500 000): mean rmax error stays at 0.265 % throughout.
        Cost is also flat (~6 ms).  The default 200 000 can safely be
        reduced to 10 000 with no accuracy penalty.

    ``num_pts_er11`` — ER11 profile points inside the bisection kernel
        The **dominant cost and accuracy driver**.  Halving from 5 000 to
        2 000 saves ~2× time (6 → 2.7 ms) with only a small accuracy cost
        (0.27 % → 0.40 % mean).  Going to 500 points gives ~6× speedup
        at the cost of ~0.82 % mean error (max ~3 %).

    ``nx_intersect`` — intersection check grid points
        Very weak effect on accuracy beyond 200 points.  Cost is also
        nearly flat (6.0 – 6.8 ms).  Reducing from 4 000 to 500 is free.

    ``max_iter`` — bisection iterations
        Below 15 iterations the solver degrades sharply (44 % mean error
        at 5 iterations).  At 15 iterations (~5 ms) accuracy is already
        within 0.30 % mean.  Going above 20 gives negligible improvement.

    Parameters
    ----------
    Nr_e04 : int
        Number of points in the E04 Euler integration grid.
    num_pts_er11 : int
        Number of ER11 profile points evaluated inside each bisection step.
    nx_intersect : int
        Number of points in the common grid used for the curve-intersection
        check inside the bisection kernel.
    max_iter : int
        Maximum number of bisection iterations.
    """

    Nr_e04: int = 200_000
    num_pts_er11: int = 5_000
    nx_intersect: int = 4_000
    max_iter: int = 50

    @classmethod
    def fast(cls) -> "SolverConfig":
        """
        ~7.5× faster than default; rmax mean error ~0.82 %, max ~3 %.

        Recommended for large ensemble runs where speed matters more than
        sub-percent accuracy in rmax.
        """
        return cls(Nr_e04=10_000, num_pts_er11=500, nx_intersect=500, max_iter=20)

    @classmethod
    def default(cls) -> "SolverConfig":
        """
        Current default settings; rmax mean error ~0.27 %, max ~1.2 %.
        """
        return cls()

    @classmethod
    def precise(cls) -> "SolverConfig":
        """
        ~2× slower than default; rmax mean error ~0.13 %, max ~1.0 %.

        Useful for reference comparisons or validating the fast preset.
        """
        return cls(Nr_e04=200_000, num_pts_er11=10_000, nx_intersect=4_000, max_iter=50)


# ---------------------------------------------------------------------------
# Numba-compiled kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _calculate_cd_nb(V: float) -> float:
    """Piecewise-linear Cd(V) — scalar version for use inside numba kernels."""
    if V <= V_THRESH1:
        return CD_LOWV
    elif V > V_THRESH2:
        return CD_HIGHV
    else:
        return CD_LOWV + LINEAR_SLOPE * (V - V_THRESH1)


@njit(cache=True)
def _e04_euler_loop_nb(
    rrfracr0: NDArray,
    M0: float,
    r0: float,
    fcor: float,
    Cdvary: int,
    C_d_input: float,
    w_cool: float,
) -> NDArray:
    """
    Euler integration of the E04 outer-wind ODE in non-dimensional form.

    Parameters
    ----------
    rrfracr0 : 1-D array  (Nr,)
        Radial grid from ``rfracr0_min`` to 1, **ascending** order.
    M0, r0, fcor : floats
        Storm parameters.
    Cdvary : int
        0 → constant Cd; 1 → variable Cd(V).
    C_d_input : float
        Drag coefficient to use when ``Cdvary == 0``.
    w_cool : float
        Radiative subsidence rate [m/s].

    Returns
    -------
    MMfracM0 : 1-D array (Nr,)
        Non-dimensional angular momentum M/M0 at each radius.
        Filled with NaN beyond the first numerical failure.
    """
    Nr = rrfracr0.shape[0]
    MMfracM0 = np.empty(Nr, dtype=np.float64)
    MMfracM0[:] = np.nan

    MMfracM0[Nr - 1] = 1.0
    MMfracM0[Nr - 2] = 1.0  # d(M/M0)/d(r/r0) == 0 at r/r0 == 1

    gamma = C_d_input * fcor * r0 / w_cool  # used when Cdvary == 0

    drfracr0 = rrfracr0[1] - rrfracr0[0]  # positive step

    for ii in range(Nr - 3, -1, -1):
        rfracr0_curr = rrfracr0[ii + 1]
        MfracM0_curr = MMfracM0[ii + 1]

        if Cdvary == 1:
            if rfracr0_curr > 1e-9:
                V_temp = (M0 / r0) * (MfracM0_curr / rfracr0_curr - rfracr0_curr)
                if V_temp < 0.0:
                    V_temp = 0.0
                C_d = _calculate_cd_nb(V_temp)
            else:
                C_d = CD_LOWV
            gamma = C_d * fcor * r0 / w_cool

        M_term = MfracM0_curr - rfracr0_curr * rfracr0_curr
        denominator = 1.0 - rfracr0_curr * rfracr0_curr

        if denominator <= 1e-9:
            dMdr = 0.0
        elif M_term < 0.0:
            dMdr = 0.0
        else:
            dMdr = gamma * (M_term * M_term) / denominator

        MMfracM0[ii] = MfracM0_curr - dMdr * drfracr0

        if MMfracM0[ii] > MMfracM0[ii + 1] + 1e-9:
            # Integration became non-monotone – truncate
            for jj in range(ii + 1):
                MMfracM0[jj] = np.nan
            break
        if MMfracM0[ii] < 0.0:
            for jj in range(ii + 1):
                MMfracM0[jj] = np.nan
            break

    return MMfracM0


@njit(cache=True)
def _er11_profile_nb(
    rr: NDArray,
    Mm: float,
    rmax: float,
    fcor: float,
    CkCd: float,
) -> NDArray:
    """
    Compute the ER11 wind-speed profile on a pre-allocated radius array.

    Returns V (m/s) at each radius, NaN at points where the formula is
    undefined, and 0 at r == 0.
    """
    n = rr.shape[0]
    V = np.empty(n, dtype=np.float64)
    V[:] = np.nan

    exp = 1.0 / (2.0 - CkCd)

    for i in range(n):
        r = rr[i]
        if r <= 1e-9:
            V[i] = 0.0
            continue
        ros2 = (r / rmax) * (r / rmax)  # (r/rmax)^2
        denom = 2.0 - CkCd + CkCd * ros2
        if denom <= 0.0:
            continue
        base = 2.0 * ros2 / denom
        if base <= 0.0:
            continue
        V[i] = (Mm / r) * (base**exp) - 0.5 * fcor * r
        if V[i] < 0.0:
            V[i] = 0.0

    return V


@njit(cache=True)
def _curve_intersect_count_nb(
    x1: NDArray,
    y1: NDArray,
    x2: NDArray,
    y2: NDArray,
    nx: int,
) -> Tuple[float, float, int]:
    """
    Count sign changes in (y1 - y2) on a common grid of ``nx`` points
    spanning the overlap of x1 and x2, and return the *mean* (x, y) of
    the first sign-change crossing together with the count.

    Uses simple linear interpolation (like MATLAB's ``interp1``).

    Returns
    -------
    x_mean, y_mean : float
        Mean intersection coordinates.  NaN if no intersections.
    count : int
        Number of sign-change crossings found.
    """
    # Overlap range
    x_lo = x1[0] if x1[0] > x2[0] else x2[0]
    x_hi = (
        x1[x1.shape[0] - 1]
        if x1[x1.shape[0] - 1] < x2[x2.shape[0] - 1]
        else x2[x2.shape[0] - 1]
    )

    if x_lo >= x_hi:
        return np.nan, np.nan, 0

    dx = (x_hi - x_lo) / (nx - 1)

    x_sum = 0.0
    y_sum = 0.0
    count = 0
    prev_diff = np.nan
    prev_x = np.nan
    prev_y1 = np.nan

    # Pointers into the two sorted input arrays
    j1 = 0
    j2 = 0

    for i in range(nx):
        xc = x_lo + i * dx

        # Interpolate y1 at xc
        # Advance j1
        while j1 + 1 < x1.shape[0] - 1 and x1[j1 + 1] <= xc:
            j1 += 1
        if x1[j1] <= xc <= x1[j1 + 1]:
            t = (xc - x1[j1]) / (x1[j1 + 1] - x1[j1])
            yc1 = y1[j1] + t * (y1[j1 + 1] - y1[j1])
        else:
            prev_diff = np.nan
            continue

        # Interpolate y2 at xc
        while j2 + 1 < x2.shape[0] - 1 and x2[j2 + 1] <= xc:
            j2 += 1
        if x2[j2] <= xc <= x2[j2 + 1]:
            t = (xc - x2[j2]) / (x2[j2 + 1] - x2[j2])
            yc2 = y2[j2] + t * (y2[j2 + 1] - y2[j2])
        else:
            prev_diff = np.nan
            continue

        diff = yc1 - yc2

        if not (prev_diff != prev_diff):  # prev_diff is not NaN
            if (prev_diff > 0.0 and diff < 0.0) or (prev_diff < 0.0 and diff > 0.0):
                # Linear interpolation of zero crossing
                if abs(diff - prev_diff) > 1e-20:
                    x_cross = prev_x - prev_diff * (xc - prev_x) / (diff - prev_diff)
                else:
                    x_cross = 0.5 * (prev_x + xc)
                y_cross = yc1  # approximate
                x_sum += x_cross
                y_sum += y_cross
                count += 1

        prev_diff = diff
        prev_x = xc
        prev_y1 = yc1

    if count > 0:
        return x_sum / count, y_sum / count, count
    return np.nan, np.nan, 0


@njit(cache=True)
def _bisect_rmaxr0_nb(
    r0: float,
    fcor: float,
    Vmax: float,
    CkCd: float,
    # E04 profile (pre-computed, passed in)
    rrfracr0_E04: NDArray,
    MMfracM0_E04: NDArray,
    M0: float,
    # bisection parameters
    rmaxr0_lo: float,
    rmaxr0_hi: float,
    max_iter: int,
    drmaxr0_thresh: float,
    # ER11 grid parameters
    rfracrm_max: float,
    num_pts_er11: int,
    # intersection grid resolution
    nx_intersect: int,
) -> Tuple[float, float, float, NDArray, NDArray]:
    """
    Bisection search for the tangent ``rmaxr0`` value.

    All the ER11 profile computation and curve intersection check are done
    inside this numba kernel, so there is **no Python overhead** between
    bisection iterations.

    The ER11 profile is converged (up to ``max_conv`` iterations) at each
    bisection step so that the profile's actual Vmax and rmax match the
    targets — matching what the Python reference code does.

    Returns
    -------
    rmaxr0_final : float
        Best estimate of the tangent rmax/r0.
    rmerger0 : float
        Non-dim merge radius r_merge/r0.
    MmergeM0 : float
        Non-dim M/M0 at merge.
    rr_ER11_best : 1-D array
        Radius array for the best ER11 profile (in metres).
    VV_ER11_best : 1-D array
        Wind-speed array for the best ER11 profile (m/s).
    """
    rmaxr0_low = rmaxr0_lo
    rmaxr0_high = rmaxr0_hi

    last_valid_rmaxr0 = np.nan
    rmerger0_last = np.nan
    MmergeM0_last = np.nan

    # Allocate ER11 arrays (fixed size across iterations)
    rr_ER11 = np.empty(num_pts_er11, dtype=np.float64)
    VV_ER11 = np.empty(num_pts_er11, dtype=np.float64)
    rr_ER11_best = np.empty(num_pts_er11, dtype=np.float64)
    VV_ER11_best = np.empty(num_pts_er11, dtype=np.float64)
    rr_ER11_best[:] = np.nan
    VV_ER11_best[:] = np.nan

    max_conv = 10  # convergence iterations per bisection step

    for _it in range(max_iter):
        rmaxr0_guess = 0.5 * (rmaxr0_low + rmaxr0_high)
        rmax_guess = rmaxr0_guess * r0

        # Build ER11 radius grid — cap at r0 so no points are wasted beyond
        # the E04 domain; this also avoids spurious boundary intersections.
        rfracrm_actual = r0 / rmax_guess  # = 1/rmaxr0_guess
        if rfracrm_actual > rfracrm_max:
            rfracrm_actual = rfracrm_max
        step = rfracrm_actual / (num_pts_er11 - 1)
        for k in range(num_pts_er11):
            rr_ER11[k] = k * step * rmax_guess

        # --- Converge ER11 profile so that profile rmax == rmax_guess ---
        # This matches what the Python reference code does with _er11_radprof().
        Vmax_cur = Vmax
        rmax_cur = rmax_guess
        dr_step = rr_ER11[1] - rr_ER11[0]

        for _conv in range(max_conv):
            Mm_cur = Vmax_cur * rmax_cur + 0.5 * fcor * rmax_cur * rmax_cur
            VV_ER11 = _er11_profile_nb(rr_ER11, Mm_cur, rmax_cur, fcor, CkCd)

            Vmax_prof = 0.0
            imax_prof = 0
            for k in range(num_pts_er11):
                v = VV_ER11[k]
                if not (v != v) and v > Vmax_prof:
                    Vmax_prof = v
                    imax_prof = k

            if Vmax_prof <= 0.0:
                break

            rmax_prof = rr_ER11[imax_prof]
            dV_err = Vmax - Vmax_prof
            dr_err = rmax_guess - rmax_prof

            if abs(dV_err / Vmax) < 1e-2 and abs(dr_err) < dr_step * 0.5:
                break

            new_r = rmax_cur + dr_err
            rmax_cur = new_r if new_r > 0.0 else 1.0
            new_V = Vmax_cur + dV_err
            Vmax_cur = new_V if new_V > 0.0 else 1.0

        # Final evaluation with converged parameters
        Mm_final = Vmax_cur * rmax_cur + 0.5 * fcor * rmax_cur * rmax_cur
        VV_ER11 = _er11_profile_nb(rr_ER11, Mm_final, rmax_cur, fcor, CkCd)

        # Convert ER11 to (r/r0, M/M0) space, clipping to r <= r0
        # (points beyond r0 cause spurious boundary intersections)
        n_valid = 0
        for k in range(num_pts_er11):
            v = VV_ER11[k]
            r = rr_ER11[k]
            if not (v != v) and r > 1e-9 and r <= r0:  # not NaN, within r0
                n_valid += 1

        if n_valid < 2:
            rmaxr0_low = rmaxr0_guess
            continue

        rrfracr0_ER11 = np.empty(n_valid, dtype=np.float64)
        MMfracM0_ER11 = np.empty(n_valid, dtype=np.float64)
        idx = 0
        for k in range(num_pts_er11):
            v = VV_ER11[k]
            r = rr_ER11[k]
            if not (v != v) and r > 1e-9 and r <= r0:
                MM = r * v + 0.5 * fcor * r * r
                rrfracr0_ER11[idx] = r / r0
                MMfracM0_ER11[idx] = MM / M0
                idx += 1

        # Find intersections between E04 and ER11 curves
        x_mean, y_mean, n_cross = _curve_intersect_count_nb(
            rrfracr0_E04,
            MMfracM0_E04,
            rrfracr0_ER11,
            MMfracM0_ER11,
            nx_intersect,
        )

        if n_cross == 0:
            rmaxr0_low = rmaxr0_guess
        else:
            rmaxr0_high = rmaxr0_guess
            last_valid_rmaxr0 = rmaxr0_guess
            rmerger0_last = x_mean
            MmergeM0_last = y_mean
            # Save best ER11 profile
            for k in range(num_pts_er11):
                rr_ER11_best[k] = rr_ER11[k]
                VV_ER11_best[k] = VV_ER11[k]

        if (rmaxr0_high - rmaxr0_low) < drmaxr0_thresh:
            break

    return last_valid_rmaxr0, rmerger0_last, MmergeM0_last, rr_ER11_best, VV_ER11_best


# ---------------------------------------------------------------------------
# Pure-Python wrappers that handle scipy calls + call numba kernels
# ---------------------------------------------------------------------------


def _e04_outerwind_r0input_nondim_mm0(
    r0: float,
    fcor: float,
    Cdvary: int,
    C_d_input: float,
    w_cool: float,
    Nr: int = 100000,
) -> Tuple[NDArray, NDArray]:
    """E04 outer profile.  Euler loop is compiled with numba."""
    fcor = abs(fcor)
    M0 = 0.5 * fcor * r0**2

    drfracr0 = 0.001
    if r0 > 2500e3 or r0 < 200e3:
        drfracr0 /= 10.0

    max_Nr = int(1.0 / drfracr0)
    Nr = min(Nr, max_Nr)
    Nr = max(Nr, 2)

    rfracr0_max = 1.0
    rfracr0_min = rfracr0_max - (Nr - 1) * drfracr0
    if rfracr0_min <= 0:
        rfracr0_min = drfracr0 / 10.0
        Nr = int((rfracr0_max - rfracr0_min) / drfracr0) + 1

    rrfracr0 = np.linspace(rfracr0_min, rfracr0_max, Nr)
    MMfracM0 = _e04_euler_loop_nb(rrfracr0, M0, r0, fcor, Cdvary, C_d_input, w_cool)

    valid = ~np.isnan(MMfracM0)
    return rrfracr0[valid], MMfracM0[valid]


def _er11_rmax_r0_relation(rmax_var, Vmax, r0, fcor, CkCd):
    """Implicit ER11 relation (root = 0).  Same as cle15.py."""
    if rmax_var <= 0 or rmax_var >= r0:
        return np.inf
    M0 = 0.5 * fcor * r0**2
    Mm = Vmax * rmax_var + 0.5 * fcor * rmax_var**2
    if Mm <= 0 or (2.0 - CkCd) == 0:
        return np.inf
    ratio_M = M0 / Mm
    ratio_r2 = (r0 / rmax_var) ** 2
    lhs = ratio_M ** (2.0 - CkCd)
    denom_rhs = 2.0 - CkCd + CkCd * ratio_r2
    if denom_rhs <= 0:
        return np.inf
    rhs = 2.0 * ratio_r2 / denom_rhs
    return lhs - rhs


def _er11_radprof_raw(
    Vmax: float,
    rmax: float,
    fcor: float,
    CkCd: float,
    rr_er11: NDArray,
) -> NDArray:
    """
    ER11 profile given *rmax* (no root-finding; rmax is already known).

    Returns V (m/s) at every point in rr_er11.
    """
    fcor = abs(fcor)
    Mm = Vmax * rmax + 0.5 * fcor * rmax**2
    V = _er11_profile_nb(rr_er11, Mm, rmax, fcor, CkCd)
    return V


def _er11_radprof_with_convergence(
    Vmax_target: float,
    rmax_target: float,
    fcor: float,
    CkCd: float,
    rr_ER11: NDArray,
) -> NDArray:
    """
    Iteratively converge the ER11 profile so the *profile* Vmax and rmax
    match the targets — matching the nested-loop structure of MATLAB's
    ``ER11_radprof.m`` and the updated ``cle15.py``.

    Outer loop (max 20): converge r_in until |Δr| < dr/2.
    Inner loop (max 20): converge Vmax until |ΔVmax/Vmax| < 1 %.
    """
    dr = rr_ER11[1] - rr_ER11[0]
    rin_thresh = dr / 2.0 if dr > 0 else 1.0
    vmax_thresh = 1e-2

    Vmax_cur = Vmax_target
    rmax_cur = rmax_target
    V = np.full_like(rr_ER11, np.nan)

    for _outer in range(20):
        # --- Inner loop: converge Vmax ---
        for _inner in range(20):
            V = _er11_radprof_raw(Vmax_cur, rmax_cur, fcor, CkCd, rr_ER11)
            valid = ~np.isnan(V) & (V >= 0)
            if not np.any(valid):
                return V  # total failure
            Vmax_prof = float(np.max(V[valid]))
            dV = Vmax_target - Vmax_prof
            if abs(dV / max(Vmax_target, 1e-9)) < vmax_thresh:
                break
            Vmax_cur = max(1.0, Vmax_cur + dV)

        # --- Check r_in convergence ---
        V = _er11_radprof_raw(Vmax_cur, rmax_cur, fcor, CkCd, rr_ER11)
        valid = ~np.isnan(V) & (V >= 0)
        if not np.any(valid):
            return V
        Vmax_prof = float(np.max(V[valid]))
        rmax_prof = float(rr_ER11[valid][np.argmax(V[valid])])
        dV = Vmax_target - Vmax_prof
        dr_err = rmax_target - rmax_prof

        if abs(dV / max(Vmax_target, 1e-9)) < vmax_thresh and abs(dr_err) < rin_thresh:
            break

        rmax_cur = max(1.0, rmax_cur + dr_err)

    return V


# ---------------------------------------------------------------------------
# Eye adjustment (unchanged from cle15.py)
# ---------------------------------------------------------------------------


def _radprof_eyeadj(
    rr_in: NDArray,
    VV_in: NDArray,
    alpha: float,
    r_eye_outer: float = None,
    V_eye_outer: float = None,
) -> NDArray:
    VV_out = VV_in.copy()
    if r_eye_outer is None or V_eye_outer is None:
        valid = ~np.isnan(VV_in) & (VV_in >= 0)
        if np.any(valid):
            r_eye_outer = rr_in[valid][np.argmax(VV_in[valid])]
            V_eye_outer = np.max(VV_in[valid])
        else:
            return VV_out
    if r_eye_outer <= 0 or V_eye_outer <= 0:
        return VV_out
    eye = rr_in <= r_eye_outer
    if not np.any(eye):
        return VV_out
    VV_out[eye] = VV_in[eye] * np.maximum(0.0, rr_in[eye] / r_eye_outer) ** alpha
    VV_out[rr_in <= 1e-9] = 0.0
    return VV_out


# ---------------------------------------------------------------------------
# Main profile function
# ---------------------------------------------------------------------------


def chavas_et_al_2015_profile(
    Vmax: float,
    r0: float,
    fcor: float,
    Cdvary: int,
    C_d: float,
    w_cool: float,
    CkCdvary: int,
    CkCd_input: float,
    eye_adj: int,
    alpha_eye: float,
    solver: Optional[SolverConfig] = None,
) -> Tuple[
    NDArray,
    NDArray,
    float,
    float,
    float,
    NDArray,
    NDArray,
    float,
    float,
    float,
    float,
]:
    """
    Chavas et al. (2015) merged TC wind profile — numba-accelerated.

    API is identical to :func:`w22.cle15.chavas_et_al_2015_profile`, with
    one additional optional argument.

    Parameters
    ----------
    solver : SolverConfig, optional
        Resolution / iteration settings.  Defaults to ``SolverConfig()``
        (the "default" preset).  Pass ``SolverConfig.fast()`` for ~7.5×
        faster evaluation at ~0.8 % rmax accuracy, or
        ``SolverConfig.precise()`` for ~2× slower but ~0.13 % accuracy.

    Examples
    --------
    >>> from cle15.cle15n import chavas_et_al_2015_profile, SolverConfig
    >>> res = chavas_et_al_2015_profile(
    ...     50.0, 800e3, 5e-5, 0, 1.5e-3, 2e-3, 0, 0.9, 0, 0.5,
    ...     solver=SolverConfig.fast(),
    ... )
    >>> rmax_km = res[2] / 1e3          # rmax in km
    >>> 5 < rmax_km < 300               # sanity: physically plausible rmax
    True
    """
    if solver is None:
        solver = SolverConfig()

    fcor = abs(fcor)
    _nan = np.array([np.nan])
    _fail = (
        _nan,
        _nan,
        np.nan,
        np.nan,
        np.nan,
        _nan,
        _nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    # --- Ck/Cd ---
    CkCd = CkCd_input
    if CkCdvary == 1:
        CkCd = CKCD_COEFQUAD * Vmax**2 + CKCD_COEFLIN * Vmax + CKCD_COEFCNST
        CkCd = max(0.5, min(1.9, CkCd))

    # --- Step 1: E04 outer profile ---
    try:
        rrfracr0_E04, MMfracM0_E04 = _e04_outerwind_r0input_nondim_mm0(
            r0, fcor, Cdvary, C_d, w_cool, Nr=solver.Nr_e04
        )
        if len(rrfracr0_E04) < 2:
            raise ValueError("E04 < 2 valid points")
    except Exception as e:
        warnings.warn(f"E04 failed: {e}")
        return _fail

    M0 = 0.5 * fcor * r0**2

    # --- Step 2: Bisection to find rmaxr0 (numba kernel) ---
    rfracrm_max = 50.0

    # Outer loop: if bisection finds no intersection, nudge CkCd upward by 0.1
    # (matching the MATLAB soln_converged fallback in ER11E04_nondim_r0input.m).
    soln_converged = False
    ckcd_adjusted = False
    while not soln_converged:
        rmaxr0_final, rmerger0, MmergeM0, rr_ER11_best, VV_ER11_best = (
            _bisect_rmaxr0_nb(
                r0,
                fcor,
                Vmax,
                CkCd,
                np.ascontiguousarray(rrfracr0_E04, dtype=np.float64),
                np.ascontiguousarray(MMfracM0_E04, dtype=np.float64),
                M0,
                0.001,
                0.75,
                solver.max_iter,
                1e-6,
                rfracrm_max,
                solver.num_pts_er11,
                solver.nx_intersect,
            )
        )

        if not np.isnan(rmaxr0_final):
            soln_converged = True
        else:
            if ckcd_adjusted and CkCd >= 3.0:
                warnings.warn(
                    f"Bisection failed even after CkCd adjustments (CkCd={CkCd:.3f}). "
                    "Returning NaNs."
                )
                return _fail
            warnings.warn(
                f"Adjusting CkCd from {CkCd:.3f} to {CkCd + 0.1:.3f} to find convergence "
                "(matching MATLAB ER11E04_nondim_r0input fallback)."
            )
            CkCd += 0.1
            ckcd_adjusted = True

    rmax = rmaxr0_final * r0
    Mm = Vmax * rmax + 0.5 * fcor * rmax**2
    MmM0 = Mm / M0

    # --- Step 3: Final high-resolution ER11 profile ---
    drfracrm_final = 0.01
    rfracrm_max_final = r0 / rmax
    num_pts_final = max(1000, int(rfracrm_max_final / drfracrm_final) * 2 + 1)
    rrfracrm_final = np.linspace(0, rfracrm_max_final, num_pts_final)
    rr_final = rrfracrm_final * rmax

    # Finer ER11 grid for the final profile (converging)
    drfracrm = 0.01
    if rmax > 100e3:
        drfracrm /= 10.0
    rfracrm_max_er11_fine = 50.0
    num_pts_er11_fine = max(500, int(rfracrm_max_er11_fine / drfracrm) + 1)
    rrfracrm_er11_fine = np.linspace(0, rfracrm_max_er11_fine, num_pts_er11_fine)
    rr_er11_fine = rrfracrm_er11_fine * rmax

    VV_ER11_fine = _er11_radprof_with_convergence(Vmax, rmax, fcor, CkCd, rr_er11_fine)

    if np.all(np.isnan(VV_ER11_fine)):
        warnings.warn("Final ER11 profile calculation failed.")
        return _fail

    # --- Step 4: Merge profiles on the final grid ---
    # ER11: M/Mm vs r/rmax
    valid_er11 = ~np.isnan(VV_ER11_fine) & (rr_er11_fine > 1e-9)
    rr_e11v = rr_er11_fine[valid_er11]
    VV_e11v = VV_ER11_fine[valid_er11]
    MM_er11 = rr_e11v * VV_e11v + 0.5 * fcor * rr_e11v**2
    rrfracrm_er11_v = rr_e11v / rmax
    MMfracMm_er11_v = MM_er11 / Mm

    # E04: M/Mm vs r/rmax
    rr_E04 = rrfracr0_E04 * r0
    MM_E04 = MMfracM0_E04 * M0
    rrfracrm_e04_v = rr_E04 / rmax
    MMfracMm_e04_v = MM_E04 / Mm

    # Interpolation functions — pchip matches MATLAB's interp1(...,'pchip')
    def _make_interp(x, y):
        order = np.argsort(x)
        xu, idx = np.unique(x[order], return_index=True)
        yu = y[order][idx]
        return PchipInterpolator(xu, yu, extrapolate=False)

    interp_ER11 = _make_interp(rrfracrm_er11_v, MMfracMm_er11_v)
    interp_E04 = _make_interp(rrfracrm_e04_v, MMfracMm_e04_v)

    rmergerfracrm = rmerger0 * r0 / rmax
    MMfracMm_merged = np.full(num_pts_final, np.nan)
    inner = rrfracrm_final < rmergerfracrm
    outer = ~inner
    MMfracMm_merged[inner] = interp_ER11(rrfracrm_final[inner])
    MMfracMm_merged[outer] = interp_E04(rrfracrm_final[outer])

    # Fill any residual NaNs
    nan_m = np.isnan(MMfracMm_merged)
    if np.any(nan_m & inner):
        MMfracMm_merged[nan_m & inner] = interp_ER11(rrfracrm_final[nan_m & inner])
    if np.any(nan_m & outer):
        MMfracMm_merged[nan_m & outer] = interp_E04(rrfracrm_final[nan_m & outer])

    # --- Step 5: Dimensional wind speed ---
    VV_final = np.full(num_pts_final, np.nan)
    valid_r = rrfracrm_final > 1e-9
    VV_final[valid_r] = (Mm / rmax) * (
        MMfracMm_merged[valid_r] / rrfracrm_final[valid_r]
    ) - 0.5 * fcor * rmax * rrfracrm_final[valid_r]
    VV_final[~valid_r] = 0.0
    VV_final[VV_final < 0] = 0.0

    # --- Vmerge ---
    rmerge = rmerger0 * r0
    try:
        Vmerge = float(
            interp1d(rr_final, VV_final, bounds_error=False, fill_value=np.nan)(rmerge)
        )
        if np.isnan(Vmerge):
            Vmerge = max(0.0, float((M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)))
    except Exception:
        Vmerge = np.nan

    # --- Eye adjustment ---
    if eye_adj == 1:
        VV_final = _radprof_eyeadj(rr_final, VV_final, alpha_eye, rmax, Vmax)

    # --- Non-dim outputs ---
    rrfracr0_final = rr_final / r0
    MMfracM0_final = MMfracMm_merged * (Mm / M0)

    # rmax from output profile
    fv = ~np.isnan(VV_final) & (VV_final >= 0)
    if np.any(fv):
        rmax_out = rr_final[fv][np.argmax(VV_final[fv])]
    else:
        rmax_out = np.nan
        warnings.warn("Final profile has no valid wind speeds.")

    return (
        rr_final,
        VV_final,
        rmax_out,
        rmerge,
        Vmerge,
        rrfracr0_final,
        MMfracM0_final,
        rmaxr0_final,
        MmM0,
        rmerger0,
        MmergeM0,
    )


# ---------------------------------------------------------------------------
# Thin wrappers (same API as cle15.py)
# ---------------------------------------------------------------------------


def process_inputs(inputs: dict) -> dict:
    """Process inputs — identical to cle15.process_inputs."""
    if inputs is None:
        inputs = {}
    ins = dict(
        w_cool=W_COOL_DEFAULT,
        p0=BACKGROUND_PRESSURE / 100,
        CkCd=CK_CD_DEFAULT,
        Cd=CD_DEFAULT,
        Cdvary=CDVARY_DEFAULT,
        CkCdvary=CKCDVARY_DEFAULT,
        Vmax=VMAX_DEFAULT,
        r0=RA_DEFAULT,
        fcor=F_COR_DEFAULT,
        eye_adj=EYE_ADJ_DEFAULT,
        alpha_eye=ALPHA_EYE_DEFAULT,
    )
    if "p0" in inputs:
        assert 900 < inputs["p0"] < 1100
    for k, v in inputs.items():
        if k in ins:
            ins[k] = v
    return ins


def run_cle15(
    plot: bool = False,
    inputs: Optional[Dict] = None,
    rho0: float = RHO_AIR_DEFAULT,
    pressure_assumption: str = "isopycnal",
    solver: Optional[SolverConfig] = None,
) -> Tuple[float, float, float]:
    """Run the numba-accelerated CLE15 model — same API as cle15.run_cle15.

    Parameters
    ----------
    solver : SolverConfig, optional
        Resolution settings.  Defaults to ``SolverConfig()`` (default preset).
    """
    if inputs is None:
        inputs = {}
    ins = process_inputs(inputs)
    o = chavas_et_al_2015_profile(
        ins["Vmax"],
        ins["r0"],
        ins["fcor"],
        ins["Cdvary"],
        ins["Cd"],
        ins["w_cool"],
        ins["CkCdvary"],
        ins["CkCd"],
        ins["eye_adj"],
        ins["alpha_eye"],
        solver=solver,
    )
    ou = {"rr": o[0], "VV": o[1], "rmax": o[2], "rmerge": o[3], "Vmerge": o[4]}
    ou["VV"][-1] = 0
    ou["VV"][np.isnan(ou["VV"])] = 0
    rr = np.array(ou["rr"], dtype="float32")
    vv = np.array(ou["VV"], dtype="float32")
    p = pressure_from_wind(
        rr,
        vv,
        p0=ins["p0"] * 100,
        rho0=rho0,
        fcor=ins["fcor"],
        assumption=pressure_assumption,
    )
    return (
        float(interp1d(rr, p)(ou["rmax"])),
        ou["rmax"],
        p[0],
    )


def profile_from_stats(
    vmax: float,
    fcor: float,
    r0: float,
    p0: float,
    rho0: float = RHO_AIR_DEFAULT,
    pressure_assumption: str = "isothermal",
    solver: Optional[SolverConfig] = None,
) -> dict:
    """Same as cle15.profile_from_stats but uses the numba-accelerated kernel.

    Parameters
    ----------
    solver : SolverConfig, optional
        Resolution settings.  Defaults to ``SolverConfig()`` (default preset).
    """
    ins = process_inputs({"Vmax": vmax, "fcor": fcor, "r0": r0, "p0": p0})
    o = chavas_et_al_2015_profile(
        ins["Vmax"],
        ins["r0"],
        ins["fcor"],
        ins["Cdvary"],
        ins["Cd"],
        ins["w_cool"],
        ins["CkCdvary"],
        ins["CkCd"],
        ins["eye_adj"],
        ins["alpha_eye"],
        solver=solver,
    )
    out = {"rr": o[0], "VV": o[1], "rmax": o[2], "rmerge": o[3], "Vmerge": o[4]}
    out["VV"][-1] = 0
    out["VV"][np.isnan(out["VV"])] = 0
    out["p"] = (
        pressure_from_wind(
            out["rr"],
            out["VV"],
            p0=p0 * 100,
            fcor=fcor,
            rho0=rho0,
            assumption=pressure_assumption,
        )
        / 100
    )
    return out


# ---------------------------------------------------------------------------
# Warm-up helper (call once at import time to trigger JIT compilation)
# ---------------------------------------------------------------------------


def warmup():
    """
    Trigger numba JIT compilation by running one cheap profile.

    Call this explicitly before benchmarking so spin-up cost is not
    counted against the timed runs.
    """
    chavas_et_al_2015_profile(
        Vmax=50.0,
        r0=500e3,
        fcor=5e-5,
        Cdvary=0,
        C_d=1.5e-3,
        w_cool=2e-3,
        CkCdvary=0,
        CkCd_input=1.0,
        eye_adj=0,
        alpha_eye=0.15,
    )


if __name__ == "__main__":
    print("Warming up numba JIT …")
    warmup()
    print("Done.")
