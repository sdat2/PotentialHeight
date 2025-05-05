"""
tc_profile_reverse.py
=====================

Inverse Chavas–Lin–Emanuel (2015) radial wind model
--------------------------------------------------

*Input*  : outer calm radius ``r0`` plus physical parameters
*Output* : radius of maximum wind ``rm`` and a continuous wind-speed
           callable ``V(r)``.

The implementation mirrors the original MATLAB routines supplied by
Chavas et al. (2015):

1. Integrate the Emanuel (2004) outer ODE for absolute momentum
   *M*(r) from ``r0`` inwards.
2. Evaluate the Emanuel–Rotunno (2011) inner solution for a **trial**
   ``rm``.  Perform a **binary search** on ``rm`` until the inner and
   outer *M*-curves touch exactly once (tangent match) at merge radius
   ``rj``.
3. Splice the inner and outer wind curves at ``rj`` to give a continuous
   profile from the centre out to ``r0``.

Dependencies
------------
* numpy ≥ 1.24
* scipy  (for Brent root finder)
* matplotlib (optional, only for ``quick_plot``)

Examples (doctest)
------------------
>>> p = {'r0': 180e3, 'Vm_guess': 45., 'f': 5e-5,
...      'Ck': 8e-4, 'Cd': 1e-3, 'Wcool': 0.002}
>>> V, info = solve_reverse(p)
>>> info['rm'] < info['r0']
True
>>> round(V(info['rm']), 1)          # ≈ Vm_guess
45.0
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
import numpy as np
from scipy.interpolate import interp1d
from math import copysign


# ---------------------------------------------------------------------
# 1. Inner-core (ER11) absolute momentum
# ---------------------------------------------------------------------
def m_inner(
    r: NDArray[np.floating] | float,
    m_max: float,
    r_max: float,
    ck_over_cd: float,
) -> NDArray[np.floating] | float:
    """
    Emanuel–Rotunno (2011) analytic momentum curve.

    Parameters
    ----------
    r : float | ndarray
        Radius (m).
    m_max : float
        Momentum at ``r_max`` (m² s⁻¹).
    r_max : float
        Radius of maximum wind (m).
    ck_over_cd : float
        ``Ck/Cd`` – must be < 1.

    Returns
    -------
    float | ndarray
        Absolute momentum M(r).

    >>> m_inner(30_000., 3.6e6, 30_000., 0.8)
    3600000.0
    """
    if ck_over_cd >= 1.0:
        raise ValueError("Ck/Cd must be < 1 for the ER11 formula")

    r_arr = np.asarray(r, float)
    m_eye = m_max * (r_arr / r_max) ** 2  # solid-body
    base = (2.0 + ck_over_cd) * ((r_arr / r_max) ** 2 - 1.0)
    expo = 1.0 / (2.0 - 2.0 * ck_over_cd)
    m_ext = m_max * np.maximum(base, 0.0) ** expo
    out = np.where(r_arr <= r_max, m_eye, m_ext)
    return out if np.ndim(r) else float(out)


# ---------------------------------------------------------------------
# 2. Emanuel (2004) outer ODE + integrator
# ---------------------------------------------------------------------
def dmdr_outer(
    r: float, m: float, r0: float, f: float, cd: float, wcool: float
) -> float:
    """dM/dr for the outer solution (→ 0 at r → r0)."""
    denom = r0**2 - r**2
    return 0.0 if denom <= 0 else 2.0 * cd * wcool * (m - 0.5 * f * r**2) ** 2 / denom


def integrate_outer(
    r0: float,
    r_stop: float,
    f: float,
    cd: float,
    wcool: float,
    steps: int = 4000,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return radii (descending) and M(r) for the outer region."""
    r0_in = r0 * (1 - 1e-6)
    dr = -(r0_in - r_stop) / steps
    rs = np.linspace(r0_in, r_stop, steps + 1)
    Ms = np.empty_like(rs)
    Ms[0] = 0.5 * f * r0**2
    for i in range(steps):
        r_c, m_c = rs[i], Ms[i]
        k1 = dmdr_outer(r_c, m_c, r0, f, cd, wcool)
        k2 = dmdr_outer(r_c + 0.5 * dr, m_c + 0.5 * dr * k1, r0, f, cd, wcool)
        k3 = dmdr_outer(r_c + 0.5 * dr, m_c + 0.5 * dr * k2, r0, f, cd, wcool)
        k4 = dmdr_outer(r_c + dr, m_c + dr * k3, r0, f, cd, wcool)
        Ms[i + 1] = m_c + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rs, Ms


# ---------------------------------------------------------------------
# 3.  Reverse solver (input r0 → rm, V(r))
# ---------------------------------------------------------------------
def solve_reverse(
    cfg: Dict[str, float],
    *,
    n_outer: int = 6000,
) -> Tuple[
    Callable[[NDArray[np.floating] | float], NDArray[np.floating] | float],
    Dict[str, float],
]:
    """
    Solve for ``rm`` and return a continuous wind-speed profile ``V(r)``.

    Expected keys in *cfg*::

        r0        : float  (m)
        Vm_guess  : float  (m s⁻¹)
        f         : float  (s⁻¹)
        Ck, Cd    : float
        Wcool     : float  (m s⁻¹)

    The solver follows Chavas et al. (2015) MATLAB logic.

    Returns
    -------
    V : callable
        Tangential wind speed V(r) for scalar/array radii.
    info : dict
        ``{'r0','rm','rj','Vm','Va'}``
    """
    r0 = cfg["r0"]
    Vm0 = cfg["Vm_guess"]
    f, Ck, Cd, Wcool = cfg["f"], cfg["Ck"], cfg["Cd"], cfg["Wcool"]
    ckcd = Ck / Cd
    if ckcd >= 1.0:
        raise ValueError("Ck/Cd must be < 1 for the ER11 inner solution")

    # --- integrate outer solution once --------------------------------
    r_outer, M_outer = integrate_outer(r0, 0.0, f, Cd, Wcool, n_outer)

    # --- search for rm such that inner & outer M curves just touch -----
    def mismatch(rm_val: float) -> float:
        M_max = Vm0 * rm_val + 0.5 * f * rm_val**2
        diff = m_inner(r_outer, M_max, rm_val, ckcd) - M_outer
        return diff.max() * diff.min()  # zero when sign change collapses

    rm = brentq(mismatch, 1.0e3, 0.9 * r0, maxiter=80)

    # --- recompute inner & find merge radius ---------------------------
    M_max = Vm0 * rm + 0.5 * f * rm**2
    r_eval = np.linspace(0.0, r0, 6000)
    M_inner = m_inner(r_eval, M_max, rm, ckcd)
    M_outer_interp = np.interp(r_eval, r_outer, M_outer)
    diff = M_inner - M_outer_interp
    idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0][0]
    r1, r2, d1, d2 = r_eval[idx], r_eval[idx + 1], diff[idx], diff[idx + 1]
    rj = r1 - d1 * (r2 - r1) / (d2 - d1)
    M_ra = m_inner(rj, M_max, rm, ckcd)
    Va = M_ra / rj - 0.5 * f * rj

    # --- build combined V(r) ------------------------------------------
    V_inner = M_inner / np.where(r_eval == 0, 1e-6, r_eval) - 0.5 * f * r_eval
    V_outer = M_outer / np.where(r_outer == 0, 1e-6, r_outer) - 0.5 * f * r_outer

    r_comb = np.concatenate([r_eval[r_eval <= rj], r_outer[r_outer > rj]])
    V_comb = np.concatenate([V_inner[r_eval <= rj], V_outer[r_outer > rj]])
    order = np.argsort(r_comb)
    r_sorted, V_sorted = r_comb[order], V_comb[order]

    def V(r: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
        """Interpolated tangential wind speed (m s⁻¹)."""
        return np.interp(np.asarray(r, float), r_sorted, V_sorted)

    return V, {"r0": r0, "rm": rm, "rj": rj, "Vm": Vm0, "Va": Va}


def CLE15_profile(r0, Vm_guess, f, Ck, Cd, Wcool):
    """
    Compute the Chavas-Lin0-Emanuel (2015) radial wind profile V(r) given r0 and parameters.
    Returns an interpolation function V(r) valid for 0 <= r <= r0.
    """
    # Constants and initial setup
    f = abs(f)  # use absolute value of Coriolis (storm is symmetric w.r.t. f sign)
    CkCd = Ck / Cd  # surface exchange coeff ratio for inner solution
    M0 = 0.5 * f * r0**2  # angular momentum at outer radius (M(r0))

    # --- 1. Integrate outer solution (E04) for M(r) from r0 inward ---
    dr_frac = 1e-3  # step in r/r0 (high resolution)
    # Adjust step for very large or very small r0 for stability (mimics MATLAB adjustment)
    if r0 > 2.5e6 or r0 < 2e5:
        dr_frac *= 0.1
    # Set up nondimensional radius array from r/r0 = 1 down to ~0
    Nr = int(1.0 / dr_frac)  # number of steps to cover [0,1]
    rr_frac = np.linspace(1.0, 0.0, Nr)  # r/r0 from 1 to 0
    M_frac = np.zeros_like(rr_frac)  # array for M/M0 values
    M_frac[0] = 1.0  # at r/r0 = 1, M/M0 = 1
    # At outer boundary, d(M/M0)/d(r/r0) = 0 (wind is zero at r0), so initialize next step
    M_frac[1] = 1.0
    # Model parameter gamma (Emanuel 2004)
    gamma = (Cd * f * r0) / Wcool
    # Euler integration inward for M/M0
    for i in range(1, Nr - 1):
        x = rr_frac[i]  # current r/r0
        Mbar = M_frac[i]  # current M/M0
        # Differential equation: d(M/M0)/d(x) = gamma * ((Mbar - x^2)^2) / (1 - x^2)
        dMbar_dx = gamma * ((Mbar - x**2) ** 2) / (1 - x**2)
        # Step inward (x is decreasing, so dr_frac is effectively negative step)
        M_frac[i + 1] = Mbar - dMbar_dx * dr_frac
    # Prepare outer solution arrays (dimensional)
    r_outer = rr_frac * r0  # radii from near 0 to r0
    M_outer = M_frac * M0  # M(r) in physical units
    # Compute outer wind profile for reference (V = M/r - 0.5*f*r)
    V_outer = (
        M_outer / np.where(r_outer == 0, 1e-6, r_outer) - 0.5 * f * r_outer
    )  # (avoid division by 0)
    V_outer[0] = 0.0  # define V(0)=0 for outer (should be very small anyway)

    # --- 2. Binary search for r_max that yields a tangent inner/outer match ---
    rmax_lower = 0.001 * r0  # lower bound for r_max (very small fraction of r0)
    rmax_upper = 0.75 * r0  # upper bound for r_max (based on CLE15 assumption)
    r_join = None
    Vmax = Vm_guess  # initial guess for Vmax (will adjust if needed)
    # If Ck/Cd is variable or needs adjustment for convergence (as in MATLAB), one could iterate CkCd here.
    # (For simplicity, we assume given Ck, Cd yield a convergent solution.)

    # Function to compute inner M/M0 curve for a given r_max
    def inner_M_curve(r_max):
        """Return radius array (0->r0), M_inner array, and V_inner array for given r_max."""
        # Define radius vector for inner solution evaluation (dense near core)
        # Use r_max as a scale for resolution: smaller r_max -> need finer spacing in inner region
        dr_inner = (
            r_max * 0.01
        )  # 1% of r_max as step (similar to 0.01 in r/rm nondim steps)
        if (
            dr_inner < 100.0
        ):  # if r_max is huge, limit absolute step to keep points manageable
            dr_inner = 100.0  # e.g., max 100 m step
        r_vals = np.arange(0, r0 + dr_inner, dr_inner)
        if r_vals[-1] < r0:  # ensure the array covers r0
            r_vals = np.append(r_vals, r0)
        # Compute V_inner using ER11 formula (Chavas et al. 2015 Eq. for inner wind):contentReference[oaicite:8]{index=8}
        M_max = Vmax * r_max + 0.5 * f * r_max**2  # M at r_max (peak)
        # Avoid division by zero at r=0 by starting from a small radius
        r_safe = np.where(r_vals == 0, 1e-6, r_vals)
        term = (2 * (r_safe / r_max) ** 2) / (2 - CkCd + CkCd * (r_safe / r_max) ** 2)
        V_inner = (M_max / r_safe) * (term ** (1.0 / (2 - CkCd))) - 0.5 * f * r_safe
        V_inner[r_vals == 0] = 0.0  # define V(0)=0 explicitly
        # Compute inner M profile from V (M = r*V + 0.5*f*r^2)
        M_inner = r_safe * V_inner + 0.5 * f * r_safe**2
        return r_vals, M_inner, V_inner

    # Binary search iterations
    for _ in range(50):  # limit to 50 iterations (sufficient for convergence)
        r_max_candidate = 0.5 * (rmax_lower + rmax_upper)
        # Compute inner solution for this candidate r_max
        r_inner, M_inner, V_inner = inner_M_curve(r_max_candidate)
        # Interpolate outer M onto inner radii (for comparison up to r0)
        M_outer_interp = np.interp(r_inner, r_outer, M_outer)
        # Find intersections by checking where the difference changes sign
        diff = M_inner - M_outer_interp
        # Exclude r=0 endpoint (both curves start at 0 there)
        diff[0] = 0.0
        # Identify zero crossings (sign changes) in diff
        sign_change_indices = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
        if sign_change_indices.size == 0:
            # No intersection: inner curve stays below outer -> r_max too small
            rmax_lower = r_max_candidate
        else:
            # At least one intersection: inner exceeds outer -> r_max too large
            rmax_upper = r_max_candidate
            # Record the first intersection point as current merge candidate
            idx = sign_change_indices[0]
            # Linear interpolation to refine the intersection radius
            r1, r2 = r_inner[idx], r_inner[idx + 1]
            M_diff1, M_diff2 = diff[idx], diff[idx + 1]
            # Compute root of diff between r1 and r2
            r_int = r1 - M_diff1 * (r2 - r1) / (M_diff2 - M_diff1)
            r_join = r_int
        # Check convergence of r_max
        if (rmax_upper - rmax_lower) < 1e-6 * r0:
            # Converged: narrow enough interval
            r_max_final = r_max_candidate
            break

    # --- 3. Construct merged wind profile using the determined r_max and r_join ---
    if r_join is None:
        # In case no merge found (should not happen if convergent), set merge to r0
        r_join = r0
        r_max_final = r_max_candidate
        r_inner, M_inner, V_inner = inner_M_curve(r_max_final)
    else:
        # Recompute inner profile at final r_max for completeness
        r_inner, M_inner, V_inner = inner_M_curve(r_max_final)
    # Compute outer wind profile (V_outer) on outer radii (already have from before)
    V_outer = M_outer / np.where(r_outer == 0, 1e-6, r_outer) - 0.5 * f * r_outer
    V_outer[0] = 0.0

    # Use r_join to splice the profiles
    # Take inner solution for r <= r_join, outer solution for r >= r_join
    inner_mask = r_inner <= r_join
    outer_mask = r_outer >= r_join
    # Concatenate the two segments
    r_combined = np.concatenate([r_inner[inner_mask], r_outer[outer_mask]])
    V_combined = np.concatenate([V_inner[inner_mask], V_outer[outer_mask]])
    # Ensure the two profiles meet continuously at r_join:
    # (They should, by construction; but we can sort by radius just in case)
    sort_idx = np.argsort(r_combined)
    r_combined = r_combined[sort_idx]
    V_combined = V_combined[sort_idx]

    # Create an interpolation function for V(r) on [0, r0]
    V_function = interp1d(r_combined, V_combined, kind="linear")
    return V_function, {
        "r0": r0,
        "rm": r_max_final,
        "rj": r_join,
        "Vm": Vmax,
        "Va": V_function(r_join),
    }  # returns a callable function V(r)


# ---------------------------------------------------------------------
# Optional quick-plot helper
# ---------------------------------------------------------------------
def quick_plot(V: Callable, info: Dict[str, float]) -> None:  # pragma: no cover
    """Matplotlib sanity check."""
    import matplotlib.pyplot as plt

    r = np.linspace(0, info["r0"], 1400)
    plt.plot(r / 1000, V(r))
    for k, c in (("rm", "r"), ("rj", "g"), ("r0", "k")):
        plt.axvline(info[k] / 1000, ls="--", c=c, label=k)
    plt.xlabel("Radius (km)")
    plt.ylabel("Wind (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    # python -m w22.cle15
    cfg_demo = {
        "r0": 1000e3,
        "Vm_guess": 45.0,
        "f": 5e-5,
        "Ck": 8e-4,
        "Cd": 1e-3,
        "Wcool": 0.002,
    }
    # V_demo, info_demo = solve_reverse(cfg_demo)
    # print(info_demo)
    # try:
    #    quick_plot(V_demo, info_demo)
    # except ImportError:
    #    pass

    print(info_demo)
    V_demo, info_demo = CLE15_profile(
        cfg_demo["r0"],
        cfg_demo["Vm_guess"],
        cfg_demo["f"],
        cfg_demo["Ck"],
        cfg_demo["Cd"],
        cfg_demo["Wcool"],
    )
    print(info_demo)
    try:
        quick_plot(V_demo, info_demo)
    except ImportError:
        pass
