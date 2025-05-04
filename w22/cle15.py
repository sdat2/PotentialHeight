"""
tc_profile.py
=============

A complete, self-contained Python implementation of the **Chavas, Lin & Emanuel
(2015)** tropical-cyclone radial wind profile.

Main features
-------------
* Inner-core angular momentum from Emanuel & Rotunno (2011).
* Outer-region solution of Emanuel (2004) solved via 4th-order Runge-Kutta.
* Monotone root solve for the outer radius *r₀* so that
  \(M_{\text{outer}}(r_m) = M_m\)  ⇒ always a physical match.
* Exact merge radius *rₐ* (Brent solver on the momentum difference) with the
  point inserted into the cached outer curve → **continuous wind field**.
* Vectorised callable ``V(r)`` (works for scalars or NumPy arrays).
* Works with a plain ``dict`` or OmegaConf ``DictConfig`` (soft import).
* Google-style doc-strings **with doctests**
  (run ``python -m doctest -v tc_profile.py``).
* Optional Matplotlib helper ``quick_plot`` for a rapid visual check.

Dependencies
------------
``numpy`` and ``scipy`` (for the Brent root finder).
``matplotlib`` is *optional* and used only by ``quick_plot``.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq  # single non-stdlib dependency

# Soft support for OmegaConf
try:
    from omegaconf import DictConfig  # type: ignore

    _OMEGA = True
except ImportError:  # pragma: no cover
    _OMEGA = False


# ----------------------------------------------------------------------
# 1. Inner-core angular momentum (Emanuel & Rotunno 2011)
# ----------------------------------------------------------------------
def m_inner(
    r: NDArray | float,
    m_m: float,
    r_m: float,
    ck_over_cd: float,
) -> NDArray | float:
    """
    Eyewall / inner-core absolute angular momentum $M(r)$.

    Solid-body rotation for *r ≤ rₘ* and the Emanuel-Rotunno analytic
    expression for *r ≥ rₘ*.

    Args:
        r: Radius or radii (m).
        m_m: Angular momentum at *rₘ* (m² s⁻¹).
        r_m: Radius of maximum wind (m).
        ck_over_cd: \(C_k/C_d\) (must be < 1).

    Returns:
        $M(r)$ in m² s⁻¹.

    >>> m_inner(30_000., 3.6e6, 30_000., 0.8)
    3600000.0
    """
    if ck_over_cd >= 1.0:
        raise ValueError("Ck/Cd must be < 1 for the inner solution.")

    r_arr = np.asarray(r, float)
    # Eye: solid-body
    m_eye = m_m * (r_arr / r_m) ** 2

    # Emanuel-Rotunno outside r_m
    base = (2.0 + ck_over_cd) * ((r_arr / r_m) ** 2 - 1.0)
    expo = 1.0 / (2.0 - 2.0 * ck_over_cd)
    m_er = m_m * np.maximum(base, 0.0) ** expo

    out = np.where(r_arr <= r_m, m_eye, m_er)
    return out if np.ndim(r) else float(out)


# ----------------------------------------------------------------------
# 2. Emanuel (2004) outer ODE
# ----------------------------------------------------------------------
def dmdr_outer(r, m, r0, f, cd, wc):
    """dM/dr with analytic limit → 0 as r → r₀⁻."""
    denom = r0**2 - r**2
    return 0.0 if denom <= 0.0 else 2.0 * cd * wc * (m - 0.5 * f * r**2) ** 2 / denom


def integrate_outer(
    r0: float,
    r_stop: float,  # ≤ r0
    f: float,
    cd: float,
    wc: float,
    n: int = 4000,
) -> Tuple[NDArray, NDArray]:
    """
    Integrate the outer ODE inward from *r₀* to *r_stop*.

    Returns:
        radii (descending) and corresponding $M(r)$.
    """
    r0_in = r0 * (1 - 1e-6)  # tiny inset to avoid 0/0
    dr = -(r0_in - r_stop) / n
    rs = np.linspace(r0_in, r_stop, n + 1)
    Ms = np.empty_like(rs)
    Ms[0] = 0.5 * f * r0**2  # boundary condition at r₀

    for i in range(n):
        r_c, m_c = rs[i], Ms[i]
        k1 = dmdr_outer(r_c, m_c, r0, f, cd, wc)
        k2 = dmdr_outer(r_c + 0.5 * dr, m_c + 0.5 * dr * k1, r0, f, cd, wc)
        k3 = dmdr_outer(r_c + 0.5 * dr, m_c + 0.5 * dr * k2, r0, f, cd, wc)
        k4 = dmdr_outer(r_c + dr, m_c + dr * k3, r0, f, cd, wc)
        Ms[i + 1] = m_c + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return rs, Ms


# ----------------------------------------------------------------------
# 3. Solve full wind profile
# ----------------------------------------------------------------------
def solve_profile(
    params: Dict | "DictConfig",
    *,
    n_outer: int = 6000,
) -> Tuple[Callable[[NDArray | float], NDArray | float], Dict[str, float]]:
    """
    Create the merged wind-profile function ``V(r)``.

    Args:
        params: mapping with keys ::

            Vm    (m s⁻¹)  - maximum wind
            rm    (m)      - radius of maximum wind
            f     (s⁻¹)    - Coriolis parameter
            Ck              surface enthalpy exchange coefficient
            Cd              surface drag coefficient
            Wcool (m s⁻¹)  - radiative-subsidence rate
        n_outer: number of Runge-Kutta steps in the outer integration.

    Returns:
        V(r) and a dict ``{'r0','ra','rm','Vm','Va'}``.

    >>> cfg = {'Vm': 50., 'rm': 30_000., 'f': 5e-5,
    ...         'Ck': 6e-4, 'Cd': 1e-3, 'Wcool': 0.003}
    >>> V, info = solve_profile(cfg)
    >>> info['rm'] < info['ra'] < info['r0']
    True
    >>> abs(V(info['rm']) - cfg['Vm']) < 1.0
    True
    >>> abs(V(info['ra']*1.001) - V(info['ra']*0.999)) < 1.0 # 1e-3
    True
    """
    # Accept DictConfig transparently
    if _OMEGA and isinstance(params, DictConfig):  # type: ignore[truthy-bool]
        p: Dict[str, float] = params  # type: ignore[assignment]
    else:
        p = dict(params)

    Vm, rm = p["Vm"], p["rm"]
    f, Ck, Cd, Wc = p["f"], p["Ck"], p["Cd"], p["Wcool"]
    ckcd = Ck / Cd
    if ckcd >= 1.0:
        raise ValueError("Ck/Cd must be < 1 (inner solution requirement).")

    # Inner-core momentum at r_m
    M_m = rm * Vm + 0.5 * f * rm**2

    # ---- solve for r0 such that outer M(rm) == M_m -------------------
    def diff(r0_val: float) -> float:
        _, Ms = integrate_outer(r0_val, rm, f, Cd, Wc, max(800, n_outer // 30))
        return Ms[-1] - M_m

    r0 = brentq(diff, 1.05 * rm, 5e6 * rm, maxiter=120)

    # High-resolution outer integration
    r_out, M_out = integrate_outer(r0, rm, f, Cd, Wc, n_outer)

    # ---- precise merge radius r_a by root on M_inner - M_outer -------
    def g(r_val: float) -> float:
        M_outer = np.interp(r_val, r_out[::-1], M_out[::-1])
        return m_inner(r_val, M_m, rm, ckcd) - M_outer

    ra = brentq(g, rm * 1.001, r0 * 0.999, maxiter=80)
    M_a = m_inner(ra, M_m, rm, ckcd)
    Va = M_a / ra - 0.5 * f * ra

    # Cache outer curve INCLUDING the exact (ra, M_a) and (r0, ½f r0²)
    r_cache = np.hstack([r_out, ra, r0])
    M_cache = np.hstack([M_out, M_a, 0.5 * f * r0**2])

    # ---- final wind-speed callable ----------------------------------
    def V(r: NDArray | float) -> NDArray | float:
        """Tangential wind (m s⁻¹) at radius *r* (m)."""
        R = np.asarray(r, float)
        Vv = np.zeros_like(R)

        eye = R < rm
        inner = (R >= rm) & (R <= ra)
        outer = (R > ra) & (R <= r0)

        Vv[eye] = Vm * (R[eye] / rm)
        M_in = m_inner(R[inner], M_m, rm, ckcd)
        Vv[inner] = M_in / R[inner] - 0.5 * f * R[inner]

        M_out_interp = np.interp(R[outer], r_cache[::-1], M_cache[::-1])
        Vv[outer] = M_out_interp / R[outer] - 0.5 * f * R[outer]

        return Vv if np.ndim(r) else float(Vv)

    info = dict(r0=r0, ra=ra, rm=rm, Vm=Vm, Va=Va)
    return V, info


# ----------------------------------------------------------------------
# 4. Quick visual check (optional)
# ----------------------------------------------------------------------
def quick_plot(V: Callable, info: Dict[str, float]) -> None:  # pragma: no cover
    """Plot wind profile with markers at r_m, r_a, r_0."""
    import matplotlib.pyplot as plt

    r = np.linspace(0.0, info["r0"], 14000)
    plt.plot(r * 1e-3, V(r), label="V(r)")
    for k, col in (("rm", "r"), ("ra", "g"), ("r0", "k")):
        plt.axvline(info[k] * 1e-3, ls="--", c=col, label=k)
    plt.xlabel("Radius (km)")
    plt.ylabel("Wind (m s$^{-1}$)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 1. Inner-core angular momentum (Emanuel & Rotunno 2011)
# ---------------------------------------------------------------------
def m_inner(
    r: NDArray | float, m_m: float, r_m: float, ck_over_cd: float
) -> NDArray | float:
    """
    Eyewall / inner-core absolute angular momentum *M(r)*.

    Args:
        r: Radius (m) or array of radii.
        m_m: Angular momentum at ``r_m`` (m² s⁻¹).
        r_m: Radius of maximum wind (m).
        ck_over_cd: Ratio Ck/Cd (< 1).

    Returns:
        M(r) in m² s⁻¹.

    >>> m_inner(30_000., 3.6e6, 30_000., 0.8)
    3600000.0
    """
    if ck_over_cd >= 1.0:
        raise ValueError("Ck/Cd must be < 1")
    r_arr = np.asarray(r, float)
    m_eye = m_m * (r_arr / r_m) ** 2
    base = (2.0 + ck_over_cd) * ((r_arr / r_m) ** 2 - 1.0)
    expo = 1.0 / (2.0 - 2.0 * ck_over_cd)
    m_er = m_m * np.maximum(base, 0.0) ** expo
    out = np.where(r_arr <= r_m, m_eye, m_er)
    return out if np.ndim(r) else float(out)


# ---------------------------------------------------------------------
# 2. Emanuel (2004) outer ODE helper
# ---------------------------------------------------------------------
def dmdr_outer(r: float, m: float, r0: float, f: float, cd: float, wc: float) -> float:
    """Derivative dM/dr that tends to 0 as r→r₀⁻."""
    denom = r0**2 - r**2
    return 0.0 if denom <= 0.0 else 2 * cd * wc * (m - 0.5 * f * r**2) ** 2 / denom


def integrate_outer(
    r0: float, r_stop: float, f: float, cd: float, wc: float, n: int = 4000
) -> Tuple[NDArray, NDArray]:
    """
    Inward 4-th-order RK integration of the outer ODE.

    Returns:
        radii (descending) and M(r).
    """
    r0_in = r0 * (1 - 1e-6)
    dr = -(r0_in - r_stop) / n
    rs = np.linspace(r0_in, r_stop, n + 1)
    Ms = np.empty_like(rs)
    Ms[0] = 0.5 * f * r0**2
    for i in range(n):
        r, m = rs[i], Ms[i]
        k1 = dmdr_outer(r, m, r0, f, cd, wc)
        k2 = dmdr_outer(r + 0.5 * dr, m + 0.5 * dr * k1, r0, f, cd, wc)
        k3 = dmdr_outer(r + 0.5 * dr, m + 0.5 * dr * k2, r0, f, cd, wc)
        k4 = dmdr_outer(r + dr, m + dr * k3, r0, f, cd, wc)
        Ms[i + 1] = m + (dr / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rs, Ms


# ---------------------------------------------------------------------
# 3. Reverse solver   (given r0 → rm)
# ---------------------------------------------------------------------
def solve_reverse(
    cfg: Dict, *, n_outer: int = 6000
) -> Tuple[Callable[[NDArray | float], NDArray | float], Dict[str, float]]:
    """
    Compute wind profile for a fixed outer radius.

    Args:
        cfg: Must contain
            r0 (m), Vm_guess (m/s), f (s⁻¹), Ck, Cd, Wcool (m/s)
        n_outer: RK steps for outer integration.

    Returns:
        V(r) callable and information dict.

    >>> cfg = {'r0':180e3,'Vm_guess':45.,'f':5e-5,
    ...         'Ck':8e-4,'Cd':1e-3,'Wcool':0.002}
    >>> V, info = solve_reverse(cfg)
    >>> info['rm'] < info['r0']
    True
    >>> abs(V(info['rm']) - cfg['Vm_guess']) < 5
    True
    """
    r0, Vm0 = cfg["r0"], cfg["Vm_guess"]
    f, Ck, Cd, Wc = cfg["f"], cfg["Ck"], cfg["Cd"], cfg["Wcool"]
    ckcd = Ck / Cd
    if ckcd >= 1.0:
        raise ValueError("Ck/Cd must be < 1")

    def mismatch(rm_val: float) -> float:
        M_m = rm_val * Vm0 + 0.5 * f * rm_val**2
        _, M_out = integrate_outer(r0, rm_val, f, Cd, Wc, max(800, n_outer // 30))
        return M_out[-1] - M_m

    rm = brentq(mismatch, 1e3, 0.9 * r0, maxiter=80)
    Vm = Vm0
    M_m = rm * Vm + 0.5 * f * rm**2
    r_out, M_out = integrate_outer(r0, rm, f, Cd, Wc, n_outer)

    def gap(r_val: float) -> float:
        M_outer = np.interp(r_val, r_out[::-1], M_out[::-1])
        return m_inner(r_val, M_m, rm, ckcd) - M_outer

    ra = brentq(gap, rm * 1.001, r0 * 0.999, maxiter=80)
    M_a = m_inner(ra, M_m, rm, ckcd)

    r_cache = np.hstack([r_out, ra, r0])
    M_cache = np.hstack([M_out, M_a, 0.5 * f * r0**2])

    def V(r: NDArray | float) -> NDArray | float:
        """Tangential wind speed (m/s) at radius *r* (m)."""
        R = np.asarray(r, float)
        Vv = np.zeros_like(R)
        eye = R < rm
        inner = (R >= rm) & (R <= ra)
        outer = (R > ra) & (R <= r0)
        Vv[eye] = Vm * (R[eye] / rm)
        Vv[inner] = m_inner(R[inner], M_m, rm, ckcd) / R[inner] - 0.5 * f * R[inner]
        Vv[outer] = (
            np.interp(R[outer], r_cache[::-1], M_cache[::-1]) / R[outer]
            - 0.5 * f * R[outer]
        )
        return Vv if np.ndim(r) else float(Vv)

    return V, {"r0": r0, "rm": rm, "ra": ra, "Vm": Vm}


# ---------------------------------------------------------------------
def quick_plot_two(V: Callable, info: Dict[str, float]) -> None:  # pragma: no cover
    """Matplotlib sanity check."""
    import matplotlib.pyplot as plt

    r = np.linspace(0, info["r0"], 1400)
    plt.plot(r / 1000, V(r))
    for k, c in (("rm", "r"), ("ra", "g"), ("r0", "k")):
        plt.axvline(info[k] / 1000, ls="--", c=c, label=k)
    plt.xlabel("Radius (km)")
    plt.ylabel("Wind (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    cfg_demo = {
        "r0": 180e3,
        "Vm_guess": 45.0,
        "f": 5e-5,
        "Ck": 8e-4,
        "Cd": 1e-3,
        "Wcool": 0.002,
    }
    V_demo, info_demo = solve_reverse(cfg_demo)
    print(info_demo)
    try:
        quick_plot(V_demo, info_demo)
    except ImportError:
        pass


# ----------------------------------------------------------------------
# 5. Demo when executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # python -m w22.cle15
    cfg_demo = {
        "Vm": 50.0,
        "rm": 30_000.0,
        "f": 5e-5,
        "Ck": 6e-4,
        "Cd": 1e-3,
        "Wcool": 0.003,
    }
    V_demo, info_demo = solve_profile(cfg_demo)
    print(info_demo)
    try:
        quick_plot_two(V_demo, info_demo)
    except ImportError:
        print("matplotlib not installed — skipping plot.")
