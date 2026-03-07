# CLE15 implementation notes

## Overview

This directory contains three implementations of the Chavas et al. (2015) tropical cyclone wind profile model, which merges the Emanuel & Rotunno (2011) inner profile with the Emanuel (2004) outer profile:

| Module | Description |
|---|---|
| `cle15.py` | Pure-Python reference implementation |
| `cle15n.py` | Numba-accelerated drop-in replacement |
| `cle15m.py` | Thin wrapper calling the original MATLAB/Octave scripts in `mcle/` |

All three expose the same public API: `run_cle15`, `profile_from_stats`, `process_inputs`.

---

## Algorithm summary

1. **E04 outer profile** — integrate the Emanuel (2004) ODE radially inward from $r_0$ (where $V = 0$) to produce $M/M_0$ vs $r/r_0$.
2. **ER11 inner profile** — evaluate the Emanuel & Rotunno (2011) analytical formula for $V(r)$ given $(V_\text{max}, r_\text{max}, C_k/C_d)$.
3. **Merge** — find the $r_\text{max}/r_0$ value at which the ER11 curve is tangent to the E04 curve in $(r/r_0, M/M_0)$ space, via bisection.
4. **Assemble** — splice ER11 (inner, $r < r_\text{merge}$) onto E04 (outer, $r \geq r_\text{merge}$), interpolate onto a uniform grid in $r/r_\text{max}$ space, recover $V(r)$.

---

## Implementation choices in `cle15.py` vs the MATLAB reference (`mcle/`)

### 1. ER11 rmax–r0 root finding (`_er11_radprof_raw`)

**MATLAB** uses the Symbolic Math Toolbox (`syms` / `solve`) to find $r_\text{max}$ from $r_0$ analytically via ER11 Eq. 37.

**Python** uses `scipy.optimize.root_scalar` with `brentq` to solve the same implicit equation numerically.  The solutions agree to well within 1 m for all physically relevant inputs.

### 2. ER11 convergence loop (`_er11_radprof`)

**MATLAB** (`ER11_radprof.m`) uses a **nested** loop:
- *Outer loop* (max 20 iterations): adjusts the internal $r_\text{in}$ estimate until the profile's $r_\text{in}$ matches the target to within half a grid step.
- *Inner loop* (max 20 iterations, nested inside each outer step): adjusts the internal $V_\text{max}$ estimate until the profile $V_\text{max}$ matches to within 1 % relative error.

**Python** replicates this nested structure exactly, with the same iteration limits and convergence thresholds.

### 3. Merge bisection (`chavas_et_al_2015_profile`)

**MATLAB** wraps the entire bisection in an outer `while soln_converged == 0` loop. If the bisection completes without ever finding an intersection (ER11 always below E04, which occurs for very low $C_k/C_d$), it increments $C_k/C_d$ by 0.1 and retries.

**Python** replicates this `while not soln_converged` outer loop with a safety valve that returns NaN arrays if $C_k/C_d$ reaches 3.0 without convergence.

### 4. Final profile interpolation

**MATLAB** uses `interp1(..., 'pchip')` — a shape-preserving piecewise cubic Hermite interpolant — to assemble the merged $M/M_m$ vs $r/r_m$ profile onto a uniform grid.

**Python** uses `scipy.interpolate.PchipInterpolator` with `extrapolate=False`, which is the direct scipy equivalent.  Earlier versions used `interp1d(..., kind='linear')`, which produced a slightly different (sharper) kink near the merge point.

### 5. Curve intersection detection (`_curve_intersect`)

**MATLAB** calls the third-party `CurveIntersect` toolbox (included in `mcle/mfiles/general/CurveIntersect/`).

**Python** uses a custom sign-change detector: both curves are interpolated onto a common grid and sign changes in their difference are located and refined by linear interpolation between adjacent grid points.

### 6. Eye adjustment (`_radprof_eyeadj`)

Identical to MATLAB `radprof_eyeadj.m`: multiply wind speeds for $r \leq r_\text{max}$ by $(r / r_\text{max})^\alpha$.

---

## `cle15n.py` — Numba-accelerated version

`cle15n.py` is a drop-in replacement for `cle15.py` that JIT-compiles the three computational hot-spots with `numba.njit(cache=True)`.

### Kernels

| Kernel | What it does |
|---|---|
| `_calculate_cd_nb` | Piecewise-linear $C_d(V)$ — scalar, used inside other kernels |
| `_e04_euler_loop_nb` | Full E04 Euler integration loop (~1000 steps for typical storms) |
| `_er11_profile_nb` | Vectorised ER11 analytical formula |
| `_curve_intersect_count_nb` | Sign-change intersection counter on a common grid |
| `_bisect_rmaxr0_nb` | Entire bisection loop with ER11 convergence inlined — no Python overhead between iterations |

The key design decision is that `_bisect_rmaxr0_nb` inlines everything (ER11 evaluation, convergence iteration, intersection check) so the ~50 bisection steps run entirely inside a single compiled kernel.

### Accuracy differences from `cle15.py`

The numba version uses the same algorithm as `cle15.py` but with differences in:
- ER11 convergence: simplified additive adjustment with max 10 iterations (vs nested loop in Python)
- ER11 is clipped to $r \leq r_0$ before the intersection check (fixes a spurious boundary intersection bug)
- Final profile assembled with `scipy.interpolate.interp1d` (linear), not pchip

These produce rmax agreement with `cle15.py` to a mean of ~0.3 % and a maximum of ~1.7 % across a 75-case benchmark grid.

### Performance (75-case benchmark: $V_\text{max} \in \{30\text{–}70\}$ m/s, $r_0 \in \{400\text{–}1200\}$ km, $f \in \{3, 5, 7\} \times 10^{-5}$ s$^{-1}$)

| Implementation | Time per call | Speedup vs Octave |
|---|---|---|
| Octave (`cle15m.py`) | ~540 ms | 1× |
| Python (`cle15.py`) | ~45 ms | ~12× |
| **Numba** (`cle15n.py`) | **~6.4 ms** | **~85×** |

JIT warm-up (first call only): ~1.2 s. Subsequent calls use the cached compiled kernel.

---

## References

- Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for tropical cyclone wind speed and rainfall profiles with physical interpretations. *Journal of the Atmospheric Sciences*, 72(9), 3403–3428.
- Emanuel, K., & Rotunno, R. (2011). Self-stratification of tropical cyclone outflow. Part I: Implications for storm structure. *Journal of the Atmospheric Sciences*, 68(10), 2236–2249.
- Emanuel, K. (2004). Tropical cyclone energetics and structure. In *Atmospheric Turbulence and Mesoscale Meteorology* (pp. 165–191). Cambridge University Press.
- Donelan, M. A., et al. (2004). On the limiting aerodynamic roughness of the ocean in very strong winds. *Geophysical Research Letters*, 31(18).
