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

The numba version uses the same high-level algorithm as `cle15.py` with one unavoidable difference: inside `_bisect_rmaxr0_nb` the ER11 convergence is simplified to a flat 10-iteration additive adjustment (the full nested 20×20 loop cannot run inside an `@njit` kernel without significant restructuring). ER11 is also clipped to $r \leq r_0$ before the intersection check to suppress a spurious boundary-intersection bug.

The Python-level wrappers (`_er11_radprof_with_convergence`, `chavas_et_al_2015_profile`, `_make_interp`) are fully consistent with `cle15.py`: nested 20×20 convergence, CkCd fallback loop, and `PchipInterpolator`.

These produce rmax agreement with `cle15.py` to a mean of ~0.22 % and a maximum of ~1.72 % across a 75-case benchmark grid.

### `SolverConfig` — resolution presets

The four internal resolution knobs are exposed through the `SolverConfig` dataclass, which can be passed as `solver=` to `chavas_et_al_2015_profile`, `run_cle15`, and `profile_from_stats`.  Three named presets are provided:

```python
from cle15.cle15n import chavas_et_al_2015_profile, SolverConfig

# ~7.5× faster than default, ~0.8 % rmax error
res = chavas_et_al_2015_profile(..., solver=SolverConfig.fast())

# default (no argument needed)
res = chavas_et_al_2015_profile(...)

# ~2× slower than default, ~0.13 % rmax error
res = chavas_et_al_2015_profile(..., solver=SolverConfig.precise())
```

### Accuracy vs. cost sweep (`bench_precision.py`)

The four knobs were swept independently over a 75-case benchmark grid ($V_\text{max} \in \{30\text{–}70\}$ m/s, $r_0 \in \{400\text{–}1200\}$ km, $f \in \{3,5,7\} \times 10^{-5}$ s$^{-1}$), with rmax error measured relative to the pure-Python reference.

#### `Nr_e04` — E04 Euler grid points (default 200 000)

Cost and accuracy are **completely flat** from 1 000 to 500 000 points (~6 ms, 0.265 % mean error throughout).  The bottleneck is not grid resolution but the number of ER11 evaluations inside the bisection.  The default can safely be reduced to 10 000.

#### `num_pts_er11` — ER11 points inside the bisection kernel (default 5 000)

This is the **dominant driver** of both cost and accuracy.

| Points | ms/call | mean err% | max err% |
|-------:|--------:|----------:|---------:|
| 50 | 1.0 | 11.7 | 48.4 |
| 100 | 0.9 | 1.75 | 8.3 |
| 200 | 0.9 | 1.25 | 5.0 |
| 500 | 1.2 | 0.82 | 3.0 |
| 1 000 | 1.7 | 0.57 | 2.0 |
| 2 000 | 2.7 | 0.40 | 1.1 |
| **5 000** | **6.3** | **0.27** | **1.2** |
| 10 000 | 12.8 | 0.13 | 1.0 |

#### `nx_intersect` — intersection check grid points (default 4 000)

Nearly **no effect** on accuracy beyond 200 points; cost increases only mildly (6.0 → 6.8 ms over the full range).  500 points is sufficient.

| Points | ms/call | mean err% | max err% |
|-------:|--------:|----------:|---------:|
| 50 | 6.0 | 0.41 | 1.1 |
| 200 | 6.0 | 0.27 | 1.2 |
| **4 000** | **6.4** | **0.27** | **1.2** |
| 8 000 | 6.8 | 0.27 | 1.2 |

#### `max_iter` — bisection iterations (default 50)

Accuracy **degrades sharply** below 15 iterations.  The solver saturates at 20 iterations — going higher gives negligible benefit.

| Iters | ms/call | mean err% | max err% |
|------:|--------:|----------:|---------:|
| 5 | 2.4 | 45.0 | 160 |
| 10 | 3.6 | 1.77 | 6.2 |
| 15 | 5.0 | 0.29 | 1.2 |
| **20** | **6.4** | **0.27** | **1.2** |
| 50 | 6.3 | 0.27 | 1.2 |

#### Preset summary

| Preset | `Nr_e04` | `num_pts_er11` | `nx_intersect` | `max_iter` | ms/call | mean err% | max err% |
|---|--:|--:|--:|--:|--:|--:|--:|
| `fast()` | 10 000 | 500 | 500 | 20 | ~0.8 | ~0.82 | ~3.0 |
| `default()` | 200 000 | 5 000 | 4 000 | 50 | ~6–7 | ~0.27 | ~1.2 |
| `precise()` | 200 000 | 10 000 | 4 000 | 50 | ~13 | ~0.13 | ~1.0 |

### Resolved: spurious boundary-crossing bug in the numba bisection kernel

**Status: fixed.** The numba solver (`cle15n`) is now fully reliable for use in bisection loops and is re-enabled in `w22/ps.py`.

**Root cause (fixed):** `_bisect_rmaxr0_nb` built the ER11 profile grid capped at $r \le r_0$.  The E04 outer curve also terminates at $r/r_0 = 1.0$ with $M/M_0 = 1.0$ (by definition).  For large $r_\text{max}/r_0$ guesses the ER11 $M/M_0$ exceeds 1.0 in the outer region (supergradient) and then drops back to 1.0 at the boundary — creating a spurious sign-change that `_curve_intersect_count_nb` detected as a valid crossing at $r_\text{merge}/r_0 = 1.0$.  The bisection then converged to an unphysical tiny-$r_\text{max}$ solution, producing a degenerate merged profile and ultimately a NaN or wildly wrong $p_m$.

**Fix (one line in `cle15n.py`):** the ER11 grid is now clipped to $r < r_0 \times (1 - 10^{-6})$ (strictly less than $r_0$), so the shared endpoint is never included and the spurious crossing cannot occur.

**Verification:** a sweep of 100 $r_0$ values in the bisection range (1.9–2.4 Mm) previously produced a ~19% NaN rate; after the fix the NaN rate is **0%**.  Regression tests covering the two previously failing $(V_\text{max},\, r_0)$ cases are in `TestNbRobustness` in `test_cle15.py`.

### Performance: `w22/ps.py` bisection — using `SolverConfig.fast()` for inner calls

**Problem:** After switching `w22/ps.py` to the numba backend, the `w22` test suite ran in **12 minutes** despite the numba kernel being 4.5× faster than Python on the standard benchmark.

**Root cause:** `calculate_ps_ufunc` finds the potential size $r_0$ by bisecting on `run_cle15` — it calls the solver ~24 times per grid point.  The benchmark grid uses $r_0 \in \{400\text{–}1200\}$ km, but `ps.py` typically operates at $r_0 \sim 2$ Mm (potential size), where `_bisect_rmaxr0_nb` takes **~14 ms** per call (vs ~10 ms at the benchmark scale).  This gave ~400 ms per `calculate_ps_ufunc` point, and the numba speedup at the leaf level was completely swamped by the 24× call multiplier.

A secondary issue that was investigated but found to be negligible: `cle15n.py` had a `drfracrm /= 10` branch that expanded the final ER11 grid to 50 001 points when $r_\text{max} > 100$ km.  This added ~0.5 ms to the *final* call (negligible) and was removed for cleanliness.

**Fix:** Use `SolverConfig.fast()` (500 ER11 points, ~2 ms/call) for all `run_cle15` calls **inside** the bisection loop, and keep the default solver for the single **final** call after convergence (which produces the output profile).  The final call uses the full 5 000-point grid and gives the same result as before; only the ~24 intermediate calls are coarsened.

In `w22/ps.py`:
```python
from cle15.cle15n import run_cle15, SolverConfig

_BISECT_SOLVER = SolverConfig.fast()   # ~2 ms/call; used only inside bisection

def try_for_r0(r0: float):
    pm_cle, rmax_cle, _ = run_cle15(..., solver=_BISECT_SOLVER)  # fast
    ...

# after bisection converges:
pm, rmax, pc = run_cle15(...)   # default precision — called once
```

**Result:**

| Stage | Before | After |
|---|---|---|
| `run_cle15` per bisection step | ~14 ms | ~2 ms |
| `calculate_ps_ufunc` per point | ~400 ms | ~71 ms |
| `w22` test suite (9 tests) | 738 s (12 min) | 17.6 s |
| **Speedup** | — | **42×** |

The <0.4 % rmax error introduced at the bisection stage is well within the uncertainty of the $r_0$ bisection itself (1 Pa pressure tolerance → much larger $r_0$ uncertainty), and the output profile is computed with full precision.

### Performance summary (75-case benchmark)

| Implementation | Time per call | Speedup vs Octave |
|---|---|---|
| Octave (`cle15m.py`) | ~500 ms | 1× |
| Python (`cle15.py`) | ~45 ms | ~11× |
| Numba default (`cle15n.py`) | ~7 ms | ~71× |
| **Numba fast** (`SolverConfig.fast()`) | **~0.8 ms** | **~625×** |

JIT warm-up (first call only): ~1.2 s. Subsequent calls use the cached compiled kernel.

---

## References

- Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for tropical cyclone wind speed and rainfall profiles with physical interpretations. *Journal of the Atmospheric Sciences*, 72(9), 3403–3428.
- Emanuel, K., & Rotunno, R. (2011). Self-stratification of tropical cyclone outflow. Part I: Implications for storm structure. *Journal of the Atmospheric Sciences*, 68(10), 2236–2249.
- Emanuel, K. (2004). Tropical cyclone energetics and structure. In *Atmospheric Turbulence and Mesoscale Meteorology* (pp. 165–191). Cambridge University Press.
- Donelan, M. A., et al. (2004). On the limiting aerodynamic roughness of the ocean in very strong winds. *Geophysical Research Letters*, 31(18).
