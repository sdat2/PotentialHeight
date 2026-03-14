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

---

## Test-suite audit (March 2026)

### Are the implementations equivalent?

A detailed equivalence analysis was performed across the full benchmark grid (75 cases: $V_\text{max} \in \{30\text{–}70\}$ m/s, $r_0 \in \{400\text{–}1200\}$ km, $f \in \{3,5,7\} \times 10^{-5}$ s$^{-1}$).

**E04 profiles are bit-for-bit identical** between `cle15.py` and `cle15n.py` — the Euler integration is straightforward and the numba kernel reproduces the Python loop exactly.

**rmax disagreement** is sub-percent for most cases (mean ~0.22 %, max ~1.72 %) but reaches up to **0.7 %** in the worst benchmark cases, with an RMS wind difference of up to **0.047 m/s** (close to the 0.05 m/s consistency threshold used in the existing tests).

The three algorithmic differences driving the rmax gap are:

| Aspect | `cle15.py` | `cle15n.py` |
|---|---|---|
| ER11 grid in bisection | Extends to `50×rmax` (can reach $r/r_0 = 15$) | Clipped to $r < r_0 \times (1 - 10^{-6})$ |
| Intersection grid | Dynamic: `max(10, len_E04, len_ER11)×2` points | Fixed 4,000 points (`nx_intersect`) |
| NaN guards in `run_cle15` | None (zeros out NaN winds and proceeds) | 3 guards: NaN rmax, >10 % NaN $V$, $r_\text{merge} \approx r_0$ |

The ER11 clipping was introduced deliberately to fix the spurious boundary-crossing bug (documented above).  The intersection grid difference has negligible practical impact (see `bench_precision.py` sweep).  The NaN guards in `cle15n.run_cle15` represent a **functional divergence**: calling `run_cle15` on a marginal input may return NaN arrays from `cle15n` but a (possibly garbage) finite result from `cle15`.

**Validation is circular** — there is no independent MATLAB ground truth for the regression tests.  The consistency tests compare `cle15.py` against `cle15n.py`; they do not verify correctness against the original MATLAB implementation.

### NaN handling — MATLAB reference semantics

The MATLAB reference (`mcle/mfiles/`) defines the following NaN semantics:

- **`ER11_radprof.m`**: if the nested convergence loop exceeds 20 outer iterations, returns `V_ER11 = NaN(size(rr_ER11))` and `r_out = NaN` — a clean NaN propagation.
- **`ER11E04_nondim_r0input.m`**: if bisection finds no intersection (ER11 always below E04), increments $C_k/C_d$ by 0.1 and retries.  If after iterating $C_k/C_d$ reaches the cap without a solution, the function returns without a crash (empty / NaN outputs).
- **Degenerate inputs** (`fcor=0`, `r0=0`, `Vmax=0`): MATLAB's `syms solve` call would return an empty root set, effectively erroring without graceful NaN propagation.  Neither MATLAB nor the Python translations specify well-defined output for these inputs.

### New tests: `TestNanHandling` (26 tests, added March 2026)

A new test class was added to `test_cle15.py` to cover NaN propagation paths systematically.  The class is organised into six sub-groups:

| Sub-group | What is tested |
|---|---|
| **A** | `_er11_radprof` non-convergence → returns NaN arrays (replicates MATLAB `ER11_radprof.m` behaviour) |
| **B** | $C_k/C_d$ upward-nudge fallback: hard-to-converge inputs trigger the outer `while not soln_converged` loop, but still produce a finite, physical $r_\text{max}$ |
| **C** | `run_cle15` NaN propagation: standard inputs give finite output; the three `cle15n`-only NaN guards are unit-tested via monkeypatching |
| **D** | `profile_from_stats` propagates finite inputs to finite outputs |
| **E** | `chavas_et_al_2015_profile` with degenerate inputs (`Vmax=0`, `r0=0`, `w_cool=0`, `fcor=0`) |
| **F** | Pressure-assumption consistency between `run_cle15` and `profile_from_stats` |

**Results: 187 passed, 0 xfailed** (as of March 2026 after fixes below).

#### Degenerate inputs — bugs found and fixed

The degenerate-input tests (sub-group E) initially exposed four cases where both implementations raised an unhandled `ZeroDivisionError` rather than returning NaN gracefully:

| Input | Module(s) | Root cause |
|---|---|---|
| `Vmax=0` | `cle15n` | `_bisect_rmaxr0_nb` divides by zero in the `abs(dV_err / Vmax)` convergence check |
| `r0=0` | `cle15n` | `_bisect_rmaxr0_nb` divides by zero computing `rfracrm_actual = r0 / rmax_guess` |
| `fcor=0` | both | $M_0 = \tfrac{1}{2} f r_0^2 = 0$ makes the ER11 power-law ratio $0^{2-C_k/C_d}$ singular |

**Fix (March 2026):** a guard block was added immediately after `fcor = abs(fcor)` at the top of `chavas_et_al_2015_profile` in both `cle15.py` and `cle15n.py`.  If `Vmax <= 0`, `r0 <= 0`, or `fcor == 0`, the function now issues a `warnings.warn` and returns the NaN sentinel tuple immediately, before any solver code is reached.  Additionally, `except ValueError` was broadened to `except (ValueError, ZeroDivisionError)` in both `_er11_rmax_r0_relation` and `_er11_r0_rmax_relation` in `cle15.py` as a belt-and-suspenders defence.

Note: `Vmax=0` and `r0=0` are physically meaningless (no storm / no outer boundary).  `fcor=0` (equator) is a genuine singularity in the ER11 theory — MATLAB's symbolic solver also returns empty roots for this case.  The correct model output for all three is NaN.

#### Pressure consistency (`run_cle15` vs. `profile_from_stats`)

The two pressure assumptions correspond to different thermodynamic closures in the cyclogeostrophic balance integral $\frac{dp}{dr} = \rho(r)\left(\frac{v^2}{r} + fv\right)$:

- **Isopycnal** (constant density): $\rho(r) = \rho_0$, so $p(r) = p_0 + \rho_0 \int_\infty^r \left(\frac{v^2}{r} + fv\right) dr$.  Linear in wind speed squared.
- **Isothermal** (ideal gas at constant temperature, so $\rho \propto p$): $\rho(r) = \frac{\rho_0}{p_0} p(r)$, yielding $p(r) = p_0 \exp\!\left(-\frac{\rho_0}{p_0} \int_\infty^r \left(\frac{v^2}{r} + fv\right) dr\right)$.  Exponential, and gives a slightly lower central pressure than isopycnal for the same wind field.

Both `run_cle15` and `profile_from_stats` default to **`isothermal`**, consistent with `w22/ps.py` and all other callers in the codebase.

A systematic **~2.7 hPa offset** exists between the central pressures returned by the two functions even when called with the same assumption.  This is independent of the assumption chosen (measured: 2.74 hPa isothermal, 2.75 hPa isopycnal) and arises purely because the two functions integrate the pressure gradient along different radial grids.  The test (`test_pressure_assumption_consistent_when_matched`) accepts up to 5 hPa divergence and asserts at least 0.5 hPa divergence (a canary for unexpected changes to the integration paths).

---

## References

- Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for tropical cyclone wind speed and rainfall profiles with physical interpretations. *Journal of the Atmospheric Sciences*, 72(9), 3403–3428.
- Emanuel, K., & Rotunno, R. (2011). Self-stratification of tropical cyclone outflow. Part I: Implications for storm structure. *Journal of the Atmospheric Sciences*, 68(10), 2236–2249.
- Emanuel, K. (2004). Tropical cyclone energetics and structure. In *Atmospheric Turbulence and Mesoscale Meteorology* (pp. 165–191). Cambridge University Press.
- Donelan, M. A., et al. (2004). On the limiting aerodynamic roughness of the ocean in very strong winds. *Geophysical Research Letters*, 31(18).
