"""
Benchmark and accuracy comparison:
  cle15.py  (pure-Python reference)
  cle15n.py (numba-accelerated)
  cle15m.py (octave/matlab -- run last, limited samples, very slow)

Usage:
    python -m w22.bench_cle15
"""

from __future__ import annotations
import time
import warnings
import numpy as np
from typing import List, Tuple

# ── Reference Python implementation ─────────────────────────────────────────
from .cle15 import chavas_et_al_2015_profile as cle15_py

# ── Numba implementation ─────────────────────────────────────────────────────
from .cle15n import chavas_et_al_2015_profile as cle15_nb
from .cle15n import warmup as nb_warmup

# ── Shared defaults ──────────────────────────────────────────────────────────
from .constants import (
    W_COOL_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    F_COR_DEFAULT,
)

warnings.filterwarnings("ignore")

# ── Parameter grid ───────────────────────────────────────────────────────────
# Cover a physically plausible range of (Vmax, r0) combinations
VMAX_VALS = [30.0, 40.0, 50.0, 60.0, 70.0]   # m/s
R0_VALS   = [400e3, 600e3, 800e3, 1000e3, 1200e3]  # m
FCOR_VALS = [3e-5, 5e-5, 7e-5]                # s^-1

# Build a flat list of test cases
ALL_CASES: List[Tuple[float, float, float]] = [
    (v, r, f)
    for v in VMAX_VALS
    for r in R0_VALS
    for f in FCOR_VALS
]
N_CASES = len(ALL_CASES)

# Shared fixed parameters (same for all cases)
CDVARY   = 0
C_D      = CD_DEFAULT
W_COOL   = W_COOL_DEFAULT
CKCDVARY = 0
CKCD     = CK_CD_DEFAULT
EYE_ADJ  = 0
ALPHA    = 0.15

# ── Helpers ──────────────────────────────────────────────────────────────────

def _call_py(vmax, r0, fcor):
    return cle15_py(vmax, r0, fcor, CDVARY, C_D, W_COOL, CKCDVARY, CKCD, EYE_ADJ, ALPHA)

def _call_nb(vmax, r0, fcor):
    return cle15_nb(vmax, r0, fcor, CDVARY, C_D, W_COOL, CKCDVARY, CKCD, EYE_ADJ, ALPHA)

def _extract_scalars(result):
    """Return (rmax_km, rmerge_km, Vmerge) from a profile result tuple."""
    _, _, rmax, rmerge, Vmerge, *_ = result
    return rmax, rmerge, Vmerge

# ── Step 1 · Warm up numba (spin-up not counted in timing) ──────────────────

print("=" * 65)
print("CLE15 benchmark: Python vs Numba vs Octave")
print("=" * 65)
print(f"\nTest cases: {N_CASES}  (Vmax × r0 × fcor grid)")
print("\nWarming up numba JIT (not counted in timing) …", flush=True)
t0 = time.perf_counter()
nb_warmup()
t_warmup = time.perf_counter() - t0
print(f"  JIT warm-up took {t_warmup:.2f} s\n")

# ── Step 2 · Time the Python implementation ──────────────────────────────────

print(f"Running {N_CASES} samples with pure-Python (cle15.py) …", flush=True)
py_results = []
t0 = time.perf_counter()
for (vmax, r0, fcor) in ALL_CASES:
    py_results.append(_call_py(vmax, r0, fcor))
t_py = time.perf_counter() - t0
print(f"  Total: {t_py:.2f} s  →  {t_py/N_CASES*1000:.1f} ms/call")

# ── Step 3 · Time the numba implementation ───────────────────────────────────

print(f"\nRunning {N_CASES} samples with numba (cle15n.py) …", flush=True)
nb_results = []
t0 = time.perf_counter()
for (vmax, r0, fcor) in ALL_CASES:
    nb_results.append(_call_nb(vmax, r0, fcor))
t_nb = time.perf_counter() - t0
print(f"  Total: {t_nb:.2f} s  →  {t_nb/N_CASES*1000:.1f} ms/call")

print(f"\n  Speedup (Python → Numba): {t_py/t_nb:.1f}×")

# ── Step 4 · Accuracy comparison ─────────────────────────────────────────────

print("\nAccuracy comparison (Python vs Numba):")
rmax_errs   = []
rmerge_errs = []
vmerge_errs = []
n_failed_py = 0
n_failed_nb = 0

for i, (case, py_res, nb_res) in enumerate(zip(ALL_CASES, py_results, nb_results)):
    vmax, r0, fcor = case
    rmax_py, rmerge_py, vm_py = _extract_scalars(py_res)
    rmax_nb, rmerge_nb, vm_nb = _extract_scalars(nb_res)

    if np.isnan(rmax_py):
        n_failed_py += 1
    if np.isnan(rmax_nb):
        n_failed_nb += 1

    if not (np.isnan(rmax_py) or np.isnan(rmax_nb)):
        rmax_errs.append(abs(rmax_py - rmax_nb) / rmax_py * 100)
        if not (np.isnan(rmerge_py) or np.isnan(rmerge_nb)):
            rmerge_errs.append(abs(rmerge_py - rmerge_nb) / rmerge_py * 100)
        if not (np.isnan(vm_py) or np.isnan(vm_nb)):
            vmerge_errs.append(abs(vm_py - vm_nb))

if rmax_errs:
    print(f"  rmax   : mean |err| = {np.mean(rmax_errs):.3f} %   "
          f"max = {np.max(rmax_errs):.3f} %")
if rmerge_errs:
    print(f"  rmerge : mean |err| = {np.mean(rmerge_errs):.3f} %   "
          f"max = {np.max(rmerge_errs):.3f} %")
if vmerge_errs:
    print(f"  Vmerge : mean |err| = {np.mean(vmerge_errs):.3f} m/s  "
          f"max = {np.max(vmerge_errs):.3f} m/s")
print(f"  Failures: Python={n_failed_py}/{N_CASES}, Numba={n_failed_nb}/{N_CASES}")

# Per-case detail for first 5 cases
print("\n  Per-case detail (first 5 cases):")
print(f"  {'Vmax':>6}  {'r0':>8}  {'fcor':>8}  "
      f"{'rmax_py':>9}  {'rmax_nb':>9}  {'Δrmax%':>7}  "
      f"{'rm_py':>8}  {'rm_nb':>8}  {'ΔVm':>7}")
for i in range(min(5, N_CASES)):
    vmax, r0, fcor = ALL_CASES[i]
    rmax_py, rmerge_py, vm_py = _extract_scalars(py_results[i])
    rmax_nb, rmerge_nb, vm_nb = _extract_scalars(nb_results[i])
    d_rmax = abs(rmax_py - rmax_nb) / rmax_py * 100 if not np.isnan(rmax_py) else np.nan
    d_vm   = abs(vm_py - vm_nb) if not np.isnan(vm_py) else np.nan
    print(f"  {vmax:6.0f}  {r0/1e3:8.0f}  {fcor:8.1e}  "
          f"  {rmax_py/1e3:7.1f}    {rmax_nb/1e3:7.1f}  {d_rmax:7.3f}  "
          f"  {rmerge_py/1e3:6.1f}  {rmerge_nb/1e3:6.1f}  {d_vm:7.3f}")

# ── Step 5 · Quick profile shape check ───────────────────────────────────────

print("\nProfile shape check (case 0, V=30 m/s, r0=400 km, fcor=3e-5):")
rr_py, VV_py = py_results[0][0], py_results[0][1]
rr_nb, VV_nb = nb_results[0][0], nb_results[0][1]
print(f"  Python: {len(rr_py)} radial points, "
      f"Vmax_profile = {np.nanmax(VV_py):.2f} m/s")
print(f"  Numba : {len(rr_nb)} radial points, "
      f"Vmax_profile = {np.nanmax(VV_nb):.2f} m/s")

# Interpolate both to a common grid and compute RMS error
r_common = np.linspace(0, min(rr_py[-1], rr_nb[-1]), 500)
V_py_c = np.interp(r_common, rr_py, VV_py)
V_nb_c = np.interp(r_common, rr_nb, VV_nb)
rms_V = np.sqrt(np.mean((V_py_c - V_nb_c) ** 2))
print(f"  RMS wind-speed error on common grid: {rms_V:.4f} m/s")

# ── Step 6 · Octave/Matlab comparison (very limited — last!) ────────────────

print("\n" + "=" * 65)
print("Octave/Matlab comparison (cle15m.py) — SLOW, 3 samples only")
print("=" * 65)

N_OCT = 3
OCT_CASES = [ALL_CASES[0], ALL_CASES[12], ALL_CASES[24]]

try:
    from .cle15m import _run_cle15_octave

    def _call_oct(vmax, r0, fcor):
        return _run_cle15_octave({"Vmax": vmax, "r0": r0, "fcor": fcor})

    print(f"\nRunning {N_OCT} samples with Octave …", flush=True)
    oct_results = []
    t0 = time.perf_counter()
    for (vmax, r0, fcor) in OCT_CASES:
        oct_results.append(_call_oct(vmax, r0, fcor))
    t_oct = time.perf_counter() - t0
    print(f"  Total: {t_oct:.2f} s  →  {t_oct/N_OCT:.2f} s/call")

    print(f"\n  Speedup (Octave → Python): {t_oct/t_py*N_CASES/N_OCT:.1f}×  "
          f"(Octave → Numba): {t_oct/t_nb*N_CASES/N_OCT:.1f}×")

    print("\n  Comparison (Octave vs Python vs Numba) for 3 samples:")
    print(f"  {'Vmax':>6}  {'r0(km)':>8}  {'fcor':>8}  "
          f"{'rmax_oct':>9}  {'rmax_py':>9}  {'rmax_nb':>9}")
    for i, (case, oct_r, py_r, nb_r) in enumerate(
        zip(OCT_CASES, oct_results,
            [py_results[0], py_results[12], py_results[24]],
            [nb_results[0], nb_results[12], nb_results[24]])
    ):
        vmax, r0, fcor = case
        rmax_oct = oct_r["rmax"]
        rmax_py, *_ = _extract_scalars(py_r)
        rmax_nb, *_ = _extract_scalars(nb_r)
        print(f"  {vmax:6.0f}  {r0/1e3:8.0f}  {fcor:8.1e}  "
              f"  {rmax_oct/1e3:7.1f}    {rmax_py/1e3:7.1f}    {rmax_nb/1e3:7.1f}")

except Exception as e:
    print(f"\n  Octave comparison skipped: {e}")

print("\nBenchmark complete.")
