"""Patch the fixed-solver rmax into fresh copies of the IBTrACS track files.

Purpose
-------
After the 2026-07 potential-size humidity/units bug fix
(``w22.w22_carnot.carnot_pm_from_y``), the potential radius-of-max-wind
(``rmax``) stored inside the IBTrACS per-track products is stale. This script
copies the three relevant track ``.nc`` files verbatim and overwrites their
``rmax`` variable in-place with the re-solved (fixed) values, so the repo
plotters (``tcpips.ibtracs``) draw the *fixed* figures without re-running the
solver.

The fixed rmax values (plus the Cat-1 wind mask ``W``) are read from the
``exact_{r1,r2,r3}.npz`` intermediates produced by the companion re-solve
script (``rerun/ibtracs/ibtracs_exact.py``). Only points where the Cat-1 mask
is set *and* the fixed solve returned a finite rmax are patched; every other
value is left exactly as stored.

Measure -> track file -> npz key:
  r3 = ``...pi_ps.nc``    (V_p / potential-intensity driven)
  r2 = ``...cps.nc``      (climate potential size)
  r1 = ``...ps_cat1.nc``  (Cat-1 filtered)

Durable inputs
--------------
- Track files (copied fresh, then patched):
  /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.{pi_ps,cps,ps_cat1}.nc
- Re-solved rmax intermediates:
  <NPZ_DIR>/exact_{r1,r2,r3}.npz   (default NPZ_DIR = /Volumes/s/tcpips/fixed_ibtracs/exact_resolve,
  matching where resolve_exceedance.py writes them)
- Untouched siblings the plotters/``get_normalized_data`` also open, symlinked
  from the source dir:
  IBTrACS.since1980.v04r01.{,.normalized,.unique,.cps_inputs}.nc

Durable outputs
---------------
/Volumes/s/tcpips/fixed_ibtracs/
  IBTrACS.since1980.v04r01.{pi_ps,cps,ps_cat1}.nc  (fresh copies, rmax patched)
  + symlinks to the untouched sibling files listed above.

Expected numbers (fixed exceedance table this pipeline feeds):
  r3 46.1 / 42.4 ; r2 19.9 / 10.2(sst) / 7.94(AF) ; r1 13.5 / 4.2
  supersize Ellen 3.86, Man-Yi 3.11.

Repo symlink note
-----------------
The patched-data plotters read tracks from the repo ``data/ibtracs`` symlink.
Point it at this output dir before plotting:
  ln -sfn /Volumes/s/tcpips/fixed_ibtracs /Users/simon/worstsurge/data/ibtracs

How to run
----------
Cheap (copy + in-place variable rewrite; no solve), but still go through the
guard + caffeinate so an interrupted append can't wedge the machine:

  export HDF5_USE_FILE_LOCKING=FALSE
  caffeinate -i -s python rerun/mem_guard.py \
      rerun/ibtracs/patch_track_files.py 2048 3600

Run the companion ``ibtracs_exact.py`` first so the ``exact_*.npz`` files exist.
"""

import os

# --- env-safety guards (exFAT/macOS): set BEFORE importing netCDF4/numpy ---
# HDF5_USE_FILE_LOCKING=FALSE is mandatory on the exFAT SSD or the netCDF4
# append below fails with an HDF5 locking error. BLAS/loky caps are harmless
# here (no heavy math, no joblib) but kept for a uniform environment.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import shutil
import numpy as np
from netCDF4 import Dataset

SRC = "/Volumes/s/tcpips/data/ibtracs"
DST = "/Volumes/s/tcpips/fixed_ibtracs"
# Durable location of the re-solved rmax intermediates (written by
# rerun/ibtracs/resolve_exceedance.py to .../exact_resolve/). Overridable via
# EXACT_NPZ_DIR; default must match resolve_exceedance.py's OUT and
# figures_tracks.py's EXACT_DIR.
NPZ_DIR = os.environ.get("EXACT_NPZ_DIR", "/Volumes/s/tcpips/fixed_ibtracs/exact_resolve")

os.makedirs(DST, exist_ok=True)
npz = {m: np.load(os.path.join(NPZ_DIR, f"exact_{m}.npz")) for m in ("r1", "r2", "r3")}
# measure -> (track file, npz key)
patch = {"IBTrACS.since1980.v04r01.pi_ps.nc": "r3",
         "IBTrACS.since1980.v04r01.cps.nc": "r2",
         "IBTrACS.since1980.v04r01.ps_cat1.nc": "r1"}
for fn, meas in patch.items():
    src = os.path.join(SRC, fn); dst = os.path.join(DST, fn)
    print(f"copying {fn} ...", flush=True); shutil.copy2(src, dst)
    d = npz[meas]; W = d["W"].astype(bool); rf = d["rmax_fixed"]  # flat S*D
    use = W & np.isfinite(rf)
    nc = Dataset(dst, "a")
    rm = nc.variables["rmax"]
    orig = np.asarray(rm[:], dtype="float64"); shape = orig.shape
    flat = orig.ravel().copy()
    assert flat.size == use.size, f"{fn}: rmax size {flat.size} != mask {use.size}"
    flat[use] = rf[use]
    rm[:] = flat.reshape(shape)
    nc.close()
    print(f"  patched {int(use.sum())} rmax values in {fn}", flush=True)
# symlink the other files get_normalized_data / plotters need
for fn in ("IBTrACS.since1980.v04r01.nc", "IBTrACS.since1980.v04r01.normalized.nc",
           "IBTrACS.since1980.v04r01.unique.nc", "IBTrACS.since1980.v04r01.cps_inputs.nc"):
    l = os.path.join(DST, fn)
    if os.path.lexists(l): os.remove(l)
    if os.path.exists(os.path.join(SRC, fn)): os.symlink(os.path.join(SRC, fn), l)
print("\nDST contents:")
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f); print(f"  {'->' if os.path.islink(p) else '  '} {f}")
print("DONE")
