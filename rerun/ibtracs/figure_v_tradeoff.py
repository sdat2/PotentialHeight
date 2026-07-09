"""Regenerate the velocity-tradeoff figure for Katrina (fixed potential-size solver).

This drives ``tcpips.ibtracs.vary_v_cps`` for hurricane Katrina. That routine
sweeps the input wind from the category-1 threshold (33 m/s, at gradient-wind
height) up to Katrina's potential intensity, re-solves the Wang-2022 potential
size at each wind with the CORRECTED humidity/units solver
(``w22.ps.parallelized_ps`` -> ``point_solution_ps`` ->
``w22.w22_carnot.carnot_pm_from_y``; the 2026-07 fix), and plots the resulting
radius-of-maximum-wind (rmax) trade-off. Three marked points are the panel's
content: r1 at the cat-1 threshold, r2 at Katrina's observed wind, r3 at the
potential-intensity endpoint. The deliverable is the figure itself, not a
scalar table.

Solver sanity check (fixed solver, canonical point): r0 = 2128.81 km,
rmax = 60.67 km at CkCd = 1, rh = 0.9.

DURABLE INPUTS (read by tcpips.ibtracs.get_tc_ds via IBTRACS_DATA_PATH =
<repo>/data/ibtracs, which must be the symlink below):
    IBTrACS.since1980.v04r01.pi_ps.nc      (FIXED pi/ps product; gives vmax, r3)
    IBTrACS.since1980.v04r01.cps_inputs.nc  (solver inputs: msl, rh, sst, t0)
    IBTrACS.since1980.v04r01.nc             (raw, for name/basin/subbasin labels)
The fixed pi_ps.nc lives in /Volumes/s/tcpips/fixed_ibtracs/; the other two are
symlinks there back to /Volumes/s/tcpips/data/ibtracs/. So point the repo's
(normally empty) data/ibtracs at the fixed directory before running:

    ln -sfn /Volumes/s/tcpips/fixed_ibtracs /Users/simon/worstsurge/data/ibtracs

OUTPUTS:
    <repo>/img/ibtracs/vary_v_ps_katrina.pdf   (written by vary_v_cps)
    /Volumes/s/tcpips/fixed_figures/vary_v_ps_katrina.pdf   (durable copy)
The curated durable copy from the fixed run is
/Volumes/s/tcpips/fixed_figures/Image-1_vtradeoff__vary_v_ps_katrina.pdf.

HOW TO RUN (caffeinated so the laptop won't sleep; under the RSS/wall watchdog
because the PS re-solve is parallel and memory-hungry):
    caffeinate -i -s python rerun/mem_guard.py rerun/ibtracs/figure_v_tradeoff.py 6144 3600

exFAT/macOS notes baked in below: HDF5 file locking must be off, BLAS must be
single-threaded, and loky is capped (vary_v_cps -> calculate_cps_ds ->
parallelized_ps defaults to jobs=-1, so LOKY_MAX_CPU_COUNT bounds the worker
count instead of grabbing every core). dask runs synchronous, never the
'processes' scheduler (BrokenProcessPool on macOS spawn).
"""

import os
import warnings
import traceback

# exFAT SSD + macOS: HDF5 file locking off; cap loky so the jobs=-1 PS solve
# doesn't grab every core; BLAS single-threaded so workers don't oversubscribe.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["LOKY_MAX_CPU_COUNT"] = "3"
for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import dask

dask.config.set(scheduler="synchronous")
import tcpips.ibtracs as ib

try:
    ib.vary_v_cps(v_reduc=0.8)
    print("OK vary_v_cps")
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    traceback.print_exc(limit=4)

# Persist the produced figure to the durable fixed-figures directory. This is a
# copy only; the figure content is exactly what vary_v_cps wrote to FIGURE_PATH.
import shutil

DURABLE_FIG_DIR = "/Volumes/s/tcpips/fixed_figures"
_produced = os.path.join(ib.FIGURE_PATH, "vary_v_ps_katrina.pdf")
if os.path.exists(_produced):
    os.makedirs(DURABLE_FIG_DIR, exist_ok=True)
    _durable = os.path.join(DURABLE_FIG_DIR, "vary_v_ps_katrina.pdf")
    shutil.copyfile(_produced, _durable)
    print(f"  durable copy -> {_durable}")

import glob
import time

for f in sorted(
    glob.glob(os.path.join(ib.FIGURE_PATH, "*.pdf")),
    key=os.path.getmtime,
    reverse=True,
)[:3]:
    print(
        f"  {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(f)))} "
        f"{os.path.basename(f)}"
    )
print("DONE")
