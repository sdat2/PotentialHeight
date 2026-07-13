"""Regenerate the IBTrACS "normalized-quad" figures and the exceedance table.

This is the stage-1 IBTrACS figure/table driver (recovered verbatim from the
session transcript, then made durable). It regenerates the published/thesis
normalized-variable figures and the exceedance table from the *already
computed* along-track datasets that were produced with the FIXED potential-size
solver (2026-07 humidity/units fix: ``w22.w22_carnot.carnot_pm_from_y`` via
``w22.ps.calculate_ps13_ufunc`` / ``point_solution_ps``; canonical point
r0 = 2128.81 km, rmax = 60.67 km at CkCd = 1, rh = 0.9).

It is a thin driver: it does NOT recompute any potential intensity/size. It only
reads the pre-computed per-track ``.nc`` files and calls the repo plotting
functions, so the numbers are fully determined by those input files.

WHAT IT PRODUCES (calls, in order):
  1. ``ib.plot_normalized_quad_dual()``  -> survival/hist "dual" panels
       - img/ibtracs/normalized_quad_dual.pdf        (Image-8b hist dual)
       - img/ibtracs/normalized_quad_cdf_dual.pdf     (Image-8 survival dual)
     Each panel legend reports the SF (sst>=26.5, |lat|<=30 sub-filter) and AF
     survival-at-1 percentages via ``_find_survival_value_at_thresh``.
  2. ``ib.plot_normalized_quad()``       -> single-filter quad + top tables
       - img/ibtracs/normalized_quad.pdf              (Image-9 quad)
       - img/ibtracs/normalized_quad_cdf.pdf          (Image-8 cdf)
       - img/ibtracs/normalized_vp_cps_cdf.pdf        (extra vp/cps cdf)
       - img/ibtracs/normalized_2d_hist.pdf           (Image-9b 2d hist)
  3. ``ib.create_exceedance_table()``    -> data/exceedance_table.tex

METRIC (preserved exactly, implemented inside the repo functions):
  per-storm-max survival-at-1 (``ibtracs._find_survival_value_at_thresh``) with
  the Cat-1 wind filter (usa_wind*0.514444 > 33 AND V_p*0.8 > 33) applied inside
  ``get_normalized_data``; SF/AF filters are sst>=26.5 and |lat|<=30.

EXPECTED NUMBERS (fixed-solver exceedance table, % of storms exceeding 1):
  r3: 46.1 / 42.4     r2: 19.9 / 10.2 (sst) / 7.94 (AF)     r1: 13.5 / 4.2
  (supersize probes elsewhere: Ellen 3.86, Man-Yi 3.11.)

DURABLE INPUTS (read by ``ib.get_normalized_data`` from ``IBTRACS_DATA_PATH``):
  IBTrACS.since1980.v04r01.{nc,pi_ps,cps,ps_cat1,normalized}.nc
  These live under /Volumes/s/tcpips/fixed_ibtracs/ (patched tracks with the
  fixed rmax). ``IBTRACS_DATA_PATH`` resolves to <repo>/data/ibtracs, so the
  patched inputs must be exposed there via the symlink:

      ln -sfn /Volumes/s/tcpips/fixed_ibtracs /Users/simon/worstsurge/data/ibtracs

  (The in-repo data/ibtracs is normally empty; without this symlink
  get_normalized_data silently reads nothing / stale data.)

DURABLE OUTPUTS:
  The repo functions hard-code their write paths to ``FIGURE_PATH``
  (<repo>/img/ibtracs) and ``DATA_PATH`` (<repo>/data/exceedance_table.tex);
  they are not overridable from here. The canonical fixed copies of these
  outputs are staged (renamed) under /Volumes/s/tcpips/fixed_figures/ , e.g.
  Image-8_*, Image-8b_*, Image-9_*, Image-9b_* and
  /Volumes/s/tcpips/data/exceedance_table.tex . After a run, copy the freshly
  written img/ibtracs/*.pdf and data/exceedance_table.tex over those.

HOW TO RUN (macOS, exFAT-safe; keep the machine awake and cap memory):
  ln -sfn /Volumes/s/tcpips/fixed_ibtracs /Users/simon/worstsurge/data/ibtracs
  caffeinate -i python /Users/simon/worstsurge/rerun/mem_guard.py \
      python /Users/simon/worstsurge/rerun/ibtracs/figures_normalized_stage1.py
  # or simply:  caffeinate -i python figures_normalized_stage1.py

exFAT/macOS notes preserved here: HDF5_USE_FILE_LOCKING=FALSE (netCDF on the
exFAT volume), LOKY_MAX_CPU_COUNT capped (some repo funcs default n_jobs=-1),
BLAS single-threaded, and dask on the synchronous scheduler (the 'processes'
scheduler raises BrokenProcessPool on macOS).
"""

import os, warnings, traceback

# --- env-safety guards (must be set BEFORE importing numpy/dask/xarray) ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # exFAT: netCDF/HDF5 file locking off
os.environ["LOKY_MAX_CPU_COUNT"] = "3"  # cap joblib/loky workers (memory)
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

print(
    "IBTRACS_DATA_PATH:",
    ib.IBTRACS_DATA_PATH,
    "->",
    os.path.realpath(ib.IBTRACS_DATA_PATH),
)
print("FIGURE_PATH:", ib.FIGURE_PATH)
for name, call in [
    (
        "plot_normalized_quad_dual (survival Image-8)",
        lambda: ib.plot_normalized_quad_dual(),
    ),
    (
        "plot_normalized_quad (histograms Image-9 + top tables)",
        lambda: ib.plot_normalized_quad(),
    ),
    ("create_exceedance_table", lambda: ib.create_exceedance_table()),
]:
    try:
        print(f"\n>>> {name}", flush=True)
        call()
        print(f"    OK")
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
print("\n=== PDFs in img/ibtracs (recent) ===")
import glob, time

for f in sorted(
    glob.glob(os.path.join(ib.FIGURE_PATH, "*.pdf")), key=os.path.getmtime, reverse=True
)[:12]:
    print(
        f"  {time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(f)))}  {os.path.basename(f)}"
    )
print("DONE")
