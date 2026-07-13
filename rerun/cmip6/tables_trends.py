"""Regenerate the CMIP6/ERA5 temporal trend correlation & fit .tex tables (fixed PS solver).

PURPOSE
-------
Faithful, durable copy of the recovered scratchpad ``tables_era5.py``. It
regenerates the per-city temporal-relationship LaTeX tables that appear in the
thesis / potential-height paper:

    {place}_temporal_correlation_pi4.tex   (Pearson rho of Vp, r0, sst vs year,
                                            and cross-correlations sst<->Vp,
                                            sst<->r0, per model/member/period)
    {place}_temporal_fit_pi4.tex           (OLS gradients / trends, with
                                            uncertainties, for the same fields)

for ``place in {new_orleans, hong_kong}``. All plotted/tabulated quantities come
from the repo's precomputed per-place potential-size timeseries, which were
produced with the FIXED potential-size solver (2026-07 humidity/units bug fix in
``w22.w22_carnot.carnot_pm_from_y``, reached via ``w22.ps``). Canonical
fixed-solver sanity point: r0 = 2128.81 km, rmax = 60.67 km at CkCd = 1,
rh = 0.9.

This driver contains NO numerical logic of its own -- it only calls the repo
functions ``w22.stats2.temporal_relationship_data`` (which internally loads the
timeseries and computes ``safe_corr`` / ``safe_grad`` via
``timeseries_relationships``) and ``w22.plot.plot_era5_timeseries``. Nothing
about the filters, correlation/gradient math, model/member list
(CESM2 r4/r10/r11, HadGEM3-GC31-MM r1..r3 i1p1f3, MIROC6 r1..r3 i1p1f1), the
future/historical period splits (2014-2100 / 1980-2014 for CMIP6, 1980-2024 and
1940-2024 for ERA5), or ``pi_version=4`` is changed here.

RELATION TO figures_multipanel.py
---------------------------------
The sibling ``rerun/cmip6/figures_multipanel.py`` regenerates the multipanel
figures and, incidentally, the ``new_orleans`` temporal table only. This script
is NOT superseded by it: it uniquely regenerates the ``hong_kong`` temporal
correlation/fit tables (and re-emits the ``new_orleans`` ones), which is its
reproducibility value.

DURABLE INPUTS (read by the repo functions, all under <repo>/w22/data)
----------------------------------------------------------------------
  - CMIP6 per-place timeseries:
      {place}_august_{historical,ssp585}_{model}_{member}_isothermal_pi4new.nc
  - ERA5 per-place timeseries:
      {place}_august_era5_isothermal.nc
  Places: new_orleans and hong_kong. (These .nc files are produced upstream by
  rerun/cmip6/regen_point_timeseries.py.)

OUTPUTS (written to the repo, durable)
--------------------------------------
Written by ``temporal_relationship_data`` to ``w22.stats2.DATA_PATH``
(= <repo>/w22/data/stats):
  - {place}_temporal_relationships_pi4.csv
  - {place}_temporal_correlation_pi4.tex
  - {place}_temporal_fit_pi4.tex
for place in {new_orleans, hong_kong}.
(FIGURE_PATH / DATA_PATH are resolved by ``w22.constants`` relative to the
installed ``w22`` package -- there is no scratchpad path to redirect.)

EXPECTED RESULT
---------------
Reproduces the fixed-solver temporal correlation/fit trend tables. This script
produces *trend tables and (attempted) an ERA5 figure*; it does NOT produce the
IBTrACS exceedance-table numbers (those come from the separate ibtracs_* scripts).

KNOWN PRESERVED BEHAVIOUR
-------------------------
The step ``plot_era5_timeseries(place="new_orleans")`` is preserved
byte-for-byte from the recovered transcript. That repo function requires a
positional ``axs`` argument, so this call raises ``TypeError`` and is reported as
``FAIL`` by the ``step`` wrapper. It is intentionally NOT "fixed" here so the run
reproduces the original behaviour exactly (the two table steps run first and
independently, so the tables are written regardless).

HOW TO RUN (macOS/exFAT-safe, under caffeinate + the RSS/wall watchdog)
----------------------------------------------------------------------
    caffeinate -is python rerun/mem_guard.py rerun/cmip6/tables_trends.py 6000 3600

or directly (env guards are set at the top of this file):

    caffeinate -is python rerun/cmip6/tables_trends.py

The env guards below (single-thread BLAS, capped LOKY workers, disabled HDF5
file locking, synchronous dask) are required on macOS/exFAT to avoid
BrokenProcessPool, file-lock errors and multi-core memory blow-ups, and must be
set BEFORE the heavy imports.
"""

import os
import warnings
import traceback

# --- macOS/exFAT env-safety guards (must be set before heavy imports) ---
os.environ["LOKY_MAX_CPU_COUNT"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
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
import w22.plot as wp
import w22.stats2 as st


def step(n, f):
    try:
        f()
        print(f"OK {n}", flush=True)
    except Exception as e:
        print(f"FAIL {n}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=3)


step("tables new_orleans", lambda: st.temporal_relationship_data(place="new_orleans"))
step("tables hong_kong", lambda: st.temporal_relationship_data(place="hong_kong"))
step("era5 timeseries fig", lambda: wp.plot_era5_timeseries(place="new_orleans"))

import glob
import time

print(
    "recent .tex:",
    (
        [
            os.path.basename(f)
            for f in glob.glob(os.path.join(st.DATA_PATH, "*.tex"))
            if os.path.getmtime(f) > time.time() - 120
        ]
        if hasattr(st, "DATA_PATH")
        else "?"
    ),
)
from w22.constants import DATA_PATH

print(
    "recent .tex in DATA_PATH:",
    [
        os.path.basename(f)
        for f in glob.glob(os.path.join(DATA_PATH, "*.tex"))
        if os.path.getmtime(f) > time.time() - 120
    ],
)
print(
    "recent era5 pdf:",
    [
        os.path.basename(f)
        for f in glob.glob(os.path.join(wp.FIGURE_PATH, "*era5*.pdf"))
        if os.path.getmtime(f) > time.time() - 120
    ],
)
print("DONE")
