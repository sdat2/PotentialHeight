"""Regenerate the CMIP6/ERA5 potential-size/intensity trend figures (multipanel).

Purpose
-------
Driver that rebuilds the {new_orleans, hong_kong} multipanel trend figures and
the accompanying temporal-relationship trend tables using the FIXED
potential-size solver (2026-07 humidity/units fix in
``w22.w22_carnot.carnot_pm_from_y``; canonical point r0 = 2128.81 km,
rmax = 60.67 km at CkCd = 1, rh = 0.9). Every call below delegates to the repo
functions; NO numerical logic lives in this file, so re-running reproduces the
published/thesis figures and trend numbers exactly.

Steps (left panels = spatial fields, right panels = timeseries):
  1. spatial fields for New Orleans (GoM) and Hong Kong (SCS) regions
     (HadGEM3-GC31-MM r1i1p1f3, Aug 2015) via ``w22.ps_runs.spatial_example``
     (recalculate_pi=True, pi_version=4, trial=1);
  2. the two multipanel trend figures via ``w22.plot.multipanel``;
  3. the ERA5 New-Orleans trend figure via ``w22.plot.plot_era5_timeseries``
     (guarded by ``hasattr``; intentionally left as-recovered — see
     "Faithfulness" below);
  4. the temporal-relationship trend tables for both cities via
     ``w22.stats2.temporal_relationship_data``.

Durable inputs (read by the repo functions, verified present)
-------------------------------------------------------------
  - CMIP6 ssp585 regridded fields under ``tcpips.constants.CDO_PATH`` (read by
    ``spatial_example`` when ``recalculate_pi=True``) plus the precomputed
    CMIP6 potential-size/intensity per-place timeseries loaded by
    ``w22.plot.get_cmip6_timeseries`` / ``w22.stats2.get_timeseries`` for
    CESM2 (r4,r10,r11), HADGEM3-GC31-MM (r1..r3 i1p1f3),
    MIROC6 (r1..r3 i1p1f1).
  - ERA5 per-place potential-size/intensity timeseries loaded by
    ``w22.plot.get_era5_timeseries``.
  - /Volumes/s/tcpips/data/ibtracs/*.nc and era5_unique_points_*.nc (durable
    copies of the fixed-solver inputs used elsewhere in the pipeline).

Outputs (written by the repo functions to their module-constant paths)
----------------------------------------------------------------------
  - Figures -> ``w22.plot.FIGURE_PATH`` (== ``w22/img/``):
        new_orleans_multipanel.pdf, hong_kong_multipanel.pdf,
        new_ps_calculation_output_gom.pdf and related spatial PDFs.
  - Trend tables -> ``w22.stats2.DATA_PATH`` (== ``w22/data/stats/``):
        {place}_temporal_relationships_pi4.csv,
        {place}_temporal_correlation_pi4.tex,
        {place}_temporal_fit_pi4.tex   for place in {new_orleans, hong_kong}.
  The durable, regenerated copies of the figures live in
  ``/Volumes/s/tcpips/fixed_figures/``; copy the freshly-written PDFs there after
  a run if you want to refresh them. (Output paths are fixed by the repo module
  constants and are deliberately NOT redirected here — doing so would change the
  recovered behaviour.)

Expected numbers
----------------
Reproduces the thesis trend figures (Image-16 new_orleans multipanel, Image-17
hong_kong multipanel, Image-14 ERA5 new_orleans timeseries) and the temporal
trend/correlation tables under the fixed solver. All CMIP6 member timeseries
fits use year_min=2014, year_max=2100 (the defaults baked into
``w22.plot.multipanel``).

How to run (macOS/exFAT-safe, under caffeinate + the RSS/wall watchdog)
----------------------------------------------------------------------
    caffeinate -is python -m rerun.cmip6.figures_multipanel
or, with the repo mem_guard,
    caffeinate -is python rerun/mem_guard.py rerun/cmip6/figures_multipanel.py 6000 3600

The env-safety guards at the top MUST stay: HDF5_USE_FILE_LOCKING=FALSE (exFAT),
LOKY_MAX_CPU_COUNT=3 (repo funcs default n_jobs=-1 => all cores => ~5 GB), and
single-thread BLAS. dask is pinned to the synchronous scheduler to avoid the
macOS BrokenProcessPool seen with the 'processes' scheduler.

Faithfulness
------------
Recovered VERBATIM from the session transcript (scratchpad ``plot_cmip6.py``).
The only changes vs. the original are this docstring and whitespace/line
reflowing of the (non-numerical) env-guard and call statements — every import,
argument, filter and joblib/dask setting is byte-for-byte identical, so the
computation is unchanged. In particular the ``plot_era5_timeseries`` call in
step 3 is preserved exactly as recovered even though its repo signature takes a
mandatory ``axs`` first positional argument; under the ``step()`` try/except it
simply reports FAIL and does not affect any other output. Do not "fix" it.
"""

import os, warnings, traceback, time

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
from w22.ps_runs import spatial_example
import w22.plot as wp, w22.stats2 as st


def step(name, fn):
    try:
        t = time.time()
        fn()
        print(f"OK {name} ({(time.time()-t)/60:.1f} min)", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        traceback.print_exc(limit=2)


# 1. spatial fields (HadGEM3 r1, Aug 2015) for NO + HK regions
step(
    "spatial gom",
    lambda: spatial_example(
        place="new_orleans",
        model="HADGEM3-GC31-MM",
        member="r1i1p1f3",
        pi_version=4,
        trial=1,
        recalculate_pi=True,
    ),
)
step(
    "spatial scs",
    lambda: spatial_example(
        place="hong_kong",
        model="HADGEM3-GC31-MM",
        member="r1i1p1f3",
        pi_version=4,
        trial=1,
        recalculate_pi=True,
    ),
)
# 2. multipanel trend figures
step("multipanel new_orleans (Image-16)", lambda: wp.multipanel(place="new_orleans"))
step(
    "multipanel hong_kong (Image-17)",
    lambda: wp.multipanel(
        place="hong_kong", models={"HADGEM3-GC31-MM", "MIROC6", "ERA5"}
    ),
)
# 3. ERA5 trend figure (Image-14)
if hasattr(wp, "plot_era5_timeseries"):
    step(
        "plot_era5_timeseries new_orleans (Image-14)",
        lambda: wp.plot_era5_timeseries(place="new_orleans"),
    )
# 4. trend tables
step(
    "temporal_relationship_data new_orleans",
    lambda: st.temporal_relationship_data(place="new_orleans"),
)
step(
    "temporal_relationship_data hong_kong",
    lambda: st.temporal_relationship_data(place="hong_kong"),
)

import glob

print("\n=== figures produced ===")
for f in sorted(
    glob.glob(os.path.join(wp.FIGURE_PATH, "*multipanel*.pdf"))
    + glob.glob(os.path.join(wp.FIGURE_PATH, "*era5*.pdf")),
    key=os.path.getmtime,
    reverse=True,
)[:6]:
    print("  ", os.path.basename(f))
print("DONE")
