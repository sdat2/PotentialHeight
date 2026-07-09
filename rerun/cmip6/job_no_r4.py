"""Regenerate the New Orleans CESM2 r4i1p1f1 ssp585 point timeseries (fixed PS solver).

Purpose
-------
Recompute the per-point (New Orleans) potential-size timeseries for CMIP6
CESM2 member ``r4i1p1f1`` under ssp585, with potential intensity (PI)
recomputed on the fly from the SSD regrid data (tcpyPI), using the FIXED
potential-size solver (2026-07 humidity/units bug fix:
``w22.w22_carnot.carnot_pm_from_y`` via ``w22.ps``). This is the single-member
version of the ``ps_for_place`` sweep; it produces one of the per-place
timeseries datasets that the multipanel figures / temporal-relationship tables
are built from (canonical point r0 = 2128.81 km, rmax = 60.67 km at CkCd = 1,
rh = 0.9).

This is a faithful, durable copy of the recovered scratchpad ``job_no_r4.py``.
All numerical logic lives in the repo function ``w22.ps_runs.point_timeseries``
(which calls ``w22.ps.parallelized_ps13_dask``); nothing is computed here.

Durable inputs (read by ``w22.ps_runs.get_processed_data_for_point``)
--------------------------------------------------------------------
- CMIP6 regrid (tcpyPI-ready) ocean + atmos data, resolved by
  ``tcpips.constants.CDO_PATH``:
  ``<repo>/data/cmip6/regrid/ssp585/{ocean,atmos}/CESM2/r4i1p1f1.nc``
  (the repo ``data`` directory is SSD-backed). PI is recomputed here
  (``recalculate_pi=True``), so no precomputed PI zarr is read.

Output (written to the repo, durable)
-------------------------------------
- ``w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc``
  (path resolved by ``w22.constants.DATA_PATH`` relative to the installed
  ``w22`` package; there is no scratchpad path to redirect).

Expected result
---------------
Regenerates the fixed-solver New Orleans / CESM2 r4 point timeseries
(``vmax_3``, ``r0_3``, ``rmax_3``, ``rmax_1`` per year). This is an
intermediate dataset feeding the figures / temporal tables, not one of the
IBTrACS exceedance-table numbers.

How to run (macOS/exFAT-safe, under caffeinate + the RSS/wall watchdog)
----------------------------------------------------------------------
    caffeinate -is python rerun/mem_guard.py rerun/cmip6/job_no_r4.py 6000 3600

or directly (env guards are set at the top of this file):

    caffeinate -is python rerun/cmip6/job_no_r4.py

The env guards below (single-thread BLAS, capped LOKY workers, disabled HDF5
file locking, synchronous dask) are required on macOS/exFAT to avoid
BrokenProcessPool, file-lock errors and multi-core memory blow-ups.
"""

import os

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

import dask

dask.config.set(scheduler="synchronous")
from w22.ps_runs import point_timeseries

# Fixed-solver regeneration: New Orleans, CESM2 r4i1p1f1, ssp585, PI recomputed
# from the SSD regrid data (tcpyPI), output -> w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc
point_timeseries(member="r4i1p1f1", model="CESM2", exp="ssp585",
                 place="new_orleans", pressure_assumption="isothermal",
                 pi_version=4, recalculate_pi=True)
print("JOB DONE")
