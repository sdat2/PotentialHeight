"""Compute the per-city CMIP6 potential-size timeseries (fixed PS solver).

Purpose
-------
Drive ``w22.ps_runs.point_timeseries`` for the coastal-city points used in the
thesis / potential-height paper, producing the per-place potential-size
timeseries datasets that the multipanel city figures are built from. All
potential-size numbers come from the FIXED solver (2026-07 humidity/units bug
fix: ``w22.w22_carnot.carnot_pm_from_y``, reached here via
``w22.ps.parallelized_ps13_dask`` -> ``calculate_ps13_ufunc``; canonical point
r0 = 2128.81 km, rmax = 60.67 km at CkCd = 1, rh = 0.9).

This is a faithful, durable copy of the recovered scratchpad ``job_cities.py``.
It only calls the repo driver ``point_timeseries`` (which itself recomputes PI
from the CMIP6 fields and then runs the fixed PS solver) -- no numerical logic
lives in this file.

Runs performed (member r4i1p1f1, model CESM2, isothermal, pi_version 4,
recalculate_pi=True):
- galverston   / ssp585
- miami        / ssp585
- new_orleans  / historical

Durable inputs (read by ``get_processed_data_for_point`` inside the driver)
--------------------------------------------------------------------------
CMIP6 CDO-regridded ocean/atmos fields under ``tcpips.constants.CDO_PATH``:
``{CDO_PATH}/{exp}/{ocean,atmos}/CESM2/r4i1p1f1.nc`` for the experiments above.
The point selection uses ``w22.constants.OFFSET_D`` (galverston / miami /
new_orleans are all defined there).

Outputs (written by the driver to ``w22.constants.DATA_PATH`` == ``w22/data``)
-----------------------------------------------------------------------------
- ``galverston_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc``
- ``miami_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc``
- ``new_orleans_august_historical_CESM2_r4i1p1f1_isothermal_pi4new.nc``
(The output path is resolved internally by ``w22.constants`` relative to the
installed ``w22`` package; there is no scratchpad path to redirect.)

Expected result
---------------
Regenerates the fixed-solver per-city CMIP6 timeseries ``.nc`` files that feed
the multipanel city figures. This script produces *figure-supporting
timeseries*, not the IBTrACS exceedance-table numbers.

How to run (macOS/exFAT-safe, under caffeinate + the RSS/wall watchdog)
----------------------------------------------------------------------
    caffeinate -is python rerun/mem_guard.py rerun/cmip6/job_cities.py 6000 7200

or directly (env guards are set at the top of this file):

    caffeinate -is python rerun/cmip6/job_cities.py

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

for place, exp in [
    ("galverston", "ssp585"),
    ("miami", "ssp585"),
    ("new_orleans", "historical"),
]:
    print(f"===== {place} / {exp} =====", flush=True)
    point_timeseries(
        member="r4i1p1f1",
        model="CESM2",
        exp=exp,
        place=place,
        pressure_assumption="isothermal",
        pi_version=4,
        recalculate_pi=True,
    )
print("JOB DONE")
