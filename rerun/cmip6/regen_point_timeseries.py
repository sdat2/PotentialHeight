"""Regenerate the New Orleans / Hong Kong point potential-size timeseries (fixed solver).

PURPOSE
-------
Driver that regenerates the per-point CMIP6 (historical + ssp585) and ERA5
potential-size (PS) timeseries for New Orleans and Hong Kong, using the FIXED
potential-size solver (2026-07 humidity/units bug fix in
``w22.w22_carnot.carnot_pm_from_y``, reached here via
``w22.ps.parallelized_ps13_dask`` -> ``calculate_ps13_ufunc`` ->
``point_solution_ps``). These point files feed the CMIP6 tradeoff / temporal
figures. The New-Orleans point is New Orleans (29.9511 N) offset by
``lat_offset = -1`` and snapped to the nearest grid cell (~28.75 N); Hong Kong
is 22.3964 N offset by ``lat_offset = -1``, ``lon_offset = +1`` (see
``w22.constants.OFFSET_D``).

COMPUTATION (imported from the repo — NOT duplicated here):
    w22.ps_runs.point_timeseries       -> writes one .nc per (place, model, member, exp)
    w22.ps_runs.point_era5_timeseries  -> writes one .nc per place for ERA5
Both call ``parallelized_ps13_dask`` (the fixed pi4 PS solver) internally.

DURABLE INPUTS
--------------
Read by the imported repo functions (not by this driver directly):
  - CMIP6 processed / PI4 fields via ``get_processed_data_for_point`` +
    ``calculate_pi`` (recalculate_pi=True, pi_version=4).
  - ERA5 regridded fields via ``get_all_regridded_data`` (1940-2024).

OUTPUTS
-------
Written by the imported repo functions to ``w22.constants.DATA_PATH``
(= <repo>/w22/data), one netCDF per case:
  - CMIP6:  ``{place}_august_{exp}_{model}_{member}_isothermal_pi4new.nc``
  - ERA5:   ``{place}_august_era5_isothermal.nc``
The skip-check below MUST match the CMIP6 filename that ``point_timeseries``
writes (pressure_assumption="isothermal", pi_version=4), so re-running only
recomputes cases whose output is missing.

EXPECTED NUMBERS
----------------
Canonical fixed-solver point check (independent sanity value): r0 = 2128.81 km,
rmax = 60.67 km at CkCd = 1, rh = 0.9. This driver produces the point-timeseries
inputs for the CMIP6 potential-size figures; it does not itself print an
exceedance table.

HOW TO RUN (macOS / exFAT-aware, long job — hours; use caffeinate + a memory guard)
-----------------------------------------------------------------------------------
    caffeinate -i python /Users/simon/worstsurge/rerun/cmip6/regen_point_timeseries.py
Or wrap under the repo memory guard (rerun/_extracted/guarded_run.py / write_mem_guard.py)
to cap resident memory. The env guards at the top of this file (single-thread
BLAS, capped joblib workers, disabled HDF5 file locking, synchronous dask
scheduler) must stay set BEFORE importing dask / xarray on this platform to
avoid file-lock failures on exFAT volumes and BrokenProcessPool on macOS.
"""

import os
import warnings
import time
import traceback

# --- Environment safety guards (macOS / exFAT). Must run before dask/xarray import. ---
os.environ["LOKY_MAX_CPU_COUNT"] = "3"  # cap joblib workers (repo funcs may default to all cores)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # required for netCDF/HDF5 on exFAT/macOS
for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"  # single-thread BLAS

warnings.filterwarnings("ignore")

import dask

dask.config.set(scheduler="synchronous")  # not "processes": avoids BrokenProcessPool on macOS

from w22.constants import DATA_PATH
from w22.ps_runs import point_timeseries, point_era5_timeseries

MODELS = {
    "CESM2": ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"],
    "HadGEM3-GC31-MM": ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"],
    "MIROC6": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
}
PLACES = ["new_orleans", "hong_kong"]

done = 0
skip = 0
fail = 0

for place in PLACES:
    for model, members in MODELS.items():
        for mem in members:
            for exp in ["historical", "ssp585"]:
                out = os.path.join(
                    DATA_PATH,
                    f"{place}_august_{exp}_{model}_{mem}_isothermal_pi4new.nc",
                )
                if os.path.exists(out):
                    skip += 1
                    continue
                try:
                    t = time.time()
                    point_timeseries(
                        member=mem,
                        model=model,
                        recalculate_pi=True,
                        exp=exp,
                        place=place,
                        pressure_assumption="isothermal",
                        pi_version=4,
                    )
                    done += 1
                    print(
                        f"OK {place}/{model}/{mem}/{exp} ({(time.time()-t):.0f}s) "
                        f"[done={done} skip={skip} fail={fail}]",
                        flush=True,
                    )
                except Exception as e:
                    fail += 1
                    print(
                        f"FAIL {place}/{model}/{mem}/{exp}: {type(e).__name__}: {str(e)[:150]}",
                        flush=True,
                    )

# ERA5 point timeseries for NO + HK
for place in PLACES:
    try:
        point_era5_timeseries(place=place)  # may need pi_version arg; try default
        print(f"OK ERA5 {place}", flush=True)
    except Exception as e:
        print(f"FAIL ERA5 {place}: {type(e).__name__}: {str(e)[:150]}", flush=True)

print(f"\nSUMMARY done={done} skip={skip} fail={fail}")
print("DONE")
