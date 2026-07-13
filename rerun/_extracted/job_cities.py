import os

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
