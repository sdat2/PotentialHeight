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

# Fixed-solver regeneration: New Orleans, CESM2 r4i1p1f1, ssp585, PI recomputed
# from the SSD regrid data (tcpyPI), output -> w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc
point_timeseries(
    member="r4i1p1f1",
    model="CESM2",
    exp="ssp585",
    place="new_orleans",
    pressure_assumption="isothermal",
    pi_version=4,
    recalculate_pi=True,
)
print("JOB DONE")
