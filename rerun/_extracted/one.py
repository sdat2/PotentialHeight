import os, warnings

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
import dask

dask.config.set(scheduler="synchronous")
from w22.ps_runs import point_timeseries

try:
    point_timeseries(
        member="r1i1p1f3",
        model="HadGEM3-GC31-MM",
        recalculate_pi=True,
        exp="historical",
        place="hong_kong",
        pressure_assumption="isothermal",
        pi_version=4,
    )
    print("OK regenerated HK HadGEM3 r1 historical")
except Exception as e:
    import traceback

    print("FAIL:", type(e).__name__, e)
    traceback.print_exc(limit=3)
print("DONE")
