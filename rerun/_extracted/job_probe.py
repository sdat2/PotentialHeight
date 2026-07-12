import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import dask
dask.config.set(scheduler="synchronous")
from w22.ps_runs import get_processed_data_for_point
ds = get_processed_data_for_point(place="new_orleans", member="r4i1p1f1", model="CESM2", exp="ssp585")
print("PROBE OK:", dict(ds.sizes), "| vars:", list(ds.data_vars)[:8])
print("august steps:", len(ds.time))
