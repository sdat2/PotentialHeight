import os, warnings, traceback

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
import w22.plot as wp, w22.stats2 as st


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
import glob, time

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
