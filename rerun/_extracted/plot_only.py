import os, warnings, traceback

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
import w22.plot as wp, w22.stats2 as st


def step(name, fn):
    try:
        fn()
        print(f"OK {name}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=4)


step("multipanel new_orleans", lambda: wp.multipanel(place="new_orleans"))
step(
    "multipanel hong_kong",
    lambda: wp.multipanel(
        place="hong_kong", models={"HADGEM3-GC31-MM", "MIROC6", "ERA5"}
    ),
)
step(
    "plot_era5_timeseries new_orleans",
    lambda: wp.plot_era5_timeseries(place="new_orleans"),
)
step(
    "temporal_relationship_data new_orleans",
    lambda: st.temporal_relationship_data(place="new_orleans"),
)
import glob

print(
    "FIGS:",
    [
        os.path.basename(f)
        for f in glob.glob(os.path.join(wp.FIGURE_PATH, "*.pdf"))
        if os.path.getmtime(f) > __import__("time").time() - 200
    ],
)
print("DONE")
