import os, warnings, traceback

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
import matplotlib

matplotlib.use("Agg")
import dask

dask.config.set(scheduler="synchronous")
import tcpips.ibtracs as ib

try:
    ib.vary_v_cps(v_reduc=0.8)
    print("OK vary_v_cps")
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    traceback.print_exc(limit=4)
import glob, time

for f in sorted(
    glob.glob(os.path.join(ib.FIGURE_PATH, "*.pdf")), key=os.path.getmtime, reverse=True
)[:3]:
    print(
        f"  {time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(f)))} {os.path.basename(f)}"
    )
print("DONE")
