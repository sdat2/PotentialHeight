import os, warnings

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
import w22.test_figures as tf

# figure_4a (single-solution crossing, Image-2) — find the producing fn
for fn in ("test_figure_4a", "figure_4a", "test_figure_4"):
    if hasattr(tf, fn):
        print("running", fn, flush=True)
        try:
            getattr(tf, fn)()
            print("  OK", fn)
        except Exception as e:
            print("  FAIL", fn, type(e).__name__, str(e)[:200])
        break
print("DONE")
