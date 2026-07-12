import os, warnings, traceback
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import dask; dask.config.set(scheduler="synchronous")
import tcpips.ibtracs as ib
print("IBTRACS_DATA_PATH:", ib.IBTRACS_DATA_PATH, "->", os.path.realpath(ib.IBTRACS_DATA_PATH))
print("FIGURE_PATH:", ib.FIGURE_PATH)
for name, call in [
    ("plot_normalized_quad_dual (survival Image-8)", lambda: ib.plot_normalized_quad_dual()),
    ("plot_normalized_quad (histograms Image-9 + top tables)", lambda: ib.plot_normalized_quad()),
    ("create_exceedance_table", lambda: ib.create_exceedance_table()),
]:
    try:
        print(f"\n>>> {name}", flush=True); call(); print(f"    OK")
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
print("\n=== PDFs in img/ibtracs (recent) ===")
import glob, time
for f in sorted(glob.glob(os.path.join(ib.FIGURE_PATH,"*.pdf")), key=os.path.getmtime, reverse=True)[:12]:
    print(f"  {time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(f)))}  {os.path.basename(f)}")
print("DONE")
