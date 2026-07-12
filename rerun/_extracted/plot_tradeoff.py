import os,warnings,traceback
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import dask; dask.config.set(scheduler="synchronous")
import tcpips.ibtracs as ib
try:
    print("running vary_v_cps(KATRINA)...", flush=True)
    ib.vary_v_cps(v_reduc=0.8)
    print("OK")
except Exception as e:
    print("FAIL:", type(e).__name__, e); traceback.print_exc(limit=4)
import glob,time
for f in sorted(glob.glob(os.path.join(ib.FIGURE_PATH,"*.pdf")),key=os.path.getmtime,reverse=True)[:4]:
    print(f"  {time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(f)))} {os.path.basename(f)}")
print("DONE")
