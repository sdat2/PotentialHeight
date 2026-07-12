import os,warnings,traceback,time
os.environ["LOKY_MAX_CPU_COUNT"]="3"; os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import dask; dask.config.set(scheduler="synchronous")
from w22.ps_runs import spatial_example
import w22.plot as wp, w22.stats2 as st
def step(name, fn):
    try: t=time.time(); fn(); print(f"OK {name} ({(time.time()-t)/60:.1f} min)",flush=True)
    except Exception as e: print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}",flush=True); traceback.print_exc(limit=2)
# 1. spatial fields (HadGEM3 r1, Aug 2015) for NO + HK regions
step("spatial gom", lambda: spatial_example(place="new_orleans",model="HADGEM3-GC31-MM",member="r1i1p1f3",pi_version=4,trial=1,recalculate_pi=True))
step("spatial scs", lambda: spatial_example(place="hong_kong",model="HADGEM3-GC31-MM",member="r1i1p1f3",pi_version=4,trial=1,recalculate_pi=True))
# 2. multipanel trend figures
step("multipanel new_orleans (Image-16)", lambda: wp.multipanel(place="new_orleans"))
step("multipanel hong_kong (Image-17)", lambda: wp.multipanel(place="hong_kong", models={"HADGEM3-GC31-MM","MIROC6","ERA5"}))
# 3. ERA5 trend figure (Image-14)
if hasattr(wp,"plot_era5_timeseries"): step("plot_era5_timeseries new_orleans (Image-14)", lambda: wp.plot_era5_timeseries(place="new_orleans"))
# 4. trend tables
step("temporal_relationship_data new_orleans", lambda: st.temporal_relationship_data(place="new_orleans"))
step("temporal_relationship_data hong_kong", lambda: st.temporal_relationship_data(place="hong_kong"))
import glob
print("\n=== figures produced ==="); 
for f in sorted(glob.glob(os.path.join(wp.FIGURE_PATH,"*multipanel*.pdf"))+glob.glob(os.path.join(wp.FIGURE_PATH,"*era5*.pdf")), key=os.path.getmtime, reverse=True)[:6]:
    print("  ", os.path.basename(f))
print("DONE")
