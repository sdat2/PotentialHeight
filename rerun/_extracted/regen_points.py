import os, warnings, time, traceback

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
from w22.constants import DATA_PATH
from w22.ps_runs import point_timeseries, point_era5_timeseries

MODELS = {
    "CESM2": ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"],
    "HadGEM3-GC31-MM": ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"],
    "MIROC6": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
}
PLACES = ["new_orleans", "hong_kong"]
done = 0
skip = 0
fail = 0
for place in PLACES:
    for model, members in MODELS.items():
        for mem in members:
            for exp in ["historical", "ssp585"]:
                out = os.path.join(
                    DATA_PATH,
                    f"{place}_august_{exp}_{model}_{mem}_isothermal_pi4new.nc",
                )
                if os.path.exists(out):
                    skip += 1
                    continue
                try:
                    t = time.time()
                    point_timeseries(
                        member=mem,
                        model=model,
                        recalculate_pi=True,
                        exp=exp,
                        place=place,
                        pressure_assumption="isothermal",
                        pi_version=4,
                    )
                    done += 1
                    print(
                        f"OK {place}/{model}/{mem}/{exp} ({(time.time()-t):.0f}s) [done={done} skip={skip} fail={fail}]",
                        flush=True,
                    )
                except Exception as e:
                    fail += 1
                    print(
                        f"FAIL {place}/{model}/{mem}/{exp}: {type(e).__name__}: {str(e)[:150]}",
                        flush=True,
                    )
# ERA5 point timeseries for NO + HK
for place in PLACES:
    try:
        point_era5_timeseries(place=place)  # may need pi_version arg; try default
        print(f"OK ERA5 {place}", flush=True)
    except Exception as e:
        print(f"FAIL ERA5 {place}: {type(e).__name__}: {str(e)[:150]}", flush=True)
print(f"\nSUMMARY done={done} skip={skip} fail={fail}")
print("DONE")
