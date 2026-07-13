import os, warnings, time

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"
warnings.filterwarnings("ignore")
import numpy as np, xarray as xr
from joblib import Parallel, delayed

IB = "/Volumes/s/tcpips/data/ibtracs"
cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc")[
    ["vmax", "msl", "sst", "t0", "rh", "ck_cd", "cd"]
].load()
r = lambda n: np.asarray(cps[n].values, dtype="float64").ravel()
vmax, msl, sst, t0, rh, ck, cd = [
    r(n) for n in ("vmax", "msl", "sst", "t0", "rh", "ck_cd", "cd")
]
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
if np.nanmax(rh) > 1.5:
    rh = rh / 100
idx = np.where(np.isfinite(vmax) & (vmax > 0.01) & np.isfinite(sst) & np.isfinite(rh))[
    0
][:500]


def one(i):
    import w22.ps as p

    return p.calculate_ps_ufunc(
        vmax[i],
        msl[i],
        sst[i],
        t0[i],
        lat[i],
        rh[i],
        ck[i] if np.isfinite(ck[i]) else 0.9,
        cd[i] if np.isfinite(cd[i]) else 0.0015,
        0.002,
        1.2,
        "isothermal",
    )[3]


t = time.time()
out = Parallel(n_jobs=4, backend="loky")(delayed(one)(i) for i in idx)
print(
    f"n_jobs=4: {len(out)} solves in {time.time()-t:.0f}s ({np.isfinite(out).sum()} finite)"
)
