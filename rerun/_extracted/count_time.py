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
import dask

dask.config.set(scheduler="synchronous")
import numpy as np, xarray as xr
import w22.ps as psmod

IB = "/Volumes/s/tcpips/data/ibtracs"
cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
pip = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.pi_ps.nc")[["vmax"]].load()
nrm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["usa_rmw", "usa_wind"]
].load()
S, D = cps.sizes["storm"], cps.sizes["date_time"]
r = lambda ds, n: np.asarray(ds[n].values, dtype="float64").ravel()
vmax = r(cps, "vmax")
sst = r(cps, "sst")
rh = r(cps, "rh")
usa_rmw = r(nrm, "usa_rmw")
usa_wind = r(nrm, "usa_wind")
Vp = r(pip, "vmax")
if np.nanmax(rh) > 1.5:
    rh = rh / 100
# comparison set: has observation (usa_rmw) + valid inputs
base = (
    np.isfinite(vmax)
    & (vmax > 0.01)
    & np.isfinite(sst)
    & np.isfinite(rh)
    & (rh >= 0)
    & (rh <= 1)
    & np.isfinite(usa_rmw)
    & (usa_rmw > 0)
)
cat1 = (usa_wind * 0.514444 > 33) & (Vp * 0.8 > 33)
print(f"points w/ obs + valid inputs (r2 grid):  {base.sum()}")
print(f"  ... also passing Cat-1 wind filter:     {(base&cat1).sum()}")
# time 800 solves
idx = np.where(base)[0]
samp = idx[:800]
msl = r(cps, "msl")
t0 = r(cps, "t0")
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
ck = r(cps, "ck_cd")
cd = r(cps, "cd")
psmod.calculate_ps_ufunc(
    vmax[samp[0]],
    msl[samp[0]],
    sst[samp[0]],
    t0[samp[0]],
    lat[samp[0]],
    rh[samp[0]],
    0.9,
    0.0015,
    0.002,
    1.2,
    "isothermal",
)  # warmup
t = time.time()
for i in samp:
    psmod.calculate_ps_ufunc(
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
    )
dt = (time.time() - t) / len(samp)
print(f"\nsteady-state: {dt*1000:.1f} ms/solve (single-thread)")
n = int(base.sum())
print(
    f"full exact r2 re-solve ({n} pts): serial ~{n*dt/60:.0f} min; n_jobs=4 ~{n*dt/4/60:.0f} min"
)
print(
    f"all 3 measures (~3x): serial ~{3*n*dt/60:.0f} min; n_jobs=4 ~{3*n*dt/4/60:.0f} min"
)
