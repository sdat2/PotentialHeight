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
import dask

dask.config.set(scheduler="synchronous")
import numpy as np, xarray as xr
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure

rng = np.random.default_rng(0)
IB = "/Volumes/s/tcpips/data/ibtracs"

cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
nrm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["usa_rmw"]
].load()
f = lambda ds, n: np.asarray(ds[n].values, dtype="float64").ravel()
vmax = f(cps, "vmax")
msl = f(cps, "msl")
sst = f(cps, "sst")
t0 = f(cps, "t0")
rh = f(cps, "rh")
rmax_b = f(cps, "rmax")
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
ckcd = f(cps, "ck_cd")
cd = f(cps, "cd")
usa_rmw = f(nrm, "usa_rmw")
storm = np.repeat(np.arange(cps.sizes["storm"]), cps.sizes["date_time"])
if np.nanmax(rh) > 1.5:
    rh = rh / 100.0

# buggy normalized (per obs), exactly as code: usa_rmw*1852/rmax
norm_b = np.full(rmax_b.shape, np.nan)
m = np.isfinite(usa_rmw) & np.isfinite(rmax_b) & (rmax_b > 0)
norm_b[m] = usa_rmw[m] * 1852.0 / rmax_b[m]


def perc_gt1(x, filt):
    x = np.where(filt, x, np.nan).ravel()
    x = x[~np.isnan(x)]
    return np.sum(x > 1) / len(x) * 100 if len(x) else np.nan, len(x)


def per_storm(x, filt):
    xf = np.where(filt, x, np.nan)
    s = storm
    v = xf
    ok = ~np.isnan(v)
    s2 = s[ok]
    v2 = v[ok]
    if not len(s2):
        return np.nan, 0
    o = np.argsort(s2, kind="stable")
    s2 = s2[o]
    v2 = v2[o]
    u, st = np.unique(s2, return_index=True)
    mx = np.maximum.reduceat(v2, st)
    return np.mean(mx > 1) * 100, len(u)


FN = np.ones_like(sst, bool)
FT = sst >= 26.5
FTL = (sst >= 26.5) & (np.abs(lat) <= 30)
print("[REPRODUCE BUGGY] r2 normalized_size_cps, per-OBSERVATION (code _perc_gt_1):")
for nm, filt, paper in [
    ("none", FN, 17.06),
    ("sst>26.5", FT, 7.94),
    ("sst>26.5 & |lat|<=30", FTL, 5.89),
]:
    p, n = perc_gt1(norm_b, filt)
    ps, _ = per_storm(norm_b, filt)
    print(
        f"   {nm:22} per-obs={p:6.2f}% (paper {paper})   per-storm-max={ps:6.2f}%   [n={n}]"
    )

# --- robust ratio re-solve ---
vi = np.where(
    m
    & np.isfinite(vmax)
    & (vmax > 0.01)
    & np.isfinite(sst)
    & np.isfinite(rh)
    & (rh >= 0)
    & (rh <= 1)
)[0]


def solve(i, buggy):
    o = psmod.carnot_pm_from_y
    if buggy:
        r = rh[i]
        psmod.carnot_pm_from_y = lambda y, pda, tns, _r=r: (
            pda - (1.0 - _r) * buck_sat_vap_pressure(tns)
        ) / y + buck_sat_vap_pressure(tns)
    try:
        _, _, _, rmx = psmod.calculate_ps_ufunc(
            vmax[i],
            msl[i],
            sst[i],
            t0[i],
            lat[i],
            rh[i],
            ckcd[i] if np.isfinite(ckcd[i]) else 0.9,
            cd[i] if np.isfinite(cd[i]) else 0.0015,
            0.002,
            1.2,
            "isothermal",
        )
        return rmx
    finally:
        psmod.carnot_pm_from_y = o


N = 6000
samp = rng.choice(vi, size=min(N, len(vi)), replace=False)
print(f"\nre-solving {len(samp)} points (buggy+fixed)...")
rb = np.array([solve(i, True) for i in samp])
rf = np.array([solve(i, False) for i in samp])
good = np.isfinite(rb) & np.isfinite(rf) & (rb > 0) & (rf > 0)
rho = rf[good] / rb[good]
inl = (rho > 0.6) & (rho < 1.02)  # drop solver-failure outliers
print(
    f"converged {good.sum()}/{len(samp)}; inliers {inl.sum()}; rho median {np.median(rho[inl]):.4f} range [{rho[inl].min():.3f},{rho[inl].max():.3f}]"
)
sg = samp[good][inl]
y = rho[inl]


def feats(i):
    return np.column_stack(
        [np.ones_like(rh[i]), rh[i], rh[i] ** 2, sst[i], vmax[i], rh[i] * sst[i]]
    )


# 70/30 split
perm = rng.permutation(len(sg))
tr, va = perm[: int(0.7 * len(sg))], perm[int(0.7 * len(sg)) :]
X = feats(sg)
coef, _, _, _ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
err = np.abs(X[va] @ coef - y[va]) / y[va]
print(
    f"fit held-out err: median {np.median(err)*100:.3f}%, 95pct {np.percentile(err,95)*100:.3f}%, max {err.max()*100:.3f}%"
)
# apply
rho_all = np.clip(feats(np.arange(len(rh))) @ coef, 0.6, 1.02)
norm_f = np.full(norm_b.shape, np.nan)
norm_f[m] = norm_b[m] / rho_all[m]
print("\n===== r2 exceedance BUGGY -> FIXED (per-observation) =====")
for nm, filt, paper in [
    ("none", FN, 17.06),
    ("sst>26.5", FT, 7.94),
    ("sst>26.5&|lat|<=30", FTL, 5.89),
]:
    pb, _ = perc_gt1(norm_b, filt)
    pf, _ = perc_gt1(norm_f, filt)
    print(f"   {nm:20} {pb:6.2f}% -> {pf:6.2f}%   (paper {paper})")
print(
    f"\n median rho {np.median(rho_all[m]):.4f} -> normalized sizes rise ~{(1/np.median(rho_all[m])-1)*100:.1f}%"
)
# supersize extremes (per-obs max ratio, sst+lat filter)
xb = np.where(FTL, norm_b, np.nan)
xf = np.where(FTL, norm_f, np.nan)
print(f" max normalized (AF): buggy {np.nanmax(xb):.2f} -> fixed {np.nanmax(xf):.2f}")
print("DONE")
