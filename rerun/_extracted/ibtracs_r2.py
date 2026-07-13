import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"
import dask

dask.config.set(scheduler="synchronous")
import numpy as np, xarray as xr
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure

rng = np.random.default_rng(0)
IB = "/Volumes/s/tcpips/data/ibtracs"

# --- load r2 (cps) inputs + buggy rmax, and observed usa_rmw (same storm x date_time grid) ---
cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
nrm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["usa_rmw"]
].load()
print(
    "cps dims",
    dict(cps.sizes),
    "| grids align:",
    cps.sizes == dict(storm=4813, date_time=360),
)


def flat(ds, name):
    return np.asarray(ds[name].values, dtype="float64").ravel()


vmax = flat(cps, "vmax")
msl = flat(cps, "msl")
sst = flat(cps, "sst")
t0 = flat(cps, "t0")
rh = flat(cps, "rh")
rmax_buggy = flat(cps, "rmax")
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
ckcd = flat(cps, "ck_cd")
cd = flat(cps, "cd")
wcool = np.full_like(vmax, 0.002)
usa_rmw = flat(nrm, "usa_rmw")  # nautical miles
storm_idx = np.repeat(np.arange(cps.sizes["storm"]), cps.sizes["date_time"])
if np.nanmax(rh) > 1.5:
    rh = rh / 100.0

# valid = have inputs, buggy rmax, and an observed rmw
valid = (
    np.isfinite(vmax)
    & (vmax > 0.01)
    & np.isfinite(sst)
    & np.isfinite(rh)
    & (rh >= 0)
    & (rh <= 1)
    & np.isfinite(rmax_buggy)
    & (rmax_buggy > 0)
    & np.isfinite(usa_rmw)
    & (usa_rmw > 0)
)
vi = np.where(valid)[0]
print(
    f"valid r2 points: {len(vi)} (of {valid.size}); storms w/ >=1 valid: {len(np.unique(storm_idx[vi]))}"
)

# --- reproduce BUGGY normalized r2 = usa_rmw[nmi]*1852 / rmax[m] ---
norm_buggy = np.full(valid.size, np.nan)
norm_buggy[vi] = usa_rmw[vi] * 1852.0 / rmax_buggy[vi]


def per_storm_exceed(norm, mask):
    """fraction of storms (with any qualifying obs) whose lifetime-max normalized > 1"""
    idx = np.where(mask)[0]
    s = storm_idx[idx]
    vals = norm[idx]
    order = np.argsort(s, kind="stable")
    s = s[order]
    vals = vals[order]
    uniq, start = np.unique(s, return_index=True)
    maxes = np.maximum.reduceat(vals, start)
    return np.mean(maxes > 1.0) * 100.0, len(uniq), uniq, maxes


# SF = all valid ; AF = valid & sst>26.5 & |lat|<30
sf_mask = valid.copy()
af_mask = valid & (sst > 26.5) & (np.abs(lat) < 30)
sf_b, ns, _, _ = per_storm_exceed(norm_buggy, sf_mask)
af_b, na, _, _ = per_storm_exceed(norm_buggy, af_mask)
print(
    f"\n[REPRODUCE BUGGY] r2 exceedance  SF={sf_b:.2f}% (paper 17.06)  AF={af_b:.2f}% (paper 5.89)  [nSF={ns}, nAF={na}]"
)


# --- re-solve a sample (buggy+fixed) to get per-point ratio rho = rmax_fixed/rmax_buggy ---
def solve(i, buggy):
    orig = psmod.carnot_pm_from_y
    if buggy:
        r = rh[i]

        def conv(y, pda, tns):
            e = buck_sat_vap_pressure(tns)
            return (pda - (1.0 - r) * e) / y + e

        psmod.carnot_pm_from_y = conv
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
            wcool[i],
            1.2,
            "isothermal",
        )
        return rmx
    finally:
        psmod.carnot_pm_from_y = orig


Nfit, Nval = 4000, 2000
samp = rng.choice(vi, size=min(Nfit + Nval, len(vi)), replace=False)
fitset, valset = samp[:Nfit], samp[Nfit : Nfit + Nval]
print(f"\nre-solving {len(samp)} sample points (buggy+fixed)...")
rb = np.array([solve(i, True) for i in samp])
rf = np.array([solve(i, False) for i in samp])
ok = np.isfinite(rb) & np.isfinite(rf) & (rb > 0)
rho = rf[ok] / rb[ok]  # shrink factor (<1)
# check resolved-buggy vs stored
resid = (rb[ok] - rmax_buggy[samp][ok]) / rmax_buggy[samp][ok]
print(
    f"resolved-buggy vs stored rmax: median {np.median(resid)*100:+.2f}%, 95pct |{np.percentile(np.abs(resid),95)*100:.2f}|%"
)
print(
    f"rho (rmax_fixed/rmax_buggy): min {rho.min():.3f}, median {np.median(rho):.3f}, max {rho.max():.3f}"
)


# --- fit rho ~ f(rh,sst,vmax); validate on held-out ---
def features(i):
    return np.column_stack(
        [np.ones_like(rh[i]), rh[i], rh[i] ** 2, sst[i], vmax[i], rh[i] * sst[i]]
    )


sok = samp[ok]
mask_fit = np.isin(sok, fitset)
mask_val = np.isin(sok, valset)
X = features(sok)
coef, _, _, _ = np.linalg.lstsq(X[mask_fit], rho[mask_fit], rcond=None)
pred_val = X[mask_val] @ coef
err = np.abs(pred_val - rho[mask_val]) / rho[mask_val]
print(
    f"fit validation (held-out {mask_val.sum()}): median err {np.median(err)*100:.3f}%, 95pct {np.percentile(err,95)*100:.3f}%, max {err.max()*100:.3f}%"
)

# --- apply fit to ALL valid points -> fixed rmax -> fixed normalized ---
rho_all = features(vi) @ coef
rho_all = np.clip(rho_all, 0.5, 1.05)
norm_fixed = np.full(valid.size, np.nan)
norm_fixed[vi] = norm_buggy[vi] / rho_all  # normalized rises as rmax shrinks

sf_f, _, _, _ = per_storm_exceed(norm_fixed, sf_mask)
af_f, _, _, _ = per_storm_exceed(norm_fixed, af_mask)
print(f"\n===== r2 (corresponding potential size) exceedance: BUGGY -> FIXED =====")
print(
    f"  SF (all storms):          {sf_b:5.2f}%  ->  {sf_f:5.2f}%   (paper quotes 17.06%)"
)
print(
    f"  AF (sst>26.5,|lat|<30):   {af_b:5.2f}%  ->  {af_f:5.2f}%   (paper quotes 5.89%)"
)
print(
    f"  median rho (shrink): {np.median(rho_all):.3f}  => normalized sizes rise ~{(1/np.median(rho_all)-1)*100:.1f}%"
)

# --- top supersize storms (per-storm max normalized, AF), buggy vs fixed ---
nm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["name", "sid"]
].load()
names = (
    np.array(
        [
            (
                b"".join(x).decode(errors="ignore").strip()
                if x.dtype.kind == "S"
                else str(x)
            )
            for x in nm["name"].values
        ]
    )
    if "name" in nm
    else None
)


def topN(norm, mask, N=6):
    idx = np.where(mask)[0]
    s = storm_idx[idx]
    vals = norm[idx]
    order = np.argsort(s, kind="stable")
    s = s[order]
    vals = vals[order]
    uniq, start = np.unique(s, return_index=True)
    maxes = np.maximum.reduceat(vals, start)
    top = np.argsort(maxes)[::-1][:N]
    return [(int(uniq[t]), float(maxes[t])) for t in top]


print("\n  top-6 supersize (AF), storm_idx: buggy_ratio -> fixed_ratio:")
tb = dict(topN(norm_buggy, af_mask))
tf = dict(topN(norm_fixed, af_mask))
allids = list(dict.fromkeys(list(tf) + list(tb)))
for sid in allids[:8]:
    nm_s = names[sid] if names is not None and sid < len(names) else "?"
    print(
        f"    storm {sid} ({nm_s}): {tb.get(sid,float('nan')):.2f} -> {tf.get(sid,float('nan')):.2f}"
    )
print("\nDONE")
