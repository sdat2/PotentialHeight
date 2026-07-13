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
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

IB = "/Volumes/s/tcpips/data/ibtracs"
OUT = "/private/tmp/claude-501/-Users-simon-thesis/3ecac1c7-3c1c-48be-9b89-cf847cc43e0d/scratchpad"
cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
pip = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.pi_ps.nc")[
    ["vmax", "rmax"]
].load()
ct1 = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.ps_cat1.nc")[
    ["vmax", "rmax"]
].load()
nrm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["usa_rmw", "usa_wind", "name"]
].load()
S, D = cps.sizes["storm"], cps.sizes["date_time"]
N = S * D
r = lambda ds, n: np.asarray(ds[n].values, dtype="float64").ravel()
sst = r(cps, "sst")
rh = r(cps, "rh")
msl = r(cps, "msl")
t0 = r(cps, "t0")
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
ck = r(cps, "ck_cd")
cd = r(cps, "cd")
if np.nanmax(rh) > 1.5:
    rh = rh / 100
Vp = r(pip, "vmax")
usa_rmw = r(nrm, "usa_rmw")
usa_wind = r(nrm, "usa_wind")
storm = np.repeat(np.arange(S), D)
W = (usa_wind * 0.514444 > 33.0) & (Vp * 0.8 > 33.0)  # Cat-1 wind filter (uniform)
envok = (
    np.isfinite(sst)
    & np.isfinite(rh)
    & (rh >= 0)
    & (rh <= 1)
    & np.isfinite(msl)
    & np.isfinite(t0)
    & np.isfinite(lat)
)


def solve_measure(vmaxarr):
    idx = np.where(W & envok & np.isfinite(vmaxarr) & (vmaxarr > 0.01))[0]

    def one(i):
        import w22.ps as p

        return p.calculate_ps_ufunc(
            vmaxarr[i],
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

    res = Parallel(n_jobs=4, backend="loky")(delayed(one)(i) for i in idx)
    rmax_fixed = np.full(N, np.nan)
    rmax_fixed[idx] = res
    return rmax_fixed


measures = {
    "r3": (Vp, r(pip, "rmax")),
    "r2": (r(cps, "vmax"), r(cps, "rmax")),
    "r1": (r(ct1, "vmax"), r(ct1, "rmax")),
}
results = {}
for name, (vmaxarr, rmax_buggy) in measures.items():
    t = time.time()
    rmax_fixed = solve_measure(vmaxarr)
    nsolved = np.isfinite(rmax_fixed).sum()
    results[name] = (rmax_buggy, rmax_fixed)
    np.savez(
        f"{OUT}/exact_{name}.npz",
        rmax_buggy=rmax_buggy,
        rmax_fixed=rmax_fixed,
        usa_rmw=usa_rmw,
        storm=storm,
        W=W,
        lat=lat,
        sst=sst,
    )
    print(f"[{name}] solved {nsolved} pts in {(time.time()-t)/60:.1f} min", flush=True)


def surv1(psm):
    x = psm[~np.isnan(psm)]
    xs = np.sort(x)
    sf = 1 - np.arange(1, len(xs) + 1) / len(xs)
    return (
        float(interp1d(xs, sf, bounds_error=False, fill_value="extrapolate")(1.0)) * 100
    )


def exc(norm, min_sst, max_lat):
    f = W.copy()
    if min_sst is not None:
        f = f & (sst >= min_sst)
    if max_lat is not None:
        f = f & (np.abs(lat) <= max_lat)
    x = np.where(f, norm, np.nan).reshape(S, D)
    psm = np.nanmax(np.where(np.isnan(x), -np.inf, x), axis=1)
    psm = np.where(np.isfinite(psm), psm, np.nan)
    return surv1(psm)


combos = {
    "r3": [(None, None, 40.80), (None, 30, 35.74)],
    "r2": [(None, None, 17.06), (26.5, None, 7.94), (26.5, 30, 5.89)],
    "r1": [(None, None, 9.85), (None, 30, 2.39)],
}
print("\n===== EXACT exceedance (fit-free): BUGGY(stored) -> FIXED(re-solved) =====")
for name in ["r3", "r2", "r1"]:
    rb, rf = results[name]
    nb = usa_rmw * 1852 / rb
    nf = usa_rmw * 1852 / rf
    for ms, ml, pap in combos[name]:
        print(
            f"  {name} sst>={ms},|lat|<={ml}:  {exc(nb,ms,ml):6.2f}% -> {exc(nf,ms,ml):6.2f}%   (paper {pap})"
        )
# supersize top-6 (r2, AF)
rb, rf = results["r2"]
nb = usa_rmw * 1852 / rb
nf = usa_rmw * 1852 / rf
try:
    nmv = nrm["name"].values
    names = np.array(
        [
            (
                "".join(
                    [
                        c.decode() if isinstance(c, bytes) else str(c)
                        for c in row
                        if c not in (b"", b" ")
                    ]
                ).strip()
                if hasattr(row, "__iter__")
                else str(row)
            )
            for row in nmv
        ]
    )
except Exception:
    names = None


def top6(norm):
    f = W & (sst >= 26.5) & (np.abs(lat) <= 30)
    x = np.where(f, norm, np.nan).reshape(S, D)
    psm = np.nanmax(np.where(np.isnan(x), -np.inf, x), axis=1)
    psm = np.where(np.isfinite(psm), psm, np.nan)
    o = np.argsort(np.where(np.isnan(psm), -1, psm))[::-1][:6]
    return [(int(i), float(psm[i])) for i in o]


print("\n  top-6 supersize (r2, AF): storm  buggy -> fixed")
tb = dict(top6(nb))
tf = dict(top6(nf))
for sidx in list(dict.fromkeys(list(tf) + list(tb)))[:8]:
    nm = names[sidx] if names is not None and sidx < len(names) else "?"
    print(
        f"    {sidx} ({nm}): {tb.get(sidx,float('nan')):.2f} -> {tf.get(sidx,float('nan')):.2f}"
    )
print("\nDONE")
