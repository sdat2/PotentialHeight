"""Exact, fit-free PS re-solve of the IBTrACS Cat-1 track points (fixed solver).

For every IBTrACS storm/time point that passes the uniform Cat-1 wind filter and
has a finite environment, this re-solves the Wang-2022 potential size from
scratch with the corrected humidity back-conversion
(w22.w22_carnot.carnot_pm_from_y — the 2026-07 units/humidity fix; see
../unit_bug_fix.md) and recomputes the survival-at-1 exceedance table. It is the
"exact" counterpart to the fit-based re-solve: no regression surrogate, every
qualifying point goes through w22.ps.calculate_ps_ufunc directly (the ~72-min
joblib job).

Three intensity "measures" are re-solved, each with its own vmax array but the
SAME uniform Cat-1 filter and the SAME re-solve:
  * r3 — vmax = potential intensity (pi_ps.vmax), buggy rmax = pi_ps.rmax
  * r2 — vmax = cps.vmax,                          buggy rmax = cps.rmax
  * r1 — vmax = ps_cat1.vmax (Cat-1 minimum),      buggy rmax = ps_cat1.rmax
The normalized size used for exceedance is usa_rmw * 1852 / rmax, evaluated on
both the stored (buggy) rmax and the freshly re-solved (fixed) rmax so the two
columns are directly comparable.

Durable inputs (read-only; ORIGINAL products, not the patched copies — this
script IS the thing that produces the fix):
    /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.cps.nc
    /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.pi_ps.nc
    /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.ps_cat1.nc
    /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.normalized.nc
(The data/ibtracs repo symlink -> fixed_ibtracs is NOT needed here: this job
reads absolute /Volumes paths and re-solves rmax itself.)

Outputs (durable):
    /Volumes/s/tcpips/fixed_ibtracs/exact_resolve/exact_{r1,r2,r3}.npz
        each holding rmax_buggy, rmax_fixed, usa_rmw, storm, W, lat, sst
    plus the exceedance table + top-6 supersize list printed to stdout.

Expected numbers (FIXED / re-solved column; reproduces the thesis/paper table):
    r3  46.1 (all) / 42.4 (|lat|<=30)
    r2  19.9 (all) / 10.2 (sst>=26.5) / 7.94 (AF: sst>=26.5 & |lat|<=30)
    r1  13.5 (all) /  4.2 (|lat|<=30)
    supersize top-6 (r2, AF): Ellen 3.86, Man-Yi 3.11
The "(paper ...)" values printed on each line are the OLD (pre-fix) reference
numbers, kept inline for the buggy->fixed comparison.

How to run (via the memory guard, caffeinated so the machine won't sleep;
~72 min, steady-state RSS ~1-2 GB at n_jobs=4):
    export HDF5_USE_FILE_LOCKING=FALSE
    caffeinate -i -s python rerun/mem_guard.py \\
        rerun/ibtracs/resolve_exceedance.py 6144 21600
"""

import os, warnings, time

# exFAT SSD + macOS: HDF5 file locking must be off or reads fail with an HDF
# error; cap loky workers so joblib can't oversubscribe cores; BLAS
# single-threaded so the per-point scalar solves don't spawn thread pools.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
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
OUT = "/Volumes/s/tcpips/fixed_ibtracs/exact_resolve"
os.makedirs(OUT, exist_ok=True)
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
