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
from scipy.interpolate import interp1d
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure

rng = np.random.default_rng(1)
IB = "/Volumes/s/tcpips/data/ibtracs"
cps = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
pip = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.pi_ps.nc")[
    ["vmax", "rmax", "t0"]
].load()
ct1 = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.ps_cat1.nc")[
    ["vmax", "rmax", "t0"]
].load()
nrm = xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[
    ["usa_rmw", "usa_wind"]
].load()
S, D = cps.sizes["storm"], cps.sizes["date_time"]
g = lambda ds, n: np.asarray(ds[n].values, dtype="float64").reshape(S, D)
# shared environment (from cps)
sst = g(cps, "sst")
rh = g(cps, "rh")
msl = g(cps, "msl")
t0 = g(cps, "t0")
lat = np.asarray(cps["lat"].values, dtype="float64").reshape(S, D)
ckcd = g(cps, "ck_cd")
cd = g(cps, "cd")
if np.nanmax(rh) > 1.5:
    rh = rh / 100.0
Vp = g(pip, "vmax")
usa_rmw = g(nrm, "usa_rmw")
usa_wind = g(nrm, "usa_wind")
W = (usa_wind * 0.514444 > 33.0) & (Vp * 0.8 > 33.0)  # uniform wind filter (obs & V_p)


def surv1(psm):
    x = psm[~np.isnan(psm)]
    xs = np.sort(x)
    sf = 1 - np.arange(1, len(xs) + 1) / len(xs)
    return (
        float(interp1d(xs, sf, bounds_error=False, fill_value="extrapolate")(1.0)) * 100
    )


def exc(norm, max_lat):
    f = W.copy()
    if max_lat is not None:
        f &= np.abs(lat) <= max_lat
    x = np.where(f, norm, np.nan)
    psm = np.nanmax(np.where(np.isnan(x), -np.inf, x), axis=1)
    return surv1(np.where(np.isfinite(psm), psm, np.nan))


def solve1(vmax_i, msl_i, sst_i, t0_i, lat_i, rh_i, ck, cdd, buggy):
    o = psmod.carnot_pm_from_y
    if buggy:
        psmod.carnot_pm_from_y = lambda y, pda, tns, _r=rh_i: (
            pda - (1.0 - _r) * buck_sat_vap_pressure(tns)
        ) / y + buck_sat_vap_pressure(tns)
    try:
        _, _, _, rx = psmod.calculate_ps_ufunc(
            vmax_i,
            msl_i,
            sst_i,
            t0_i,
            lat_i,
            rh_i,
            ck if np.isfinite(ck) else 0.9,
            cdd if np.isfinite(cdd) else 0.0015,
            0.002,
            1.2,
            "isothermal",
        )
        return rx
    finally:
        psmod.carnot_pm_from_y = o


fl = lambda a: a.ravel()


def do_measure(name, vmax2d, rmax_b2d, paper):
    norm_b = usa_rmw * 1852.0 / rmax_b2d
    print(f"\n===== {name} =====")
    print(
        f"[GATE] (None,None)={exc(norm_b,None):.2f} (paper {paper[0]});  (·,lat<=30)={exc(norm_b,30):.2f} (paper {paper[1]})"
    )
    fv, fm, fs, ft, frh, frx, flt, fck, fcd, fW = [
        fl(a) for a in (vmax2d, msl, sst, t0, rh, rmax_b2d, lat, ckcd, cd, W)
    ]
    vi = np.where(
        np.isfinite(fv)
        & (fv > 0.01)
        & np.isfinite(fs)
        & np.isfinite(frh)
        & (frh >= 0)
        & (frh <= 1)
        & np.isfinite(frx)
        & (frx > 0)
        & fW
    )[0]
    samp = rng.choice(vi, size=min(3000, len(vi)), replace=False)
    rb = np.array(
        [
            solve1(fv[i], fm[i], fs[i], ft[i], flt[i], frh[i], fck[i], fcd[i], True)
            for i in samp
        ]
    )
    rf = np.array(
        [
            solve1(fv[i], fm[i], fs[i], ft[i], flt[i], frh[i], fck[i], fcd[i], False)
            for i in samp
        ]
    )
    gd = np.isfinite(rb) & np.isfinite(rf) & (rb > 0) & (rf > 0)
    rho = rf[gd] / rb[gd]
    inl = (rho > 0.6) & (rho < 1.02)
    resid = np.abs(rb[gd][inl] - frx[samp][gd][inl]) / frx[samp][gd][inl]
    sg = samp[gd][inl]
    y = rho[inl]
    feats = lambda i: np.column_stack(
        [np.ones_like(frh[i]), frh[i], frh[i] ** 2, fs[i], fv[i], frh[i] * fs[i]]
    )
    p = rng.permutation(len(sg))
    tr, va = p[: int(0.7 * len(sg))], p[int(0.7 * len(sg)) :]
    X = feats(sg)
    coef, _, _, _ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    e = np.abs(X[va] @ coef - y[va]) / y[va]
    print(
        f"  resolved-buggy vs stored: median {np.median(resid)*100:+.2f}%; rho median {np.median(y):.4f}; fit err median {np.median(e)*100:.2f}%"
    )
    rho_all = np.clip(feats(np.arange(frh.size)) @ coef, 0.6, 1.02).reshape(S, D)
    norm_f = norm_b / rho_all
    print(
        f"  BUGGY->FIXED: (None,None) {exc(norm_b,None):.2f}->{exc(norm_f,None):.2f}%;  (lat<=30) {exc(norm_b,30):.2f}->{exc(norm_f,30):.2f}%"
    )
    print(
        f"  median rho {np.nanmedian(rho_all[np.isfinite(norm_b)]):.4f} -> normalized rise ~{(1/np.nanmedian(rho_all[np.isfinite(norm_b)])-1)*100:.1f}%"
    )


do_measure("r3 (PI potential size)", Vp, g(pip, "rmax"), (40.80, 35.74))
do_measure("r1 (cat-1 potential size)", g(ct1, "vmax"), g(ct1, "rmax"), (9.85, 2.39))
print("\nDONE")
