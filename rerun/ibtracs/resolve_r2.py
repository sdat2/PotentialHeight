"""Fit-based cross-check of the r2 (corresponding potential size, cps) exceedance.

AUTHORITATIVE SOURCE NOTE: the r2 numbers (19.9 / 10.2 / 7.94) are produced
*exactly and fit-free* by ``resolve_exceedance.py`` (which re-solves every point).
This script is the earlier fit-based route — it re-solves a 3500-point sample,
fits the per-point shrink ratio rho=rmax_fixed/rmax_buggy, and applies it — kept
as an independent cross-check that validated the exact solve (held-out error <1%).
Cite ``resolve_exceedance.py`` for the numbers; use this to corroborate the method.

Final variant of the r2 re-solve (supersedes the earlier ``ibtracs_r2.py`` /
``ibtracs_r2b.py`` attempts). Recovered verbatim from the session transcript; the
numerical logic is preserved byte-for-byte so the published/thesis numbers
reproduce exactly. It uses the FIXED potential-size solver
(2026-07 humidity/units fix: ``w22.w22_carnot.carnot_pm_from_y``, reached here
through ``w22.ps.calculate_ps_ufunc``).

What it computes
----------------
For the r2 measure (normalized_size_cps = usa_rmw[nmi]*1852 / rmax[m]) it reports
the exact paper metric: the per-storm-max survival-at-1 (survival function of the
per-storm lifetime-max normalized size evaluated at 1.0, via linear
extrapolation of the empirical SF) under the Cat-1 wind filter
(``usa_wind*0.514444 > 33`` AND ``vmax*0.8 > 33``) and the SF/AF sub-filters
(sst>=26.5, |lat|<=30). It then re-solves rmax for a 3500-point Cat-1 sample
under both the buggy and fixed solvers, fits the per-point shrink ratio
rho = rmax_fixed / rmax_buggy to (rh, sst, vmax) features, applies it to every
point, and re-reports the exceedance BUGGY -> FIXED plus the AF supersize max
ratio. The buggy solver is reproduced by monkeypatching
``psmod.carnot_pm_from_y`` to the pre-2026-07-07 conversion (an extra
(1-rh)*e_sat term, i.e. the old effectively-rh=1 behaviour).

Durable inputs (verified to exist)
----------------------------------
- /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.cps.nc
- /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.normalized.nc

Outputs
-------
None written to disk; all results are printed to stdout (the exceedance numbers
and the supersize ratios ARE the deliverable).

Expected numbers
----------------
Fixed r2 exceedance table this pipeline reproduces (per-storm-max survival-at-1):
~19.9% (no sst/lat filter), ~10.2% (sst>=26.5), ~7.94% (AF: sst>=26.5 & |lat|<=30).
The printed "(paper ...)" baselines (17.06 / 7.94 / 5.89) are the originally
quoted figures; the AF supersize max ratio is reported against "paper Ele 3.59".
Canonical fixed-solver point for cross-checking: r0=2128.81 km, rmax=60.67 km at
CkCd=1, rh=0.9.

How to run
----------
    caffeinate -is python -m mem_guard rerun/ibtracs/resolve_r2.py
(or plain ``python rerun/ibtracs/resolve_r2.py``). Runs serially (no joblib);
re-solving 3500 points takes a few minutes. The exFAT/macOS env guards below
(HDF5 file locking off, single-thread BLAS, LOKY cap) must stay set.
"""

import os, warnings
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"  # exFAT/macOS: netCDF4/h5py file locking must be off
os.environ.setdefault("LOKY_MAX_CPU_COUNT","4")  # cap joblib workers on macOS (memory guard)
warnings.filterwarnings("ignore")
import dask; dask.config.set(scheduler="synchronous")
import numpy as np, xarray as xr
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure
rng=np.random.default_rng(0); IB="/Volumes/s/tcpips/data/ibtracs"
cps=xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.cps.nc").load()
nrm=xr.open_dataset(f"{IB}/IBTrACS.since1980.v04r01.normalized.nc")[["usa_rmw","usa_wind"]].load()
S,D=cps.sizes["storm"],cps.sizes["date_time"]
g=lambda ds,n: np.asarray(ds[n].values,dtype="float64").reshape(S,D)
vmax=g(cps,"vmax");sst=g(cps,"sst");rmax_b=g(cps,"rmax");rh=g(cps,"rh");msl=g(cps,"msl");t0=g(cps,"t0")
lat=np.asarray(cps["lat"].values,dtype="float64").reshape(S,D)
ckcd=g(cps,"ck_cd");cd=g(cps,"cd");usa_rmw=g(nrm,"usa_rmw");usa_wind=g(nrm,"usa_wind")
if np.nanmax(rh)>1.5: rh=rh/100.0
norm_b=usa_rmw*1852.0/rmax_b
# Cat-1 wind filter (as get_normalized_data): obs 10m wind >33 AND vmax*0.8>33
W=(usa_wind*0.514444>33.0)&(vmax*0.8>33.0)
def surv_at_1(ps_max):
    x=ps_max[~np.isnan(ps_max)]; xs=np.sort(x); sf=1-np.arange(1,len(xs)+1)/len(xs)
    from scipy.interpolate import interp1d
    return float(interp1d(xs,sf,bounds_error=False,fill_value="extrapolate")(1.0))*100
def exc(norm,min_sst,max_lat):
    f=W.copy()
    if min_sst is not None: f&=(sst>=min_sst)
    if max_lat is not None: f&=(np.abs(lat)<=max_lat)
    x=np.where(f,norm,np.nan); psm=np.nanmax(np.where(np.isnan(x),-np.inf,x),axis=1)
    psm=np.where(np.isfinite(psm),psm,np.nan)
    return surv_at_1(psm)
print("[GATE] r2 (normalized_size_cps), per-storm-max survival-at-1, Cat-1 wind filter:")
for ms,ml,pap in [(None,None,17.06),(26.5,None,7.94),(26.5,30,5.89)]:
    print(f"   sst>={ms}, |lat|<={ml}: {exc(norm_b,ms,ml):6.2f}%  (paper {pap})")
# ratio re-solve (3500) with outlier drop + fit
flat=lambda a:a.ravel()
fv,fm,fs,ft,fr,frx,fl,fck,fcd=[flat(a) for a in (vmax,msl,sst,t0,rh,rmax_b,lat,ckcd,cd)]
vi=np.where(np.isfinite(fv)&(fv>0.01)&np.isfinite(fs)&np.isfinite(fr)&(fr>=0)&(fr<=1)&np.isfinite(frx)&(frx>0)&flat(W))[0]
def solve(i,buggy):
    o=psmod.carnot_pm_from_y
    if buggy:
        r=fr[i]; psmod.carnot_pm_from_y=lambda y,pda,tns,_r=r:(pda-(1.0-_r)*buck_sat_vap_pressure(tns))/y+buck_sat_vap_pressure(tns)
    try:
        _,_,_,rx=psmod.calculate_ps_ufunc(fv[i],fm[i],fs[i],ft[i],fl[i],fr[i],fck[i] if np.isfinite(fck[i]) else 0.9,fcd[i] if np.isfinite(fcd[i]) else 0.0015,0.002,1.2,"isothermal")
        return rx
    finally: psmod.carnot_pm_from_y=o
samp=rng.choice(vi,size=min(3500,len(vi)),replace=False)
print(f"\nre-solving {len(samp)} Cat-1 points...")
rb=np.array([solve(i,True) for i in samp]);rf=np.array([solve(i,False) for i in samp])
gd=np.isfinite(rb)&np.isfinite(rf)&(rb>0)&(rf>0);rho=rf[gd]/rb[gd];inl=(rho>0.6)&(rho<1.02)
sg=samp[gd][inl];y=rho[inl]
feats=lambda i:np.column_stack([np.ones_like(fr[i]),fr[i],fr[i]**2,fs[i],fv[i],fr[i]*fs[i]])
p=rng.permutation(len(sg));tr,va=p[:int(.7*len(sg))],p[int(.7*len(sg)):]
X=feats(sg);coef,_,_,_=np.linalg.lstsq(X[tr],y[tr],rcond=None)
e=np.abs(X[va]@coef-y[va])/y[va]
print(f"rho median {np.median(y):.4f}; fit held-out err median {np.median(e)*100:.3f}% 95pct {np.percentile(e,95)*100:.3f}%")
rho_all=np.clip(feats(np.arange(fr.size))@coef,0.6,1.02).reshape(S,D)
norm_f=norm_b/rho_all
print("\n===== r2 exceedance BUGGY -> FIXED (exact paper metric) =====")
for ms,ml,pap in [(None,None,17.06),(26.5,None,7.94),(26.5,30,5.89)]:
    print(f"   sst>={ms}, |lat|<={ml}: {exc(norm_b,ms,ml):6.2f}% -> {exc(norm_f,ms,ml):6.2f}%  (paper {pap})")
print(f"\n median rho {np.nanmedian(rho_all):.4f} -> normalized rise ~{(1/np.nanmedian(rho_all[np.isfinite(norm_b)])-1)*100:.1f}%")
xb=np.where((W)&(sst>=26.5)&(np.abs(lat)<=30),norm_b,np.nan)
xf=np.where((W)&(sst>=26.5)&(np.abs(lat)<=30),norm_f,np.nan)
print(f" max supersize ratio (AF): {np.nanmax(xb):.2f} -> {np.nanmax(xf):.2f}   (paper Ele 3.59)")
print("DONE")
