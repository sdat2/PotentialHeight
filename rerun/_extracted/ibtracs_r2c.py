import os, warnings
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
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
