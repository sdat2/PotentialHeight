"""Regenerate the 16 per-storm IBTrACS track figures with the FIXED PS solver.

Purpose
-------
This is the final plotting driver for the per-storm potential-size track
figures (``tcpips.ibtracs.plot_tc_example`` -> one PDF per storm). Before
plotting it rebuilds the three IBTrACS track ``.nc`` files with the corrected
potential radius-of-max-wind (``rmax``) coming from the 2026-07 humidity/units
bug fix (``w22.w22_carnot.carnot_pm_from_y``, via
``w22.ps.calculate_ps_ufunc``), in TWO stages:

  1. Cat-1 exact patch. Fresh-copies the source ``.nc`` files, then overwrites
     ``rmax`` at every Cat-1 track point with the re-solved value read from the
     ``exact_{r1,r2,r3}.npz`` intermediates (produced by the companion
     ``rerun/ibtracs/resolve_exceedance.py``). This is exactly what
     ``rerun/ibtracs/patch_track_files.py`` does.
  2. Full-track re-solve (the extra step, needed for smooth plots). For the 16
     named storms below, re-solves ``rmax`` along their WHOLE track (not just
     the Cat-1 points) via ``w22.ps.calculate_ps_ufunc`` and patches those in
     on top of stage 1, so the plotted curves have no Cat-1-filter gaps. This
     per-measure track re-solve is cached to ``track_{r1,r2,r3}.npz``.

Then ``plot_tc_example`` is called for each of the 16 storms.

Measure -> track file -> npz key:
  r3 = ...pi_ps.nc     (V_p / potential-intensity driven)
  r2 = ...cps.nc       (climate potential size)
  r1 = ...ps_cat1.nc   (Cat-1 filtered)

Durable inputs
--------------
- Source track files (copied fresh, then patched):
  /Volumes/s/tcpips/data/ibtracs/IBTrACS.since1980.v04r01.{cps,pi_ps,ps_cat1}.nc
- Cat-1 re-solved rmax intermediates (stage 1), from resolve_exceedance.py:
  <EXACT_NPZ_DIR>/exact_{r1,r2,r3}.npz   (holding W, rmax_fixed, ...)
  default EXACT_NPZ_DIR = /Volumes/s/tcpips/fixed_ibtracs/exact_resolve
- The base + sibling files opened for names/env and by the plotters, symlinked
  into DST (must already be present at DST — see Prerequisites):
  IBTrACS.since1980.v04r01.{,.normalized,.unique,.cps_inputs}.nc

Intermediates (durable, this script's own cache)
------------------------------------------------
  /Volumes/s/tcpips/fixed_ibtracs/track_resolve/track_{r1,r2,r3}.npz
  (idx = flat storm*date_time indices of the 16 storms' valid track points;
   res = re-solved rmax). If present they are reused, so a re-plot is cheap.

Durable outputs
---------------
- Re-patched (stage 1 + stage 2) track files:
  /Volumes/s/tcpips/fixed_ibtracs/IBTrACS.since1980.v04r01.{cps,pi_ps,ps_cat1}.nc
- 16 track PDFs written by plot_tc_example to the repo figure dir:
  /Users/simon/worstsurge/img/ibtracs/tracks/<name>_track.pdf
  (the reviewed/durable copies of these live under
   /Volumes/s/tcpips/fixed_figures/).

Expected numbers
----------------
Solver sanity (fixed carnot_pm_from_y): canonical point r0 = 2128.81 km,
rmax = 60.67 km at CkCd = 1, rh = 0.9. The stage-1 Cat-1 patch this pipeline
shares with resolve_exceedance.py reproduces the fixed exceedance table
(r3 46.1/42.4 ; r2 19.9/10.2(sst)/7.94(AF) ; r1 13.5/4.2 ; supersize Ellen
3.86, Man-Yi 3.11). Primary deliverable here: 16 track PDFs (all 16 storms
should plot OK).

Repo symlink note
-----------------
plot_tc_example reads its tracks from the repo ``data/ibtracs`` symlink, so it
must point at the patched output dir before plotting:
  ln -sfn /Volumes/s/tcpips/fixed_ibtracs /Users/simon/worstsurge/data/ibtracs

Prerequisites
-------------
1. resolve_exceedance.py has been run so exact_{r1,r2,r3}.npz exist under
   EXACT_NPZ_DIR.
2. DST already contains the base/sibling IBTrACS...nc symlinks (this driver
   opens {DST}/IBTrACS.since1980.v04r01.nc at load time). Run
   patch_track_files.py first, or create the symlinks manually. The DST symlink
   step at the end of this script only fills in any that are still missing.

How to run (caffeinate so the mac won't sleep, mem_guard watchdog; the stage-2
track re-solve is a joblib job, minutes, RSS ~1-2 GB at n_jobs=3 -- once the
track_*.npz caches exist a re-run is just the plotting):
  export HDF5_USE_FILE_LOCKING=FALSE
  caffeinate -i -s python rerun/mem_guard.py \
      rerun/ibtracs/figures_tracks.py 4096 7200
"""

import os, warnings, time, shutil

# --- env-safety guards (exFAT/macOS): set BEFORE importing netCDF4/numpy ---
# HDF5_USE_FILE_LOCKING=FALSE is mandatory on the exFAT SSD or the netCDF4
# append below fails with an HDF5 locking error. BLAS single-threaded so the
# per-point scalar solves don't spawn thread pools; LOKY capped so joblib can't
# oversubscribe cores (n_jobs=3 is also hard-set on the Parallel calls below).
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
os.environ["LOKY_MAX_CPU_COUNT"]="3"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy as np, xarray as xr
from netCDF4 import Dataset
from joblib import Parallel, delayed
SRC="/Volumes/s/tcpips/data/ibtracs"; DST="/Volumes/s/tcpips/fixed_ibtracs"
# Durable .npz locations (were the volatile scratchpad in the original run):
#   EXACT_DIR  - stage-1 Cat-1 re-solve, produced by resolve_exceedance.py
#   TRACK_DIR  - stage-2 full-track re-solve, this script's own cache
EXACT_DIR=os.environ.get("EXACT_NPZ_DIR","/Volumes/s/tcpips/fixed_ibtracs/exact_resolve")
TRACK_DIR="/Volumes/s/tcpips/fixed_ibtracs/track_resolve"
os.makedirs(DST, exist_ok=True); os.makedirs(TRACK_DIR, exist_ok=True)
fname={"r2":"IBTrACS.since1980.v04r01.cps.nc","r3":"IBTrACS.since1980.v04r01.pi_ps.nc","r1":"IBTrACS.since1980.v04r01.ps_cat1.nc"}
# 1. FRESH copies (overwrite read-tainted ones)
for m,fn in fname.items(): print("fresh-copy",fn,flush=True); shutil.copy2(f"{SRC}/{fn}",f"{DST}/{fn}")
# 2. storm matching + track indices
STORMNAMES={"KATRINA","IDA","HELENE","IAN","HARVEY","FIONA","SAOLA","MANGKHUT","HATO","VICENTE","YORK","ELLEN","BEBINCA","JEBI","MERANTI","FREDDY"}
base=xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.nc")
names=np.array([x.decode().strip() if isinstance(x,bytes) else str(x).strip() for x in base["name"].values])
S=base.sizes["storm"];D=base.sizes["date_time"]
want=np.where(np.isin(names,list(STORMNAMES)))[0]
print(f"matched {len(want)} storms",flush=True)
trackmask=np.zeros(S*D,bool)
for s in want: trackmask[s*D:(s+1)*D]=True
cps=xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.cps.nc")
r=lambda ds,n:np.asarray(ds[n].values,dtype="float64").ravel()
sst=r(cps,"sst");rh=r(cps,"rh");msl=r(cps,"msl");t0=r(cps,"t0")
lat=np.asarray(cps["lat"].values,dtype="float64").ravel();ck=r(cps,"ck_cd");cd=r(cps,"cd")
if np.nanmax(rh)>1.5: rh=rh/100
vmap={"r2":r(cps,"vmax"),"r3":r(xr.open_dataset(f'{DST}/{fname["r3"]}')[["vmax"]],"vmax"),"r1":r(xr.open_dataset(f'{DST}/{fname["r1"]}')[["vmax"]],"vmax")}
def one(vmax,i):
    import w22.ps as p
    return p.calculate_ps_ufunc(vmax[i],msl[i],sst[i],t0[i],lat[i],rh[i],ck[i] if np.isfinite(ck[i]) else 0.9,cd[i] if np.isfinite(cd[i]) else 0.0015,0.002,1.2,"isothermal")[3]
# 3. build fixed rmax per measure = orig, then cat-1 patch (exact npz), then track patch
for meas,fn in fname.items():
    exact=np.load(os.path.join(EXACT_DIR,f"exact_{meas}.npz")); W=exact["W"].astype(bool); rf_cat1=exact["rmax_fixed"]
    tf=os.path.join(TRACK_DIR,f"track_{meas}.npz")
    if os.path.exists(tf):
        d=np.load(tf); tidx=d["idx"]; tres=d["res"]
    else:
        vmax=vmap[meas]; tidx=np.where(trackmask&np.isfinite(vmax)&(vmax>0.01)&np.isfinite(sst)&np.isfinite(rh)&(rh>=0)&(rh<=1))[0]
        t=time.time(); tres=np.array(Parallel(n_jobs=3,backend="loky")(delayed(one)(vmax,i) for i in tidx))
        np.savez(tf,idx=tidx,res=tres); print(f"[{meas}] re-solved {len(tidx)} track pts in {(time.time()-t)/60:.1f} min",flush=True)
    nc=Dataset(f"{DST}/{fn}","a");rm=nc.variables["rmax"];orig=np.asarray(rm[:],dtype="float64");sh=orig.shape;flat=orig.ravel()
    ok=W&np.isfinite(rf_cat1); flat[ok]=rf_cat1[ok]
    tok=np.isfinite(tres); flat[tidx[tok]]=tres[tok]
    rm[:]=flat.reshape(sh);nc.close(); print(f"[{meas}] patched cat1({int(ok.sum())})+track({int(tok.sum())})",flush=True)
# 4. symlink others
for fn in ("IBTrACS.since1980.v04r01.nc","IBTrACS.since1980.v04r01.normalized.nc","IBTrACS.since1980.v04r01.unique.nc","IBTrACS.since1980.v04r01.cps_inputs.nc"):
    l=f"{DST}/{fn}"
    if os.path.lexists(l) and not os.path.islink(l): pass
    elif not os.path.exists(l) and os.path.exists(f"{SRC}/{fn}"):
        if os.path.lexists(l): os.remove(l)
        os.symlink(f"{SRC}/{fn}",l)
# 5. plot
import tcpips.ibtracs as ib
STORMS=[("KATRINA","NA"),("IDA","NA"),("HELENE","NA"),("IAN","NA"),("HARVEY","NA"),("FIONA","NA"),("SAOLA","WP"),("MANGKHUT","WP"),("HATO","WP"),("VICENTE","WP"),("YORK","WP"),("ELLEN","WP"),("BEBINCA","WP"),("JEBI","WP"),("MERANTI","WP"),("FREDDY","SI")]
ok=0
for nm,bs in STORMS:
    try: ib.plot_tc_example(name=nm.encode(),basin=bs.encode(),subbasin=b"WPAC",bbox=None); ok+=1; print(f"  {nm} OK")
    except Exception as e: print(f"  {nm} FAIL: {type(e).__name__}: {str(e)[:100]}")
import glob; print(f"\nplotted {ok}/16; tracks dir: {len(glob.glob(os.path.join(ib.FIGURE_PATH,'tracks','*.pdf')))} pdfs")
print("DONE")
