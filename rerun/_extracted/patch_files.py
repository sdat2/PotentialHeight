import os, shutil, numpy as np
from netCDF4 import Dataset
SRC="/Volumes/s/tcpips/data/ibtracs"; DST="/Volumes/s/tcpips/fixed_ibtracs"
os.makedirs(DST, exist_ok=True)
npz={m:np.load(f"/private/tmp/claude-501/-Users-simon-thesis/3ecac1c7-3c1c-48be-9b89-cf847cc43e0d/scratchpad/exact_{m}.npz") for m in ("r1","r2","r3")}
# measure -> (track file, npz key)
patch={"IBTrACS.since1980.v04r01.pi_ps.nc":"r3","IBTrACS.since1980.v04r01.cps.nc":"r2","IBTrACS.since1980.v04r01.ps_cat1.nc":"r1"}
for fn,meas in patch.items():
    src=os.path.join(SRC,fn); dst=os.path.join(DST,fn)
    print(f"copying {fn} ...", flush=True); shutil.copy2(src,dst)
    d=npz[meas]; W=d["W"].astype(bool); rf=d["rmax_fixed"]  # flat S*D
    use=W&np.isfinite(rf)
    nc=Dataset(dst,"a")
    rm=nc.variables["rmax"]
    orig=np.asarray(rm[:],dtype="float64"); shape=orig.shape
    flat=orig.ravel().copy()
    assert flat.size==use.size, f"{fn}: rmax size {flat.size} != mask {use.size}"
    flat[use]=rf[use]
    rm[:]=flat.reshape(shape)
    nc.close()
    print(f"  patched {int(use.sum())} rmax values in {fn}", flush=True)
# symlink the other files get_normalized_data / plotters need
for fn in ("IBTrACS.since1980.v04r01.nc","IBTrACS.since1980.v04r01.normalized.nc",
           "IBTrACS.since1980.v04r01.unique.nc","IBTrACS.since1980.v04r01.cps_inputs.nc"):
    l=os.path.join(DST,fn)
    if os.path.lexists(l): os.remove(l)
    if os.path.exists(os.path.join(SRC,fn)): os.symlink(os.path.join(SRC,fn),l)
print("\nDST contents:")
for f in sorted(os.listdir(DST)): 
    p=os.path.join(DST,f); print(f"  {'->' if os.path.islink(p) else '  '} {f}")
print("DONE")
