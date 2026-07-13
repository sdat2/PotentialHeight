import os, warnings, time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["LOKY_MAX_CPU_COUNT"] = "3"
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
import tcpips.era5 as e5
from tcpips.era5 import mon_increase, select_seasonal_hemispheric_data
from w22.ps import parallelized_ps13_dask

t0 = time.time()
ds = e5.get_all_data(start_year=1980, end_year=1989)
print(
    f"get_all_data(1980,1989): {'None' if ds is None else 'loaded (lazy)'} in {time.time()-t0:.0f}s"
)
if ds is None:
    raise SystemExit("no data")
if "valid_time" in ds.dims:
    ds = ds.rename({"valid_time": "time"})
ds = (
    mon_increase(select_seasonal_hemispheric_data(ds, months_to_average=1))
    .sel(latitude=slice(-40, 40))
    .rename({"latitude": "lat", "longitude": "lon"})
)
for k in ("t", "q", "pressure_level"):
    if k in ds:
        del ds[k]
ds = ds.rename({"vmax": "vmax_3"})
ds["vmax_1"] = (ds.vmax_3.dims, np.ones(ds.vmax_3.shape) * 33 / 0.8)
print("prepped dims:", dict(ds.sizes), "| dims of vmax_3:", ds.vmax_3.dims)
print("has 'year' dim:", "year" in ds.dims, "| coords:", list(ds.coords)[:8])
print(
    "PS-input vars present:",
    [v for v in ("vmax_3", "vmax_1", "sst", "msl", "rh", "t0") if v in ds],
)
# count valid ocean cells across the decade (small: just vmax_3, lazy->compute one var)
vmax = np.asarray(ds["vmax_3"].values)
nvalid = int(np.sum(np.isfinite(vmax) & (vmax > 0.01)))
print(f"total cells={vmax.size}, valid(ocean)={nvalid} ({100*nvalid/vmax.size:.0f}%)")
# time a small block via the real path
ydim = "year" if "year" in ds.dims else ("time" if "time" in ds.dims else None)
sub = ds.isel({ydim: 0}) if ydim else ds
sub = sub.isel(lat=slice(80, 110), lon=slice(600, 650))
tt = time.time()
out = parallelized_ps13_dask(sub)
out = out.compute() if hasattr(out, "compute") else out
dt = time.time() - tt
sv = np.asarray(sub["vmax_3"].values)
nsub = int(np.sum(np.isfinite(sv) & (sv > 0.01)))
rate = dt / max(nsub, 1)
print(
    f"subset {dict(sub.sizes)}: {nsub} valid cells in {dt:.1f}s -> {rate*1000:.0f} ms/cell"
)
print(
    f"\nESTIMATE decade: ~{nvalid*rate/3600:.0f} h serial, ~{nvalid*rate/3600/6:.0f} h at n_jobs=6"
)
print("DONE")
