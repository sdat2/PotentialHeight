import os, warnings, time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"
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

dask.config.set(scheduler="synchronous")  # safe serial for the lazy data load
import numpy as np, xarray as xr
from joblib import Parallel, delayed
import tcpips.era5 as e5
from tcpips.era5 import mon_increase, select_seasonal_hemispheric_data

OUT = "/Volumes/s/tcpips/data/era5/ps_fixed"
os.makedirs(OUT, exist_ok=True)
OUTVARS = ["r0_1", "pm_1", "pc_1", "rmax_1", "r0_3", "pm_3", "pc_3", "rmax_3"]
print(
    "=== ERA5 first decade (1980-1989) PS re-solve, FIXED solver (joblib) ===",
    flush=True,
)
ds = e5.get_all_data(start_year=1980, end_year=1989)
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
lat1d = np.asarray(ds["lat"].values)
lon1d = np.asarray(ds["lon"].values)
years = [int(y) for y in np.asarray(ds["year"].values)]
print("years:", years, "grid", len(lat1d), "x", len(lon1d), flush=True)


def one(v1, v3, msl, sst, t0, lat, rh):
    import w22.ps as p

    return p.calculate_ps13_ufunc(
        float(v1),
        float(v3),
        float(msl),
        float(sst),
        float(t0),
        float(lat),
        float(rh),
        0.9,
        0.0015,
        0.002,
        1.2,
        "isothermal",
    )


for i, yr in enumerate(years):
    out = os.path.join(OUT, f"era5_ps_fixed_{yr}.nc")
    if os.path.exists(out):
        print(f"[{yr}] done, skip", flush=True)
        continue
    t = time.time()
    g = lambda n: np.asarray(ds[n].isel(year=i).values, dtype="float64")
    v3 = g("vmax_3")
    v1 = g("vmax_1")
    sst = g("sst")
    msl = g("msl")
    rh = g("rh")
    t0 = g("t0")
    LAT = np.broadcast_to(lat1d[:, None], v3.shape)
    shp = v3.shape
    N = v3.size
    fv3, fv1, fsst, fmsl, frh, ft0, fLAT = [
        a.ravel() for a in (v3, v1, sst, msl, rh, t0, LAT)
    ]
    valid = np.where(
        np.isfinite(fv3)
        & (fv3 > 0.01)
        & np.isfinite(fsst)
        & np.isfinite(frh)
        & (frh >= 0)
        & (frh <= 1)
        & np.isfinite(fmsl)
        & np.isfinite(ft0)
    )[0]
    print(f"[{yr}] solving {len(valid)} ocean cells ...", flush=True)
    res = Parallel(n_jobs=6, backend="loky")(
        delayed(one)(fv1[j], fv3[j], fmsl[j], fsst[j], ft0[j], fLAT[j], frh[j])
        for j in valid
    )
    res = np.array(res)  # (nvalid, 8)
    outarrs = {k: np.full(N, np.nan) for k in OUTVARS}
    for c, k in enumerate(OUTVARS):
        outarrs[k][valid] = res[:, c]
    dso = xr.Dataset(
        {k: (("lat", "lon"), outarrs[k].reshape(shp)) for k in OUTVARS},
        coords={"lat": lat1d, "lon": lon1d, "year": yr},
    )
    for iv in ("vmax_3", "vmax_1", "sst", "msl", "rh", "t0"):
        dso[iv] = (
            ("lat", "lon"),
            (
                g(iv).reshape(shp)
                if False
                else locals()[
                    {
                        "vmax_3": "v3",
                        "vmax_1": "v1",
                        "sst": "sst",
                        "msl": "msl",
                        "rh": "rh",
                        "t0": "t0",
                    }[iv]
                ]
            ),
        )
    tmp = out + ".tmp"
    dso.to_netcdf(tmp)
    os.replace(tmp, out)
    print(
        f"[{yr}] DONE {len(valid)} cells in {(time.time()-t)/3600:.2f} h -> {out}",
        flush=True,
    )
print(
    "\nALL DONE. Combine: xr.open_mfdataset('ps_fixed/era5_ps_fixed_*.nc', combine='nested', concat_dim='year')",
    flush=True,
)
print("DONE")
