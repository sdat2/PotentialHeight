"""Local, laptop-safe ERA5 gridded potential-size re-solve (fixed solver).

Recomputes Wang-2022 potential size on the ERA5 grid with the corrected
humidity back-conversion (w22.w22_carnot.carnot_pm_from_y — see unit_bug_fix.md),
replacing the buggy ARCHER2 product. It is the local/AWS stand-in for the
ARCHER2 dask-mpi path in tcpips/era5.py:calculate_potential_sizes, rewritten to
be safe on a single machine:

  * synchronous dask for the lazy data load (the `processes` scheduler
    BrokenProcessPool's on macOS spawn + netCDF-backed dask — don't use it);
  * joblib(loky) over ocean cells for the actual solve (the pattern that ran
    the 168k-point IBTrACS re-solve cleanly);
  * per-year checkpoints, so it is safe to kill and resume — finished years are
    skipped. Run it under rerun/mem_guard.py so RAM is capped externally.

The eight outputs per cell are r0/pm/pc/rmax for the category-1 (``_1``) and
potential-intensity (``_3``) hurricanes, plus the inputs for provenance.

Usage (via the guard, caffeinated so the machine won't sleep):
    caffeinate -i -s python rerun/mem_guard.py \\
        rerun/era5_ps_local.py 7168 259200 -- --start-year 1980 --end-year 1989

Resume is the identical command (finished-year .nc files are skipped).
"""

import argparse
import os
import time
import warnings

# exFAT SSD + macOS: HDF5 file locking must be off, and BLAS single-threaded so
# the joblib workers don't oversubscribe cores.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

warnings.filterwarnings("ignore")

import dask

dask.config.set(scheduler="synchronous")  # safe serial lazy load

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

import tcpips.era5 as e5
from tcpips.era5 import mon_increase, select_seasonal_hemispheric_data

# Fixed-solver output variables, in the order calculate_ps13_ufunc returns them.
OUTVARS = ["r0_1", "pm_1", "pc_1", "rmax_1", "r0_3", "pm_3", "pc_3", "rmax_3"]
INVARS = ["vmax_3", "vmax_1", "sst", "msl", "rh", "t0"]
DEFAULT_OUT = "/Volumes/s/tcpips/data/era5/ps_fixed"


def _solve_one(v1, v3, msl, sst, t0, lat, rh):
    """One cell -> 8-tuple. Imported inside so it pickles cleanly to workers."""
    import w22.ps as p

    return p.calculate_ps13_ufunc(
        float(v1), float(v3), float(msl), float(sst), float(t0),
        float(lat), float(rh), 0.9, 0.0015, 0.002, 1.2, "isothermal",
    )


def load_decade(start_year: int, end_year: int) -> xr.Dataset:
    """Lazily load + season-select ERA5, drop the heavy pressure-level fields."""
    ds = e5.get_all_data(start_year=start_year, end_year=end_year)
    if ds is None:
        raise SystemExit("no ERA5 data found for that range")
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
    # category-1 threshold wind (33 m/s / 0.8 supergradient) as the "_1" intensity
    ds["vmax_1"] = (ds.vmax_3.dims, np.full(ds.vmax_3.shape, 33.0 / 0.8))
    return ds


def solve_year(ds: xr.Dataset, i: int, year: int, out_dir: str, n_jobs: int) -> None:
    out = os.path.join(out_dir, f"era5_ps_fixed_{year}.nc")
    if os.path.exists(out):
        print(f"[{year}] checkpoint exists, skip", flush=True)
        return
    t = time.time()
    fields = {n: np.asarray(ds[n].isel(year=i).values, dtype="float64") for n in INVARS}
    shp = fields["vmax_3"].shape
    lat2d = np.broadcast_to(np.asarray(ds["lat"].values)[:, None], shp)
    flat = {n: a.ravel() for n, a in fields.items()}
    flat_lat = lat2d.ravel()

    valid = np.where(
        np.isfinite(flat["vmax_3"]) & (flat["vmax_3"] > 0.01)
        & np.isfinite(flat["sst"])
        & np.isfinite(flat["rh"]) & (flat["rh"] >= 0) & (flat["rh"] <= 1)
        & np.isfinite(flat["msl"]) & np.isfinite(flat["t0"])
    )[0]
    print(f"[{year}] solving {len(valid)} ocean cells (n_jobs={n_jobs}) ...", flush=True)

    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one)(
            flat["vmax_1"][j], flat["vmax_3"][j], flat["msl"][j], flat["sst"][j],
            flat["t0"][j], flat_lat[j], flat["rh"][j],
        )
        for j in valid
    )
    res = np.asarray(res)  # (n_valid, 8)

    data_vars = {}
    for c, k in enumerate(OUTVARS):
        arr = np.full(flat["vmax_3"].size, np.nan)
        arr[valid] = res[:, c]
        data_vars[k] = (("lat", "lon"), arr.reshape(shp))
    for n in INVARS:  # keep inputs alongside outputs for provenance
        data_vars[n] = (("lat", "lon"), fields[n])

    dso = xr.Dataset(
        data_vars,
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values, "year": year},
    )
    tmp = out + ".tmp"
    dso.to_netcdf(tmp)
    os.replace(tmp, out)  # atomic: a killed write never leaves a half file
    print(f"[{year}] DONE {len(valid)} cells in {(time.time() - t) / 3600:.2f} h -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=1989)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"=== ERA5 {args.start_year}-{args.end_year} PS re-solve (FIXED solver, joblib) ===",
        flush=True,
    )
    ds = load_decade(args.start_year, args.end_year)
    years = [int(y) for y in np.asarray(ds["year"].values)]
    print(f"years: {years}  grid {ds.sizes['lat']}x{ds.sizes['lon']}", flush=True)
    for i, year in enumerate(years):
        solve_year(ds, i, year, args.out_dir, args.n_jobs)
    print(
        "\nALL DONE. Combine with:\n"
        f"  xr.open_mfdataset('{args.out_dir}/era5_ps_fixed_*.nc', "
        "combine='nested', concat_dim='year')",
        flush=True,
    )


if __name__ == "__main__":
    main()
