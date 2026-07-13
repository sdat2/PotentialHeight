import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"
import dask

dask.config.set(scheduler="synchronous")
import numpy as np, xarray as xr
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure, coriolis_parameter_from_lat

# read only 5 unique points (tiny)
ds = (
    xr.open_dataset("/Volumes/s/tcpips/era5_unique_points_ps_0.nc")
    .isel(u=slice(0, 5))
    .load()
)
print(
    "loaded 5 points; vmax units:",
    ds.vmax.attrs.get("units", "?"),
    "| rh units:",
    ds.rh.attrs.get("units", "?"),
)


def solve_one(vmax, msl, sst, t0, lat, rh, ckcd, cd, wcool, buggy):
    orig = psmod.carnot_pm_from_y
    if buggy:

        def conv(y, pda, tns):
            e = buck_sat_vap_pressure(tns)
            return (pda - (1.0 - rh) * e) / y + e

        psmod.carnot_pm_from_y = conv
    try:
        r0, pm, pc, rmax = psmod.calculate_ps_ufunc(
            float(vmax),
            float(msl),
            float(sst),
            float(t0),
            float(lat),
            float(rh),
            float(ckcd),
            float(cd),
            float(wcool),
            1.2,
            "isothermal",
        )
        return r0, rmax
    finally:
        psmod.carnot_pm_from_y = orig


def g(ds, i, name, default=None):
    if name in ds:
        v = float(ds[name].isel(u=i).values.ravel()[0])
        return v
    return default


print(
    f"\n{'pt':>3} {'lat':>6} {'rh':>6} {'sst':>6} {'vmax':>6} | {'stored r0':>10} {'buggy r0':>10} {'fixed r0':>10} {'dr0%':>6} | {'stored rmax':>11} {'fixed rmax':>10} {'drmax%':>7}"
)
for i in range(5):
    lat = g(ds, i, "lat")
    rh = g(ds, i, "rh")
    sst = g(ds, i, "sst")
    msl = g(ds, i, "msl")
    t0 = g(ds, i, "t0")
    vmax = g(ds, i, "vmax")
    ckcd = g(ds, i, "ck_cd", 0.9)
    cd = g(ds, i, "cd", 0.0015)
    wcool = g(ds, i, "w_cool", 0.002)
    stored_r0 = g(ds, i, "r0")
    stored_rmax = g(ds, i, "rmax")
    # rh may be percent; normalize
    rh_frac = rh / 100 if rh and rh > 1.5 else rh
    b = solve_one(vmax, msl, sst, t0, lat, rh_frac, ckcd, cd, wcool, True)
    f = solve_one(vmax, msl, sst, t0, lat, rh_frac, ckcd, cd, wcool, False)
    dr0 = 100 * (f[0] / b[0] - 1) if b[0] and not np.isnan(b[0]) else float("nan")
    drmax = 100 * (f[1] / b[1] - 1) if b[1] and not np.isnan(b[1]) else float("nan")
    print(
        f"{i:>3} {lat:>6.2f} {rh_frac:>6.3f} {sst:>6.2f} {vmax:>6.1f} | {stored_r0/1000 if stored_r0 else float('nan'):>10.1f} {b[0]/1000 if b[0] and not np.isnan(b[0]) else float('nan'):>10.1f} {f[0]/1000 if f[0] and not np.isnan(f[0]) else float('nan'):>10.1f} {dr0:>6.1f} | {stored_rmax/1000 if stored_rmax else float('nan'):>11.2f} {f[1]/1000 if f[1] and not np.isnan(f[1]) else float('nan'):>10.2f} {drmax:>7.1f}"
    )
print("\n(buggy r0 should ~= stored r0 if input extraction is faithful)")
