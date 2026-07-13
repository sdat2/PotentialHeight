"""Post-regeneration analysis. Reads ONLY small point files (no big data).
1. Isolate the humidity fix (buggy vs fixed) on identical inputs, at BOTH the
   paper's grid point (lat 29.25, from the archived file) and the current
   code's point (lat 28.75, from the fresh regen).
2. Compare fresh-regen (current point) vs archived (paper point) for each city.
"""

import numpy as np, xarray as xr
import w22.ps as psmod
from w22.utils import buck_sat_vap_pressure

DATA = "/Users/simon/worstsurge/w22/data"
SSD = "/Volumes/s/tcpips"


def solve(vmax, rh, sst, msl, t0, lat, buggy):
    orig = psmod.carnot_pm_from_y
    if buggy:

        def conv(y, pda, tns):
            e = buck_sat_vap_pressure(tns)
            return (pda - (1.0 - rh) * e) / y + e  # numerator reverts rh->1

        psmod.carnot_pm_from_y = conv
    try:
        ds = xr.Dataset(
            dict(
                sst=sst,
                supergradient_factor=1.2,
                t0=t0,
                w_cool=0.002,
                vmax=vmax,
                msl=msl,
                rh=rh,
                cd=0.0015,
            ),
            coords=dict(lat=lat),
        )
        o = psmod.point_solution_ps(ds, pressure_assumption="isothermal")
        return float(o.r0.values) / 1000, float(o.rmax.values) / 1000
    finally:
        psmod.carnot_pm_from_y = orig


def inp(ds, yr):
    sel = ds.isel(time=[t.year == yr for t in ds.indexes["time"]])
    g = lambda *ks: next(float(sel[k].values.ravel()[0]) for k in ks if k in sel)
    lat = float(ds.lat.values)
    return dict(
        vmax=g("vmax_3", "vmax"),
        rh=g("rh"),
        sst=g("sst"),
        msl=g("msl"),
        t0=g("t0"),
        lat=lat,
    )


print("=" * 78)
print("PART 1 — isolate the humidity fix (identical inputs, buggy vs fixed)")
print("=" * 78)
for label, path in [
    (
        "PAPER point (lat 29.25)",
        f"{SSD}/new_orleans_august_ssp585_r4i1p1f1_isothermal_pi4new.nc",
    ),
    (
        "CURRENT point (lat 28.75)",
        f"{DATA}/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc",
    ),
]:
    ds = xr.open_dataset(path)
    print(f"\n{label}:")
    for yr in (2015, 2100):
        iv = inp(ds, yr)
        b = solve(
            **{k: iv[k] for k in ("vmax", "rh", "sst", "msl", "t0", "lat")}, buggy=True
        )
        f = solve(
            **{k: iv[k] for k in ("vmax", "rh", "sst", "msl", "t0", "lat")}, buggy=False
        )
        print(
            f"  {yr} (rh={iv['rh']:.3f}): r0 {b[0]:.1f}->{f[0]:.1f} km ({100*(f[0]/b[0]-1):+.1f}%), "
            f"rmax {b[1]:.2f}->{f[1]:.2f} km ({100*(f[1]/b[1]-1):+.1f}%)"
        )

print("\n" + "=" * 78)
print("PART 2 — fresh regen (fixed, current pt) vs archived (buggy, paper pt), by city")
print("=" * 78)
cities = [
    ("new_orleans", f"{SSD}/new_orleans_august_ssp585_r4i1p1f1_isothermal_pi4new.nc"),
    ("galverston", f"{SSD}/galverston_august_ssp585_r4i1p1f1_isothermal_pi4new.nc"),
    ("miami", f"{SSD}/miami_august_ssp585_r4i1p1f1_isothermal_pi4new.nc"),
]
import os

for city, oldpath in cities:
    newpath = f"{DATA}/{city}_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc"
    if not os.path.exists(newpath):
        print(f"\n{city}: fresh file not present yet")
        continue
    hasold = os.path.exists(oldpath)
    new = xr.open_dataset(newpath)
    old = xr.open_dataset(oldpath) if hasold else None
    print(
        f"\n{city}: new pt lat={float(new.lat.values)}"
        + (
            f" | archived pt lat={float(old.lat.values)}"
            if hasold
            else " | (no archived file)"
        )
    )
    for yr in (2015, 2100):
        ni = inp(new, yr)
        s = f"  {yr}: NEW vmax={ni['vmax']:.2f} rh={ni['rh']:.3f} sst={ni['sst']:.2f} -> r0={float(new.isel(time=[t.year==yr for t in new.indexes['time']]).r0_3.values.ravel()[0])/1000:.1f} km, rmax={float(new.isel(time=[t.year==yr for t in new.indexes['time']]).rmax_3.values.ravel()[0])/1000:.2f} km"
        if hasold:
            oi = inp(old, yr)
            oldr0 = (
                float(
                    old.isel(
                        time=[t.year == yr for t in old.indexes["time"]]
                    ).r0.values.ravel()[0]
                )
                / 1000
            )
            s += f" || OLD vmax={oi['vmax']:.2f} rh={oi['rh']:.3f} -> r0={oldr0:.1f} km"
        print(s)
print("\nDONE")
