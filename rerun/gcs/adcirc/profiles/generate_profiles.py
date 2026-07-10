"""Generate the profile pair(s) for the ADCIRC materiality check.

For each year (2015, 2100) this produces:
  profile_old_{year}.json    — byte-copy of the SHIPPED pre-fix profile the paper's
                               ADCIRC/BO runs actually used (w22/data/{year}_new_orleans_
                               profile_r4i1p1f1.json, March-2025 vintage, r0 solved with
                               the buggy rh=1 back-conversion).
  profile_fixed_{year}.json  — same storm regenerated with the FIXED solver
                               (w22.w22_carnot.carnot_pm_from_y, 2026-07 humidity fix):
                               fixed r0 from the stored CESM2 point environment, then the
                               same cle15.profile_from_stats call the shipped profiles used.

Environment inputs come from the stored point file (CESM2 r4i1p1f1 ssp585, August,
29.25N -90.25E — the paper's grid point):
  /Volumes/s/tcpips/new_orleans_august_ssp585_r4i1p1f1_isothermal_pi4new.nc
whose 2015 values (Vp 105.012 m/s, msl 1019.55 hPa, sst 31.26 C, t0 204.90 K,
rh 0.7466) reproduce the paper's tab:pips row. Vp is UNAFFECTED by the humidity fix,
so old and fixed profiles share the same intensity; only r0 (and hence the wind/pressure
structure) changes. Expected at 2015: r0 1665->~1527 km (-8.7%), rmax 28.6->~24.9 km.

Solver parameters are the production point-timeseries values (ck_cd=0.9, cd=0.0015,
w_cool=0.002, supergradient=1.2, isothermal) — the same family that produced the stored
point file, NOT the CkCd=1 canonical-test values.

Run from the repo root (SSD mounted): python rerun/gcs/adcirc/profiles/generate_profiles.py
Then use via fort22.py's literal-.json branch:
  tc.profile_name.value=/work/profiles/profile_fixed_2015.json
"""

import json
import os
import shutil
import sys
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore")

# make cle15/w22 importable when run as a plain script from anywhere
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import xarray as xr

from cle15.cle15n import profile_from_stats
from w22.ps import calculate_ps_ufunc
from w22.utils import coriolis_parameter_from_lat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
POINT_NC = "/Volumes/s/tcpips/new_orleans_august_ssp585_r4i1p1f1_isothermal_pi4new.nc"
# production solver parameters (match the stored point file's family)
CK_CD, CD, W_COOL, SUPERGRADIENT = 0.9, 0.0015, 0.002, 1.2
ASSUMPTION = "isothermal"


def main() -> None:
    ds = xr.open_dataset(POINT_NC)
    lat = float(ds["lat"].values)
    fcor = abs(float(coriolis_parameter_from_lat(lat)))
    meta = {"point_nc": POINT_NC, "lat": lat,
            "solver": {"ck_cd": CK_CD, "cd": CD, "w_cool": W_COOL,
                       "supergradient_factor": SUPERGRADIENT,
                       "pressure_assumption": ASSUMPTION},
            "years": {}}

    for year in (2015, 2100):
        sel = ds.sel(time=ds.time.dt.year == year)
        vmax = float(sel["vmax"].values.ravel()[0])   # Vp, gradient level [m/s]
        msl = float(sel["msl"].values.ravel()[0])     # [hPa]
        sst = float(sel["sst"].values.ravel()[0])     # [C]
        t0 = float(sel["t0"].values.ravel()[0])       # [K]
        rh = float(sel["rh"].values.ravel()[0])       # [frac]
        r0_stored = float(sel["r0"].values.ravel()[0])
        rmax_stored = float(sel["rmax"].values.ravel()[0])

        # fixed-solver potential size at PI intensity
        r0_f, pm_f, pc_f, rmax_f = calculate_ps_ufunc(
            vmax, msl, sst, t0, lat, rh, CK_CD, CD, W_COOL, SUPERGRADIENT, ASSUMPTION
        )
        print(f"[{year}] Vp={vmax:.3f} m/s  msl={msl:.2f} hPa  sst={sst:.2f} C  rh={rh:.4f}")
        print(f"        r0:   stored {r0_stored/1000:8.1f} km -> fixed {r0_f/1000:8.1f} km "
              f"({100*(r0_f-r0_stored)/r0_stored:+.1f}%)")
        print(f"        rmax: stored {rmax_stored/1000:8.2f} km -> fixed {rmax_f/1000:8.2f} km "
              f"({100*(rmax_f-rmax_stored)/rmax_stored:+.1f}%)")

        # old = the shipped artifact, byte-copied
        shipped = os.path.join(REPO, "w22", "data",
                               f"{year}_new_orleans_profile_r4i1p1f1.json")
        old_out = os.path.join(HERE, f"profile_old_{year}.json")
        shutil.copyfile(shipped, old_out)

        # fixed = same profile call the shipped ones used, with the fixed r0
        prof = profile_from_stats(vmax, fcor, float(r0_f), msl,
                                  pressure_assumption=ASSUMPTION)
        fixed_out = os.path.join(HERE, f"profile_fixed_{year}.json")
        with open(fixed_out, "w") as f:
            json.dump({k: (list(np.asarray(v).astype(float)) if np.ndim(v) else float(v))
                       for k, v in prof.items()}, f)

        old = json.load(open(old_out))
        print(f"        profile rmax: old {old['rmax']/1000:6.2f} km -> fixed "
              f"{prof['rmax']/1000:6.2f} km;  r0: old {old['rr'][-1]/1000:7.1f} -> "
              f"fixed {prof['rr'][-1]/1000:7.1f} km")
        meta["years"][year] = {
            "vmax_ms": vmax, "msl_hPa": msl, "sst_C": sst, "t0_K": t0, "rh": rh,
            "r0_stored_m": r0_stored, "r0_fixed_m": float(r0_f),
            "rmax_stored_m": rmax_stored, "rmax_fixed_m": float(rmax_f),
            "pm_fixed_Pa": float(pm_f), "pc_fixed_Pa": float(pc_f),
            "profile_old": os.path.basename(old_out),
            "profile_fixed": os.path.basename(fixed_out),
        }

    with open(os.path.join(HERE, "PROVENANCE.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nwritten:", ", ".join(sorted(os.listdir(HERE))))


if __name__ == "__main__":
    main()
