import os, warnings, time

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
import matplotlib

matplotlib.use("Agg")
import numpy as np, xarray as xr
from netCDF4 import Dataset
from joblib import Parallel, delayed
import w22.ps as psmod

DST = "/Volumes/s/tcpips/fixed_ibtracs"
STORMS = [
    ("KATRINA", "NA"),
    ("IDA", "NA"),
    ("HELENE", "NA"),
    ("IAN", "NA"),
    ("HARVEY", "NA"),
    ("FIONA", "NA"),
    ("SAOLA", "WP"),
    ("MANGKHUT", "WP"),
    ("HATO", "WP"),
    ("VICENTE", "WP"),
    ("YORK", "WP"),
    ("ELLEN", "WP"),
    ("BEBINCA", "WP"),
    ("JEBI", "WP"),
    ("MERANTI", "WP"),
    ("FREDDY", "SI"),
]
# --- identify storm indices by name+basin ---
base = xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.nc")


def decode(arr):
    v = arr.values
    if v.ndim == 2:
        return np.array(
            [
                "".join(
                    c.decode() if isinstance(c, bytes) else str(c) for c in row
                ).strip()
                for row in v
            ]
        )
    return np.array(
        [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in v]
    )


names = decode(base["name"])
basins = decode(base["basin"])
S = base.sizes["storm"]
D = base.sizes["date_time"]
want = set()
for nm, bs in STORMS:
    hit = np.where((names == nm) & (basins == bs))[0]
    want.update(hit.tolist())
want = sorted(want)
print(f"matched {len(want)} storms for {len(STORMS)} (name,basin) targets", flush=True)
# --- env + per-measure vmax on flat grid ---
cps = xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.cps.nc")
pip = xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.pi_ps.nc")[["vmax"]]
ct1 = xr.open_dataset(f"{DST}/IBTrACS.since1980.v04r01.ps_cat1.nc")[["vmax"]]
r = lambda ds, n: np.asarray(ds[n].values, dtype="float64").ravel()
sst = r(cps, "sst")
rh = r(cps, "rh")
msl = r(cps, "msl")
t0 = r(cps, "t0")
lat = np.asarray(cps["lat"].values, dtype="float64").ravel()
ck = r(cps, "ck_cd")
cd = r(cps, "cd")
if np.nanmax(rh) > 1.5:
    rh = rh / 100
vmap = {"r2": r(cps, "vmax"), "r3": r(pip, "vmax"), "r1": r(ct1, "vmax")}
fname = {
    "r2": "IBTrACS.since1980.v04r01.cps.nc",
    "r3": "IBTrACS.since1980.v04r01.pi_ps.nc",
    "r1": "IBTrACS.since1980.v04r01.ps_cat1.nc",
}
# flat indices for wanted storms' full valid tracks
mask = np.zeros(S * D, bool)
for s in want:
    mask[s * D : (s + 1) * D] = True


def one(vmax, i):
    import w22.ps as p

    return p.calculate_ps_ufunc(
        vmax[i],
        msl[i],
        sst[i],
        t0[i],
        lat[i],
        rh[i],
        ck[i] if np.isfinite(ck[i]) else 0.9,
        cd[i] if np.isfinite(cd[i]) else 0.0015,
        0.002,
        1.2,
        "isothermal",
    )[3]


for meas, vmax in vmap.items():
    idx = np.where(
        mask
        & np.isfinite(vmax)
        & (vmax > 0.01)
        & np.isfinite(sst)
        & np.isfinite(rh)
        & (rh >= 0)
        & (rh <= 1)
    )[0]
    t = time.time()
    res = Parallel(n_jobs=3, backend="loky")(delayed(one)(vmax, i) for i in idx)
    # patch into copy
    nc = Dataset(f"{DST}/{fname[meas]}", "a")
    rm = nc.variables["rmax"]
    orig = np.asarray(rm[:], dtype="float64")
    sh = orig.shape
    flat = orig.ravel()
    flat[idx] = res
    rm[:] = flat.reshape(sh)
    nc.close()
    print(
        f"[{meas}] re-solved+patched {len(idx)} track pts in {(time.time()-t)/60:.1f} min",
        flush=True,
    )
# --- plot each storm ---
import tcpips.ibtracs as ib

ok = 0
for nm, bs in STORMS:
    try:
        ib.plot_tc_example(
            name=nm.encode(),
            basin=bs.encode(),
            subbasin=b"GM" if bs == "NA" else b"WPAC",
            bbox=None,
        )
        ok += 1
    except Exception as e:
        print(f"  {nm} FAIL: {type(e).__name__}: {str(e)[:120]}")
print(f"\nplotted {ok}/{len(STORMS)} track figs")
import glob

print(
    "tracks dir:",
    len(glob.glob(os.path.join(ib.FIGURE_PATH, "tracks", "*.pdf"))),
    "pdfs",
)
print("DONE")
