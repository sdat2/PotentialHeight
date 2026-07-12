"""Convert the fixed-solver per-year ps_fixed/*.nc into the zarr layout the
repo's ERA5 figure code expects.

tcpips/era5.py's plotters (plot_snapshot_map, era5_ps_trends, plot_trend_maps,
plot_trend_lineplots) read ``ERA5_PS_OG_PATH/era5_potential_sizes_*.zarr`` with
dims (year, latitude, longitude). The 2026-07 fixed-solver recomputation
(era5_ps_local.py, laptop + GCP) produced per-year netCDFs with dims (lat, lon)
and only the solver inputs/outputs — so this script, per decade:

  1. opens the 10 per-year files, concatenates on ``year``;
  2. renames lat/lon -> latitude/longitude;
  3. lazily re-opens the decade's raw/PI inputs via ``load_decade`` and merges
     the environment fields the plots also want (otl, ifl, pmin where present);
  4. writes ``era5_potential_sizes_{s}_{e}.zarr`` into ERA5_PS_OG_PATH.

ARCHIVE THE BUGGY ORIGINALS FIRST (this script refuses to overwrite):
    mv data/era5/ps_og data/era5/ps_og_buggy_archer2 && mkdir data/era5/ps_og

Run (SSD mounted, repo root):
    HDF5_USE_FILE_LOCKING=FALSE python rerun/mem_guard.py \
        rerun/era5/make_ps_og_zarrs.py 7168 7200
"""

import glob
import os
import sys
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import dask

dask.config.set(scheduler="synchronous")

import xarray as xr

from tcpips.constants import ERA5_PS_OG_PATH

# rerun/ is deliberately not a package; import the sibling module by path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era5_ps_local import load_decade  # lazy env loader (same prep as the solve)

PS_FIXED = "/Volumes/s/tcpips/data/era5/ps_fixed"
DECADES = [(1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]
ENV_EXTRAS = ["otl", "ifl", "pmin"]  # figure code wants these; merge if available


def main() -> None:
    for start, end in DECADES:
        out = os.path.join(ERA5_PS_OG_PATH, f"era5_potential_sizes_{start}_{end}.zarr")
        if os.path.exists(out):
            print(f"[{start}-{end}] {out} exists — refusing to overwrite "
                  "(archive ps_og first)", flush=True)
            continue
        files = [os.path.join(PS_FIXED, f"era5_ps_fixed_{y}.nc")
                 for y in range(start, end + 1)]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"missing inputs: {missing[:3]}")
        parts = []
        for f in files:
            d = xr.open_dataset(f)
            if "year" not in d.dims:  # our writer stored year as a scalar coord
                d = d.expand_dims("year")
            parts.append(d)
        ds = xr.concat(parts, dim="year").rename(
            {"lat": "latitude", "lon": "longitude"}
        )

        env = load_decade(start, end).rename({"lat": "latitude", "lon": "longitude"})
        extras = [v for v in ENV_EXTRAS if v in env]
        if extras:
            ds = xr.merge([ds, env[extras]], join="left", compat="override")
        print(f"[{start}-{end}] vars: {sorted(ds.data_vars)}", flush=True)
        ds.chunk({"year": 1, "latitude": -1, "longitude": -1}).to_zarr(out)
        print(f"[{start}-{end}] wrote {out}", flush=True)
    print("ALL ZARRS WRITTEN", flush=True)


if __name__ == "__main__":
    main()
