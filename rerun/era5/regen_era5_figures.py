"""Regenerate all ERA5 potential-size figures from the fixed-solver zarrs.

Runs the repo's own plotters (tcpips.era5) over the 1980-2024 fixed-solver
product written by make_ps_og_zarrs.py:

  1. era5_ps_trends()          -> Newey-West linear trends product zarr
  2. plot_snapshot_map(2020)   -> single-year spatial snapshot (Image-13 class)
  3. plot_trend_maps()         -> per-variable trend maps (Image-15 class)
  4. plot_trends_all_lineplots -> per-site (NO/HK/...) trend lineplots

Figures land in ERA5_FIGURE_PATH (img/era5/). Run guarded from repo root:
    HDF5_USE_FILE_LOCKING=FALSE python rerun/mem_guard.py \
        rerun/era5/regen_era5_figures.py 7168 14400
"""

import os
import sys
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore")

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib

matplotlib.use("Agg")

from tcpips import era5


def step(label, fn):
    print(f"### {label} ###", flush=True)
    try:
        fn()
        print(f"    OK {label}", flush=True)
    except Exception as e:  # keep going; one broken plot shouldn't sink the rest
        import traceback

        print(f"    FAIL {label}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def main():
    step("1. era5_ps_trends (compute trends product)", era5.era5_ps_trends)
    step("2. plot_snapshot_map(2020)", lambda: era5.plot_snapshot_map(year=2020))
    step("3. plot_trend_maps", era5.plot_trend_maps)
    step("4. plot_trends_all_lineplots", era5.plot_trends_all_lineplots)
    print("FIGURE PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
