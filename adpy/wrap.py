"""Wrap."""
import os
import shutil
from .fort22 import return_new_input
from .fort63 import xr_loader
from sithom.time import timeit

path = "/work/n01/n01/sithom/adcirc-swan/NWS13ex"


def setup_new(new_path: str):
    og_path = "/work/n01/n01/sithom/adcirc-swan/NWS13ex"
    files = ["fort.13", "fort.14", "fort.15", "fort.63.nc", "fort.64.nc", "fort.73.nc", "fort.74.nc", "submit.slurm"]
    os.makedirs(new_path, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(og_path, file), os.path.join(new_path, file))


@timeit
def run_new(out_path=path):
    # add new forcing
    forcing_dt = return_new_input()
    forcing_dt.to_netcdf(os.path.join(out_path, "fort.22.nc"))
    # set off sbatch.

    # look at results.

def read_results():
    mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
    print(mele_ds)


if __name__ == "__main__":
    # python -m adpy.wrap
    # python adpy/wrap.py
    setup_new("/work/n01/n01/sithom/adcirc-swan/NWS13set3")
