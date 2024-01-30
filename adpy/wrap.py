"""Wrap."""
import os
from adpy.fort22 import return_new_input
from adpy.fort63 import xr_loader
from sithom.time import timeit

path = "/work/n01/n01/sithom/adcirc-swan/NWS13ex"


@timeit
def run_new():
    # add new forcing
    forcing_dt = return_new_input()
    forcing_dt.to_dataset(os.path.join(path, "fort.22.nc"))
    # set off sbatch.

    # look at results.


def read_results():
    mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
    print(mele_ds)


if __name__ == "__main__":
    # python adpy/wrap.py
    run_new()
