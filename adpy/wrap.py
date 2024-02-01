"""Wrap.

To set off Slurm I could use:
- Fireworks (https://materialsproject.github.io/fireworks/)
@article {CPE:CPE3505,
author = {Jain, Anubhav and Ong, Shyue Ping and Chen, Wei and Medasani, Bharat and Qu, Xiaohui and Kocher, Michael and Brafman, Miriam and Petretto, Guido and Rignanese, Gian-Marco and Hautier, Geoffroy and Gunter, Daniel and Persson, Kristin A.},
title = {FireWorks: a dynamic workflow system designed for high-throughput applications},
journal = {Concurrency and Computation: Practice and Experience},
volume = {27},
number = {17},
issn = {1532-0634},
url = {http://dx.doi.org/10.1002/cpe.3505},
doi = {10.1002/cpe.3505},
pages = {5037--5059},
keywords = {scientific workflows, high-throughput computing, fault-tolerant computing},
year = {2015},
note = {CPE-14-0307.R2},
-
"""
import os
import numpy as np
import shutil
from .fort22 import return_new_input, save_forcing
from .fort63 import xr_loader
from sithom.time import timeit

OG_PATH = "/work/n01/n01/sithom/adcirc-swan/NWS13ex"


@timeit
def setup_new(new_path: str, angle: float):
    """
    Set up a new ADCIRC folder and add in the
    forcing data.

    Args:
        new_path (str): _description_
    """
    # original path to copy setting files from
    files = ["fort.13", "fort.14", "fort.15", "fort.64.nc", "fort.73.nc", "fort.74.nc", "submit.slurm"]
    os.makedirs(new_path, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(OG_PATH, file), os.path.join(new_path, file))

    save_forcing(new_path, angle=angle)


@timeit
def run_new(out_path=OG_PATH):
    # add new forcing
    forcing_dt = return_new_input()
    forcing_dt.to_netcdf(os.path.join(out_path, "fort.22.nc"))
    # set off sbatch.

    # look at results.

def read_results(path=OG_PATH):
    mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
    print(mele_ds)


if __name__ == "__main__":
    # python -m adpy.wrap
    # python adpy/wrap.py
    root = "/work/n01/n01/sithom/adcirc-swan/"
    exp_dir = os.path.join(root, "angle_test")
    os.makedirs(exp_dir, exist_ok=True)
    for i, angle in enumerate(np.linspace(-90, 90, num=10)):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        setup_new(tmp_dir, angle)
