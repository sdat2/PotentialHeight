"""Wrap.

To set off Slurm I could use (https://stackoverflow.com/questions/61704713/best-practice-submitting-slurm-jobs-via-python):
- Slurmpy https://github.com/brentp/slurmpy
- Snakemake
@article{molder2021sustainable,
  title={Sustainable data analysis with Snakemake},
  author={M{\"o}lder, Felix and Jablonski, Kim Philipp and Letcher, Brice and Hall, Michael B and Tomkins-Tinch, Christopher H and Sochat, Vanessa and Forster, Jan and Lee, Soohyun and Twardziok, Sven O and Kanitz, Alexander and others},
  journal={F1000Research},
  volume={10},
  year={2021},
  publisher={Faculty of 1000 Ltd}
}
- Joblib
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
from src.constants import NEW_ORLEANS

ROOT = "/work/n01/n01/sithom/adcirc-swan/"
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
    files = [
        "fort.13",
        "fort.14",
        "fort.15",
        "fort.64.nc",
        "fort.73.nc",
        "fort.74.nc",
        "submit.slurm",
    ]
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


@timeit
def read_results(path=OG_PATH):
    mele_ds = xr_loader(os.path.join(path, "maxele.63.nc"))
    # read zeta_max point closes to the
    # work out closest point to NEW_ORLEANS
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    dist = ((xs - NEW_ORLEANS.lon) ** 2 + (ys - NEW_ORLEANS.lat) ** 2) ** 0.5
    min_p = np.argmin(dist)
    # Read the maximum elevation for that point
    return mele_ds["zeta_max"].values[min_p]


@timeit
def run_angle_exp():
    exp_dir = os.path.join(ROOT, "angle_test")
    os.makedirs(exp_dir, exist_ok=True)
    for i, angle in enumerate(np.linspace(-90, 90, num=10)):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        setup_new(tmp_dir, angle)


@timeit
def read_angle_exp():
    results = []
    exp_dir = os.path.join(ROOT, "angle_test")
    for i, angle in enumerate(np.linspace(-90, 90, num=10)):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        res = read_results(tmp_dir)
        print(i, angle, res)
        results += [[i, angle, res]]
    results = np.array(results)

    import matplotlib.pyplot as plt
    from sithom.plot import plot_defaults

    plot_defaults()
    plt.plot(results[:, 1], results[:, 2])
    plt.xlabel("Angle [$^{\circ}$]")
    plt.ylabel("Height [m]")
    plt.savefig("angle_test.png")


if __name__ == "__main__":
    # python -m adpy.wrap
    # python adpy/wrap.py
    read_angle_exp()
