from setuptools import setup
from typing import Dict, List

# Core dependencies: what the pure-python, paper-relevant modules
# (cle15, worst, comp core, w22) need from a bare `pip install .`.
# Heavy stacks (Bayesian optimisation, CMIP6 ingestion, MPI, surge
# comparison data) are split into the extras_require lists below.
REQUIRED: List[str] = [
    # tensorflow (<2.17, pulled in via the [bo] extra) requires numpy<2;
    # mirrored from env.yml so pip metadata matches the conda solve.
    "numpy<2",
    "xarray[complete]",  # to process netCDF4 files (adforce, tcpips). No version floor pinned yet; see xarray-datatree note below.
    # archived shim; only used when xarray < 2024.10 (see
    # adforce/fort22datatree._open_datatree); drop when xarray floor is raised
    "xarray-datatree",
    "netCDF4",  # to process netCDF4 files (adforce, tcpips).
    "h5netcdf",  # to process netCDF4 files (adforce, tcpips).
    "pyproj",  # to process geospatial data (adforce, tcpips).
    # dask stays in core: tcpips.era5 imports dask at module level and is in
    # turn imported at module level by w22.ps_runs/w22.stats2 (paper-relevant);
    # xarray[complete] would pull it in anyway.
    "dask[complete]",  # to process netCDF4 files lazily (adforce, tcpips).
    "scipy",  # nearest-node KD-tree for gauge matching (comp)
    "uncertainties",  # common utility for linear error propagation
    # "sithom @ git+https://github.com/sdat2/sithom",
    "sithom >= 0.1.1",  # personal common utilities for timing, plotting, and fitting
    "osqp==1.0.1",  # pip install problem 2nd April 2025 https://github.com/astral-sh/uv/issues/12618
    "imageio",  # to make animations (adforce, adbo)
    "requests",  # to download data (tcpips)
    "hydra-core",  # to read the yaml config files (adforce, worst)
    "tcpypi",  # to calculate potential intensity (tcpips)
    "joblib",  # to parallize more easily potential size calculation (tcpips/w22)
    "ujson",  # to read/write json dictionaries more flexibly (tcpips)
    "statsmodels",  # to calculate trends with Newey-West standard errors (tcpips)
    "tqdm",  # progress bars; module-level import in worst/* and w22.ps (previously only satisfied transitively)
]

# "bo": Bayesian optimisation stack (adbo; also worst.tens TF fitting).
BO: List[str] = [
    "trieste[plotting,qhsri]",  # to run bayesian optimisation (adbo)
    "slurmpy",  # to run SLURM jobs (adforce)
    # Pin a modern, internally consistent TensorFlow stack on Linux (CI).
    # Without a pin the resolver can land on an EOL tensorflow (<2.12) whose
    # _pb2 files were built with protoc <3.19, which crashes on modern
    # protobuf ("Descriptors cannot be created directly"). TF 2.15 ships
    # protobuf 4.x protos and bundles Keras 2 (no tf-keras needed); TFP 0.23
    # is the matching release. macOS keeps trieste's own tensorflow-macos
    # resolution (mirrors env.yml's pip section).
    "tensorflow==2.15.1; sys_platform=='linux'",
    "tensorflow-probability==0.23.0; sys_platform=='linux'",
]

# "cmip": CMIP6/ERA5 download, regridding and preprocessing (tcpips).
CMIP: List[str] = [
    "intake",  # to read CMIP6 data (tcpips)
    "aiohttp",  # to read CMIP6 data (tcpips)
    "intake-esm",  # to read CMIP6 data (tcpips)
    "intake-xarray",  # to read CMIP6 data (tcpips)
    "gcsfs",  # to read CMIP6 data (tcpips)
    "xmip",  # to preprocess CMIP6 data (tcpips)
    "xesmf",  # to regrid CMIP6 data (tcpips) # really this needs to be installed by conda
    "cdsapi",  # download ERA5 monthly averages (tcpips)
]

# "mpi": distributed runs on an HPC cluster (dask[complete] itself is core).
MPI: List[str] = [
    "dask_mpi",  # to run dask on an MPI cluster (tcpips)
    "mpi4py",  # to run dask on an MPI cluster (tcpips)
    "dask_jobqueue",  # SLURMCluster in tcpips/run_dask_calculation.py
]

# "comp": historical surge validation against tide gauges (comp).
COMP: List[str] = [
    "huggingface_hub",  # to download historical surge datasets (comp)
    "utide",  # to de-tide tide-gauge records for surge validation (comp)
    "pyarrow",  # Parquet cache of de-tided gauge time series (comp)
]

EXTRAS: Dict[str, List[str]] = {
    "bo": BO,
    "cmip": CMIP,
    "mpi": MPI,
    "comp": COMP,
}
EXTRAS["all"] = BO + CMIP + MPI + COMP  # union of all extras


setup(
    name="PotentialHeight",
    description="A set of tools to calculate the potential height of tropical cyclone storm surges",
    version="0.1.3",
    author_email="sdat2@cam.ac.uk",
    author="Simon D.A. Thomas",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    # TF 2.15.1 (pinned in the [bo] extra) has no python 3.12 wheels, and
    # env.yml fixes python=3.10.
    python_requires=">=3.10,<3.12",
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=["tcpips", "w22", "adforce", "adbo", "worst", "cle15", "comp"],
    package_dir={
        "tcpips": "tcpips",  # Calculate potential intensity and prerequisites for potential size
        "w22": "w22",  # Calculate the Chavas, Lin and Emanuel (2015) profile using matlab (octave), calculate potential size
        "adforce": "adforce",  # All of the interfacing with ADCIRC
        "adbo": "adbo",  # All of the tensorflow/trieste Bayesian optimization stuff
        "worst": "worst",  # Extreme value theory using the upper bound limit using tensorflow for fitting.
        "cle15": "cle15",  # Chavas, Lin & Emanuel (2015) TC wind profile implementations
        "comp": "comp",  # Compare historical ADCIRC surge against de-tided NOAA tide gauges
    },
)
