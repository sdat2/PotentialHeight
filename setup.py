from setuptools import setup
from typing import List

REQUIRED: List[str] = [
    "xarray[complete]",  # to process netCDF4 files (adforce, tcpips).
    "xarray-datatree",  # to process netCDF4 files with groups (adforce).
    "netCDF4",  # to process netCDF4 files (adforce, tcpips).
    "h5netcdf",  # to process netCDF4 files (adforce, tcpips).
    "pyproj",  # to process geospatial data (adforce, tcpips).
    "dask[complete]",  # to process netCDF4 files lazily (adforce, tcpips).
    "dask_mpi",  # to run dask on an MPI cluster (tcpips)
    "mpi4py",  # to run dask on an MPI cluster (tcpips)
    "uncertainties",  # common utility for linear error propagation
    # "sithom @ git+https://github.com/sdat2/sithom",
    "sithom >= 0.1.1",  # personal common utilities for timing, plotting, and fitting
    "osqp==1.0.1",  # pip install problem 2nd April 2025 https://github.com/astral-sh/uv/issues/12618
    "cdsapi",  # download ERA5 monthly averages (tcpips)
    "imageio",  # to make animations (adforce, adbo)
    "requests",  # to download data (tcpips)
    "intake",  # to read CMIP6 data (tcpips)
    "aiohttp",  # to read CMIP6 data (tcpips)
    "intake-esm",  # to read CMIP6 data (tcpips)
    "intake-xarray",  # to read CMIP6 data (tcpips)
    "xesmf",  # to regrid CMIP6 data (tcpips) # really this needs to be installed by conda
    "gcsfs",  # to read CMIP6 data (tcpips)
    "xmip",  # to preprocess CMIP6 data (tcpips)
    "hydra-core",  # to read the yaml config files (adforce, worst)
    "trieste",  # to run bayesian optimisation (adbo)
    "trieste[plotting]",  # to run bayesian optimisation (adbo)
    "trieste[qhsri]",  # to run bayesian optimisation (adbo)
    "slurmpy",  # to run SLURM jobs (adforce)
    "tcpypi",  # to calculate potential intensity (tcpips)
    "joblib",  # to parallize more easily potential size calculation (tcpips/w22)
    "ujson",  # to read/write json dictionaries more flexibly (tcpips)
    "statsmodels",  # to calculate trends with Newey-West standard errors (tcpips)
]


setup(
    name="PotentialHeight",
    description="A set of tools to calculate the potential height of tropical cyclone storm surges",
    version="0.1.1",
    author_email="sdat2@cam.ac.uk",
    author="Simon D.A. Thomas",
    install_requires=REQUIRED,
    python_requires=">=3.9",
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=["tcpips", "w22", "adforce", "adbo", "worst"],
    package_dir={
        "tcpips": "tcpips",  # Calculate potential intensity and prerequisites for potential size
        "w22": "w22",  # Calculate the Chavas, Lin and Emanuel (2015) profile using matlab (octave), calculate potential size
        "adforce": "adforce",  # All of the interfacing with ADCIRC
        "adbo": "adbo",  # All of the tensorflow/trieste Bayesian optimization stuff
        "worst": "worst",  # Extreme value theory using the upper bound limit using tensorflow for fitting.
    },
)
