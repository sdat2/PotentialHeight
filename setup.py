from setuptools import setup
from typing import List

REQUIRED: List[str] = [
    "xarray[complete]",  # to process netCDF4 files (adforce, tcpips).
    "xarray-datatree",  # to process netCDF4 files with groups (adforce).
    "netCDF4",  # to process netCDF4 files (adforce, tcpips).
    "dask[complete]",  # to process netCDF4 files lazily (adforce, tcpips).
    "uncertainties",  # common utility
    "sithom>=0.0.5",  # common utilities
    "imageio",  # to make animations
    "requests",  # to download data
    "intake",  # to read CMIP6 data
    "aiohttp",  # to read CMIP6 data
    "intake-esm",  # to read CMIP6 data
    "intake-xarray",  # to read CMIP6 data
    "gcsfs",  # to read CMIP6 data
    "xmip",  # to preprocess CMIP6 data
    "hydra-core",  # to read the config files
    "trieste",  # to run bayesian optimisation (adbo)
    "trieste[plotting]",  # to run bayesian optimisation (adbo)
    "trieste[qhsri]",  # to run bayesian optimisation (adbo)
    "slurmpy",  # to run SLURM jobs (adforce)
]

setup(
    name="tcpips",
    description="TC potential intensity and potential size.",
    version="0.0.0",
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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=["tcpips", "chavas15", "adforce", "adbo"],
    package_dir={
        "tcpips": "tcpips",  # Calculate potential intensity and potential size from CMIP6 data
        "chavas15": "chavas15",  # Calculate the Chavas, Lin and Emanuel (2015) profile
        "adforce": "adforce",  # All of the interfacing with ADCIRC
        "adbo": "adbo",  # all of the tensorflow/trieste stuff
        "worst": "worst",  # Extreme value theory using the upper bound limit
    },
)
