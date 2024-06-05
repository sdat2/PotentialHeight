from setuptools import setup
from typing import List

REQUIRED: List[str] = [
    # "sithom"
    "xarray[complete]",  # to process netCDF4 files (adforce, tcpips).
    "xarray-datatree",  # to process netCDF4 files with groups (adforce).
    "netCDF4",  # to process netCDF4 files (adforce, tcpips).
    "h5netcdf",  # to process netCDF4 files (adforce, tcpips).
    "dask[complete]",  # to process netCDF4 files lazily (adforce, tcpips).
    "uncertainties",  # common utility for linear error propagation
    #"sithom>=0.0.5",  # common utilities
    "sithom @ git+https://github.com/sdat2/sithom",
    "imageio",  # to make animations (adforce, adbo)
    "requests",  # to download data (tcpips)
    "intake",  # to read CMIP6 data (tcpips)
    "aiohttp",  # to read CMIP6 data (tcpips)
    "intake-esm",  # to read CMIP6 data (tcpips)
    "intake-xarray",  # to read CMIP6 data (tcpips)
    "dask[complete]",  # to read CMIP6 data (tcpips)
    "xesmf",  # to regrid CMIP6 data (tcpips) # really this needs to be installed by
    "gcsfs",  # to read CMIP6 data (tcpips)
    "xmip",  # to preprocess CMIP6 data (tcpips)
    "hydra-core",  # to read the config files
    "trieste",  # to run bayesian optimisation (adbo)
    "trieste[plotting]",  # to run bayesian optimisation (adbo)
    "trieste[qhsri]",  # to run bayesian optimisation (adbo)
    "slurmpy", # to run SLURM jobs (adforce)
    "adcircpy", # to process ADIRC inputs (adforce)
    "tcpypi", # to calculate potential intensity (tcpips)
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
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=["tcpips", "chavas15", "cle", "adforce", "adbo"],
    package_dir={
        "tcpips": "tcpips",  # Calculate potential intensity and potential size from CMIP6 data
        "chavas15": "chavas15",  # Calculate the Chavas, Lin and Emanuel (2015) profile
        "cle": "cle",  # Calculate the Chavas, Lin and Emanuel (2015) profile
        "adforce": "adforce",  # All of the interfacing with ADCIRC
        "adbo": "adbo",  # all of the tensorflow/trieste stuff
        "worst": "worst",  # Extreme value theory using the upper bound limit
    },
)
