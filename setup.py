from setuptools import setup
from typing import List

REQUIRED: List[str] = [
    "xarray[complete]",
    "netCDF4",
    "dask[complete]",
    "uncertainties",
    "sithom>=0.0.5",
    "imageio",
    "requests",
    "intake",
    "aiohttp",
    "intake-esm",
    "intake-xarray",
    "gcsfs",
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
    packages=["tcpips"],
    package_dir={"tcpips": "tcpips"},
)
