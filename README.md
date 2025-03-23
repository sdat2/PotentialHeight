# WorstSurge: Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![Python package](https://github.com/sdat2/worstsurge/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdat2/worstsurge/actions/workflows/python-package.yml)[![Documentation Status](https://readthedocs.org/projects/worstsurge/badge/?version=latest)](https://worstsurge.readthedocs.io/en/latest/?badge=latest)

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips` & `w22`), and then use a Bayesian optimization loop (`adbo`) to drive an storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an evt fit (`worst`). 

Most/all of the key experiments are carried out as `slurm` jobs, so go to `slurm/` to see these. `data/` contains some of the key data, and `img/` most of the key figures. `docs/` contains the source for the ![readthedocs documentation](https://worstsurge.readthedocs.io/en/latest/MAIN_README.html).

## tcpips

Tropical Cyclone Potential Intensity (PI) and precursors for Potential Size (PS) calculations.
Includes pangeo script to download and process data from CMIP6 for calculations. Currently regrids using `cdo` or `xESMf`.
Uses the `tcpypi` pypi package to calculate potential intensity.

## w22

Chavas et al. 2015 profile calculation in matlab (using octave instead).

Used to calculate tropical cyclone potential size as in Wang et al. (2022).


## adforce

ADCIRC forcing and processing. Some generic mesh processing. Assume gradient wind reduction factor V_reduc=0.8 for 10m wind. Runs using `hydra` for config management.

```bash
python -m adforce.wrap
```

Needs an executatable directory for ADCIRC that includes a `padcirc` and `adcprep` executable stored in `adforce/config/files/`. 

A repo with our compilation settings for ADCIRC, and small edits, is available at <https://github.com/sdat2/adcirc-v55.02>.

## adbo

Bayesian optimization using `adforce` to force ADCIRC with a `trieste` Bayesian optimization loop. Runs using `argparse` for config management, and then calls `adbo` which uses `hydra`.

```bash
python -m adbo.exp_3d
```

## worst

Statistical worst-case GEV fit using `tensorflow`. Varies whether to assume an upper bound ahead of time. Uses `hydra` for the config management.


## Getting started

Developer install:

```bash
# Use conda to install a python virtual environment
conda env create -n tcpips -f env.yml
conda activate tcpips

# or use micromamba (faster)
# maybe you need to activate micromamba "$(micromamba shell hook --shell zsh)"
micromamba create -n tcpips -f env.yml
micromamba activate tcpips

# Install repository in editable version
pip install -e .

```

