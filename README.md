# WorstSurge repository

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips`), and then use a Bayesian optimization loop (`adbo`) to drive an storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an evt fit (`worst`).

## tcpips

Tropical Cyclone Potential Intensity (PI) and precursors for Potential Size (PS) calculations.
Includes pangeo script to download and process data from CMIP6 for calculations. Currently regrids using cdo or xESMf.
Uses the `tcpypi` pypi package to calculate potential intensity.

## w22

Chavas et al. 2015 profile calculation in matlab (using octave instead).

Used to calculate tropical cyclone potential size as in Wang et al. (2022).


## adforce

ADCIRC forcing and processing. Some generic mesh processing. Assume gradient wind reduction factor V_reduc=0.8 for 10m wind.

```bash
python -m adforce.wrap
```

Needs an executatable directory for ADCIRC that includes a `padcirc` and `adcprep` executable stored in `adforce/config/files/`.

Our compilation settings for ADCIRC is available at <https://github.com/sdat2/adcirc-v55.02>.

## adbo

Bayesian optimization using `adforce` to force ADCIRC with a `trieste` Bayesian optimization loop.

```bash
python -m adbo.exp_3d
```

## worst

Statistical worst-case GEV fit using `tensorflow`.


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

