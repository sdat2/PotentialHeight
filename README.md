# WorstSurge repository

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips`), and then use a Bayesian optimization loop (`adbo`) to drive an storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an evt fit (`worst`).

## adforce

ADCIRC forcing and processing. Some generic mesh processing. Assume gradient wind reduction factor V_reduc=0.8 for 10m wind.

## adbo

Bayesian optimization using `adforce` to force ADCIRC with a `trieste` Bayesian optimization loop.

## w22

Chavas et al. 2015 profile calculation in matlab (using octave instead).

Used to calculate tropical cyclone potential size as in Wang et al. (2022).

## tcpips

Tropical Cyclone Potential Intensity (PI) and Potential Size (PS) Calculations.
Includes pangeo script to download and process data from CMIP6 for calculations.
Uses the `tcpypi` package to calculate potential intensity.

## worst

Statistical worst-case GEV/GPD fit using `scipy` or `tensorflow`.


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

## Bayesian optimization loop

Run `ADFORCE` 
```bash
python -m adbo.exp &> logs/bo_test3.txt
```