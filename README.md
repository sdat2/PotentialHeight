# WorstSurge repository


## adforce
ADCIRC forcing and processing.

# adbo
Bayesian optimisation.

## tcpips
Tropical Cyclone Potential Intensity (PI) and Potential Size (PS) Calculations

## worst
Statistical worst-case GEV/GPD fit.


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

```bash
python -m adbo.exp &> logs/bo_test3.txt
```