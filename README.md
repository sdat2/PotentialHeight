# WorstSurge repository

## adforce
ADCIRC forcing and processing. Some generic mesh processing.

## adbo
Bayesian optimization using `adforce` to force adcirc with a `trieste` Bayesian optimization loop.

## chavas15
Chavas et al. 2015 profile. (Currently not working properly)

## tcpips
Tropical Cyclone Potential Intensity (PI) and Potential Size (PS) Calculations.
Includes pangeo script to download and process data from CMIP6 for calculations.
Uses 

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

## Extension priorities

Done:

 - Get 430k node mesh working.
 - BayesOpt animations/graphs.
    - Impact animation (still need to add winds in).
    - GP convergence animation (2D).

In Progress:

 - CMIP6 processing for potential size and intensity.
 - BayesOpt for different places along the coast.
 - Fix 430k wrapping.
 - Sync times up.

TODO:

 - Get Chavas et al. 2015 profile to work in Python rather than just Matlab (~x100 speed up).
 - Scipy distribution fitting for CMIP6.
 - Better tidal gauge comparisons for different events.
 - 

## SurgeNet exploration

 - Calculate dual graph as in Benteviglio et al. 2024 SWE-GNN.
 - Consider NeuralODE to fix the smoothing problem of deep GNN.
 - Would diffusion/GANs also help?
 - How to include the dynamic features and tides?
