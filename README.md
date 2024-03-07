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

## Extension priorities

 - BayesOpt animations/graphs.
 - Get Chavas et al. 2015 profile to work in python rather than just Matlab (~x100 speed up).
 - CMIP6 processing for potential size and intensity.
 - Scipy distribution fitting for CMIP6.
 - Better tidal gauges.
 - BayesOpt for different places along the coast.
 - 430k node mesh.

 # SurgeNet

 - Calculate dual graph as in Benteviglio et al. 2024. SWE-GNN
 - Consider NeuralODE to fix the smoothing problem of deep GNN.
 - Would diffusion/GANs also help?
 - How to include the dynamic features and tides?
