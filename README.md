# WorstSurge repository

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips`), and then use a Bayesian optimization loop (`adbo`) to drive an storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an evt fit (`worst`).

## adforce

ADCIRC forcing and processing. Some generic mesh processing.

Animating an adcirc experiment:
```bash
python -m adforce.ani --path_in . --step_size 1
```

## adbo

Bayesian optimization using `adforce` to force ADCIRC with a `trieste` Bayesian optimization loop.

## cle

Chavas et al. 2015 profile calculation in matlab. (fairly slow coupling, now using `oct2py`)

## chavas15

Chavas et al. 2015 profile calculation in python. (Currently not working properly)

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
micromamba create -n worstsurge -f env.yml

micromamba create -n worst -f env.yml
pip install -e ../sithom


micromamba activate tcpips

# Install repository in editable version
pip install -e .

```

## Bayesian optimization loop

Run `ADFORCE` 
```bash
python -m adbo.exp &> logs/bo_test3.txt
```

## Task list:

### Done:

 - Get 430k node mesh working.
 - BayesOpt animations/graphs.
    - Impact animation (still need to add winds in).
    - GP convergence animation (2D).
 - Sync times up & animate winds.
 - GEV with maxima Scipy distribution fitting for CMIP6.
 - Reformat adforce/adbo to allow:
   - flexible profile forcing.
   - keep time/coordinates of moving cyclones in repo.
 - CMIP6 processing for potential size and intensity.
    - Good folder structure for processing.
    - Make sure regridding is working and efficient.
    - Parallelize potential size calculation as much as possible.
 - BayesOpt for different places along the coast.
 - Fix 430k wrapping.
 - How to quantify the performance of BO better.

### In progress

Last week:
 - Discussion vs. Introduction for WorstSurge (meet with Talea).
 - potential size vs W22 (done-ish).
 - fix potential size for variable surface relative humidity.

This week:
 - Discussion vs. Introduction for WorstSurge (meet with Talea).
 - plot seasonal cycle, global picture.

Next week:
 - Gulf of Mexico August (maybe additionaly South East Asia (e.g. Hong Kong)):
   - historical-CESM2
   - ERA5 PIPS (~observations)
   - compare to find biases.

The week after:
 - Compare ERA5 potential size to IBTrACS.
 - maybe focus on GOM and Hong Kong.

### TODO

 - Could we produce uncertainty estimates in the upper bound based on MES.
 - Get Chavas et al. 2015 profile to work in Python rather than just Matlab (~x100 speed up).

### Nice to have

 - How to distribute optimisation points more uniformly along the coast.
 - Better tidal gauge comparisons for different events.

## SurgeNet exploration

### TODO

 - Calculate dual graph as in Benteviglio et al. 2024 SWE-GNN.
 - Run initial training/test of SWE-GGN or GCN or MLP

### (nice to have)

 - Consider NeuralODE to fix the smoothing problem of deep GNN.
 - Would diffusion/GANs also help?

### Probably excluded
 - How to include the dynamic features and tides?

