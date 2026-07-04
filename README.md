# Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![Python package](https://github.com/sdat2/PotentialHeight/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdat2/PotentialHeight/actions/workflows/python-package.yml)[![Documentation Status](https://readthedocs.org/projects/worstsurge/badge/?version=latest)](https://worstsurge.readthedocs.io/en/latest/?badge=latest)[![Code DOI](https://zenodo.org/badge/718141777.svg)](https://doi.org/10.5281/zenodo.15073504)[![EarthArXiv Preprint](https://img.shields.io/badge/EarthArXiv-doi:10.31223/X57T5R-blue?style=flat)](https://doi.org/10.31223/X57T5R)

EarthArXiv preprint available at <https://doi.org/10.31223/X57T5R>.

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips` & `w22`), and then use a Bayesian optimization loop (`adbo`) to drive a storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an EVT fit (`worst`).

All (or almost all) of the key experiments are carried out as `slurm` jobs — see `slurm/` and [`slurm/README.md`](https://github.com/sdat2/worstsurge/blob/main/slurm/README.md). `data/` contains some of the key data, and `img/` most of the key figures. `docs/` contains the source for the ReadTheDocs documentation at <https://worstsurge.readthedocs.io/en/latest/MAIN_README.html>.

**Reproducing the paper:** see [REPRODUCE.md](https://github.com/sdat2/worstsurge/blob/main/REPRODUCE.md) for a figure-by-figure mapping from each paper figure/table to the command, inputs, outputs, and SLURM job that produce it, plus the full pipeline (CMIP6/ERA5 → PI → PS → ADCIRC → BO → EVT) and which stages need ARCHER2/ADCIRC versus a local machine.

## Modules

### `tcpips`

Tropical Cyclone Potential Intensity (PI) and precursors for Potential Size (PS) calculations. Includes a pangeo script to download and process CMIP6 data. Regrids using `cdo` or `xESMF`. Uses the `tcpypi` package to calculate potential intensity. Also includes `tcpips/ibtracs.py` to compare IBTrACS observations with potential sizes and intensities calculated from ERA5 monthly averages (post-1980), with filters for storms undergoing extratropical transition.

### `w22`

Calculates tropical cyclone potential size following Wang et al. (2022). The implementation of Potential Size is in `w22/ps.py`, which can now calculate both potential sizes together. `w22/stats2.py` generates tables describing CMIP6 results (for HadGEM3-GC31-MM, MIROC6, and CESM2).

```bash
# Calculate example potential sizes
python -m w22.ps
# Run tests against W22
python -m w22.test_figures
```

### `cle15`

A standalone Python re-implementation of the Chavas, Lin & Emanuel (2015) tropical cyclone wind profile model. Three variants are provided:

| Module | Description |
|---|---|
| `cle15.py` | Pure-Python reference implementation |
| `cle15n.py` | Numba-accelerated drop-in replacement (~10× faster) |
| `cle15m.py` | Thin wrapper around the original MATLAB/Octave scripts in `mcle/` |

All three expose the same public API: `run_cle15`, `profile_from_stats`, `process_inputs`. The Numba implementation is now the default in `w22`.

```bash
# Run benchmarks
python -m cle15.bench_cle15
# Run tests
python -m cle15.test_cle15
```

### `adforce`

An ADCIRC wrapper for forcing and post-processing. Handles idealised axisymmetric tropical cyclones (NWS=13, ADCIRC ≥ 55.02) with a gradient wind reduction factor of $V_\text{reduc} = 0.8$ for 10 m winds. Uses `hydra` for config management.

Key files:
- `fort22.py` — write wind/pressure `fort.22.nc` input files; supports `lc12` asymmetry and Ide et al. (2022) curved parabolic tracks.
- `fort61.py` / `fort63.py` — read tide gauge and SSH/wind output.
- `mesh.py` — fast mesh reading/processing; converts ADCIRC output to dual-graph format for GNN training.
- `dual_graph.py` — dual-graph construction for ML training datasets.
- `generate_training_data.py` — forces ADCIRC with IBTrACS historical U.S. landfalling storms (1980–2024) to generate SurgeNet training data.
- `wrap.py` — orchestrates parallel ADCIRC runs on Archer2; the main file to edit when porting to a new machine.

Supports `pyproj` and a fast `sphere` approximation for distance calculations (controlled by `geoid` in `adforce/config/grid/grid_fort22.yaml`; defaults to `sphere`).

```bash
python -m adforce.wrap
```

Requires `padcirc` and `adcprep` executables placed in `adforce/config/files/`. A repo with compilation settings and minor edits is available at <https://github.com/sdat2/adcirc-v55.02>.

### `adbo`

Bayesian optimization using `adforce` to drive ADCIRC via a `trieste` optimization loop. Supports multiple GP kernels and acquisition functions. Uses `argparse` for top-level config management, then delegates to `hydra` inside `adbo`.

Key scripts:
- `exp_1d.py` — 1D Bayesian optimization experiment.
- `exp_2d.py` / `exp_3d.py` — 2D and 3D experiments.
- `gp_exp.py` — exploration of different GP kernels.
- `create_test_set.py` — builds the potential-height test set (best BO run per location) for SurgeNet training.

```bash
python -m adbo.exp_1d --test True --exp_name test
```

### `comp`

Validates the historical ADCIRC surge simulations (the SurgeNet training set, published on Hugging Face) against de-tided NOAA CO-OPS tide-gauge observations, scoring peak skill, time-series skill, and peak timing, with permutation/negative-control null tests and sensitivity checks. See [`comp/README.md`](https://github.com/sdat2/worstsurge/blob/main/comp/README.md) for the validation methodology, negative controls, and results.

### `worst`

Statistical worst-case GEV fit with `scipy` (`sci.py`) and `tensorflow` (`tens.py`). Explores the effect of knowing the upper bound ahead of time on sampling uncertainty and bias. Uses `hydra` for config management.

GEV parameterisation used: location $\alpha$, scale $\beta > 0$, shape $\gamma < 0$ (Weibull), giving upper bound $z^\star = \alpha - \beta/\gamma$.

```bash
python -m worst.vary_gamma_beta
python -m worst.vary_samples_ns
```

### `surgenet`

A motivating toy example (`toy_example.py`): a single small Keras ReLU MLP is trained to emulate $y = -f_X(x)$ for Gaussian $f_X$, and its empirical exceedance probabilities are compared against the true analytical curve, illustrating the pathological tail extrapolation of ReLU networks. The actual GNN surge-emulator work (mSWE-GNN retrained on the ADCIRC training data generated by `adforce`) lives outside this repository.

## Repository layout

```
adbo/        Bayesian optimization loop (trieste)
adforce/     ADCIRC wrapper and forcing utilities
cle15/       CLE15 wind profile implementations (pure-Python & Numba)
comp/        Historical surge validation against NOAA tide gauges (see comp/README.md)
data/        Key input data (fort.22.nc, IBTrACS, ERA5, CMIP6, etc.)
docs/        ReadTheDocs source
img/         Key figures
slurm/       SLURM job scripts for all major experiments (see slurm/README.md)
surgenet/    ReLU-extrapolation toy example (motivation for the external mSWE-GNN emulator)
tcpips/      Potential intensity & size from CMIP6/ERA5
w22/         Wang et al. (2022) potential size calculation
worst/       EVT upper-bound GEV fitting
```

## Getting started

Use conda to create the environment:

```bash
conda env create -n tcpips -f env.yml
conda activate tcpips
```

Or with micromamba (faster):

```bash
micromamba create -n tcpips -f env.yml
micromamba activate tcpips
```

`env.yml` is the portable environment specification. The lock files at the repo
root record the exact solve that was validated locally (macOS):
`environment.lock.yml` (micromamba env export) and `requirements.lock.txt`
(pip freeze). Use `env.yml` for new installs and consult the lock files to
reproduce the exact validated versions. An export of the ARCHER2 production
environment is still TODO before the final Zenodo tag.

Install the repository in editable mode:

```bash
pip install -e .
```

The core install covers data processing and the lighter analyses. Heavier
dependency groups are available as extras:

```bash
pip install -e .[bo]    # trieste/TensorFlow Bayesian optimization (adbo, worst)
pip install -e .[cmip]  # intake/xESMF CMIP6 download + regridding (tcpips)
pip install -e .[mpi]   # dask_mpi/mpi4py HPC parallelism (tcpips)
pip install -e .[comp]  # utide/huggingface_hub tide-gauge validation (comp)
pip install -e .[all]   # everything
```

Then see [REPRODUCE.md](https://github.com/sdat2/worstsurge/blob/main/REPRODUCE.md) for how to regenerate each paper figure
and table.

## Citations

If you use this code, please cite the preprint:

```bibtex
@article{potential_height_paper_2025,
  title={Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian optimization},
  author={Thomas, Simon D. A. and Jones, Dani C. and Mayo, Talea and Taylor, John R. and Moss, Henry B. and Munday, David R. and Haigh, Ivan D. and Gopinathan, Devaraj},
  journal={EarthArXiv (Submitted to Environmental Data Science)},
  year={2025},
  doi={10.31223/X57T5R}
}
```

And cite the code as:

```bibtex
@software{potential_height_code,
  author = {Thomas, Simon D. A.},
  doi = {10.5281/zenodo.15073504},
  month = {Nov},
  title = {{Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian Optimization}},
  url = {https://github.com/sdat2/PotentialHeight},
  version = {v0.1.3},
  year = {2025}
}
```

The CLE15 wind profile model (used via `cle15/`) should also be cited as:

```bibtex
@software{cle_2015_code,
  title = {Code for tropical cyclone wind profile model of Chavas et al (2015, JAS)},
  month = {Jun},
  url = {https://purr.purdue.edu/publications/4066/1},
  year = {2022},
  doi = {10.4231/CZ4P-D448},
  author = {Daniel Robert Chavas}
}
```
