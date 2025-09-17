# Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![Python package](https://github.com/sdat2/worstsurge/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdat2/worstsurge/actions/workflows/python-package.yml)[![Documentation Status](https://readthedocs.org/projects/worstsurge/badge/?version=latest)](https://worstsurge.readthedocs.io/en/latest/?badge=latest)[![Code DOI](https://zenodo.org/badge/718141777.svg)](https://doi.org/10.5281/zenodo.15073504)[![EarthArXiv Preprint](https://img.shields.io/badge/EarthArXiv-doi:10.31223/X57T5R-blue?style=flat)](https://doi.org/10.31223/X57T5R)

EarthArxiv preprint now available <https://doi.org/10.31223/X57T5R>!

We want to answer the question of what the potential height of a storm surge could be now and in a changing climate. To do this we first calculate the potential intensity and size from CMIP6 (`tcpips` & `w22`), and then use a Bayesian optimization loop (`adbo`) to drive an storm surge model ADCIRC with idealised tropical cyclones (`adforce`). We then show that knowing the upper bound can be useful in the context of an evt fit (`worst`). 

All (or almost all) of the key experisments are carried out as `slurm` jobs, so go to `slurm/` to see these. `data/` contains some of the key data, and `img/` most of the key figures. `docs/` contains the source for the readthedocs documentation <https://worstsurge.readthedocs.io/en/latest/MAIN_README.html>.

## tcpips

Tropical Cyclone Potential Intensity (PI) and precursors for Potential Size (PS) calculations.
Includes pangeo script to download and process data from CMIP6 for calculations. Currently regrids using `cdo` or `xESMf`.
Uses the `tcpypi` pypi package to calculate potential intensity.

## w22

Used to calculate tropical cyclone potential size as in Wang et al. (2022). Uses Chavas et al. 2015 profile calculation in matlab (using octave instead).
Now we have a new python implementation that should work better.

```bash
# to calculate the example potential size
python -m w22.ps
# to calculate the tests against W22
python -m w22.test
```


## adforce

ADCIRC forcing and processing. Some generic mesh processing. Assumes gradient wind reduction factor V_reduc=0.8 for 10m wind. Runs using `hydra` for config management.

```bash
python -m adforce.wrap
```

Needs an executatable directory for ADCIRC that includes a `padcirc` and `adcprep` executable stored in `adforce/config/files/`. 

A repo with our compilation settings for ADCIRC, and small edits, is available at <https://github.com/sdat2/adcirc-v55.02>.

## adbo

Bayesian optimization using `adforce` to force ADCIRC with a `trieste` Bayesian optimization loop. Runs using `argparse` for config management, and then calls `adbo` which uses `hydra`.

```bash
python -m adbo.exp_1d --test True --exp_name test
```

## worst

Statistical worst-case GEV fit using `tensorflow`. Varies whether to assume an upper bound ahead of time. Uses `hydra` for the config management.


## Getting started

Developer install:

Use conda to install a python virtual environment:

```bash
conda env create -n tcpips -f env.yml
conda activate tcpips
```

Alternatively use micromamba (faster):

```bash
micromamba create -n tcpips -f env.yml
micromamba activate tcpips
```

Install repository in editable version:

```bash
pip install -e .
```

## Citations

If you use this code, please cite the following paper:

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
  month = {Sep},
  title = {{Finding the potential height of tropical cyclone storm surges in a changing climate using Bayesian Optimization}},
  url = {https://github.com/sdat2/PotentialHeight},
  version = {v0.1.1},
  year = {2025}
}
```

We used to use the matlab code from Chavas, Lin, and Emanuel (2015), which should be cited as:

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
