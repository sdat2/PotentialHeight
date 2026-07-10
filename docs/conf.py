# pylint: disable=missing-module-docstring
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "worstsurge"
copyright = "2025"  # pylint: disable=redefined-builtin
author = "Simon Thomas"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "nbsphinx_link",
]


## intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

## autodoc

# Heavy optional dependencies mocked so the API docs build with only the core
# install (ReadTheDocs installs the package without the [bo]/[cmip]/[mpi]
# extras). Only packages actually imported at module level by the documented
# packages are listed (checked by grep):
#   tensorflow/tensorflow_probability - adbo.exp, adbo.gp_exp, worst.tens,
#       worst.vary_nonstationary, surgenet.toy_example
#   trieste, gpflow                   - adbo.exp, adbo.gp_exp
#   intake, xmip                      - tcpips.pangeo
#   xesmf                             - tcpips.regrid
#   cdsapi                            - tcpips.era5
#   dask_mpi                          - tcpips.era5_ps_calc
#   dask_jobqueue                     - tcpips.run_dask_calculation
#   adcircpy                          - adforce.training.inputs (via .driver; shim adforce.generate_training_data is lazy)
#       (not in install_requires at all)
#   numba                             - cle15.cle15n
#       (not in install_requires; remove from this list if added there)
# Not mocked because they are only imported inside functions and so cannot
# break the docs build: utide, huggingface_hub (comp), mpi4py, oct2py.
# Not mocked because they are core install_requires: imageio, datatree
# (xarray-datatree), slurmpy, tcpyPI (tcpypi), hydra/omegaconf.
autodoc_mock_imports = [
    "tensorflow",
    "tensorflow_probability",
    "trieste",
    "gpflow",
    "intake",
    "xmip",
    "xesmf",
    "cdsapi",
    "dask_mpi",
    "dask_jobqueue",
    "adcircpy",
    "numba",
]

# mathjax.

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# README.md, lang.txt and requirements.txt are docs-tooling files, not
# documentation sources (they otherwise trigger "not in any toctree" warnings).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "lang.txt",
    "requirements.txt",
]


# filetype mapping

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# No custom static files are shipped; an empty list avoids the sphinx warning
# about a nonexistent _static directory.
html_static_path = []


# html_logo = "images/geograph_logo.png"

html_theme_options = {
    "style_nav_header_background": "#a0a0a0",
    "logo_only": True,
}

# html_favicon = 'images/geograph_logo_small.png'

# This adds the 'edit on github' banner on top right corner

html_context = {
    "display_github": True,
    "github_user": "sdat2",
    "github_repo": "worstsurge",
    "github_version": "main/docs/",
}

# Latex options
# latex_logo = "./images/geograph_logo.png"

latex_elements = {
    "extraclassoptions": "openany,oneside",
    "papersize": "a4paper",
}
