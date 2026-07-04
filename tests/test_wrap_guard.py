"""Unit tests for the NaN/fill-value guard in adforce.wrap.observe_max_point.

A failed or dry ADCIRC run can leave NaN or a netCDF fill value (e.g. 9.96921e36)
in maxele.63.nc; the guard must raise ValueError so the Bayesian optimization
loop fails loudly instead of feeding a fake surge maximum to the GP. Uses a tiny
synthetic xarray dataset and monkeypatches xr_loader, so no ADCIRC output files
are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adforce import wrap

NETCDF_FILL = 9.96921e36  # default netCDF float32 fill value


def _synthetic_maxele(zeta_at_obs: float) -> xr.Dataset:
    """Three-node maxele.63.nc stand-in; node 1 is nearest the observation point."""
    return xr.Dataset(
        {
            "zeta_max": ("node", np.array([1.0, zeta_at_obs, 2.0])),
            "depth": ("node", np.array([5.0, 10.0, 20.0])),
            "x": ("node", np.array([-90.0, -89.9, -89.8])),
            "y": ("node", np.array([29.0, 29.1, 29.2])),
        }
    )


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "files": {"run_folder": "/tmp/fake_run"},
            "adcirc": {
                "attempted_observation_location": {"value": [-89.9, 29.1]},
                "actual_observation_location": {"value": [0.0, 0.0]},
            },
        }
    )


def test_valid_value_passes(monkeypatch) -> None:
    monkeypatch.setattr(wrap, "xr_loader", lambda path: _synthetic_maxele(3.7))
    assert float(wrap.observe_max_point(_cfg())) == pytest.approx(3.7)


@pytest.mark.parametrize("bad", [NETCDF_FILL, -NETCDF_FILL, np.nan, np.inf, 1e5])
def test_bad_value_raises(monkeypatch, bad) -> None:
    monkeypatch.setattr(wrap, "xr_loader", lambda path: _synthetic_maxele(bad))
    with pytest.raises(ValueError, match="fake_run"):
        wrap.observe_max_point(_cfg())
