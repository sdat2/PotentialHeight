"""Unit tests for the NaN/fill-value guard in adforce.wrap.observe_max_point.

A NaN or netCDF fill value (e.g. 9.96921e36) at the observation node in
maxele.63.nc has two very different causes, which the guard must separate:

* the run succeeded but the node never wetted -> DryObservationPoint
  (a ValueError subclass; BO loops may score it as 0.0 m of surge), vs.
* the run crashed/was truncated, leaving fill values across the mesh ->
  AdcircRunFailure (NOT a ValueError, so it cannot be swallowed by callers
  catching the dry case and silently fed to the GP — this poisoned the
  no3d-mes-s42 BO run when a worker disk filled mid-campaign).

Uses a tiny synthetic xarray dataset and monkeypatches xr_loader, so no ADCIRC
output files are needed.
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


def _synthetic_maxele(zeta_at_obs: float, zeta_elsewhere=(1.0, 2.0)) -> xr.Dataset:
    """Three-node maxele.63.nc stand-in; node 1 is nearest the observation point."""
    return xr.Dataset(
        {
            "zeta_max": (
                "node",
                np.array([zeta_elsewhere[0], zeta_at_obs, zeta_elsewhere[1]]),
            ),
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
def test_dry_node_on_healthy_mesh_raises_dry(monkeypatch, bad) -> None:
    """Bad value at the obs node only (mesh mostly valid) -> dry point."""
    monkeypatch.setattr(wrap, "xr_loader", lambda path: _synthetic_maxele(bad))
    with pytest.raises(wrap.DryObservationPoint, match="fake_run"):
        wrap.observe_max_point(_cfg())


@pytest.mark.parametrize("bad", [NETCDF_FILL, np.nan])
def test_invalid_mesh_raises_run_failure(monkeypatch, bad) -> None:
    """Bad values across the whole mesh -> the run itself failed."""
    monkeypatch.setattr(
        wrap,
        "xr_loader",
        lambda path: _synthetic_maxele(bad, zeta_elsewhere=(bad, bad)),
    )
    with pytest.raises(wrap.AdcircRunFailure, match="fake_run"):
        wrap.observe_max_point(_cfg())


def test_run_failure_is_not_a_value_error() -> None:
    """A `except ValueError` (the dry-point catch in adbo.exp) must never
    swallow an infrastructure failure."""
    assert not issubclass(wrap.AdcircRunFailure, ValueError)
    assert issubclass(wrap.DryObservationPoint, ValueError)
