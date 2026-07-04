"""Size--intensity tradeoff curve r(V) from the potential size model.

The potential size model (Wang et al. 2022; see ``w22.ps``) has the maximum
azimuthal wind speed ``V`` as a free parameter, so for a fixed thermodynamic
environment it defines a tradeoff curve between intensity and size
(thesis ``pips:fig:v-tradeoff``): for each ``V`` there is a corresponding
potential (outer) size ``r0(V)`` and potential inner size ``rmax(V)``.

This module precomputes that curve on a grid of intensities and provides
interpolators plus per-intensity CLE15 profile generation, so that the
Bayesian optimization in ``adbo`` can treat intensity as an additional
degree of freedom while keeping every sampled storm exactly on the
tradeoff frontier predicted by the potential size model.

Velocity convention: everywhere in this module ``vmax`` is the maximum
azimuthal wind at the *gradient level* (as in ``w22.ps``). The Category-1
threshold (33 m/s) is a 10 m wind speed, so the default lower limit of the
curve is ``33 / v_reduc`` with ``v_reduc = 0.8`` (the same reduction factor
used when forcing ADCIRC in ``adforce``).

Example::

    python -m w22.tradeoff   # writes + prints a demo curve (synthetic env)

"""

import os
import warnings
from typing import Optional

import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator
from sithom.io import write_json
from sithom.misc import get_git_revision_hash
from sithom.time import time_stamp, timeit

from cle15.cle15n import profile_from_stats
from .constants import (
    DATA_PATH,
    PROJECT_PATH,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    W_COOL_DEFAULT,
    SUPERGRADIENT_FACTOR,
    ENVIRONMENTAL_HUMIDITY_DEFAULT,
)
from .ps import calculate_ps_ufunc
from .utils import coriolis_parameter_from_lat

# Category 1 threshold at 10 m height [m/s] (Saffir-Simpson).
CAT1_10M = 33.0
# Default gradient-wind to 10 m reduction factor (matches adforce v_reduc).
V_REDUC_DEFAULT = 0.8

CURVE_DIR = os.path.join(DATA_PATH, "curves")

# environment keys that must be provided (units as in w22.ps.calculate_ps_ufunc)
REQUIRED_ENV_KEYS = ("vmax", "msl", "sst", "t0", "lat", "rh")


def curve_path(name: str) -> str:
    """Canonical path for a stored tradeoff curve.

    Args:
        name (str): Curve name, e.g. "2025_new_orleans_r4i1p1f1".

    Returns:
        str: Path of the netCDF file under ``w22/data/curves``.
    """
    return os.path.join(CURVE_DIR, f"{name}.nc")


@timeit
def tradeoff_ds(
    env: dict,
    num: int = 40,
    v_grid: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """Evaluate the potential size model along a grid of intensities.

    Args:
        env (dict): Environmental scalars. Required keys: ``vmax``
            (potential intensity, gradient level, m/s), ``msl`` (ambient
            pressure, mbar), ``sst`` (degC), ``t0`` (outflow temperature, K),
            ``lat`` (degrees N), ``rh`` (0-1). Optional keys with defaults:
            ``ck_cd``, ``cd``, ``w_cool``, ``supergradient_factor``,
            ``pressure_assumption``, ``v_reduc``.
        num (int, optional): Number of intensity grid points. Defaults to 40.
        v_grid (np.ndarray, optional): Explicit gradient-level intensity grid
            [m/s]; overrides ``num``. Defaults to
            ``linspace(CAT1_10M / v_reduc, env["vmax"], num)``.

    Returns:
        xr.Dataset: Coordinates ``vmax`` (gradient wind, m/s); variables
            ``r0``, ``rmax`` [m] and ``pm``, ``pc`` [Pa]; the environment and
            provenance (git hash, timestamp) stored in ``attrs``.
    """
    missing = [k for k in REQUIRED_ENV_KEYS if k not in env]
    if missing:
        raise ValueError(f"tradeoff_ds: env missing required keys {missing}")

    v_reduc = float(env.get("v_reduc", V_REDUC_DEFAULT))
    v_pi = float(env["vmax"])
    v_lo = CAT1_10M / v_reduc
    if v_grid is None:
        if v_pi <= v_lo:
            raise ValueError(
                f"Potential intensity {v_pi} m/s (gradient) is below the "
                f"Category-1 lower limit {v_lo:.2f} m/s; no tradeoff curve."
            )
        v_grid = np.linspace(v_lo, v_pi, num)
    v_grid = np.asarray(v_grid, dtype=np.float64)

    r0 = np.full_like(v_grid, np.nan)
    pm = np.full_like(v_grid, np.nan)
    pc = np.full_like(v_grid, np.nan)
    rmax = np.full_like(v_grid, np.nan)
    for i, v in enumerate(v_grid):
        r0[i], pm[i], pc[i], rmax[i] = calculate_ps_ufunc(
            v,
            float(env["msl"]),
            float(env["sst"]),
            float(env["t0"]),
            float(env["lat"]),
            float(env["rh"]),
            float(env.get("ck_cd", CK_CD_DEFAULT)),
            float(env.get("cd", CD_DEFAULT)),
            float(env.get("w_cool", W_COOL_DEFAULT)),
            float(env.get("supergradient_factor", SUPERGRADIENT_FACTOR)),
            str(env.get("pressure_assumption", "isothermal")),
        )

    ds = xr.Dataset(
        data_vars={
            "r0": (["vmax"], r0, {"units": "m", "long_name": "potential size"}),
            "rmax": (
                ["vmax"],
                rmax,
                {"units": "m", "long_name": "potential inner size"},
            ),
            "pm": (
                ["vmax"],
                pm,
                {"units": "Pa", "long_name": "pressure at maximum winds"},
            ),
            "pc": (["vmax"], pc, {"units": "Pa", "long_name": "central pressure"}),
        },
        coords={
            "vmax": (
                ["vmax"],
                v_grid,
                {"units": "m s-1", "long_name": "maximum gradient wind speed"},
            )
        },
    )
    # store the environment + provenance so the curve is self-describing
    for key in REQUIRED_ENV_KEYS:
        ds.attrs[f"env_{key}"] = float(env[key])
    ds.attrs["env_v_reduc"] = v_reduc
    ds.attrs["env_pressure_assumption"] = str(
        env.get("pressure_assumption", "isothermal")
    )
    for key, default in [
        ("ck_cd", CK_CD_DEFAULT),
        ("cd", CD_DEFAULT),
        ("w_cool", W_COOL_DEFAULT),
        ("supergradient_factor", SUPERGRADIENT_FACTOR),
    ]:
        ds.attrs[f"env_{key}"] = float(env.get(key, default))
    try:
        ds.attrs["curve_calculated_at_git_hash"] = get_git_revision_hash(
            path=str(PROJECT_PATH)
        )
    except Exception:  # git not available (e.g. archived copy)
        pass
    ds.attrs["curve_calculated_at_time"] = time_stamp()
    return ds


def generate_curve(
    env: dict,
    name: str,
    num: int = 40,
    regenerate: bool = False,
) -> str:
    """Compute and store a tradeoff curve, skipping if it already exists.

    Args:
        env (dict): Environment (see :func:`tradeoff_ds`).
        name (str): Curve name (file stem under ``w22/data/curves``).
        num (int, optional): Intensity grid points. Defaults to 40.
        regenerate (bool, optional): Recompute even if the file exists.

    Returns:
        str: Path of the stored netCDF file.
    """
    path = curve_path(name)
    if os.path.exists(path) and not regenerate:
        print(f"generate_curve: {path} exists; pass regenerate=True to recompute.")
        return path
    ds = tradeoff_ds(env, num=num)
    os.makedirs(CURVE_DIR, exist_ok=True)
    ds.to_netcdf(path)
    print(f"generate_curve: wrote {path}")
    return path


class TradeoffCurve:
    """Interpolated size--intensity tradeoff curve with profile generation.

    Wraps the dataset produced by :func:`tradeoff_ds` with monotone (PCHIP)
    interpolators for ``r0(V)``, ``rmax(V)`` and ``pm(V)``, and generates the
    CLE15 wind/pressure profile for any intensity on the curve.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        v = np.asarray(ds["vmax"].values, dtype=np.float64)
        r0 = np.asarray(ds["r0"].values, dtype=np.float64)
        rmax = np.asarray(ds["rmax"].values, dtype=np.float64)
        pm = np.asarray(ds["pm"].values, dtype=np.float64)
        good = np.isfinite(v) & np.isfinite(r0) & np.isfinite(rmax) & np.isfinite(pm)
        dropped = int((~good).sum())
        if dropped:
            warnings.warn(
                f"TradeoffCurve: dropping {dropped}/{len(v)} non-finite curve "
                "points (potential size bisection failed there)."
            )
        v, r0, rmax, pm = v[good], r0[good], rmax[good], pm[good]
        if len(v) < 4:
            raise ValueError(
                f"TradeoffCurve: only {len(v)} valid points; need >= 4. "
                "Recompute the curve with a finer grid or check the environment."
            )
        order = np.argsort(v)
        v, r0, rmax, pm = v[order], r0[order], rmax[order], pm[order]
        if not np.all(np.diff(rmax) < 0):
            # the model predicts smaller cores for more intense storms;
            # non-monotonicity indicates solver noise worth knowing about
            warnings.warn(
                "TradeoffCurve: rmax(V) is not strictly decreasing; "
                "the curve may be noisy at this resolution."
            )
        self.ds = ds
        self.v = v
        self.v_min = float(v[0])
        self.v_max = float(v[-1])
        self._r0 = PchipInterpolator(v, r0)
        self._rmax = PchipInterpolator(v, rmax)
        self._pm = PchipInterpolator(v, pm)

    @classmethod
    def from_file(cls, path: str) -> "TradeoffCurve":
        """Load a stored curve (see :func:`generate_curve`)."""
        return cls(xr.open_dataset(path))

    def r0(self, vmax: float) -> float:
        """Potential (outer) size [m] at gradient-level intensity ``vmax``."""
        self._check_range(vmax)
        return float(self._r0(vmax))

    def rmax(self, vmax: float) -> float:
        """Potential inner size [m] at gradient-level intensity ``vmax``."""
        self._check_range(vmax)
        return float(self._rmax(vmax))

    def pm(self, vmax: float) -> float:
        """Pressure at maximum winds [Pa] at intensity ``vmax``."""
        self._check_range(vmax)
        return float(self._pm(vmax))

    def _check_range(self, vmax: float) -> None:
        if not (self.v_min - 1e-6 <= vmax <= self.v_max + 1e-6):
            raise ValueError(
                f"vmax={vmax} m/s outside tradeoff curve domain "
                f"[{self.v_min:.2f}, {self.v_max:.2f}] m/s (gradient wind)."
            )

    def profile(self, vmax: float, out_path: Optional[str] = None) -> dict:
        """CLE15 wind/pressure profile for a storm on the tradeoff curve.

        Args:
            vmax (float): Gradient-level intensity [m/s] within the curve
                domain.
            out_path (str, optional): If given, also write the profile as a
                JSON usable by ``adforce`` (same format as the profiles in
                ``w22/data``). Defaults to None.

        Returns:
            dict: Profile with keys ``rr`` [m], ``VV`` [m/s], ``p`` [hPa],
                ``rmax``, ``rmerge``, ``Vmerge`` (see
                ``cle15.cle15n.profile_from_stats``).
        """
        fcor = abs(float(coriolis_parameter_from_lat(self.ds.attrs["env_lat"])))
        profile = profile_from_stats(
            float(vmax),
            fcor,
            self.r0(vmax),
            float(self.ds.attrs["env_msl"]),  # hPa
            pressure_assumption=str(
                self.ds.attrs.get("env_pressure_assumption", "isothermal")
            ),
        )
        if out_path is not None:
            write_json(profile, out_path)
        return profile


if __name__ == "__main__":
    # python -m w22.tradeoff
    # Demo with a synthetic warm-pool environment (Wang 2022-like); real
    # curves should be generated from the PS pipeline's point_timeseries
    # environmental scalars for the site/year of interest.
    demo_env = {
        "vmax": 83.0,  # potential intensity, gradient level [m/s]
        "msl": 1015.0,  # [mbar]
        "sst": 26.85,  # [degC] (near-surface air temp 299 K with 1 K offset)
        "t0": 200.0,  # outflow temperature [K]
        "lat": 29.5,  # New Orleans-ish [degN]
        "rh": ENVIRONMENTAL_HUMIDITY_DEFAULT,
    }
    demo = tradeoff_ds(demo_env, num=11)
    curve = TradeoffCurve(demo)
    print(demo)
    for v_ in np.linspace(curve.v_min, curve.v_max, 5):
        print(
            f"V={v_:6.2f} m/s  r0={curve.r0(v_)/1000:8.1f} km  "
            f"rmax={curve.rmax(v_)/1000:7.1f} km  pm={curve.pm(v_)/100:7.1f} hPa"
        )
