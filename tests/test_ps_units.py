"""Unit-consistency regression tests for the potential-size (w22) path.

Written 2026-07-07 alongside the fix of the humidity inconsistency in the
Wang (2022) Carnot back-conversion: the y-solve received the rh-corrected
dry inflow pressure while the y -> p_m conversion silently reverted to
rh = 1 in its numerator (three copy-pasted sites in w22/ps.py). The solved
potential size was effectively the rh = 1 answer regardless of the input
humidity (canonical point: r0 = 2136.9 km "at rh = 0.9" vs the consistent
2079.5 km; the fixed solver's rh = 1.0 answer is 2142.0 km).

These tests pin the single shared implementation (w22_carnot.carnot_pm_from_y)
against the hand formula, assert the physically required humidity
monotonicity of the solver, and guard the input-key plumbing.
"""

import numpy as np
import pytest
import xarray as xr

from w22.constants import TEMP_0K
from w22.utils import buck_sat_vap_pressure
from w22.w22_carnot import carnot_pm_from_y


def test_carnot_pm_from_y_matches_hand_formula():
    """p_m = p_da / y + e_sat(T): same p_da as the solve, saturation at the eyewall."""
    t_ns = 299.0  # [K]
    e_sat = buck_sat_vap_pressure(t_ns)  # [Pa]
    for rh in (0.7, 0.9, 1.0):
        p_da = 1015e2 - rh * e_sat  # [Pa] ambient dry partial pressure
        for y in (1.01, 1.05, 1.2):
            expected = p_da / y + e_sat
            assert carnot_pm_from_y(y, p_da, t_ns) == pytest.approx(
                expected, rel=1e-12
            )


def test_carnot_pm_from_y_uses_supplied_dry_pressure():
    """The regression that would have caught the bug: p_m must move 1:1 (in 1/y)
    with the supplied dry inflow pressure — i.e. the conversion cannot secretly
    rebuild its own numerator from rh = 1."""
    t_ns = 299.0
    e_sat = buck_sat_vap_pressure(t_ns)
    y = 1.05
    pm_rh09 = carnot_pm_from_y(y, 1015e2 - 0.9 * e_sat, t_ns)
    pm_rh10 = carnot_pm_from_y(y, 1015e2 - 1.0 * e_sat, t_ns)
    # difference must be exactly (1 - 0.9) * e_sat / y, not zero
    assert (pm_rh09 - pm_rh10) == pytest.approx(0.1 * e_sat / y, rel=1e-9)


def _canonical_inputs(rh: float) -> xr.Dataset:
    omega = 2 * np.pi / (60 * 60 * 24)
    lat = float(np.degrees(np.arcsin(5e-5 / (2 * omega))))
    return xr.Dataset(
        data_vars={
            "sst": 300 - TEMP_0K,  # degC
            "supergradient_factor": 1.2,  # dimensionless
            "t0": 200,  # K
            "w_cool": 0.002,  # m/s
            "vmax": 83 / 1.2,  # m/s (gradient-level)
            "msl": 1015,  # hPa
            "rho_air": 1.225,  # kg m-3
            "rh": rh,  # fraction
            "ck_cd": 1,  # canonical key (cd_ck alias also accepted since 2026-07-07; before that, cd_ck was silently ignored -> default 0.9)
            "cd": 0.0015,
        },
        coords={"lat": lat},
    )


@pytest.mark.slow
def test_potential_size_monotonic_in_humidity():
    """Drier environment -> smaller dry inflow pressure -> smaller potential
    storm. Under the pre-2026-07-07 bug the solver was almost insensitive to
    rh (it effectively solved the rh = 1 problem for every input)."""
    from w22.ps import point_solution_ps

    r0 = {}
    for rh in (0.8, 0.9, 1.0):
        out = point_solution_ps(_canonical_inputs(rh), pressure_assumption="isothermal")
        r0[rh] = float(out.r0.values)
    assert r0[0.8] < r0[0.9] < r0[1.0]
    # the spread must be macroscopic (the buggy solver gave ~0 spread):
    assert (r0[1.0] - r0[0.8]) > 50_000  # > 50 km across rh 0.8 -> 1.0


@pytest.mark.slow
def test_env_humidity_alias_equals_rh_key():
    """point_solution_ps accepts 'rh' (canonical) and 'env_humidity' (alias);
    before 2026-07-07 the alias was silently ignored."""
    from w22.ps import point_solution_ps

    ds_rh = _canonical_inputs(0.8)
    ds_alias = ds_rh.rename({"rh": "env_humidity"})
    out_rh = point_solution_ps(ds_rh, pressure_assumption="isothermal")
    out_alias = point_solution_ps(ds_alias, pressure_assumption="isothermal")
    assert float(out_rh.r0.values) == pytest.approx(
        float(out_alias.r0.values), rel=1e-9
    )
