"""Consistency guards for the thermodynamic formulas duplicated across packages.

The repo (deliberately or historically) contains three saturation-vapour-pressure
approximations:

1. ``tcpips.rh.saturation_pressure`` -- Magnus-Tetens (T in degC -> hPa).
2. ``w22.utils.buck_sat_vap_pressure`` -- Arden Buck (T in K -> Pa).
3. The Bolton (1980) form inlined in ``w22.utils.qair2rh``
   (``6.112 * exp(17.67 * T / (T + 243.5))``, T in degC -> hPa).

They are not identical, but they must stay close: relative humidity and moist
air density feed the potential-intensity/size pipeline, so silent drift between
these formulas would make results depend on which code path computed them.
These tests pin the current sub-percent agreement so any future edit that
widens the spread fails loudly. See the cross-referencing comments at each
definition site.
"""

import numpy as np

from tcpips.rh import saturation_pressure
from w22.constants import TEMP_0K
from w22.utils import buck_sat_vap_pressure, rho_air_f


def bolton_sat_vap_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    """Bolton (1980) form as inlined in w22.utils.qair2rh (keep in sync)."""
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def test_saturation_vapour_pressure_formulas_agree() -> None:
    """The three es approximations agree to <1% over 0-40 degC.

    Tolerance is deliberately tight (1%): at the time of writing the maximum
    pairwise spread over this range is ~0.5%, so a failure here means one
    of the three implementations has actually been changed.
    """
    temps_c = np.linspace(0.0, 40.0, 81)  # tropical-relevant range

    es_magnus_hpa = np.array([saturation_pressure(t) for t in temps_c])
    es_buck_hpa = (
        np.array([buck_sat_vap_pressure(t + TEMP_0K) for t in temps_c]) / 100.0
    )
    es_bolton_hpa = bolton_sat_vap_pressure_hpa(temps_c)

    for name_a, a, name_b, b in [
        ("magnus", es_magnus_hpa, "buck", es_buck_hpa),
        ("magnus", es_magnus_hpa, "bolton", es_bolton_hpa),
        ("buck", es_buck_hpa, "bolton", es_bolton_hpa),
    ]:
        rel = np.max(np.abs(a - b) / b)
        assert rel < 0.01, (
            f"{name_a} vs {name_b} saturation vapour pressure diverge by "
            f"{rel:.2%} (>1%) over 0-40 degC; the duplicated formulas in "
            "tcpips.rh / w22.utils have drifted apart."
        )


def test_sat_vap_pressure_reference_values() -> None:
    """Pin es at 0/20/30 degC against standard reference values (hPa)."""
    # Reference: ~6.11 hPa at 0C, ~23.4 hPa at 20C, ~42.4 hPa at 30C
    for temp_c, expected_hpa in [(0.0, 6.11), (20.0, 23.4), (30.0, 42.4)]:
        es_hpa = buck_sat_vap_pressure(temp_c + TEMP_0K) / 100.0
        assert abs(es_hpa - expected_hpa) / expected_hpa < 0.01, (
            f"buck_sat_vap_pressure({temp_c} degC) = {es_hpa:.2f} hPa, "
            f"expected ~{expected_hpa} hPa"
        )


def test_moist_air_density_bounds() -> None:
    """rho_air_f: dry-air standard value, and moisture always lowers density."""
    rho_dry = rho_air_f(1013.25, 288.15, 0.0)
    assert abs(rho_dry - 1.225) < 0.001  # ISA sea-level density

    # tropical case: 30 degC, 80% RH
    t_k = 30.0 + TEMP_0K
    wvp_pa = 0.8 * buck_sat_vap_pressure(t_k)
    rho_moist = rho_air_f(1013.25, t_k, wvp_pa)
    assert rho_moist < rho_air_f(1013.25, t_k, 0.0)  # moisture lightens air
    assert 1.10 < rho_moist < 1.20  # sanity band for warm moist surface air
