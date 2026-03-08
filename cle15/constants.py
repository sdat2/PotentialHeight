"""
Constants for the cle15 package.

Re-exports all symbols from :mod:`w22.constants` so that the moved
cle15 modules can use ``from .constants import ...`` without change.

``SRC_PATH`` is *overridden* to point at this package's own directory
(i.e. ``cle15/``) so that :mod:`cle15.cle15m` can locate ``mcle/``
via ``os.path.join(SRC_PATH, 'mcle', ...)``.
"""

import pathlib as _pathlib

# Override SRC_PATH before the wildcard re-export so it shadows the w22 value.
SRC_PATH = _pathlib.Path(__file__).resolve().parent

# Re-export everything else verbatim from w22.constants.
from w22.constants import (  # noqa: F401, E402
    PROJECT_PATH,
    FIGURE_PATH,
    DATA_PATH,
    TMP_PATH,
    TMPS_PATH,
    TEMP_0K,
    LATENT_HEAT_OF_VAPORIZATION,
    GAS_CONSTANT_FOR_WATER_VAPOR,
    GAS_CONSTANT,
    BACKGROUND_PRESSURE,
    F_COR_DEFAULT,
    RHO_AIR_DEFAULT,
    DEFAULT_SURF_TEMP,
    NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,
    OUTFLOW_TEMPERATURE_DEFAULT,
    ENVIRONMENTAL_HUMIDITY_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    W_COOL_DEFAULT,
    RA_DEFAULT,
    VMAX_DEFAULT,
    ALPHA_EYE_DEFAULT,
    EYE_ADJ_DEFAULT,
    CDVARY_DEFAULT,
    CKCDVARY_DEFAULT,
    SUPERGRADIENT_FACTOR,
    BETA_LIFT_PARAMETERIZATION_DEFAULT,
    EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,
    PRESSURE_DRY_AT_INFLOW_DEFAULT,
    MAX_WIND_SPEED_DEFAULT,
    RADIUS_OF_INFLOW_DEFAULT,
    RADIUS_OF_MAX_WIND_DEFAULT,
    A_DEFAULT,
    B_DEFAULT,
    C_DEFAULT,
    LOWER_Y_WANG_BISECTION,
    UPPER_Y_WANG_BISECTION,
    W22_BISECTION_TOLERANCE,
)
