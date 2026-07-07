"""CLE15 python package constants."""

import pathlib
import os

constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(SRC_PATH, "img"))
DATA_PATH = os.path.join(SRC_PATH, "data")
SUP_PATH = os.path.join(SRC_PATH, "sup")
TMP_PATH = os.path.join(DATA_PATH, "tmp")
os.makedirs(TMP_PATH, exist_ok=True)
TMPS_PATH = os.path.join(DATA_PATH, "tmps")
os.makedirs(TMPS_PATH, exist_ok=True)

# real physical constants
TEMP_0K = 273.15  # [K]
# Latent heat of vaporization of water [J/kg]. 2.5e6 J/kg is the value at 0 degC
# (2.257e6 J/kg would be the value at 100 degC; at the ~26 degC sea surface used
# here the physical value is ~2.44e6 J/kg). We keep 2.5e6 because the Wang (2022)
# coefficients (a, b, c = 0.062, 0.031, 0.008; see w22_carnot.wang_consts doctest)
# were derived with it — the derivation neglects the temperature variation of L_v.
LATENT_HEAT_OF_VAPORIZATION = 2_500_000  # [J/kg] (at 0 degC; matches Wang 2022)
GAS_CONSTANT_FOR_WATER_VAPOR = 461.5  # [J/kg/K]
GAS_CONSTANT = 287  # [J/kg/K] (dry air)


# could probably move some of these to a config file.

# General defaults
BACKGROUND_PRESSURE = 1016 * 100  # [Pa]
F_COR_DEFAULT = 5e-5  # [s-1] Coriolis parameter (~lat 20 deg)
RHO_AIR_DEFAULT = 1.225  # [kg m-3]

# temperatures
DEFAULT_SURF_TEMP = 299  # [K]
NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT = DEFAULT_SURF_TEMP  # [K]
OUTFLOW_TEMPERATURE_DEFAULT = 200  # [K]

ENVIRONMENTAL_HUMIDITY_DEFAULT = 0.9  # [dimensionless]

# CLE15
CK_CD_DEFAULT = (
    0.9  # [dimensionless], chosen to match standard potential intensity assumptions
)
CD_DEFAULT = 0.0015  # drag coefficient
W_COOL_DEFAULT = 0.002  # m s-1
RA_DEFAULT = 734735.8325246028  # outer radius [m].
VMAX_DEFAULT = 49.45  # [m s-1], maximum wind speed
ALPHA_EYE_DEFAULT = 0.5  # [dimensionless], influences real ps
EYE_ADJ_DEFAULT = 0
# [dimensionless] bool
# whether to use the eye adjustment
CDVARY_DEFAULT = 0  # do not vary cd
CKCDVARY_DEFAULT = 0  # do not vary ck/cd


# supergradient
SUPERGRADIENT_FACTOR = 1.2  # [dimensionless], influences real ps


# W22
BETA_LIFT_PARAMETERIZATION_DEFAULT = 1.25  # [dimensionless], influences real ps 5/4
EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT = 0.5  # [dimensionless], influences real ps
PRESSURE_DRY_AT_INFLOW_DEFAULT = 985_00  # [Pa]
# vp * gamma_sg
MAX_WIND_SPEED_DEFAULT = 83  # [m s-1]

# radii defaults for W22
RADIUS_OF_INFLOW_DEFAULT = 2193_000  # [m]
RADIUS_OF_MAX_WIND_DEFAULT = 64_000  # [m]


# W22 carnot engine functions (not used)
A_DEFAULT: float = 0.062
B_DEFAULT: float = 0.031
C_DEFAULT: float = 0.008

# WANG defaults
LOWER_Y_WANG_BISECTION = 0.3  # [dimensionless]
UPPER_Y_WANG_BISECTION = 1.5  # [dimensionless]
W22_BISECTION_TOLERANCE = 1e-6  # [dimensionless]


# Bisection for potential intensity
LOWER_RADIUS_BISECTION = 200_000  # [m] 200 km
UPPER_RADIUS_BISECTION = 5_000_000  # [m] 5000 km
# Stopping tolerance for the r0 bisection. NOTE the semantics: solve.bisection
# converges on the width of the x-bracket, NOT on |f(x)| — so this is a 1 METRE
# tolerance on the outer radius r0, not a pressure tolerance (the old name
# PRESSURE_DIFFERENCE_BISECTION_TOLERANCE and its "[mbar]" comment were wrong
# on both counts).
R0_BISECTION_TOLERANCE = 1  # [m] width of the r0 bracket at which to stop

from adforce.constants import NEW_ORLEANS, MIAMI, GALVERSTON, HONG_KONG, SHANGHAI, HANOI

OFFSET_D = {
    "galverston": {"point": GALVERSTON, "lon_offset": 0, "lat_offset": -0.9},
    "miami": {"point": MIAMI, "lon_offset": 0.2, "lat_offset": 0},
    "new_orleans": {"point": NEW_ORLEANS, "lon_offset": 0, "lat_offset": -1},
    # these ones are not checked
    "shanghai": {"point": SHANGHAI, "lon_offset": 0.5, "lat_offset": -0.3},
    "hong_kong": {"point": HONG_KONG, "lon_offset": 1, "lat_offset": -1},
    "hanoi": {"point": HANOI, "lon_offset": 2, "lat_offset": -0.3},
}
