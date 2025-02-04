"""CLE15 python package constants."""

import pathlib
import os

constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))
FIGURE_PATH = pathlib.Path(os.path.join(SRC_PATH, "img"))
DATA_PATH = os.path.join(SRC_PATH, "data")
SUP_PATH = os.path.join(SRC_PATH, "sup")

# real constants
TEMP_0K = 273.15  # [K]
LATENT_HEAT_OF_VAPORIZATION = 2.27e6  # [J/kg/K]
GAS_CONSTANT_FOR_WATER_VAPOR = 461  # [J/kg/K]
GAS_CONSTANT = 287  # [J/kg/K]

# could probably move some of these to a config file.
BACKGROUND_PRESSURE = 1015 * 100  # [Pa]
DEFAULT_SURF_TEMP = 299  # [K]
W_COOL_DEFAULT = 0.002  # [K m s-1]
RHO_AIR_DEFAULT = 1.15  # [kg m-3]
F_COR_DEFAULT = 5e-5  # [s-2]
SUPERGRADIENT_FACTOR = 1.2  # [dimensionless]

BETA_LIFT_PARAMETERIZATION_DEFAULT = 1.25  # [dimensionless]
EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT = 0.5  # [dimensionless]
PRESSURE_DRY_AT_INFLOW_DEFAULT = 985_00  # [Pa]

# vp * gamma_sg
MAX_WIND_SPEED_DEFAULT = 83  # [m s-1]

# radii
RADIUS_OF_INFLOW_DEFAULT = 2193_000  # [m]
RADIUS_OF_MAX_WIND_DEFAULT = 64_000  # [m]

# temperatures
NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT = 299  # [K]
OUTFLOW_TEMPERATURE_DEFAULT = 200  # [k]

LOWER_RADIUS_BISECTION = 200_000  # [m] 200 km
UPPER_RADIUS_BISECTION = 5_000_000  # [m] 5000 km
PRESSURE_DIFFERENCE_BISECTION_TOLERANCE = 1  # [mbar]

LOWER_Y_WANG_BISECTION = 0.3  # [dimensionless]
UPPER_Y_WANG_BISECTION = 1.2  # [dimensionless]
W22_BISECTION_TOLERANCE = 1e-6  # [dimensionless]


# W22 carnot engine functions
A_DEFAULT: float = 0.062
B_DEFAULT: float = 0.031
C_DEFAULT: float = 0.008
