"""Some functions for the Wang 2022 "Tropical Cyclone Potential Size" sub-Carnot engine model."""

from typing import Callable, Tuple
import numpy as np
from .constants import (
    F_COR_DEFAULT,
    LATENT_HEAT_OF_VAPORIZATION,
    GAS_CONSTANT_FOR_WATER_VAPOR,
    GAS_CONSTANT,
    BETA_LIFT_PARAMETERIZATION_DEFAULT,
    EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,
    PRESSURE_DRY_AT_INFLOW_DEFAULT,
    MAX_WIND_SPEED_DEFAULT,
    RADIUS_OF_MAX_WIND_DEFAULT,
    RADIUS_OF_INFLOW_DEFAULT,
    NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,
    OUTFLOW_TEMPERATURE_DEFAULT,
)
from .utils import (
    absolute_angular_momentum,
    carnot_efficiency,
    buck_sat_vap_pressure,
)


def wang_diff(
    a: float = 0.062, b: float = 0.031, c: float = 0.008
) -> Callable[[float], float]:
    """
    Wang et al. 2022 difference function to find roots of.

    Args:
        a (float, optional): a. Defaults to 0.062.
        b (float, optional): b. Defaults to 0.031.
        c (float, optional): c. Defaults to 0.008.

    Returns:
        Callable[[float], float]: Function to find root of.

    Example::
        >>> f = wang_diff(a=0.062, b=0.031, c=0.008)
        >>> f"{f(1.081):.3f}"
        '0.000'
        >>> f"{f(19.1829):.3f}" # root said to be around 18 in paper. Is 19.18 close enough?
        '0.000'
    """

    def f(y: float) -> float:
        # y = exp(a*y + b*log(y)*y + c)
        return y - np.exp(a * y + b * np.log(y) * y + c)

    return f


def wang_consts(
    near_surface_air_temperature: float = NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,  # K
    outflow_temperature: float = OUTFLOW_TEMPERATURE_DEFAULT,  # K
    latent_heat_of_vaporization: float = LATENT_HEAT_OF_VAPORIZATION,  # J/kg
    gas_constant_for_water_vapor: float = GAS_CONSTANT_FOR_WATER_VAPOR,  # J/kg/K
    gas_constant: float = GAS_CONSTANT,  # J/kg/K
    beta_lift_parameterization: float = BETA_LIFT_PARAMETERIZATION_DEFAULT,  # dimensionless
    efficiency_relative_to_carnot: float = EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,  # dimensionless
    pressure_dry_at_inflow: float = PRESSURE_DRY_AT_INFLOW_DEFAULT,  # Pa
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
    maximum_wind_speed: float = MAX_WIND_SPEED_DEFAULT,  # m/s
    radius_of_inflow: float = RADIUS_OF_INFLOW_DEFAULT,  # m
    radius_of_max_wind: float = RADIUS_OF_MAX_WIND_DEFAULT,  # m
) -> Tuple[float, float, float]:  # a, b, c
    """
    Wang 2022 sub-Carnot engine model parameters. This function calculates the constants that are then used to calculate the difference function whose root we want to find.

    Args:
        near_surface_air_temperature (float, optional): Defaults to 299 [K].
        outflow_temperature (float, optional): Defaults to 200 [K].
        latent_heat_of_vaporization (float, optional): Defaults to 2.27e6 [J/kg].
        gas_constant_for_water_vapor (float, optional): Defaults to 461 [J/kg/K].
        gas_constant (float, optional): Defaults to 287 [J/kg/K].
        beta_lift_parameterization (float, optional): Defaults to 1.25 [dimesionless].
        efficiency_relative_to_carnot (float, optional): Defaults to 0.5 [dimensionless].
        pressure_dry_at_inflow (float, optional): Defaults to 985 * 100 [Pa].
        coriolis_parameter (float, optional): Defaults to 5e-5 [s-1].
        maximum_wind_speed (float, optional): Defaults to 83 [m/s].
        radius_of_inflow (float, optional): Defaults to 2193 * 1000 [m].
        radius_of_max_wind (float, optional): Defaults to 64 * 1000 [m].

    Returns:
        Tuple[float, float, float]: a, b, c (equations 17--19 in Wang et al. 2022).

    # only works if you use dodgy value for constant of latent heat of vaporization of water (2_500_000 J kg-1)
    # and not the more standard value (2_268_000 J kg-1)
    # this is because at a point in the derivation they ignore temperature variations, and so pick to use values at 0C
    # 2_500_000 J kg-1 is the value at 0C (https://met.nps.edu/~bcreasey/mr3222/files/helpful/UnitsandConstantsUsefulInMeteorology-PSU.html)
    >>> a, b, c = wang_consts(near_surface_air_temperature=299, outflow_temperature=200, latent_heat_of_vaporization=2_500_000+ 0*2_268_000, gas_constant_for_water_vapor=461, gas_constant=287, beta_lift_parameterization=5/4, efficiency_relative_to_carnot=0.5, pressure_dry_at_inflow=985_00, coriolis_parameter=5e-5, maximum_wind_speed=83, radius_of_inflow=2193_000, radius_of_max_wind=64_000)
    >>> f"{c:.3f}"
    '0.008'
    >>> f"{b:.3f}"
    '0.031'
    >>> f"{a:.3f}"
    '0.062'
    """
    # a, b, c
    absolute_angular_momentum_at_vmax = absolute_angular_momentum(
        maximum_wind_speed, radius_of_max_wind, coriolis_parameter
    )
    carnot_eff = carnot_efficiency(near_surface_air_temperature, outflow_temperature)
    near_surface_saturation_vapour_presure = buck_sat_vap_pressure(
        near_surface_air_temperature
    )

    return (
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            * (
                efficiency_relative_to_carnot
                * carnot_eff
                * latent_heat_of_vaporization
                / gas_constant_for_water_vapor
                - near_surface_air_temperature
            )
            / (
                (
                    beta_lift_parameterization
                    - efficiency_relative_to_carnot * carnot_eff
                )
                * near_surface_air_temperature
            )
        ),
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            / (beta_lift_parameterization - efficiency_relative_to_carnot * carnot_eff)
        ),
        (
            beta_lift_parameterization
            * (
                0.5 * maximum_wind_speed**2
                - 0.25 * coriolis_parameter**2 * radius_of_inflow**2
                + 0.5 * coriolis_parameter * absolute_angular_momentum_at_vmax
            )
            / (
                (
                    beta_lift_parameterization
                    - efficiency_relative_to_carnot * carnot_eff
                )
                * near_surface_air_temperature
                * gas_constant
            )
        ),
    )


def wang_carnot_velocity(
    efficiency_relative_to_carnot: float = EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,  # dimensionless
    near_surface_air_temperature: float = NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,  # K
    outflow_temperature: float = OUTFLOW_TEMPERATURE_DEFAULT,  # K
    latent_heat_of_vaporization: float = LATENT_HEAT_OF_VAPORIZATION,  # J/kg
    gas_constant_for_water_vapor: float = GAS_CONSTANT_FOR_WATER_VAPOR,  # J/kg/K
    gas_constant: float = GAS_CONSTANT,  # J/kg/K
    pressure_dry_at_inflow: float = PRESSURE_DRY_AT_INFLOW_DEFAULT,  # Pa
) -> float:
    """
    Calculate the sub-Carnot velocity from the Wang 2022 sub-Carnot engine model.

    In equation 27 they make a series of assumptions, including that intensity does not signicantly change with potential size to get the formula.

    V_carnot**2 = (eta * epsilon_c * L_v - R_v * T_s) * q_va^{*}

    q_va^{star} = R/R_v (e*s)

    They interpret this as the largest possible wind speed that can be reached in the outflow at radius r_a.

    Args:
        efficiency_relative_to_carnot (float, optional): Efficiency relative to Carnot. Defaults
            to EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT.
        near_surface_air_temperature (float, optional): Near surface air temperature in Kelvin. Defaults to NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT.
        outflow_temperature (float, optional): Outflow temperature in Kelvin. Defaults to OUTFLOW_TEMPERATURE_DEFAULT.
        latent_heat_of_vaporization (float, optional): Latent heat of vaporization in J/kg. Defaults to LATENT_HEAT_OF_VAPORIZATION.
        gas_constant_for_water_vapor (float, optional): Gas constant for water vapor in J/kg/K. Defaults to GAS_CONSTANT_FOR_WATER_VAPOR.
        gas_constant (float, optional): Gas constant in J/kg/K. Defaults to GAS_CONSTANT.
        pressure_dry_at_inflow (float, optional): Dry air pressure at inflow in Pa. Defaults to PRESSURE_DRY_AT_INFLOW_DEFAULT.
    """
    q_va_star = (
        gas_constant
        / gas_constant_for_water_vapor
        * buck_sat_vap_pressure(near_surface_air_temperature)
        / pressure_dry_at_inflow
    )
    return np.sqrt(
        (
            efficiency_relative_to_carnot
            * carnot_efficiency(near_surface_air_temperature, outflow_temperature)
            * latent_heat_of_vaporization
            - near_surface_air_temperature * gas_constant_for_water_vapor
        )
        * q_va_star
    )


def wang_outer_radius_approx(
    efficiency_relative_to_carnot: float = EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT,  # dimensionless
    near_surface_air_temperature: float = NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT,  # K
    outflow_temperature: float = OUTFLOW_TEMPERATURE_DEFAULT,  # K
    latent_heat_of_vaporization: float = LATENT_HEAT_OF_VAPORIZATION,  # J/kg
    gas_constant_for_water_vapor: float = GAS_CONSTANT_FOR_WATER_VAPOR,  # J/kg/K
    gas_constant: float = GAS_CONSTANT,  # J/kg/K
    pressure_dry_at_inflow: float = PRESSURE_DRY_AT_INFLOW_DEFAULT,  # Pa
    coriolis_parameter: float = F_COR_DEFAULT,  # s-1
):
    """
    Calculate the length scale from the Wang 2022 sub-Carnot engine model. The factor of 2 is from Appendix B.

    Most of the time the length scale is referred to as V_carnot/f_cor, but there is a factor of 2 in derivation in Appendix B, so we include that here for completeness.

    Args:
        efficiency_relative_to_carnot (float, optional): Efficiency relative to Carnot. Defaults
            to EFFICIENCY_RELATIVE_TO_CARNOT_DEFAULT.
        near_surface_air_temperature (float, optional): Near surface air temperature in Kelvin. Defaults to NEAR_SURFACE_AIR_TEMPERATURE_DEFAULT.
        outflow_temperature (float, optional):
            Outflow temperature in Kelvin. Defaults to OUTFLOW_TEMPERATURE_DEFAULT.
        latent_heat_of_vaporization (float, optional): Latent heat of vaporization in J/kg. Defaults to LATENT_HEAT_OF_VAPORIZATION.
        gas_constant_for_water_vapor (float, optional):
            Gas constant for water vapor in J/kg/K. Defaults to GAS_CONSTANT_FOR_WATER_VAPOR.
        gas_constant (float, optional): Gas constant in J/kg/K. Defaults to GAS_CONSTANT.
        pressure_dry_at_inflow (float, optional): Dry air pressure at inflow in Pa. Defaults to PRESSURE_DRY_AT_INFLOW_DEFAULT.
        coriolis_parameter (float, optional): Coriolis parameter in s-1. Defaults to F_COR_DEFAULT.
    """
    return 2 * (
        wang_carnot_velocity(
            efficiency_relative_to_carnot,
            near_surface_air_temperature,
            outflow_temperature,
            latent_heat_of_vaporization,
            gas_constant_for_water_vapor,
            gas_constant,
            pressure_dry_at_inflow,
        )
        / coriolis_parameter
    )
