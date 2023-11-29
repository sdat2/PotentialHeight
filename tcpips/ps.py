from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def g(y):
    return 5 * y - np.exp(y)


def w22_func(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:
        return np.exp(a * y + b * np.log(y) * y + c)

    return f


def wang_diff(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:
        return y - np.exp(a * y + b * np.log(y) * y + c)

    return f


def bisection(f: Callable, left: float, right: float, tol: float) -> float:
    # https://en.wikipedia.org/wiki/Root-finding_algorithms#Bisection_method
    fleft = f(left)
    fright = f(right)
    if fleft * fright > 0:
        print("Error: f(left) and f(right) must have opposite signs.")
        return None

    while fleft * fright < 0 and right - left > tol:
        mid = (left + right) / 2
        fmid = f(mid)
        if fleft * fmid < 0:
            right = mid
            fright = fmid
        else:
            left = mid
            fleft = fmid
    return (left + right) / 2


def plot_w22_func():
    f = w22_func()
    y = np.linspace(0.1, 21, 1000)
    plt.plot(y, y, label="y=y")
    plt.plot(y, f(y), label="y=exp(a*y + b*log(y)*y + c)")
    lower = bisection(wang_diff(), 0.01, 3, 1e-5)
    if lower is not None:
        print(lower, wang_diff()(lower))
    upper = bisection(wang_diff(), 6, 21, 1e-5)
    if upper is not None:
        print(upper, wang_diff()(upper))
    plt.plot(lower, f(lower), "o", label="lower, y={:.3f}".format(lower))
    plt.plot(upper, f(upper), "o", label="upper, y={:.3f}".format(upper))
    plt.legend()
    plt.show()


def buck(temp: float) -> float:  # temp in K -> saturation vapour pressure in Pa
    # https://en.wikipedia.org/wiki/Arden_Buck_equation
    temp = temp - 273.15
    return 0.61121 * np.exp((18.678 - temp / 234.5) * (temp / (257.14 + temp))) * 1000


def carnot(temp_hot: float, temp_cold: float) -> float:
    return (temp_hot - temp_cold) / temp_hot


def absolute_angular_momentum(v: float, r: float, f: float) -> float:
    return v * r + 0.5 * f * r**2


def a_param(
    near_surface_air_temperature: float,
    latent_heat_of_vaporization: float,
    gas_constant_for_water_vapor: float,
    beta_lift_parameterization,
    outflow_temperature: float,
    efficiency_relative_to_carnot: float,
    pressure_dry_at_inflow: float,
) -> float:
    near_surface_saturation_vapour_presure = buck(near_surface_air_temperature)
    carnot_efficiency = carnot(near_surface_air_temperature, outflow_temperature)
    return (
        near_surface_saturation_vapour_presure
        / pressure_dry_at_inflow
        * (
            efficiency_relative_to_carnot
            * carnot_efficiency
            * latent_heat_of_vaporization
            / gas_constant_for_water_vapor
            / near_surface_air_temperature
            - 1
        )
        / (
            (
                beta_lift_parameterization
                - efficiency_relative_to_carnot * carnot_efficiency
            )
        )
    )


def b_param(
    near_surface_air_temperature: float,
    beta_lift_parameterization,
    outflow_temperature: float,
    efficiency_relative_to_carnot: float,
    pressure_dry_at_inflow: float,
) -> float:
    near_surface_saturation_vapour_presure = buck(near_surface_air_temperature)
    carnot_efficiency = carnot(near_surface_air_temperature, outflow_temperature)
    return (
        near_surface_saturation_vapour_presure
        / pressure_dry_at_inflow
        / (
            beta_lift_parameterization
            - efficiency_relative_to_carnot * carnot_efficiency
        )
    )


def c_param(
    near_surface_air_temperature: float,
    gas_constant: float,
    coriolis_parameter: float,
    outflow_temperature: float,
    efficiency_relative_to_carnot: float,
    beta_lift_parameterization: float,
    maximum_wind_speed: float,
    radius_of_inflow: float,
    radius_of_outflow: float,
    absolute_angular_momentum_at_outflow: float,
    absolute_angular_momentum_at_vmax: float,
) -> float:
    carnot_efficiency = carnot(near_surface_air_temperature, outflow_temperature)
    return (
        beta_lift_parameterization
        * (
            0.5 * maximum_wind_speed**2
            - 0.25 * coriolis_parameter**2 * radius_of_inflow**2
            + (
                absolute_angular_momentum_at_outflow**2
                - absolute_angular_momentum_at_vmax**2
            )
            / radius_of_outflow**2
            + 0.5 * coriolis_parameter * absolute_angular_momentum_at_vmax
        )
        / (
            (
                beta_lift_parameterization
                - efficiency_relative_to_carnot * carnot_efficiency
            )
            * near_surface_air_temperature
            * gas_constant
        )
    )


# w_cool = 0.002 [m/s]
# relative_humidity_environment = 0.9 [dimensionless]
# supergradient_wind = 1.2 [dimensionless]
# r_0 = np.inf [m]
# P_da = 1014 [hPa]
# SST = 300 [K]
# Ts = 299 [K]
# To = 200 [K]
# rm = 64 [km]
# ra = 2193 km
# coriolis_parameter = 5e-5 [s^-1]
# efficiency_relative_to_carnot = 0.5 [dimensionless]
# beta_lift_parameterization = 5/4 [dimensionless]
# vmax = 83 [m/s]
# v_max = supergradient_wind * emanuel_vp


print("buck299", buck(299))
print("buck300", buck(300))

a = a_param(
    near_surface_air_temperature=299,
    latent_heat_of_vaporization=2.27e6,
    gas_constant_for_water_vapor=461,
    beta_lift_parameterization=1.25,
    outflow_temperature=200,
    efficiency_relative_to_carnot=0.5,
    pressure_dry_at_inflow=985 * 100,
)
print("a", a)

print("0.062/a", 0.062 / a)

b = b_param(
    near_surface_air_temperature=299,
    beta_lift_parameterization=1.25,
    outflow_temperature=200,
    efficiency_relative_to_carnot=0.5,
    pressure_dry_at_inflow=985 * 100,
)
print("b", b)

c = c_param(
    near_surface_air_temperature=299,
    gas_constant=287,
    coriolis_parameter=5e-5,
    outflow_temperature=200,
    efficiency_relative_to_carnot=0.5,
    beta_lift_parameterization=1.25,
    maximum_wind_speed=83,
    radius_of_inflow=2193 * 1000,
    radius_of_outflow=np.inf,
    absolute_angular_momentum_at_outflow=0,
    absolute_angular_momentum_at_vmax=absolute_angular_momentum(83, 64 * 1000, 5e-5),
    # 83 * 64 * 1000 + 0.5 * 5e-5 * (64 * 1000) ** 2,  # Vmax*rmax + f*rmax**2
)


print("c", c)


def abc(
    near_surface_air_temperature=299,  # K
    outflow_temperature=200,  # K
    latent_heat_of_vaporization=2.27e6,  # J/kg
    gas_constant_for_water_vapor=461,  # J/kg/K
    gas_constant=287,  # J/kg/K
    beta_lift_parameterization=1.25,  # dimensionless
    efficiency_relative_to_carnot=0.5,
    pressure_dry_at_inflow=985 * 100,
    coriolis_parameter=5e-5,
    maximum_wind_speed=83,
    radius_of_inflow=2193 * 1000,
    radius_of_max_wind=64 * 1000,
):
    absolute_angular_momentum_at_vmax = absolute_angular_momentum(
        maximum_wind_speed, radius_of_max_wind, coriolis_parameter
    )
    carnot_efficiency = carnot(near_surface_air_temperature, outflow_temperature)
    near_surface_saturation_vapour_presure = buck(near_surface_air_temperature)

    return (
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            * (
                efficiency_relative_to_carnot
                * carnot_efficiency
                * latent_heat_of_vaporization
                / gas_constant_for_water_vapor
                / near_surface_air_temperature
                - 1
            )
            / (
                (
                    beta_lift_parameterization
                    - efficiency_relative_to_carnot * carnot_efficiency
                )
            )
        ),
        (
            near_surface_saturation_vapour_presure
            / pressure_dry_at_inflow
            / (
                beta_lift_parameterization
                - efficiency_relative_to_carnot * carnot_efficiency
            )
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
                    - efficiency_relative_to_carnot * carnot_efficiency
                )
                * near_surface_air_temperature
                * gas_constant
            )
        ),
    )


print("abc", abc())
# Run TCPI -> get emanuel_vp

if __name__ == "__main__":
    # python ps.py
    #
    bi = bisection(g, 1, 10, 1e-5)
    print(bi, g(bi))
    bi = bisection(wang_diff(), 0.01, 3, 1e-5)
    if bi is not None:
        print(bi, wang_diff()(bi))

    bi = bisection(wang_diff(), 6, 21, 1e-5)
    if bi is not None:
        print(bi, wang_diff()(bi))
    plot_w22_func()
