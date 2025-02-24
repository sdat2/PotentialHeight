"""
Test solutions against Wang 2022s figure and paper data
(make sure that everything works as expected)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults
from .constants import DATA_PATH, TEMP_0K, GAS_CONSTANT, GAS_CONSTANT_FOR_WATER_VAPOR
from .ps import point_solution_ps
from .utils import buck_sat_vap_pressure


def test_figure_4():
    plot_defaults()
    # read csv test data
    cle15_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-cle15.csv"))
    w22_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-w22.csv"))
    # first column of each = x = radius [km]
    # second column of each = y = central pressure [mbar] (equiv to [hPa])
    # plot test data
    # plot_defaults()
    plt.plot(cle15_test.iloc[:, 0], cle15_test.iloc[:, 1], label="CLE15")
    plt.plot(w22_test.iloc[:, 0], w22_test.iloc[:, 1], label="W22")
    plt.xlabel("Outer Radius, $\tilde{r}_a$ [km]")
    plt.ylabel("Central pressure, $p_m$ [mbar]")
    plt.legend()
    # plt.show()
    # now calculate the same pressure curves based on the same radii
    # and compare the results
    # w_cool = 0.002 m s-1, gamma_supergradient_factor = 1.2, r_0 = inf km
    # enviromental humidity = 0.9, T0 = 200 K, Ts  = SST - 1 = 299 K
    # solution ends up at exactly 2193 km pa=944 mbar

    print(
        "pressure inflow",
        985,
        "mbar",
        1015 - 0.9 * buck_sat_vap_pressure(299) / 100,
        "mbar",
    )
    omega = 2 * np.pi / (60 * 60 * 24)
    lat = np.degrees(np.arcsin(5e-5 / (2 * omega)))
    # print("lat", np.degrees(lat))
    # now test for that single value
    # by ideal gas law for dry air
    # rho = p / (Rd * T)
    # assume dry air
    p_a = 1015  # mbar
    env_humidity = 0.9
    near_surface_air_temperature = 299  # K

    water_vapour_pressure = env_humidity * buck_sat_vap_pressure(
        near_surface_air_temperature
    )
    rho_air = (p_a * 100 - water_vapour_pressure) / (
        GAS_CONSTANT * near_surface_air_temperature
    ) + water_vapour_pressure / (
        GAS_CONSTANT_FOR_WATER_VAPOR * near_surface_air_temperature
    )
    print("rho_air", rho_air, "kg m-3")
    soln_ds = point_solution_ps(
        xr.Dataset(
            data_vars={
                "sst": 300 - TEMP_0K,  # deg C
                "supergradient_factor": 1.2,  # dimensionless
                "t0": 200,  # K
                "w_cool": 0.002,  # m s-1
                "vmax": 83,  # m s-1
                "msl": 1015,  # mbar
                # "rho_air": rho,  # kg m-3
                "env_humidity": 0.9,
            },
            coords={"lat": lat},
        ),
        match_center=False,
    )
    print(soln_ds)
    print("r0, 2193 km", soln_ds.r0.values / 1000, "km")
    print("rmax, 64 km", soln_ds.rmax.values / 1000, "km")
    print("pa, 1015 mbar", soln_ds.msl.values, "mbar")
    print("pc, 944 mbar", soln_ds.pc.values / 100, "mbar")
    print("pm, 944 mbar", soln_ds.pm.values / 100, "mbar")


if __name__ == "__main__":
    # python -m cle.test
    test_figure_4()
