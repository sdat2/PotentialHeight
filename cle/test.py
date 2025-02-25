"""
Test solutions against Wang 2022 figure and paper data
(make sure that everything works as expected)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults
from .constants import (
    DATA_PATH,
    FIGURE_PATH,
    TEMP_0K,
    GAS_CONSTANT,
    GAS_CONSTANT_FOR_WATER_VAPOR,
)
from .potential_size import run_cle15, wang_consts, wang_diff
from .ps import point_solution_ps
from .utils import buck_sat_vap_pressure
from .solve import bisection


def test_figure_4():
    plot_defaults()
    # read csv test data
    cle15_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-cle15.csv"))
    w22_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-w22.csv"))
    # first column of each = x = radius [km]
    # second column of each = y = central pressure [mbar] (equiv to [hPa])
    # plot test data
    # plot_defaults()
    plt.plot(cle15_test.iloc[:, 0], cle15_test.iloc[:, 1], label="CLE15 from paper")
    plt.plot(w22_test.iloc[:, 0], w22_test.iloc[:, 1], label="W22 from paper")
    plt.xlabel(r"Outer Radius, $\tilde{r}_a$ [km]")
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

    vsg = 83 / 1.2

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
                "vmax": vsg,  # m s-1
                "msl": 1015,  # mbar
                "rho_air": rho_air,  # kg m-3
                "env_humidity": 0.9,
                "cd_ck": 1,
                "cd": 0.0015,
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

    plt.plot(
        soln_ds.r0.values / 1000,
        soln_ds.pm.values / 100,
        marker="x",
        label="Our bisection point solution",
    )

    r0s = cle15_test.iloc[:, 0] * 1000  # convert to meters
    pm_cle15 = []
    pc_cle15 = []
    rmax_cle15 = []
    for r0 in r0s:
        pm_cle, rmax_cle, pc_cle = run_cle15(
            plot=False,
            inputs={
                "r0": r0,
                "Vmax": vsg,
                "w_cool": 0.002,
                "fcor": 5e-5,
                "p0": 1015,
                "Cd": 0.0015,
                "CkCd": 1,
            },
        )
        pm_cle15.append(pm_cle / 100)
        pc_cle15.append(pc_cle / 100)
        rmax_cle15.append(rmax_cle / 1000)
    pd.DataFrame(
        {
            "r0": r0s / 1000,
            "pm_cle15": pm_cle15,
            "pc_cle15": pc_cle15,
        }
    ).to_csv(os.path.join(DATA_PATH, "w22", "4a-cle15-ours.csv"), index=False)

    print("pm_cle15 [mbar]", pm_cle15)
    print("pc_cle15 [mbar]", pc_cle15)
    print("rmax_cle15 [km]", rmax_cle15)

    plt.plot(r0s / 1000, pm_cle15, label="Our CLE15 $p_m$")
    plt.plot(r0s / 1000, pc_cle15, label="Our CLE15 $p_c$")

    r0s = w22_test.iloc[:, 0] * 1000  # convert to meters
    pm_w22 = []

    for r0 in r0s:
        pm, rmax_cle, pc = run_cle15(
            inputs={
                "r0": r0,
                "Vmax": vsg,
                "w_cool": 0.002,
                "fcor": 5e-5,
                "p0": 1015,
                "Cd": 0.0015,
                "CkCd": 1,
            },
            rho0=rho_air,
        )
        print(
            "r0", r0, "rmax", rmax_cle, "pm", pm / 100, "mbar", "pc", pc / 100, "mbar"
        )
        assert rmax_cle > 1000  # should be in meters
        yval = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=vsg * 1.2,
                    coriolis_parameter=5e-5,
                    pressure_dry_at_inflow=1015 * 100
                    - 0.9 * buck_sat_vap_pressure(299),
                    near_surface_air_temperature=299,
                    outflow_temperature=200,
                )
            ),
            0.5,
            1.5,
            1e-6,
        )
        # convert solution to pressure
        pm_w22_car = (
            1015 * 100 - buck_sat_vap_pressure(299)
        ) / yval + buck_sat_vap_pressure(299)
        print("rmax", rmax_cle, "pm_w22_car", pm_w22_car / 100, "mbar")
        pm_w22.append(pm_w22_car / 100)

    print("pm_w22", pm_w22)

    plt.plot(r0s / 1000, pm_w22, label="Our W22 $p_m$")
    # write a csv with the results
    os.makedirs(os.path.join(DATA_PATH, "w22"), exist_ok=True)
    pd.DataFrame(
        {
            "r0": r0s / 1000,
            "pm_w22": pm_w22,
        }
    ).to_csv(os.path.join(DATA_PATH, "w22", "4a-w22-ours.csv"), index=False)
    plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    plt.xlim(200, 5000)
    os.makedirs(os.path.join(FIGURE_PATH, "w22"), exist_ok=True)
    plt.savefig(os.path.join(FIGURE_PATH, "w22", "figure_4.pdf"))


if __name__ == "__main__":
    # python -m cle.test
    test_figure_4()
