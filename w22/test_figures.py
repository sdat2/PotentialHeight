"""
Test solutions against Wang 2022 figure and paper data
(make sure that everything works as expected)
"""

import os
import shutil
import numpy as np
import pytest
import xarray as xr
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so tests can run headless
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults, label_subplots
from .constants import (
    DATA_PATH,
    FIGURE_PATH,
    TEMP_0K,
    GAS_CONSTANT,
    GAS_CONSTANT_FOR_WATER_VAPOR,
)
from .w22_carnot import wang_consts, wang_diff
from .ps import point_solution_ps
from .utils import buck_sat_vap_pressure
from .solve import bisection
from .utils import absolute_angular_momentum, carnot_efficiency

# need to compare w_pbl, wout, Q_s, -Q_gibbs, W_PBL surplus

# Resource guards for pytest.mark.skipif: the figure reproduction tests need
# the data digitized from the Wang (2022) figures, and test_figure_4 also
# needs GNU Octave to run the original CLE15 implementation (cle15.cle15m).
W22_DATA_PATH = os.path.join(DATA_PATH, "w22")
OCTAVE_MISSING = shutil.which("octave") is None
FIGURE_4_DATA_MISSING = [
    name
    for name in (
        "4a-cle15.csv",
        "4a-w22.csv",
        "4b-Wpbl.csv",
        "4b-Qs.csv",
        "4b-Qgibbs.csv",
        "4b-Wout.csv",
        "4b-Wpbl-surplus.csv",
    )
    if not os.path.exists(os.path.join(W22_DATA_PATH, name))
]
FIGURE_5_DATA_MISSING = [
    name
    for name in ("5a-vp-div-f.csv", "5a-vcarnot-div-f.csv")
    if not os.path.exists(os.path.join(W22_DATA_PATH, name))
]
RUN_SLOW_TESTS = os.environ.get("W22_SLOW_TESTS", "0").lower() in ("1", "true")


def w_out(
    coriolis_parameter=5e-5,
    outer_radius=2193_000,
    maximum_velocity=83,
    radius_of_maximum_winds=64_000,
    radius_of_outflow=np.inf,
):
    return (
        1 / 4 * (coriolis_parameter**2) * (outer_radius**2)
        - 0.5
        * coriolis_parameter
        * absolute_angular_momentum(
            maximum_velocity, radius_of_maximum_winds, coriolis_parameter
        )
        - 1
        / 2
        * (
            absolute_angular_momentum(0, outer_radius, coriolis_parameter) ** 2
            - absolute_angular_momentum(
                maximum_velocity, radius_of_maximum_winds, coriolis_parameter
            )
            ** 2
        )
        / (radius_of_outflow**2)
    )


def w_pbl(
    gas_constant=287,
    near_surface_air_temperature=299,
    dry_pressure_at_maximum_winds=944_00,
    dry_pressure_at_inflow=985_00,
    maximum_velocity=83,
):
    return (
        gas_constant
        * near_surface_air_temperature
        * np.log(dry_pressure_at_inflow / dry_pressure_at_maximum_winds)
        - 0.5 * maximum_velocity**2
    )


def q_gibbs(
    near_surface_air_temperature=299,
    dry_pressure_at_inflow=985_00,
    dry_pressure_at_maximum_winds=944_00,
    gas_constant=287,
):
    return (
        gas_constant
        * near_surface_air_temperature
        * buck_sat_vap_pressure(near_surface_air_temperature)
        / dry_pressure_at_maximum_winds
        * (-1 + np.log(dry_pressure_at_inflow / dry_pressure_at_maximum_winds))
    )


def w_p(
    coriolis_parameter=5e-5,
    gas_constant=287,
    beta_lift_parameterization=1.25,
    near_surface_air_temperature=299,
    outer_radius=2193_000,
    maximum_velocity=83,
    dry_pressure_at_inflow=985_00,
    radius_of_maximum_winds=64_000,
    dry_pressure_at_maximum_winds=944_00,
):
    # W_p = (beta-1) * (W_out + W_pbl)
    return (beta_lift_parameterization - 1) * (
        w_out(
            coriolis_parameter=coriolis_parameter,
            outer_radius=outer_radius,
            maximum_velocity=maximum_velocity,
            radius_of_maximum_winds=radius_of_maximum_winds,
        )
        + w_pbl(
            gas_constant=gas_constant,
            near_surface_air_temperature=near_surface_air_temperature,
            dry_pressure_at_inflow=dry_pressure_at_inflow,
            dry_pressure_at_maximum_winds=dry_pressure_at_maximum_winds,
            maximum_velocity=maximum_velocity,
        )
    )


def q_s(
    efficiency_relative_to_carnot: float = 0.5,
    near_surface_air_temperature: float = 299,
    outflow_temperature: float = 200,
    dry_pressure_at_inflow: float = 985_00,
    dry_pressure_at_maximum_winds: float = 944_00,
    gas_constant: float = 287,
    latent_heat_of_vaporization: float = 2_500_000,
    gas_constant_for_water_vapor: float = 461.5,
):
    # Q_s = W_p / (1 - efficiency)
    return (
        efficiency_relative_to_carnot
        * carnot_efficiency(near_surface_air_temperature, outflow_temperature)
        * (
            gas_constant
            * near_surface_air_temperature
            * np.log(dry_pressure_at_inflow / dry_pressure_at_maximum_winds)
            + latent_heat_of_vaporization
            * gas_constant
            / gas_constant_for_water_vapor
            / dry_pressure_at_maximum_winds
            * buck_sat_vap_pressure(near_surface_air_temperature)
        )
    )


"""def vp_no_dissipation(
    near_surface_air_temperature: float = 299,
    outflow_temperature: float = 200,
    ck_cd: float = 1,
    k_boundary_layer: float = 0.002,
    k_
):
"""


def v_carnot(
    efficiency_relative_to_carnot: float = 0.5,
    near_surface_air_temperature: float = 299,
    outflow_temperature: float = 200,
    latent_heat_of_vaporization: float = 2_500_000,
    gas_constant: float = 287,
    gas_constant_for_water_vapor: float = 461.5,
    pressure_dry_at_inflow: float = 985_00,
) -> float:
    """
    Calculate the carnot engine velocity approximated in Wang 2022.

    Args:
        efficiency_relative_to_carnot (float): The efficiency relative to Carnot.
        near_surface_air_temperature (float): The near surface air temperature in Kelvin.
        outflow_temperature (float): The outflow temperature in Kelvin.
        latent_heat_of_vaporization (float): The latent heat of vaporization in J/kg
        gas_constant (float): The gas constant for dry air in J/(kg*K).
        gas_constant_for_water_vapor (float): The gas constant for water vapor in J/(kg*K).
        pressure_dry_at_inflow (float): The dry pressure at inflow in
            Pascals (Pa).

    Returns:
        float: The velocity at which the Carnot efficiency is achieved in m/s.
    """
    return np.sqrt(
        (
            (
                efficiency_relative_to_carnot
                * carnot_efficiency(near_surface_air_temperature, outflow_temperature)
                * latent_heat_of_vaporization
                - gas_constant_for_water_vapor * near_surface_air_temperature
            )
            * gas_constant
            / gas_constant_for_water_vapor
            * buck_sat_vap_pressure(near_surface_air_temperature)
            / pressure_dry_at_inflow
        )
    )


@pytest.mark.skipif(
    OCTAVE_MISSING,
    reason="GNU Octave not found on PATH (required by cle15.cle15m.run_cle15)",
)
@pytest.mark.skipif(
    bool(FIGURE_4_DATA_MISSING),
    reason=f"Missing Wang (2022) digitized data files: {FIGURE_4_DATA_MISSING}",
)
@pytest.mark.skipif(
    not RUN_SLOW_TESTS,
    reason="Slow: ~95 Octave subprocess calls take minutes; set W22_SLOW_TESTS=1 "
    "to run (test_wang_2022_canonical_point covers the fast numeric check)",
)
def test_figure_4():
    from cle15.cle15m import run_cle15

    plot_defaults()
    # read csv test data
    cle15_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-cle15.csv"))
    w22_test = pd.read_csv(os.path.join(DATA_PATH, "w22", "4a-w22.csv"))
    # first column of each = x = radius [km]
    # second column of each = y = central pressure [mbar] (equiv to [hPa])
    # plot test data
    # plot_defaults()
    plt.plot(
        cle15_test.iloc[:, 0],
        cle15_test.iloc[:, 1],
        color="red",
        label="$p_m$ CLE15 from paper",
    )
    plt.plot(
        w22_test.iloc[:, 0],
        w22_test.iloc[:, 1],
        color="blue",
        label="$p_m$ W22 from paper",
    )
    plt.xlabel(r"Outer Radius, $\tilde{r}_a$ [km]")
    plt.ylabel("Pressure [mbar]")
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

    vsg = 83 / 1.2  # vmax = 83 m s-1, supergradient factor = 1.2
    pressure_assumption = "isothermal"  # "isopycnal"
    print("pressure_assumption", pressure_assumption)
    water_vapour_pressure = env_humidity * buck_sat_vap_pressure(
        near_surface_air_temperature
    )
    rho_air = (p_a * 100 - water_vapour_pressure) / (
        GAS_CONSTANT * near_surface_air_temperature
    ) + water_vapour_pressure / (
        GAS_CONSTANT_FOR_WATER_VAPOR * near_surface_air_temperature
    )
    print("rho_air", rho_air, "kg m-3")
    rho_air = 1.225  # kg m-3 # set back to default to match Wang 2022
    soln_ds = point_solution_ps(
        xr.Dataset(
            data_vars={
                "sst": 300 - TEMP_0K,  # deg C
                "supergradient_factor": 1.2,  # dimensionless
                "t0": 200,  #  K
                "w_cool": 0.002,  # m s-1
                "vmax": vsg,  # m s-1
                "msl": 1015,  # mbar
                "rho_air": rho_air,  # kg m-3
                "rh": 0.9,  # canonical humidity key (env_humidity also accepted as alias)
                "ck_cd": 1,  # canonical key (cd_ck alias also accepted since 2026-07-07; before that, cd_ck was silently ignored -> default 0.9)
                "cd": 0.0015,
            },
            coords={"lat": lat},
        ),
        pressure_assumption=pressure_assumption,
    )
    print(soln_ds)
    plt.plot(2193, 944, "+", label="Paper's solution")
    print("r0, 2193 km vs.", soln_ds.r0.values / 1000, "km")
    print("rmax, 64 km vs.", soln_ds.rmax.values / 1000, "km")
    print("pa, 1015 mbar vs.", soln_ds.msl.values, "mbar")
    print("pc, 944 mbar vs.", soln_ds.pc.values / 100, "mbar")
    print("pm, 944 mbar vs.", soln_ds.pm.values / 100, "mbar")

    plt.plot(
        soln_ds.r0.values / 1000,
        soln_ds.pm.values / 100,
        "x",
        color="green",
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
            rho0=rho_air,
            pressure_assumption=pressure_assumption,
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

    plt.plot(
        r0s / 1000, pm_cle15, "--", color="red", alpha=0.5, label="Our CLE15 $p_m$"
    )
    plt.plot(
        r0s / 1000, pc_cle15, "--", color="orange", alpha=0.5, label="Our CLE15 $p_c$"
    )

    r0s = w22_test.iloc[:, 0] * 1000  # convert to meters
    pm_w22 = []
    w_pbl_l = []
    q_s_l = []
    w_p_l = []
    q_gibbs_l = []
    w_out_l = []

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
            pressure_assumption=pressure_assumption,
        )
        # print(
        #    "r0", r0, "rmax", rmax_cle, "pm", pm / 100, "mbar", "pc", pc / 100, "mbar"
        # )
        assert rmax_cle > 1000  # should be in meters
        pressure_dry_at_inflow = 1015 * 100 - 0.9 * buck_sat_vap_pressure(299)
        yval = bisection(
            wang_diff(
                *wang_consts(
                    radius_of_max_wind=rmax_cle,
                    radius_of_inflow=r0,
                    maximum_wind_speed=vsg * 1.2,
                    coriolis_parameter=5e-5,
                    pressure_dry_at_inflow=pressure_dry_at_inflow,
                    near_surface_air_temperature=299,
                    outflow_temperature=200,
                )
            ),
            0.5,
            1.5,
            1e-6,
        )
        # convert solution to pressure
        pm_w22_car = pressure_dry_at_inflow / yval + buck_sat_vap_pressure(299)
        w_pbl_val = w_pbl(
            gas_constant=GAS_CONSTANT,
            near_surface_air_temperature=299,
            dry_pressure_at_maximum_winds=pm_w22_car,
            dry_pressure_at_inflow=pressure_dry_at_inflow,
            maximum_velocity=vsg * 1.2,
        )
        w_out_val = w_out(
            coriolis_parameter=5e-5,
            outer_radius=r0,
            maximum_velocity=vsg * 1.2,
            radius_of_maximum_winds=rmax_cle,
        )
        q_gibbs_val = -q_gibbs(
            near_surface_air_temperature=299,
            dry_pressure_at_inflow=pressure_dry_at_inflow,
            dry_pressure_at_maximum_winds=pm_w22_car,
            gas_constant=GAS_CONSTANT,
        )
        w_p_val = w_p(
            coriolis_parameter=5e-5,
            gas_constant=GAS_CONSTANT,
            beta_lift_parameterization=1.25,
            near_surface_air_temperature=299,
            outer_radius=r0,
            maximum_velocity=vsg * 1.2,
            dry_pressure_at_inflow=pressure_dry_at_inflow,
            radius_of_maximum_winds=rmax_cle,
            dry_pressure_at_maximum_winds=pm_w22_car,
        )
        q_s_val = q_s(
            efficiency_relative_to_carnot=0.5,
            near_surface_air_temperature=299,
            outflow_temperature=200,
            dry_pressure_at_inflow=pressure_dry_at_inflow,
            dry_pressure_at_maximum_winds=pm_w22_car,
            gas_constant=GAS_CONSTANT,
            latent_heat_of_vaporization=2_500_000,
            gas_constant_for_water_vapor=461.5,
        )
        w_pbl_l.append(w_pbl_val)
        q_s_l.append(q_s_val)
        w_p_l.append(w_p_val)
        q_gibbs_l.append(q_gibbs_val)
        w_out_l.append(w_out_val)

        print("rmax [m]", rmax_cle, "pm_w22_car [mbar]", pm_w22_car / 100, "mbar")
        pm_w22.append(pm_w22_car / 100)  # in hPa

    print("pm_w22", pm_w22)

    plt.plot(r0s / 1000, pm_w22, "--", alpha=0.5, color="blue", label="Our W22 $p_m$")
    # write a csv with the results
    os.makedirs(os.path.join(DATA_PATH, "w22"), exist_ok=True)
    pd.DataFrame(
        {
            "r0": r0s / 1000,
            "pm_w22": pm_w22,
        }
    ).to_csv(os.path.join(DATA_PATH, "w22", "4a-w22-ours.csv"), index=False)

    df = pd.DataFrame(
        {
            "r0": r0s / 1000,
            "w_pbl": w_pbl_l,
            "q_s": q_s_l,
            "w_p": w_p_l,
            "q_gibbs": q_gibbs_l,
            "w_out": w_out_l,
        }
    )
    df.to_csv(os.path.join(DATA_PATH, "w22", "4b-w22-ours.csv"), index=False)

    plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    plt.xlim(200, 5000)
    os.makedirs(os.path.join(FIGURE_PATH, "w22"), exist_ok=True)
    plt.savefig(os.path.join(FIGURE_PATH, "w22", "figure_4a.pdf"))
    plt.clf()
    plt.close()

    file_names = {
        "w_pbl": "4b-Wpbl",
        "q_s": "4b-Qs",
        "w_p": None,
        "q_gibbs": "4b-Qgibbs",
        "w_out": "4b-Wout",
        "w_pbl_surplus": "4b-Wpbl-surplus",
    }
    labels = {
        "w_pbl": "$W_{PBL}$",
        "q_s": "$Q_S$",
        "w_p": "$W_P$",
        "q_gibbs": "$-Q_{\mathrm{Gibbs}}$",
        "w_out": "$W_{\mathrm{out}}$",
        "w_pbl_surplus": "$W_{PBL}$ surplus",
    }
    colors = {
        "w_pbl": "red",
        "q_s": "green",
        "w_p": "orange",
        "q_gibbs": "purple",
        "w_out": "blue",
        "w_pbl_surplus": "grey",
    }

    for key, val in file_names.items():
        if val is not None:
            read_df = pd.read_csv(os.path.join(DATA_PATH, "w22", f"{val}.csv"))
            plt.plot(
                read_df["r0"],
                read_df.iloc[:, 1],
                label=labels[key],
                color=colors[key],
                alpha=0.5,
            )
        if key in df.columns:
            plt.plot(
                df["r0"],
                df[key],
                "--",
                label=f"Our {labels[key]}",
                color=colors[key],
                alpha=0.5,
            )
    plt.xlim(200, 5000)
    plt.xlabel(r"Outer Radius, $\tilde{r}_a$ [km]")
    plt.ylabel("W [J kg$^{-1}$]")
    plt.xlim(200, 5000)
    plt.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    plt.savefig(os.path.join(FIGURE_PATH, "w22", "figure_4b.pdf"))
    # df["r0"]
    plt.clf()
    plt.close()


def test_wang_2022_canonical_point():
    """Golden-value test for the canonical Wang (2022) example of figure 4a.

    Uses the pure python/numba CLE15 implementation (cle15.cle15n) via
    point_solution_ps, so it needs no Octave and no input data files.
    Inputs (as in test_figure_4): w_cool = 0.002 m s-1, supergradient
    factor = 1.2, vmax = 83 m s-1, T0 = 200 K, Ts = SST - 1 = 299 K,
    environmental humidity = 0.9, msl = 1015 mbar, rho_air = 1.225 kg m-3,
    fcor = 5e-5 s-1. Wang (2022) finds r0 = 2193 km, rmax = 64 km and
    pm = 944 mbar for this case.
    """
    omega = 2 * np.pi / (60 * 60 * 24)
    lat = np.degrees(np.arcsin(5e-5 / (2 * omega)))
    vsg = 83 / 1.2  # vmax = 83 m s-1, supergradient factor = 1.2
    soln_ds = point_solution_ps(
        xr.Dataset(
            data_vars={
                "sst": 300 - TEMP_0K,  # deg C
                "supergradient_factor": 1.2,  # dimensionless
                "t0": 200,  #  K
                "w_cool": 0.002,  # m s-1
                "vmax": vsg,  # m s-1
                "msl": 1015,  # mbar
                "rho_air": 1.225,  # kg m-3
                "rh": 0.9,  # canonical humidity key (env_humidity also accepted as alias)
                "ck_cd": 1,  # canonical key (cd_ck alias also accepted since 2026-07-07; before that, cd_ck was silently ignored -> default 0.9)
                "cd": 0.0015,
            },
            coords={"lat": lat},
        ),
        pressure_assumption="isothermal",
    )
    r0_km = float(soln_ds.r0.values) / 1000
    rmax_km = float(soln_ds.rmax.values) / 1000
    pm_mbar = float(soln_ds.pm.values) / 100
    pc_mbar = float(soln_ds.pc.values) / 100
    # (a) loose check against the values quoted in Wang (2022).
    assert np.isclose(r0_km, 2193, rtol=0.05), f"r0 {r0_km} km != W22 2193 km"
    assert np.isclose(rmax_km, 64, rtol=0.10), f"rmax {rmax_km} km != W22 64 km"
    assert np.isclose(pm_mbar, 944, atol=5), f"pm {pm_mbar} mbar != W22 944 mbar"
    # (b) tight regression pins on our numba solution to catch numerical drift.
    # Pins regenerated 2026-07-07 after fixing TWO silent input bugs:
    # (i) the rh inconsistency in the Carnot back-conversion (the y -> p_m
    #     numerator dropped the rh factor; w22_carnot.carnot_pm_from_y is now
    #     the single implementation), and
    # (ii) this test passed "cd_ck": 1, a key point_solution_ps silently
    #     ignored, so the old pins were computed at CkCd = 0.9 (the default),
    #     not the intended Wang canonical CkCd = 1.
    # History: original pins (both bugs) r0 2136.94, rmax 59.14, pm 940.44,
    # pc 880.84; fixing (i) alone at CkCd = 0.9 gives 2079.49 / 56.50 /
    # 940.90 / 881.38; fixing both gives the pins below — 2.9% under Wang's
    # 2193 km, back inside the original rtol = 0.05.
    assert np.isclose(r0_km, 2128.81, atol=2.0), f"r0 {r0_km} km drifted"
    assert np.isclose(rmax_km, 60.67, atol=0.5), f"rmax {rmax_km} km drifted"
    assert np.isclose(pm_mbar, 942.04, atol=0.5), f"pm {pm_mbar} mbar drifted"
    assert np.isclose(pc_mbar, 887.93, atol=0.5), f"pc {pc_mbar} mbar drifted"


@pytest.mark.skipif(
    bool(FIGURE_5_DATA_MISSING),
    reason=f"Missing Wang (2022) digitized data files: {FIGURE_5_DATA_MISSING}",
)
def test_figure_5():
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    plot_defaults()

    vp = 83 / 1.2  # m s-1
    v_car = v_carnot(
        efficiency_relative_to_carnot=0.5,
        near_surface_air_temperature=299,
        outflow_temperature=200,
        latent_heat_of_vaporization=2_500_000,
        gas_constant=287,
        gas_constant_for_water_vapor=461.5,
        pressure_dry_at_inflow=985_00,
    )
    print("v_car", v_car)
    print("vp", vp)

    fig, ax = plt.subplots()

    # column 1 = 1/f [s], column 2 = r0 [km]
    vpdiv_df = pd.read_csv(os.path.join(DATA_PATH, "w22", "5a-vp-div-f.csv"))
    ax.plot(
        vpdiv_df.iloc[:, 0],
        vpdiv_df.iloc[:, 1],
        color="grey",
        label=r"$v_p / f$ (W22)",
    )
    ax.plot(
        vpdiv_df.iloc[:, 0],
        vp * vpdiv_df.iloc[:, 0] / 1000,
        ":",
        color="grey",
        alpha=0.7,
        label=r"$V_p / f$ (ours)",
    )
    vcarnot_div_df = pd.read_csv(os.path.join(DATA_PATH, "w22", "5a-vcarnot-div-f.csv"))
    ax.plot(
        vcarnot_div_df.iloc[:, 0],
        vcarnot_div_df.iloc[:, 1],
        color="black",
        label=r"$v_{\mathrm{carnot}} / f$ (W22)",
    )
    ax.plot(
        vcarnot_div_df.iloc[:, 0],
        v_car * vcarnot_div_df.iloc[:, 0] / 1000,
        ":",
        color="black",
        alpha=0.7,
        label=r"$V_{\mathrm{carnot}} / f$ (ours)",
    )

    # Format x-axis: values in seconds (~5e3–8e4), display as ×10³ s
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e-3:.0f}"))
    ax.xaxis.set_major_locator(MultipleLocator(10_000))
    ax.set_xlabel(
        r"$f^{-1}$, larger at lower latitudes" "\n" r"$({\times}10^{3}\ \mathrm{s})$"
    )

    # Format y-axis: values in km (~350–6500), display as ×10³ km
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 1e-3:.1f}"))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.set_ylabel(r"Potential Size, $r_a$" "\n" r"$({\times}10^{3}\ \mathrm{km})$")

    ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.join(FIGURE_PATH, "w22"), exist_ok=True)
    fig.savefig(os.path.join(FIGURE_PATH, "w22", "figure_5a.pdf"), bbox_inches="tight")
    plt.clf()
    plt.close()


def octave_vs_python(add_name: str = ""):
    from cle15.cle15m import run_cle15 as run_cle15_octave
    from cle15.cle15 import run_cle15 as run_cle15_python
    from .constants import (
        RA_DEFAULT,
        BACKGROUND_PRESSURE,
        VMAX_DEFAULT,
        W_COOL_DEFAULT,
        CK_CD_DEFAULT,
        CD_DEFAULT,
        F_COR_DEFAULT,
        CDVARY_DEFAULT,
    )
    import time

    inputs = {
        "r0": RA_DEFAULT,
        "p0": BACKGROUND_PRESSURE / 100,
        "Vmax": VMAX_DEFAULT,
        "w_cool": W_COOL_DEFAULT,
        "fcor": F_COR_DEFAULT,
        "Cd": CD_DEFAULT,
        "CkCd": CK_CD_DEFAULT,
        "Cdvary": CDVARY_DEFAULT,
    }

    def compare_with_inp(inp: dict) -> dict:
        time_start = time.perf_counter()
        out_octave = run_cle15_octave(inputs=inp, pressure_assumption="isothermal")
        print(f"Octave output, {out_octave}")
        time_end = time.perf_counter()
        time_octave = time_end - time_start
        print(f"Octave time {time_octave:.3f} s")
        time_start = time.perf_counter()

        out_python = run_cle15_python(inputs=inp, pressure_assumption="isothermal")
        time_end = time.perf_counter()
        time_python = time_end - time_start
        print(f"Python time {time_python:.3f} s")
        print("Octave output", out_octave)
        print("Python output", out_python)
        return {
            "time_octave": time_octave,
            "time_python": time_python,
            "pm_octave": out_octave[0] / 100,
            "pm_python": out_python[0] / 100,
            "rmax_octave": out_octave[1] / 1000,
            "rmax_python": out_python[1] / 1000,
            "pc_octave": out_octave[2] / 100,
            "pc_python": out_python[2] / 100,
        }

    out_d = {}
    for r0 in np.linspace(500 * 1000, 5000 * 1000, 200):
        inputs["r0"] = r0
        out_d[r0] = compare_with_inp(inputs)

    df = pd.DataFrame.from_dict(out_d, orient="index")
    df = df.rename_axis("r0").reset_index()
    os.makedirs(os.path.join(DATA_PATH, "cle15"), exist_ok=True)
    if add_name != "":
        add_name = f"-{add_name}"
    df.to_csv(
        os.path.join(DATA_PATH, "cle15", f"octave-vs-python{add_name}.csv"), index=False
    )

    make_matlab_v_python_plot(add_name=add_name)


def make_matlab_v_python_plot(add_name: str = ""):
    # read csv test data
    if add_name != "" and add_name[0] != "-":
        add_name = f"-{add_name}"
    df = pd.read_csv(
        os.path.join(DATA_PATH, "cle15", f"octave-vs-python{add_name}.csv")
    )
    fig, axs = plt.subplots(1, 3, sharex=True)
    axs[0].plot(
        df["r0"] / 1000,
        df["pm_octave"],
        label="Octave",
        color="red",
    )
    axs[0].plot(
        df["r0"] / 1000,
        df["pm_python"],
        label="Python",
        color="blue",
    )
    axs[0].set_ylabel("Pressure, $p_m$ [hPa]")
    axs[0].set_xlabel("Outer Radius, $r_a$ [km]")
    # axs[0].legend()
    axs[1].plot(
        df["r0"] / 1000,
        df["rmax_octave"],
        label="Octave",
        color="red",
    )
    axs[1].plot(
        df["r0"] / 1000,
        df["rmax_python"],
        label="Python",
        color="blue",
    )
    axs[1].set_ylabel("Radius of max wind, $r_{\mathrm{max}}$ [m]")
    axs[1].set_xlabel("Outer Radius, $r_a$ [km]")
    axs[2].plot(
        df["r0"] / 1000,
        df["pc_octave"],
        label="Octave",
        color="red",
    )
    axs[2].plot(
        df["r0"] / 1000,
        df["pc_python"],
        label="Python",
        color="blue",
    )
    axs[2].set_ylabel("Central pressure, $p_c$ [hPa]")
    axs[2].set_xlabel("Outer Radius, $r_a$ [km]")

    label_subplots(axs, override="outside")
    avg_time_octave = df["time_octave"].mean()
    sem_time_octave = df["time_octave"].std() / np.sqrt(len(df))
    avg_time_python = df["time_python"].mean()
    sem_time_python = df["time_python"].std() / np.sqrt(len(df))
    # plot title over whole figure
    fig.suptitle(
        f"Octave (red, {avg_time_octave:.3f} ± {sem_time_octave:.3f} s) vs. Python (blue, {avg_time_python:.3f} ± {sem_time_python:.3f} s)"
    )

    os.makedirs(os.path.join(FIGURE_PATH, "cle15"), exist_ok=True)

    plt.savefig(
        os.path.join(FIGURE_PATH, "cle15", f"octave-vs-python{add_name}.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


# if __name__ == "__main__":
# python -m w22.test_figures --name archer2-slurm
# test_figure_4()
# test_figure_5()
# import argparse

# parser = argparse.ArgumentParser(description="Process some inputs.")
# parser.add_argument("--name", type=str, help="Input for octave_vs_python")
# args = parser.parse_args()
# octave_vs_python(args.name)


if __name__ == "__main__":
    # python -m w22.test_figures

    print(v_carnot())
    test_wang_2022_canonical_point()
    if not FIGURE_5_DATA_MISSING:
        test_figure_5()
    if os.path.exists(os.path.join(DATA_PATH, "cle15", "octave-vs-python.csv")):
        make_matlab_v_python_plot()
