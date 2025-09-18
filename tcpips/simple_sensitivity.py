"""Work out the sensitivity of potential intensity to changes in temperature."""

import numpy as np
import pandas as pd


def simple_sensitivity(
    delta_t: float = 1,
    k: float = 0.07,
    m: float = 1,
    avg_to_tropical_ocean: float = 0.8,
    T_s0=300,
    T_00=200,
) -> None:
    """
    Simple sensitivity analysis for the change in potential intensity
    based on the change in temperature and humidity.

    Args:
        delta_t (float): Change in average global surface temperature in Kelvin.
        k (float): Scaling factor for humidity change based on CC (default 0.07).
        m (float): Scaling factor for the change in outflow temperature (default 1).
        avg_to_tropical_ocean (float): Average temperature change
           for tropical ocean compared to global temperature (default 0.8).
    """
    delta_t_sst = (
        delta_t * avg_to_tropical_ocean
    )  # scale to tropical ocean average temperature change
    # let's start by doing a simple sensitivty for potential intensity
    T_s0 = 300  # K
    T_00 = 200  # K
    # approx as Bister 1998 and use 7% scaling for humidity CC per degree to scale enthalpy change
    # T_s = T_s0 + delta_t_sst  # K
    # T_0 = T_00 + m* delta_t_sst  # K

    vp1_div_vp0 = np.sqrt(
        (T_s0 - T_00 + (1 - m) * delta_t_sst)
        / (T_00 + m * delta_t_sst)
        * T_00
        / (T_s0 - T_00)
        * (1 + k * delta_t_sst)
    )
    print(
        f"vp1/vp0 = {vp1_div_vp0:.3f} for delta_t_sst = {delta_t_sst} K, T_s0 = {T_s0} K, T_00 = {T_00} K, m = {m}, n = {avg_to_tropical_ocean}, k = {k}"
    )
    print(f"fractional change = {(vp1_div_vp0 - 1)* 100:.3f}%")
    return T_s0, T_00, delta_t, m, avg_to_tropical_ocean, k, (vp1_div_vp0 - 1) * 100


# vp1/vp0 = 1.042 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = -1.7, k = 0.07
# fractional change = 4.221%
# vp1/vp0 = 1.038 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = -1, k = 0.07
# fractional change = 3.788%
# vp1/vp0 = 1.026 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1, k = 0.07
# fractional change = 2.557%
# vp1/vp0 = 1.024 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1.2, k = 0.07
# fractional change = 2.434%
# vp1/vp0 = 1.022 for delta_t = 0.8 K, T_s0 = 300 K, T_00 = 200 K, m = 1.5, k = 0.07
# fractional change = 2.250%


if __name__ == "__main__":
    # python -m tcpips.simple_sensitivity
    results_l = []
    results_l += [simple_sensitivity(1, k=0.07, m=-1.7)]
    results_l += [simple_sensitivity(1, k=0.07, m=-1)]
    results_l += [simple_sensitivity(1)]
    results_l += [simple_sensitivity(1, k=0.07, m=1.2)]
    results_l += [simple_sensitivity(1, k=0.07, m=1.5)]

    # simple_sensitivity(1, k=0.07, m=-1)
    # simple_sensitivity(1)
    # simple_sensitivity(1, k=0.07, m=1.2)
    # simple_sensitivity(1, k=0.07, m=1.5)

    pd.DataFrame(
        results_l,
        columns=[
            "\(T_{s0}\)",
            "\(T_{o0}\)",
            "\(\Delta T\)",
            "\(m\)",
            "\(n\)",
            "\(k\)",
            "\% change in \(V_p\)",
        ],
    ).to_latex("pi_sensitivity.tex", index=False, float_format="%.2f")
