import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import pchip_interpolate
from sithom.time import timeit


@timeit
def ER11_radprof_raw(
    Vmax: float, r_in: float, rmax_or_r0: str, fcor: float, CkCd: float, rr_ER11
):
    fcor = abs(fcor)
    r_out = None

    if rmax_or_r0 == "r0":
        r0 = r_in
        # Assume V>>fr (ER11, e.q. 38)
        rmax_simple = ((0.5 * fcor * r0**2) / Vmax) * (
            (0.5 * CkCd) ** (1 / (2 - CkCd))
        )

        def equation(rmax_var):
            # Full solution for rmax-r0 relationship (ER11 Eq. 37 with M_max including fcor term).
            left_side = (
                (0.5 * fcor * r0**2) / (Vmax * rmax_var + 0.5 * fcor * rmax_var**2)
            ) ** (2 - CkCd)
            right_side = (2 * (r0 / rmax_var) ** 2) / (
                2 - CkCd + CkCd * (r0 / rmax_var) ** 2
            )
            return left_side - right_side

        rmax = fsolve(equation, rmax_simple)
        rmax = rmax[rmax < r0 and rmax > 0]
        r_out = rmax

    elif rmax_or_r0 == "rmax":
        rmax = r_in

    else:
        raise ValueError('rmax_or_r0 must be set to either "r0" or "rmax"')

    V_ER11 = (1 / rr_ER11) * (Vmax * rmax + 0.5 * fcor * rmax**2) * (
        (2 * (rr_ER11 / rmax) ** 2) / (2 - CkCd + CkCd * (rr_ER11 / rmax) ** 2)
    ) ** (1 / (2 - CkCd)) - 0.5 * fcor * rr_ER11
    V_ER11[rr_ER11 == 0] = 0

    if rmax_or_r0 == "r0":
        rmax_profile = rr_ER11[np.argmax(V_ER11)]
        r_out = rmax_profile

    elif rmax_or_r0 == "rmax":
        i_rmax = np.argmax(V_ER11)
        if True:
            import matplotlib.pyplot as plt
            plt.plot(rr_ER11[i_rmax + 1 :], V_ER11[i_rmax + 1 :])
            print(rr_ER11[i_rmax + 1 :])
            plt.xlabel("rr_ER11")
            plt.ylabel("V_ER11")
            plt.savefig("er11_raw_test.pdf")
            plt.clf()
            plt.plot(rr_ER11[i_rmax + 1 :-1] - rr_ER11[i_rmax + 2 :])
            plt.savefig("er11_raw_test2.pdf")
            plt.clf()
        r0_profile = pchip_interpolate(V_ER11[i_rmax + 1 :][::-1], rr_ER11[i_rmax + 1 :][::-1], [0])
        r_out = r0_profile

    return V_ER11, r_out


# Example usage
# V_ER11, r_out = ER11_radprof_raw(Vmax_ER11, r0_ER11, 'r0', fcor, CkCd, rr_mean)
