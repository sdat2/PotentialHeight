import numpy as np


def E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr=100000):
    # intialisation
    fcor = abs(fcor)
    M0 = 0.5 * fcor * r0**2  # M at outer radius

    drfracr0 = 0.001
    if r0 > 2500 * 1000 or r0 < 200 * 1000:
        drfracr0 /= 10  # Extra precision for very large or small storms

    # Either use default input or 1/drfracr0, whichver is smaller
    Nr = min(Nr, int(1 / drfracr0))

    rfracr0_max = 1  # [dimensionless]; start at r0, move radially inwards
    rfracr0_min = rfracr0_max - (Nr - 1) * drfracr0  # [dimensionless]
    rrfracr0 = np.linspace(rfracr0_min, rfracr0_max, Nr)  # [dimensionless] r/r0 vector
    MMfracM0 = np.full_like(rrfracr0, np.nan)  # [dimensionless] M/M0 vector
    MMfracM0[-1] = 1  # M/M0 = 1 at r/r0 = 1

    rrfracr0_temp = rrfracr0[-2]  # one step in from outer radius
    # d(M/M0)/d(r/r0)=0 at r/r0 = 1
    MfracM0_temp = MMfracM0[-1]
    MMfracM0[-2] = MfracM0_temp

    # Variable C_d parameters from Donelan et al. (2004)
    C_d_lowV = 6.2e-4
    V_thresh1 = 6  # m/s
    V_thresh2 = 35.4  # m/s
    C_d_highV = 2.35e-3
    linear_slope = (C_d_highV - C_d_lowV) / (V_thresh2 - V_thresh1)

    for ii in range(1, Nr - 1):
        # Calculate C_d varying with V, if desired
        if Cdvary == 1:
            # V_temp = (M0 / r0) * (MfracM0_temp / rrfracr0[-ii] - rrfracr0[-ii])
            V_temp = (M0 / r0) * (MfracM0_temp / rrfracr0_temp - rrfracr0_temp)

            if V_temp <= V_thresh1:
                C_d = C_d_lowV
            elif V_temp > V_thresh2:
                C_d = C_d_highV
            else:
                C_d = C_d_lowV + linear_slope * (V_temp - V_thresh1)

        # Calculate model parameter, gamma
        gamma = C_d * fcor * r0 / w_cool

        # Update M/M0
        dMfracM0_drfracr0_temp = (
            gamma
            * ((MfracM0_temp - rrfracr0[-ii] ** 2) ** 2)
            / (1 - rrfracr0[-ii] ** 2)
        )
        MfracM0_temp -= dMfracM0_drfracr0_temp * drfracr0
        MMfracM0[-ii - 1] = MfracM0_temp

    if True:
        import matplotlib.pyplot as plt

        plt.plot(rrfracr0, MMfracM0)
        plt.xlabel("rrfracr0")
        plt.ylabel("MMfracM0")
        plt.savefig("e04_r0input_MM0.pdf")

    return rrfracr0, MMfracM0


# Example usage:
# rrfracr0, MMfracM0 = E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr)
