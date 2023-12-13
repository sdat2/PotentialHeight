from typing import Tuple, Union
import numpy as np
from sithom.time import timeit
from typeguard import typechecked
from chavas15.cd import cd_donelan


@timeit
@typechecked
def E04_outerwind_r0input_nondim_MM0(
    r0: float,
    fcor: float,
    Cdvary: Union[int, bool],
    C_d: float,
    w_cool: float,
    Nr: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the outer wind profile for a given storm size, r0, using the E04 model.

    Args:
        r0 (float): Outer radius [m]
        fcor (float): Coriolis parameter [s^-1]
        Cdvary (bool): Whether to vary C_d with V or not.
        C_d (float): Drag coefficient [dimensionless]
        w_cool (float): Cooling rate [m/s]
        Nr (int, optional): Number of radii. Defaults to 10000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: rrfracr0, MMfracM0
    """
    # intialisation
    fcor = abs(fcor)  # [s^-1]
    M0 = 0.5 * fcor * r0**2  # M at outer radius

    drfracr0 = 0.001
    if r0 > 2500 * 1000 or r0 < 200 * 1000:
        print("Large or small storm detected. Increasing drfracr0.")
        drfracr0 /= 10  # Extra precision for very large or small storms

    # Either use default input or 1/drfracr0, whichver is smaller
    Nr = min(Nr, int(1 / drfracr0))  # [dimensionless]

    rfracr0_max = 1  # [dimensionless]; start at r0, move radially inwards
    rfracr0_min = rfracr0_max - (Nr - 1) * drfracr0  # [dimensionless]
    rrfracr0 = np.arange(
        rfracr0_min,
        rfracr0_max + drfracr0,
        drfracr0,
    )  # [dimensionless] r/r0 vector
    MMfracM0 = np.full_like(
        rrfracr0, np.nan
    )  # [dimensionless] M/M0 vector full of NaNs
    MMfracM0[-1] = 1  # M/M0 = 1 at r/r0 = 1

    rrfracr0_temp = rrfracr0[-2]  # one step in from outer radius
    # d(M/M0)/d(r/r0)=0 at r/r0 = 1
    MfracM0_temp = MMfracM0[-1]
    MMfracM0[-2] = MfracM0_temp  # go in one and copy value

    # Integrate inwards from r0 to obtain profile of M/M0 vs. r/r0
    for i in range(1, Nr - 3):
        # Calculate C_d varying with V, if desired
        if Cdvary:
            V_temp = (M0 / r0) * (MfracM0_temp / rrfracr0_temp - rrfracr0_temp)
            C_d = cd_donelan(C_d, V_temp)

        # Calculate model parameter, gamma
        gamma = C_d * fcor * r0 / w_cool  # [dimensionless]

        # Update dMfracM0/drfracr0 at next step inwards
        dMfracM0_drfracr0_temp = (
            gamma
            * ((MfracM0_temp - rrfracr0_temp**2) ** 2)
            / (1 - rrfracr0_temp**2)
        )

        # Integrate M/M0 radially inwards
        MfracM0_temp -= dMfracM0_drfracr0_temp * drfracr0

        # update r/r0 to follow M/M0
        rrfracr0_temp -= drfracr0  # dimensionless - move one step inwards
        # used in next round of integration

        # save updated values
        # Seeks to replicate MMfracM0(end-i-1) = MfracM0_temp; from matlab.
        # -1 = end, i = new index
        MMfracM0[-1 - i - 1] = MfracM0_temp

    if True:
        import matplotlib.pyplot as plt

        plt.plot(rrfracr0, MMfracM0, "blue")
        plt.plot(1, 1, "r*", label="M0")
        plt.xlabel("$r$/$r_0$ [dimensionless]")
        plt.ylabel("$M$/$M_0$ [dimensionless]")
        plt.title(
            "E04 outer wind model, r0={:.0f} km, wcool={:.3e}, fcor={:.2e}".format(
                r0 / 1000,
                w_cool,
                fcor,
            ),
        )
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig("test/e04_r0input_nondim.pdf")
        plt.close()

    return rrfracr0, MMfracM0


# Example usage:
# rrfracr0, MMfracM0 = E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr)
