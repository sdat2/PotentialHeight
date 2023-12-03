import numpy as np
from scipy.interpolate import pchip_interpolate
from .radprof_raw import ER11_radprof_raw


def ER11_radprof(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11):
    dr = rr_ER11[1] - rr_ER11[0]

    # Call ER11_radprof_raw (This function needs to be defined in Python)
    V_ER11, r_out = ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11)

    # Calculate error in r_in
    if rmax_or_r0 == "rmax":
        drin_temp = r_in - rr_ER11[np.argmax(V_ER11)]
    elif rmax_or_r0 == "r0":
        drin_temp = r_in - pchip_interpolate(V_ER11[2:], rr_ER11[2:], 0)

    # Calculate error in Vmax
    dVmax_temp = Vmax - np.max(V_ER11)

    # Check if errors are too large and adjust accordingly
    r_in_save = r_in
    Vmax_save = Vmax

    n_iter = 0
    while abs(drin_temp) > dr / 2 or abs(dVmax_temp / Vmax_save) >= 1e-2:
        n_iter += 1
        if n_iter > 20:
            # Convergence not achieved, return NaNs
            return np.full_like(rr_ER11, np.nan), np.nan

        r_in += drin_temp

        while abs(dVmax_temp / Vmax) >= 1e-2:
            Vmax += dVmax_temp
            V_ER11, r_out = ER11_radprof_raw(
                Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11
            )
            Vmax_prof = np.max(V_ER11)
            dVmax_temp = Vmax_save - Vmax_prof

        V_ER11, r_out = ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11)
        Vmax_prof = np.max(V_ER11)
        dVmax_temp = Vmax_save - Vmax_prof
        if rmax_or_r0 == "rmax":
            drin_temp = r_in_save - rr_ER11[np.argmax(V_ER11 == Vmax_prof)]
        elif rmax_or_r0 == "r0":
            drin_temp = r_in_save - pchip_interpolate(V_ER11[2:], rr_ER11[2:], 0)

    return V_ER11, r_out


# Example usage
# V_ER11, r_out = ER11_radprof(Vmax_ER11, r0_ER11, 'r0', fcor, CkCd, rr_mean)

# Note: You will need to define the function `ER11_radprof_raw` in Python as well.
