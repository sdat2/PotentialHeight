"""Cd and Ck/Cd parameterisations"""


def cd_donelan(C_d: float, V_temp: float) -> float:
    # Variable C_d parameters from Donelan et al. (2004)
    C_d_lowV = 6.2e-4  # [dimensionless]
    V_thresh1 = 6  # [m/s]
    V_thresh2 = 35.4  # [m/s]
    C_d_highV = 2.35e-3  # [dimensionless]
    linear_slope = (C_d_highV - C_d_lowV) / (V_thresh2 - V_thresh1)  # [s/m]

    if V_temp <= V_thresh1:
        C_d = C_d_lowV
    elif V_temp > V_thresh2:
        C_d = C_d_highV
    else:
        C_d = C_d_lowV + linear_slope * (V_temp - V_thresh1)
    return C_d


def ck_cd(Vmax: float, CkCd: float, CkCdvary: bool) -> float:
    # Overwrite CkCd if want varying (quadratic fit to Vmax from Chavas et al. 2015)
    if CkCdvary:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst

    if CkCd > 1.9:
        CkCd = 1.9
        print("Ck/Cd is capped at 1.9 and has been set to this value.")

    return CkCd
