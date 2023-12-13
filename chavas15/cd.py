def cd_donelan(C_d: float, V_temp: float):
    # Variable C_d parameters from Donelan et al. (2004)
    C_d_lowV = 6.2e-4  # [dimensionless]
    V_thresh1 = 6  # [m/s]
    V_thresh2 = 35.4  # [m/s]
    C_d_highV = 2.35e-3  # [dimensionless]
    linear_slope = (C_d_highV - C_d_lowV) / (V_thresh2 - V_thresh1)  # [s/m]
    # V_temp = (M0 / r0) * (MfracM0_temp / rrfracr0[-i] - rrfracr0[-i])
    if V_temp <= V_thresh1:
        C_d = C_d_lowV
    elif V_temp > V_thresh2:
        C_d = C_d_highV
    else:
        C_d = C_d_lowV + linear_slope * (V_temp - V_thresh1)
    return C_d
