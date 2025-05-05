# -*- coding: utf-8 -*-
"""
Python implementation of the Chavas et al. (2015) tropical cyclone radial
profile model, merging Emanuel (2004) and Emanuel & Rotunno (2011) solutions.

This script reads parameters from a configuration file and outputs the
radial profile of wind speed.

References:
  - Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for tropical
    cyclone size. Journal of the Atmospheric Sciences, 72(9), 3418-3434.
  - Emanuel, K. (2004). Tropical cyclone energetics and structure. In
    Atmospheric Turbulence and Mesoscale Meteorology (pp. 165-191).
    Cambridge University Press.
  - Emanuel, K., & Rotunno, R. (2011). Self-stratification of tropical
    cyclone outflow. Part I: Implications for storm structure. Journal
    of the Atmospheric Sciences, 68(10), 2236-2249.
"""
import sys
import numpy as np
import configparser
import warnings
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar

# --- Configuration ---
CONFIG_FILE = "cle15.ini"  # Default name, can be changed (e.g., 'cle15.ini')


# --- Constants & Helper Functions ---
def _calculate_donelan_cd(V):
    """
    Calculates drag coefficient (Cd) based on wind speed (V) using the
    piecewise linear fit from Donelan et al. (2004), as implemented in the
    MATLAB code.

    Args:
        V (float or np.ndarray): Wind speed (m/s).

    Returns:
        float or np.ndarray: Drag coefficient (dimensionless).
    """
    cd_low_v = 6.2e-4
    v_thresh1 = 6.0  # m/s
    v_thresh2 = 35.4  # m/s
    cd_high_v = 2.35e-3
    linear_slope = (cd_high_v - cd_low_v) / (v_thresh2 - v_thresh1)

    if isinstance(V, (int, float)):
        if V <= v_thresh1:
            return cd_low_v
        elif V > v_thresh2:
            return cd_high_v
        else:
            return cd_low_v + linear_slope * (V - v_thresh1)
    else:  # Assuming numpy array
        cd = np.full_like(V, cd_low_v, dtype=float)
        mask_high = V > v_thresh2
        mask_mid = (V > v_thresh1) & (V <= v_thresh2)
        cd[mask_high] = cd_high_v
        cd[mask_mid] = cd_low_v + linear_slope * (V[mask_mid] - v_thresh1)
        return cd


def _find_intersections(x1, y1, x2, y2, tol=1e-6):
    """
    Find intersection points of two curves defined by (x1, y1) and (x2, y2).
    This is a simplified numerical approach inspired by MATLAB's curveintersect.
    (Corrected Version)

    Args:
        x1, y1 (np.ndarray): Coordinates of the first curve.
        x2, y2 (np.ndarray): Coordinates of the second curve.
        tol (float): Tolerance for detecting zero crossings/roots.

    Returns:
        tuple: (intersection_x, intersection_y) as numpy arrays.
    """
    # Ensure x-arrays are sorted and unique for interpolation
    sort_idx1 = np.argsort(x1)
    x1_s, idx1 = np.unique(x1[sort_idx1], return_index=True)
    y1_s = y1[sort_idx1][idx1]

    sort_idx2 = np.argsort(x2)
    x2_s, idx2 = np.unique(x2[sort_idx2], return_index=True)
    y2_s = y2[sort_idx2][idx2]

    # Need at least 2 points for interpolation
    if len(x1_s) < 2 or len(x2_s) < 2:
        return np.array([]), np.array([])

    # Create interpolators
    # Use bounds_error=False and fill_value=np.nan to handle extrapolation
    try:
        # Use Pchip if enough points, else linear
        if len(x1_s) < 3:
            interp1 = np.interp  # Fallback to linear
            args1 = (x1_s, y1_s)
        else:
            interp1_obj = PchipInterpolator(x1_s, y1_s, extrapolate=False)
            interp1 = interp1_obj  # Use the object directly
            args1 = ()

        if len(x2_s) < 3:
            interp2 = np.interp
            args2 = (x2_s, y2_s)
        else:
            interp2_obj = PchipInterpolator(x2_s, y2_s, extrapolate=False)
            interp2 = interp2_obj
            args2 = ()

    except ValueError as e:
        warnings.warn(f"Interpolator creation failed in _find_intersections: {e}")
        return np.array([]), np.array([])

    # Define the difference function (for scalar input x)
    def diff_func(x_scalar):
        # Handle potential NaNs from interpolation outside bounds
        if (
            callable(interp1) and interp1.__name__ == "interp"
        ):  # Linear interpolation call check
            val1 = interp1(x_scalar, *args1)
        else:  # Pchip object call
            val1 = interp1(x_scalar, *args1)

        if callable(interp2) and interp2.__name__ == "interp":
            val2 = interp2(x_scalar, *args2)
        else:
            val2 = interp2(x_scalar, *args2)

        # root_scalar needs NaN to be handled; return large number if NaN?
        if np.isnan(val1) or np.isnan(val2):
            # Return a large residual if outside interpolation range
            return 1e20  # Or some indicator that root finding should stop
        return val1 - val2

    # Find intervals where intersection might occur
    # Use a common, fine grid within the overlapping range
    x_min = max(np.min(x1_s), np.min(x2_s))
    x_max = min(np.max(x1_s), np.max(x2_s))
    if x_min >= x_max:
        return np.array([]), np.array([])  # No overlap

    # Create a finer grid for checking sign changes
    num_points_eval = max(len(x1_s), len(x2_s)) * 10  # Increased density
    x_eval = np.linspace(x_min, x_max, num_points_eval)

    # Evaluate difference function element-wise on the grid
    y_diff = np.array([diff_func(x) for x in x_eval])

    # Find where the sign of the difference changes
    # Ignore large values indicating points outside interpolation range
    valid_indices = np.abs(y_diff) < 1e19
    x_eval_valid = x_eval[valid_indices]
    y_diff_valid = y_diff[valid_indices]

    if len(y_diff_valid) < 2:
        return np.array([]), np.array([])

    # Find indices *before* sign changes
    sign_changes = np.where(np.diff(np.sign(y_diff_valid)))[0]

    intersection_x = []
    intersection_y = []

    # Find roots within the intervals identified by sign changes
    for idx in sign_changes:
        x_interval = (x_eval_valid[idx], x_eval_valid[idx + 1])
        try:
            # Use root_scalar with the scalar diff_func
            sol = root_scalar(diff_func, bracket=x_interval, method="bisect", xtol=tol)
            if sol.converged:
                root_x = sol.root
                # Get y-value from one of the interpolators
                if callable(interp1) and interp1.__name__ == "interp":
                    root_y = float(interp1(root_x, *args1))
                else:
                    root_y = float(interp1(root_x, *args1))

                # Avoid adding duplicate roots very close to each other
                is_duplicate = False
                for existing_x in intersection_x:
                    # Check relative and absolute tolerance for duplicates
                    if np.isclose(existing_x, root_x, rtol=tol * 10, atol=tol * 100):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    intersection_x.append(root_x)
                    intersection_y.append(root_y)
        except ValueError as e:
            # This bracket might be invalid if diff_func returned large value, skip it
            # print(f"Skipping bracket {x_interval} due to ValueError: {e}")
            pass
        except Exception as e:
            warnings.warn(f"Root finding failed in bracket {x_interval}: {e}")

    return np.array(intersection_x), np.array(intersection_y)


# --- Core Model Functions ---


def e04_outer_wind_nondim(r0, fcor, cd_vary, c_d_const, w_cool, nr=None):
    """
    Calculates the non-dimensional outer wind profile (M/M0 vs r/r0) based on
    Emanuel (2004). Corresponds to E04_outerwind_r0input_nondim_MM0.m.

    Args:
        r0 (float): Outer radius where V=0 (m).
        fcor (float): Coriolis parameter (s^-1).
        cd_vary (bool): True if Cd varies with wind speed, False for constant Cd.
        c_d_const (float): Constant drag coefficient (used if cd_vary is False).
        w_cool (float): Radiative subsidence rate (m/s).
        nr (int, optional): Number of radial nodes inward from r0.
                             Defaults to a value based on drfracr0.

    Returns:
        tuple: (rr_frac_r0, mm_frac_m0)
               - rr_frac_r0 (np.ndarray): Dimensionless radius (r/r0).
               - mm_frac_m0 (np.ndarray): Dimensionless angular momentum (M/M0).
    """
    fcor = abs(fcor)
    m0 = 0.5 * fcor * r0**2

    # Determine grid spacing, finer for very large/small r0
    drfracr0 = 0.001
    if r0 > 2500e3 or r0 < 200e3:
        drfracr0 = drfracr0 / 10.0

    if nr is None:
        nr = int(1.0 / drfracr0)  # Default Nr based on step size
    elif nr > int(1.0 / drfracr0):
        # This warning was hit by user, keep it.
        warnings.warn(
            f"Requested Nr ({nr}) is too large for drfracr0 ({drfracr0}). Setting Nr to max possible ({int(1.0/drfracr0)})."
        )
        nr = int(1.0 / drfracr0)

    rfracr0_max = 1.0
    rfracr0_min = rfracr0_max - (nr - 1) * drfracr0
    # Ensure min radius is slightly above zero if nr is large enough
    rfracr0_min = max(rfracr0_min, drfracr0 / 2.0)

    rr_frac_r0 = np.arange(rfracr0_min, rfracr0_max + drfracr0 / 2.0, drfracr0)
    # Ensure the last point is exactly 1.0
    if not np.isclose(rr_frac_r0[-1], 1.0):
        # Append 1.0 and remove any element very close to 1.0 if it resulted from arange
        rr_frac_r0 = rr_frac_r0[~np.isclose(rr_frac_r0, 1.0)]
        rr_frac_r0 = np.append(rr_frac_r0, 1.0)

    nr_actual = len(rr_frac_r0)
    mm_frac_m0 = np.full(nr_actual, np.nan)

    # Boundary condition at r/r0 = 1
    mm_frac_m0[-1] = 1.0

    # First step inwards: d(M/M0)/d(r/r0) = 0 at r/r0=1 implies M/M0 is constant
    # for the first small step.
    if nr_actual > 1:
        mm_frac_m0[-2] = 1.0
        # Initialize values for the loop starting from the second point from end
        m_frac_m0_temp = mm_frac_m0[-2]
        r_frac_r0_temp = rr_frac_r0[-2]
    else:
        # Handle the case where nr=1 (only the r/r0=1 point)
        return rr_frac_r0, mm_frac_m0

    # Integrate inwards using Euler method (as in MATLAB code)
    # Loop from the third-to-last point backwards to the first point
    for ii in range(nr_actual - 3, -1, -1):
        c_d = c_d_const  # Start with constant Cd
        if cd_vary:
            # Calculate V at the *previous* (outer) point to determine Cd for this step
            # Avoid division by zero if r_frac_r0_temp is very small
            if r_frac_r0_temp > 1e-9:
                v_temp = (m0 / r0) * (m_frac_m0_temp / r_frac_r0_temp - r_frac_r0_temp)
                # Ensure V is not negative (can happen in theory near r0)
                v_temp = max(0, v_temp)
                c_d = _calculate_donelan_cd(v_temp)
            else:
                c_d = _calculate_donelan_cd(0.0)  # Use low-wind Cd if radius is tiny

        # Avoid division by zero if w_cool is zero or r_frac_r0_temp is 1
        if w_cool == 0:
            gamma = np.inf  # Or handle as a special case (no friction)
            warnings.warn("w_cool is zero, E04 gamma parameter is infinite.")
        else:
            gamma = c_d * fcor * r0 / w_cool  # Dimensionless parameter

        # Avoid division by zero if r_frac_r0_temp is close to 1
        denominator = 1.0 - r_frac_r0_temp**2
        if abs(denominator) < 1e-12:
            # At r/r0 = 1, the derivative should be 0. If numerically close, enforce it.
            dm_dr_temp = 0.0
        else:
            # Ensure M/M0 >= (r/r0)^2, as M = rV + 0.5*f*r^2 and V >= 0
            # This term is V/(0.5*f*r), which should be positive
            m_term = max(0.0, m_frac_m0_temp - r_frac_r0_temp**2)
            dm_dr_temp = gamma * (m_term**2) / denominator

        # Integrate: M_new = M_old - dM/dr * dr (moving inwards, dr is negative)
        # Our loop goes backwards, so drfracr0 is positive step size.
        # M(i) = M(i+1) - dM/dr(i+1) * drfracr0
        m_frac_m0_new = m_frac_m0_temp - dm_dr_temp * drfracr0

        # Update values for the next iteration (moving one step further in)
        r_frac_r0_temp = rr_frac_r0[ii]
        m_frac_m0_temp = m_frac_m0_new
        mm_frac_m0[ii] = m_frac_m0_temp

    # Ensure M/M0 does not go significantly below 0 due to numerical errors
    mm_frac_m0 = np.maximum(0, mm_frac_m0)

    return rr_frac_r0, mm_frac_m0


def er11_radprof_raw(vmax, r_in, rmax_or_r0, fcor, ck_cd, rr_eval):
    """
    Calculates the raw Emanuel and Rotunno (2011) radial wind profile
    without iterative convergence. Corresponds to ER11_radprof_raw.m.
    Uses numerical root finding to determine rmax or r0 if needed.

    Args:
        vmax (float): Maximum wind speed (m/s).
        r_in (float): Input radius (either rmax or r0) (m).
        rmax_or_r0 (str): Specifies if r_in is 'rmax' or 'r0'.
        fcor (float): Coriolis parameter (s^-1).
        ck_cd (float): Ratio of exchange coefficients (Ck/Cd).
        rr_eval (np.ndarray): Radii (m) at which to evaluate the profile.

    Returns:
        tuple: (v_er11, r_out)
               - v_er11 (np.ndarray): ER11 wind speeds (m/s) at rr_eval.
               - r_out (float): The other radius (rmax if r0 was input,
                                r0 if rmax was input), calculated numerically
                                or directly from the profile. Returns NaN if
                                calculation fails.
    """
    fcor = abs(fcor)

    # Avoid issues with Ck/Cd = 2
    if abs(ck_cd - 2.0) < 1e-9:
        warnings.warn("Ck/Cd is very close to 2.0, ER11 calculation might be unstable.")
        ck_cd = 2.0 + np.sign(ck_cd - 2.0) * 1e-9 if ck_cd != 2.0 else 2.00001

    rmax = np.nan
    r0 = np.nan
    r_out = np.nan

    # --- Determine rmax and r0 ---
    if rmax_or_r0.lower() == "r0":
        r0 = r_in
        # Need to find rmax corresponding to this r0 and Vmax.
        # ER11 Eq. 37 relates M0/Mm = ((0.5*f*r0^2) / (Vm*rm + 0.5*f*rm^2))
        # to r0/rm. Rearranging:
        # ( (0.5*f*r0^2) / (Vm*rm + 0.5*f*rm^2) )^(2-CkCd) =
        #      (2*(r0/rm)^2) / (2 - CkCd + CkCd*(r0/rm)^2)

        def rmax_eqn(rm):
            if rm <= 1e-3 or rm >= r0:  # Physical constraints, avoid rm=0
                return np.inf  # Return large value outside valid range
            mm = vmax * rm + 0.5 * fcor * rm**2
            m0_val = 0.5 * fcor * r0**2  # Renamed from m0 to avoid confusion
            if mm <= 0 or m0_val <= 0:  # Avoid log/power issues
                return np.inf

            # Calculate LHS safely
            try:
                # Check base for power
                base_lhs = m0_val / mm
                if base_lhs <= 0:
                    return np.inf
                lhs = base_lhs ** (2.0 - ck_cd)
            except (
                ValueError
            ):  # Handles complex numbers if base negative and power non-integer
                return np.inf

            r0_rm_sq = (r0 / rm) ** 2
            rhs_denom = 2.0 - ck_cd + ck_cd * r0_rm_sq
            if rhs_denom <= 0:  # Avoid division by zero/negative
                return np.inf
            rhs = (2.0 * r0_rm_sq) / rhs_denom
            return lhs - rhs

        # Find rmax numerically. Bracket should be (0, r0).
        # Start with a reasonable guess, e.g., based on simplified formula.
        # Simple guess (Eq 38 approx): rmax_guess = (0.5*fcor*r0**2 / vmax) * ( (0.5*ck_cd)**(1/(2-ck_cd)) ) if ck_cd != 2 else r0/2
        # Provide a generous bracket [small_value, r0 - small_value]
        rmax_bracket = (1e-3 * r0, 0.999 * r0)  # Avoid exactly 0 and r0
        try:
            # Check if function changes sign in bracket
            f_a = rmax_eqn(rmax_bracket[0])
            f_b = rmax_eqn(rmax_bracket[1])
            if np.sign(f_a) == np.sign(f_b):
                warnings.warn(
                    f"ER11 rmax_eqn function does not change sign over bracket {rmax_bracket}. Cannot find root. Check Vmax, r0, CkCd."
                )
                rmax = np.nan
            else:
                sol = root_scalar(
                    rmax_eqn, bracket=rmax_bracket, method="bisect", xtol=1e-6 * r0
                )
                if sol.converged:
                    rmax = sol.root
                else:
                    warnings.warn(
                        f"ER11 rmax numerical solve did not converge for r0={r0/1000:.1f} km, Vmax={vmax:.1f} m/s, Ck/Cd={ck_cd:.2f}."
                    )
                    rmax = np.nan  # Indicate failure
        except ValueError as e:
            warnings.warn(
                f"ER11 rmax numerical solve failed (ValueError: {e}) for r0={r0/1000:.1f} km, Vmax={vmax:.1f} m/s, Ck/Cd={ck_cd:.2f}. Check inputs/bracket."
            )
            rmax = np.nan  # Indicate failure

        r_out = rmax  # This is the value we calculated

    elif rmax_or_r0.lower() == "rmax":
        rmax = r_in
        # If rmax is input, r0 is determined directly from the calculated profile below.
        # The analytical solution for r0 given rmax is complex/implicit.
        # The MATLAB code seems to calculate r0 from the profile V=0 intercept.
        # So, r_out will be calculated after computing V_ER11.
    else:
        raise ValueError("rmax_or_r0 must be 'rmax' or 'r0'")

    # --- Calculate Profile ---
    v_er11 = np.full_like(rr_eval, np.nan, dtype=float)

    if np.isnan(rmax):  # If rmax couldn't be determined
        warnings.warn("Cannot calculate ER11 profile because rmax is NaN.")
        return v_er11, np.nan

    # Calculate M at rmax
    mm = vmax * rmax + 0.5 * fcor * rmax**2

    # Avoid division by zero at r=0
    rr_eval_safe = np.maximum(rr_eval, 1e-9)  # Use a small positive radius instead of 0

    rr_rm = rr_eval_safe / rmax
    rr_rm_sq = rr_rm**2

    # Denominator term: 2 - CkCd + CkCd*(r/rm)^2
    denom_term = 2.0 - ck_cd + ck_cd * rr_rm_sq

    # Check for non-positive denominator which causes issues with the power
    valid_denom = denom_term > 1e-12  # Check denom is positive

    # Term inside the power: (2*(r/rm)^2) / (2 - CkCd + CkCd*(r/rm)^2)
    # This term represents M/Mm
    m_mm_term = np.full_like(rr_eval, np.nan, dtype=float)
    m_mm_term[valid_denom] = (2.0 * rr_rm_sq[valid_denom]) / denom_term[valid_denom]

    # Check for non-positive base for the power (1/(2-CkCd))
    valid_base = m_mm_term >= 0  # Allow base=0
    valid_indices = valid_denom & valid_base

    # Calculate the full M/Mm term including the power
    m_mm_pow = np.full_like(rr_eval, np.nan, dtype=float)
    if abs(2.0 - ck_cd) > 1e-9:  # Avoid division by zero if ck_cd == 2
        exponent = 1.0 / (2.0 - ck_cd)
        # Use np.power carefully for potentially negative bases if exponent is non-integer
        # However, m_mm_term should physically be >= 0
        # Handle base=0 separately if exponent is negative
        zero_base_idx = valid_indices & np.isclose(m_mm_term, 0.0)
        pos_base_idx = valid_indices & ~zero_base_idx

        if exponent < 0:
            m_mm_pow[pos_base_idx] = np.power(m_mm_term[pos_base_idx], exponent)
            m_mm_pow[zero_base_idx] = np.inf  # 0^(-ve) -> inf
        else:
            m_mm_pow[valid_indices] = np.power(m_mm_term[valid_indices], exponent)

    else:  # Special case Ck/Cd = 2 (limit) -> M/Mm = exp(1 - (rm/r)^2) ? No, formula structure changes.
        # The MATLAB code doesn't handle CkCd=2 specifically here, it might rely on numerical precision.
        # Let's return NaN or error for CkCd=2 for now.
        warnings.warn(
            "ER11 profile calculation for Ck/Cd exactly 2 is not explicitly handled."
        )
        # If we proceed, use a value very close to 2. (Handled by perturbation earlier)

    # Calculate V = (M / r) - 0.5 * f * r = (Mm * (M/Mm) / r) - 0.5 * f * r
    # V = (Mm / rmax) * (M/Mm / (r/rmax)) - 0.5 * f * rmax * (r/rmax)

    # Only calculate V where M/Mm could be calculated
    v_er11_calc = np.full_like(rr_eval, np.nan, dtype=float)
    # Avoid division by zero for rr_rm
    non_zero_rr_rm = rr_rm > 1e-12
    # Valid indices where M/Mm power calculation worked
    valid_pow = ~np.isnan(m_mm_pow) & ~np.isinf(m_mm_pow)

    valid_calc = valid_pow & non_zero_rr_rm

    v_er11_calc[valid_calc] = (mm / rmax) * (
        m_mm_pow[valid_calc] / rr_rm[valid_calc]
    ) - 0.5 * fcor * rr_eval_safe[valid_calc]

    # Handle r=0 explicitly: V(0) = 0
    zero_radius_idx = np.isclose(rr_eval, 0.0)
    v_er11_calc[zero_radius_idx] = 0.0

    # Assign calculated values, ensuring V doesn't go negative
    v_er11 = np.maximum(0.0, v_er11_calc)  # Physical constraint V>=0

    # Replace NaNs resulting from invalid base/denominator with 0 wind?
    # Or keep as NaN? Keep NaN to indicate calculation failure region.

    # --- Calculate r_out ---
    if rmax_or_r0.lower() == "rmax":
        # Find r0 by interpolating where V drops to 0
        # Apply the same robust interpolation logic as in er11_radprof
        r0_profile = np.nan
        valid_v = ~np.isnan(v_er11) & (
            rr_eval > max(rmax, 1e-6)
        )  # Look outside rmax, avoid r=0
        if np.any(valid_v):
            rr_outer = rr_eval[valid_v]
            vv_outer = v_er11[valid_v]

            # Check if wind speed drops to zero or below
            if np.min(vv_outer) <= 1e-3:  # Check if wind gets close to zero
                try:
                    # Sort by radius first
                    sort_r_idx = np.argsort(rr_outer)
                    rr_o_s = rr_outer[sort_r_idx]
                    vv_o_s = vv_outer[sort_r_idx]

                    # Find first index where V is near zero
                    zero_cross_idx = np.where(vv_o_s <= 1e-3)[0]
                    if len(zero_cross_idx) > 0:
                        first_zero_idx = zero_cross_idx[0]
                        if first_zero_idx > 0:  # Need point before zero crossing
                            last_pos_idx = first_zero_idx - 1
                            # --- Start r(V) interpolation ---
                            v_interp_segment = vv_o_s[last_pos_idx:]
                            r_interp_segment = rr_o_s[last_pos_idx:]
                            # Sort by V for monotonicity
                            sort_v_idx = np.argsort(v_interp_segment)
                            v_interp_sorted = v_interp_segment[sort_v_idx]
                            r_interp_sorted = r_interp_segment[sort_v_idx]
                            # Use unique V values
                            v_unique, unique_idx = np.unique(
                                v_interp_sorted, return_index=True
                            )
                            if len(v_unique) >= 2:
                                r_unique = r_interp_sorted[unique_idx]
                                try:
                                    if len(v_unique) >= 3:
                                        interp_r_of_v = PchipInterpolator(
                                            v_unique, r_unique, extrapolate=False
                                        )
                                        r0_prof_candidate = float(interp_r_of_v(0.0))
                                    else:  # Linear fallback
                                        r0_prof_candidate = float(
                                            np.interp(0.0, v_unique, r_unique)
                                        )

                                    # Check validity
                                    if (
                                        not np.isnan(r0_prof_candidate)
                                        and r0_prof_candidate > rmax
                                    ):
                                        r0_profile = r0_prof_candidate
                                    else:
                                        warnings.warn(
                                            f"ER11_raw r0 interpolation resulted in invalid value ({r0_prof_candidate/1000:.1f} km)."
                                        )
                                        # Leave r0_profile as NaN
                                except ValueError as e_interp:
                                    warnings.warn(
                                        f"ER11_raw r0 Pchip/linear interpolation failed: {e_interp}"
                                    )
                                    # Leave r0_profile as NaN
                            else:
                                warnings.warn(
                                    "ER11_raw could not find enough unique points in V near zero for r0 interpolation."
                                )
                        else:  # V is near zero at the first point outside rmax
                            r0_profile = rr_o_s[0]
                    # else: V never drops low enough

                except Exception as e:
                    warnings.warn(
                        f"ER11_raw r0 interpolation failed with exception: {e}"
                    )
                    # Leave r0_profile as NaN
            # else: Wind speed never drops near zero
            if np.isnan(r0_profile):
                warnings.warn(
                    f"ER11_raw V profile does not reach zero within the evaluated radius range (max r = {np.max(rr_eval)/1000:.1f} km). Cannot determine r0."
                )

        # else: No valid points outside rmax

        r_out = r0_profile

    elif rmax_or_r0.lower() == "r0":
        # r_out is rmax, which we tried to calculate numerically earlier.
        # Here we can also find rmax from the profile itself for consistency check?
        # MATLAB's ER11_radprof_raw returns the numerically solved rmax.
        # Let's return the numerically solved one stored in `rmax`.
        # Add consistency check:
        if np.any(~np.isnan(v_er11)):
            rmax_prof_idx = np.nanargmax(v_er11)
            if not np.isnan(rmax_prof_idx):  # Check if argmax found a valid index
                rmax_prof = rr_eval[rmax_prof_idx]
                vmax_prof = v_er11[rmax_prof_idx]  # Use V at the found rmax_prof index
                if not np.isnan(rmax) and not np.isclose(
                    rmax, rmax_prof, rtol=0.05, atol=100
                ):  # Added atol
                    warnings.warn(
                        f"ER11_raw numerical rmax ({rmax/1000:.1f} km) differs significantly from profile rmax ({rmax_prof/1000:.1f} km)."
                    )
                if not np.isclose(vmax, vmax_prof, rtol=0.05, atol=0.5):  # Added atol
                    warnings.warn(
                        f"ER11_raw input Vmax ({vmax:.1f} m/s) differs significantly from profile Vmax ({vmax_prof:.1f} m/s). This raw function doesn't iterate."
                    )
            else:
                warnings.warn(
                    "ER11_raw could not find profile rmax for consistency check."
                )
        # r_out remains the numerically calculated rmax
        pass

    return v_er11, r_out


def er11_radprof(
    vmax_target,
    r_in_target,
    rmax_or_r0,
    fcor,
    ck_cd,
    rr_eval,
    max_iter=20,
    r_tol_frac=0.005,
    v_tol_frac=0.01,
):
    """
    Iteratively calls `er11_radprof_raw` to find an ER11 profile that
    converges to the target Vmax and r_in (rmax or r0). Corresponds to
    ER11_radprof.m.
    (Includes corrected r0 interpolation)

    Args:
        vmax_target (float): Target maximum wind speed (m/s).
        r_in_target (float): Target input radius (rmax or r0) (m).
        rmax_or_r0 (str): Specifies if r_in_target is 'rmax' or 'r0'.
        fcor (float): Coriolis parameter (s^-1).
        ck_cd (float): Ratio of exchange coefficients (Ck/Cd).
        rr_eval (np.ndarray): Radii (m) at which to evaluate the profile.
        max_iter (int): Maximum number of iterations for convergence.
        r_tol_frac (float): Fractional tolerance for r_in convergence.
        v_tol_frac (float): Fractional tolerance for Vmax convergence.

    Returns:
        tuple: (v_er11_conv, r_out_conv)
               - v_er11_conv (np.ndarray): Converged ER11 wind speeds (m/s).
                                           Returns NaNs if convergence fails.
               - r_out_conv (float): The other radius (rmax or r0) from the
                                     converged profile. Returns NaN if fails.
    """
    dr = rr_eval[1] - rr_eval[0] if len(rr_eval) > 1 else 1.0

    # Initial guess is the target value
    vmax_current = vmax_target
    r_in_current = r_in_target

    v_er11 = np.full_like(rr_eval, np.nan)
    r_out = np.nan
    v_err = np.inf  # Initialize errors
    r_err = np.inf

    for iteration in range(max_iter):
        # Call the raw profile function with current estimates
        v_er11_temp, r_out_temp = er11_radprof_raw(
            vmax_current, r_in_current, rmax_or_r0, fcor, ck_cd, rr_eval
        )

        if np.all(np.isnan(v_er11_temp)):
            warnings.warn(
                f"ER11 raw calculation failed during iteration {iteration+1}. Stopping convergence."
            )
            return np.full_like(rr_eval, np.nan), np.nan  # Return NaNs on failure

        # --- Calculate current profile's Vmax and r_in ---
        vmax_prof = np.nanmax(v_er11_temp)
        if np.isnan(vmax_prof) or vmax_prof <= 0:
            warnings.warn(
                f"ER11 profile calculation yielded invalid Vmax ({vmax_prof}) in iteration {iteration+1}. Stopping."
            )
            # Return the last valid profile? Or fail completely? Let's fail.
            # Keep v_er11 from previous iteration if it was valid?
            return np.full_like(rr_eval, np.nan), np.nan

        r_in_prof = np.nan  # Initialize profile's input radius
        if rmax_or_r0.lower() == "rmax":
            # Profile's r_in is rmax
            rmax_prof_idx = np.nanargmax(v_er11_temp)
            if not np.isnan(rmax_prof_idx):
                r_in_prof = rr_eval[rmax_prof_idx]
            r_out = r_out_temp  # r_out should be r0 from raw function
        elif rmax_or_r0.lower() == "r0":
            # Profile's r_in is r0
            # r_out_temp from raw function is rmax_prof
            # We need to find r0_prof from the profile V=0 intercept
            # --- Start r0 interpolation logic (Corrected) ---
            valid_v = ~np.isnan(v_er11_temp) & (rr_eval > 1e-6)  # Exclude r=0
            if np.any(valid_v):
                rr_outer_all = rr_eval[valid_v]
                vv_outer_all = v_er11_temp[valid_v]
                # Find profile rmax to look outside of it
                rmax_prof_idx_local = np.nanargmax(vv_outer_all)
                rmax_prof = rr_outer_all[rmax_prof_idx_local]

                # Look for V=0 outside rmax_prof
                outer_indices = rr_outer_all > rmax_prof
                if np.any(outer_indices):
                    rr_outer_v0 = rr_outer_all[outer_indices]
                    vv_outer_v0 = vv_outer_all[outer_indices]

                    if np.min(vv_outer_v0) <= 1e-3:  # Check if wind gets close to zero
                        try:
                            # Sort by radius
                            sort_r_idx = np.argsort(rr_outer_v0)
                            rr_o_s = rr_outer_v0[sort_r_idx]
                            vv_o_s = vv_outer_v0[sort_r_idx]
                            # Find first index near zero
                            zero_cross_idx = np.where(vv_o_s <= 1e-3)[0]

                            if len(zero_cross_idx) > 0 and zero_cross_idx[0] > 0:
                                last_pos_idx = zero_cross_idx[0] - 1
                                # --- Start r(V) interpolation ---
                                v_interp_segment = vv_o_s[last_pos_idx:]
                                r_interp_segment = rr_o_s[last_pos_idx:]
                                # Sort by V
                                sort_v_idx = np.argsort(v_interp_segment)
                                v_interp_sorted = v_interp_segment[sort_v_idx]
                                r_interp_sorted = r_interp_segment[sort_v_idx]
                                # Use unique V
                                v_unique, unique_idx = np.unique(
                                    v_interp_sorted, return_index=True
                                )

                                if len(v_unique) >= 2:  # Need >= 2 points
                                    r_unique = r_interp_sorted[unique_idx]
                                    try:
                                        if len(v_unique) >= 3:
                                            interp_r_of_v = PchipInterpolator(
                                                v_unique, r_unique, extrapolate=False
                                            )
                                            r0_prof_candidate = float(
                                                interp_r_of_v(0.0)
                                            )
                                        else:  # Linear fallback
                                            r0_prof_candidate = float(
                                                np.interp(0.0, v_unique, r_unique)
                                            )

                                        # Check validity
                                        if (
                                            not np.isnan(r0_prof_candidate)
                                            and r0_prof_candidate > rmax_prof
                                        ):
                                            r_in_prof = r0_prof_candidate  # Assign successfully found r0
                                        else:
                                            # Failed validity check, r_in_prof remains NaN
                                            warnings.warn(
                                                f"Iter {iteration+1}: Interpolated r0 ({r0_prof_candidate/1000:.1f} km) invalid."
                                            )
                                            pass

                                    except ValueError as e_interp:
                                        warnings.warn(
                                            f"Iter {iteration+1}: ER11 r0 Pchip/linear interpolation failed: {e_interp}"
                                        )
                                        # r_in_prof remains NaN
                                else:
                                    warnings.warn(
                                        f"Iter {iteration+1}: Not enough unique points for r0 interpolation."
                                    )
                                    # r_in_prof remains NaN

                            elif len(zero_cross_idx) > 0 and zero_cross_idx[0] == 0:
                                r_in_prof = rr_o_s[0]  # Zero at first point past rmax

                            # else: V never drops low enough, r_in_prof remains NaN

                        except Exception as e:
                            warnings.warn(
                                f"Iter {iteration+1}: ER11 r0 interpolation failed with exception: {e}"
                            )
                            # r_in_prof remains NaN
                    # else: V never reaches zero, r_in_prof remains NaN
                    if np.isnan(r_in_prof):
                        warnings.warn(
                            f"ER11 V profile doesn't reach zero in iter {iteration+1}, cannot check r0 convergence accurately."
                        )

                # else: No points outside rmax_prof, r_in_prof remains NaN
            # else: No valid points at all, r_in_prof remains NaN
            # --- End r0 interpolation logic ---

            # r_out is rmax from the raw function
            r_out = r_out_temp
        else:
            raise ValueError("rmax_or_r0 must be 'rmax' or 'r0'")

        # --- Check Convergence ---
        # Check Vmax convergence
        v_err = vmax_target - vmax_prof
        v_converged = (
            abs(v_err / vmax_target) < v_tol_frac
            if vmax_target != 0
            else abs(v_err) < v_tol_frac
        )

        # Check r_in convergence
        r_converged = False  # Default if profile r_in couldn't be found
        if not np.isnan(r_in_prof):
            r_err = r_in_target - r_in_prof
            # Use fractional error for r0, add absolute check (like MATLAB's dr/2) for rmax?
            # Let's use combined relative and absolute tolerance for both for robustness.
            # Relative check: abs(r_err / r_in_target) < r_tol_frac if r_in_target != 0 else False
            # Absolute check: abs(r_err) < dr # Use dr instead of dr/2? Let's use dr.
            # Combine: (relative or absolute)
            rel_conv = (
                abs(r_err / r_in_target) < r_tol_frac
                if abs(r_in_target) > 1e-6
                else False
            )
            abs_conv = abs(r_err) < max(
                dr, 1.0
            )  # Use dr or 1m as minimum absolute tolerance
            r_converged = rel_conv or abs_conv

        else:
            r_err = np.nan  # Keep track that r_err is NaN
            warnings.warn(
                f"Could not determine profile r_in in iteration {iteration+1}. Cannot check r_in convergence."
            )

        # Store the latest valid profile
        v_er11 = v_er11_temp.copy()

        if v_converged and r_converged:
            # print(f"ER11 converged in {iteration+1} iterations.")
            return v_er11, r_out

        # --- Update Estimates for Next Iteration ---
        # Simple additive correction based on error (as in MATLAB)
        # Apply correction to Vmax first
        if not v_converged:
            vmax_current += v_err

        # Adjust r_in based on its error if it's not NaN
        if not r_converged and not np.isnan(r_err):
            r_in_current += r_err

        # Add basic checks to prevent divergence (e.g., negative Vmax/r_in)
        if vmax_current <= 0 or r_in_current <= 0:
            warnings.warn(
                f"ER11 iteration {iteration+1} resulted in non-physical Vmax/r_in guess ({vmax_current=}, {r_in_current=}). Stopping."
            )
            # Return last valid profile? Or fail? Let's fail.
            return np.full_like(rr_eval, np.nan), np.nan

    # If loop finishes without converging
    # Ensure errors are calculated based on the *last* profile attempt
    if not np.isnan(r_err):
        r_err_str = (
            f"{r_err/r_in_target*100:.2f}%"
            if abs(r_in_target) > 1e-6
            else f"{r_err:.2f} m"
        )
    else:
        r_err_str = "N/A"
    v_err_str = (
        f"{v_err/vmax_target*100:.2f}%" if vmax_target != 0 else f"{v_err:.2f} m/s"
    )

    warnings.warn(
        f"ER11 profile did not converge within {max_iter} iterations. "
        f"Final Vmax error: {v_err_str}, "
        f"Final r_in error: {r_err_str} (target Vm={vmax_target:.1f}, target r_in={r_in_target/1000:.1f} km). "
        f"Returning last calculated profile."
    )
    return v_er11, r_out  # Return the last attempt


def radprof_eyeadj(rr, vv, alpha_eye, r_eye_outer=None, v_eye_outer=None):
    """
    Applies an empirical adjustment to the wind profile in the eye region.
    V(r < r_eye_outer) is multiplied by (r / r_eye_outer)**alpha_eye.
    Corresponds to radprof_eyeadj.m.

    Args:
        rr (np.ndarray): Radius vector (m).
        vv (np.ndarray): Wind speed vector (m/s).
        alpha_eye (float): Exponent for eye adjustment.
        r_eye_outer (float, optional): Outer radius for eye adjustment (m).
                                       Defaults to rmax from the profile.
        v_eye_outer (float, optional): Wind speed at r_eye_outer (m/s).
                                       Defaults to Vmax from the profile.

    Returns:
        np.ndarray: Adjusted wind speed vector (m/s).
    """
    vv_out = vv.copy()

    if alpha_eye == 0:  # No adjustment if alpha is zero
        return vv_out

    if r_eye_outer is None or v_eye_outer is None:
        if np.any(~np.isnan(vv)):
            vmax_prof_idx = np.nanargmax(vv)
            if not np.isnan(vmax_prof_idx):
                vmax_prof = vv[vmax_prof_idx]
                rmax_prof = rr[vmax_prof_idx]
            else:  # Handle case where argmax fails (e.g. all NaNs)
                warnings.warn(
                    "Cannot determine rmax/Vmax for eye adjustment from profile."
                )
                return vv_out  # Return original if profile is unusable
        else:  # Handle case of all NaNs
            warnings.warn(
                "Cannot determine rmax/Vmax for eye adjustment from NaN profile."
            )
            return vv_out  # Return original if profile is NaN

        if r_eye_outer is None:
            r_eye_outer = rmax_prof
        if v_eye_outer is None:
            v_eye_outer = vmax_prof

    # Check for valid radius/velocity before proceeding
    if r_eye_outer <= 0 or v_eye_outer <= 0:
        warnings.warn(
            f"Invalid r_eye_outer ({r_eye_outer}) or v_eye_outer ({v_eye_outer}) for eye adjustment. Skipping."
        )
        return vv_out

    indices_eye = rr <= r_eye_outer
    if not np.any(indices_eye):
        return vv_out  # No points within the eye radius

    rr_eye = rr[indices_eye]
    vv_eye = vv_out[indices_eye]

    # Avoid division by zero if r_eye_outer is zero (handled by check above)
    rr_eye_norm = rr_eye / r_eye_outer

    # Ensure base for power is non-negative
    rr_eye_norm_safe = np.maximum(0.0, rr_eye_norm)

    # Multiplicative factor: (r/rm)^alpha
    # Handle potential 0^alpha issues if alpha < 0 (unlikely here)
    try:
        # Use np.errstate to suppress potential warnings for 0**negative
        with np.errstate(divide="ignore", invalid="ignore"):
            eye_factor = np.power(rr_eye_norm_safe, alpha_eye)
            # Replace inf resulting from 0**negative with 0? Or keep inf? Let's keep it for now.
            # Replace NaN resulting from 0**0 with 1? (convention)
            eye_factor[np.isnan(eye_factor)] = 1.0

    except ValueError as e:
        warnings.warn(
            f"Error calculating eye factor (r/rm)^alpha: {e}. Skipping eye adjustment."
        )
        return vv_out

    # Apply adjustment: V_new = V_old * factor
    vv_eye_adjusted = vv_eye * eye_factor

    # Replace eye values, handle potential Infs from factor calc
    vv_out[indices_eye] = np.where(
        np.isinf(vv_eye_adjusted), 0.0, vv_eye_adjusted
    )  # Set V=0 if factor was inf

    # Ensure V(0)=0 if r=0 is included
    zero_idx = np.isclose(rr, 0.0)
    vv_out[zero_idx] = 0.0

    return vv_out


# --- Main Merging Function ---


def er11_e04_merge(
    vmax_target,
    r0_target,
    fcor,
    cd_vary,
    c_d_const,
    w_cool,
    ck_cd_vary,
    ck_cd_const,
    eye_adj,
    alpha_eye,
):
    """
    Merges the ER11 inner profile and E04 outer profile following Chavas et al.
    (2015), using r0 as a primary input. Corresponds largely to
    ER11E04_nondim_r0input.m.
    (Includes corrected _find_intersections and er11_radprof)

    Args:
        vmax_target (float): Target maximum wind speed (m/s).
        r0_target (float): Target outer radius where V=0 (m).
        fcor (float): Coriolis parameter (s^-1).
        cd_vary (bool): True if outer Cd varies with wind speed.
        c_d_const (float): Constant outer drag coefficient (if cd_vary=False).
        w_cool (float): Radiative subsidence rate (m/s).
        ck_cd_vary (bool): True if inner Ck/Cd varies with Vmax.
        ck_cd_const (float): Constant inner Ck/Cd (if ck_cd_vary=False).
        eye_adj (bool): True to apply empirical eye adjustment.
        alpha_eye (float): Exponent for eye adjustment (if eye_adj=True).

    Returns:
        dict: A dictionary containing results:
              'rr': Radius vector (m) (np.ndarray)
              'vv': Wind speed vector (m/s) (np.ndarray)
              'rmax': Radius of maximum wind (m) (float)
              'rmerge': Radius of merge point (m) (float)
              'vmerge': Wind speed at merge point (m/s) (float)
              'rr_frac_r0': Dimensionless radius (r/r0) (np.ndarray)
              'mm_frac_m0': Dimensionless angular momentum (M/M0) (np.ndarray)
              'rmax_r0': Dimensionless rmax (rmax/r0) (float)
              'mm_m0_at_rmax': Dimensionless M at rmax (Mm/M0) (float)
              'rmerge_r0': Dimensionless merge radius (rmerge/r0) (float)
              'mmerge_m0': Dimensionless M at merge point (Mmerge/M0) (float)
              Returns None if the calculation fails to converge.
    """
    fcor = abs(fcor)

    # --- Determine Ck/Cd ---
    ck_cd = ck_cd_const
    if ck_cd_vary:
        # Quadratic fit from Chavas et al. 2015 (as in MATLAB code)
        ck_cd_coefquad = 5.5041e-04
        ck_cd_coeflin = -0.0259
        ck_cd_coefcnst = 0.7627
        ck_cd = (
            ck_cd_coefquad * vmax_target**2
            + ck_cd_coeflin * vmax_target
            + ck_cd_coefcnst
        )

        # Apply bounds as in MATLAB comments/code
        # Lower bound? MATLAB commented out CkCd<0.5 check. Add small positive lower bound.
        ck_cd = max(0.1, ck_cd)  # Avoid non-positive CkCd

        # Upper bound
        if ck_cd > 1.9:
            warnings.warn(
                f"Calculated Ck/Cd ({ck_cd:.2f}) > 1.9. Capping at 1.9. Vmax ({vmax_target}) may be outside range used for fit."
            )
            ck_cd = 1.9

    current_ck_cd = ck_cd  # Store the potentially adjusted CkCd

    # --- Step 1: Calculate E04 Outer Profile (M/M0 vs r/r0) ---
    # Use a sufficient number of points for accurate interpolation later
    nr_e04 = 2000  # Or adjust based on r0? Let's keep this value.
    rr_frac_r0_e04, mm_frac_m0_e04 = e04_outer_wind_nondim(
        r0_target, fcor, cd_vary, c_d_const, w_cool, nr=nr_e04
    )
    m0_e04 = 0.5 * fcor * r0_target**2

    # Filter out NaNs if any occurred in E04 calculation
    valid_e04 = ~np.isnan(rr_frac_r0_e04) & ~np.isnan(mm_frac_m0_e04)
    rr_frac_r0_e04 = rr_frac_r0_e04[valid_e04]
    mm_frac_m0_e04 = mm_frac_m0_e04[valid_e04]

    if len(rr_frac_r0_e04) < 2:
        warnings.warn("E04 outer profile calculation failed. Cannot proceed.")
        return None

    # --- Step 2: Find rmax/r0 where ER11 is tangent to E04 ---
    # Iteratively guess rmax/r0, calculate ER11, find intersection(s).
    # Goal: find rmax/r0 where there's tangency (ideally 1 intersection).

    soln_converged = False
    max_ckcd_adjust_attempts = 5  # Max attempts to increase CkCd if ER11 fails
    ckcd_adjust_step = 0.1

    # Store results from the rmax/r0 search loop
    rmax_r0_final = np.nan
    rmerge_r0_final = np.nan
    mmerge_m0_final = np.nan
    v_er11_final = None
    rr_er11_final = None
    rmax_final = np.nan  # Store the dimensional rmax corresponding to rmax_r0_final

    for ckcd_attempt in range(max_ckcd_adjust_attempts):

        rmax_r0_min = 0.001
        rmax_r0_max = 0.75  # Upper bound for rmax/r0 guess
        rmax_r0_guess = (rmax_r0_max + rmax_r0_min) / 2.0

        # Use bisection method for finding tangent rmax_r0
        iter_rmax = 0
        max_iter_rmax = 50  # Max iterations for rmax/r0 search
        tol_rmax_r0 = 1e-6  # Tolerance for rmax/r0 convergence

        # Store state for the bisection search
        r_low = rmax_r0_min
        r_high = rmax_r0_max
        n_int_low = -1  # Number of intersections at r_low (unknown initially)
        n_int_high = -1  # Number of intersections at r_high (unknown initially)

        # Function to get number of intersections for a given rmax_r0
        def get_num_intersections(rmax_r0_test, ck_cd_test):
            rmax_test = rmax_r0_test * r0_target
            # Define radius grid for ER11
            drfracrm = 0.01
            if rmax_test > 100e3:
                drfracrm /= 10.0
            rfracrm_min = 0.0
            rfracrm_max = max(50.0, 1.1 / rmax_r0_test if rmax_r0_test > 0 else 50.0)
            rfracrm_max = min(
                rfracrm_max, 200.0 / rmax_r0_test if rmax_r0_test > 0.01 else 20000
            )
            rr_er11_eval_test = (
                np.arange(rfracrm_min, rfracrm_max + drfracrm / 2.0, drfracrm)
                * rmax_test
            )
            if not np.isclose(rr_er11_eval_test[0], 0.0):
                rr_er11_eval_test = np.insert(rr_er11_eval_test, 0, 0.0)

            # Calculate ER11 profile
            v_er11_test, _ = er11_radprof(
                vmax_target, rmax_test, "rmax", fcor, ck_cd_test, rr_er11_eval_test
            )

            if np.all(np.isnan(v_er11_test)):
                # ER11 failed -> treat as case where ER11 is "too low"? (Needs larger rmax_r0)
                # Or treat as invalid? Let's return -1 to indicate failure.
                return -1, None, None  # Num intersections, x, y

            # Convert to M/M0 vs r/r0
            valid_er11 = ~np.isnan(v_er11_test)
            rr_er11_t = rr_er11_eval_test[valid_er11]
            vv_er11_t = v_er11_test[valid_er11]
            if len(rr_er11_t) < 2:
                return -1, None, None  # Invalid
            mm_er11_t = rr_er11_t * vv_er11_t + 0.5 * fcor * rr_er11_t**2
            rr_frac_r0_er11_t = rr_er11_t / r0_target
            mm_frac_m0_er11_t = mm_er11_t / m0_e04
            valid = ~np.isnan(rr_frac_r0_er11_t) & ~np.isnan(mm_frac_m0_er11_t)
            rr_frac_r0_er11_t = rr_frac_r0_er11_t[valid]
            mm_frac_m0_er11_t = mm_frac_m0_er11_t[valid]
            if len(rr_frac_r0_er11_t) < 2:
                return -1, None, None  # Invalid

            # Find intersections
            ix, iy = _find_intersections(
                rr_frac_r0_e04, mm_frac_m0_e04, rr_frac_r0_er11_t, mm_frac_m0_er11_t
            )
            return len(ix), ix, iy

        # Initial check for bounds
        n_int_low, _, _ = get_num_intersections(r_low, current_ck_cd)
        n_int_high, _, _ = get_num_intersections(r_high, current_ck_cd)

        # Check if bounds are valid (e.g., low gives 0, high gives >=1)
        # Expected: n_int=0 for low rmax/r0, n_int=2 (or 1) for high rmax/r0
        if n_int_low == -1 or n_int_high == -1:
            warnings.warn(
                f"ER11 calculation failed at initial bounds for Ck/Cd={current_ck_cd:.2f}. Adjusting Ck/Cd."
            )
            # Break inner loop, go to CkCd adjustment
            break
        # If low already has intersections, or high has none, bounds might be wrong or no tangent exists
        if n_int_low > 0 or n_int_high == 0:
            warnings.warn(
                f"Intersection pattern at bounds ({n_int_low=}, {n_int_high=}) suggests no unique tangent point or invalid bounds for Ck/Cd={current_ck_cd:.2f}. Trying Ck/Cd adjustment."
            )
            # Break inner loop, go to CkCd adjustment
            break

        # Bisection loop
        while (r_high - r_low) / 2.0 > tol_rmax_r0 and iter_rmax < max_iter_rmax:
            iter_rmax += 1
            rmax_r0_mid = (r_low + r_high) / 2.0
            n_int_mid, ix_mid, iy_mid = get_num_intersections(
                rmax_r0_mid, current_ck_cd
            )

            if n_int_mid == -1:  # ER11 failure
                warnings.warn(
                    f"ER11 failed at rmax/r0={rmax_r0_mid:.4f}. Assuming too few intersections. Adjusting lower bound."
                )
                # Treat failure as n_int = 0 case? Increase rmax_r0.
                r_low = rmax_r0_mid
                n_int_low = 0  # Assume 0 intersections
            elif n_int_mid == 0:  # No intersections -> need larger rmax/r0
                r_low = rmax_r0_mid
                n_int_low = 0
            else:  # n_int_mid >= 1 -> need smaller rmax/r0
                r_high = rmax_r0_mid
                n_int_high = n_int_mid
                # Store this as a potential solution
                rmax_r0_final = rmax_r0_mid
                rmerge_r0_final = np.mean(ix_mid) if n_int_mid > 0 else np.nan
                mmerge_m0_final = np.mean(iy_mid) if n_int_mid > 0 else np.nan
                # Store corresponding rmax and calculate V profile for later use?
                rmax_final = rmax_r0_final * r0_target
                # Need to recalculate V profile outside this helper function

        # --- Check convergence of rmax/r0 bisection loop ---
        if iter_rmax >= max_iter_rmax:
            warnings.warn(
                f"rmax/r0 bisection did not converge within {max_iter_rmax} iterations for Ck/Cd = {current_ck_cd:.2f}. Using last valid estimate."
            )
            # Check if the last stored values are valid
            if np.isnan(rmax_r0_final) or np.isnan(rmerge_r0_final):
                warnings.warn(
                    "No valid merge point found during unconverged rmax/r0 search."
                )
                # Let it try increasing CkCd
                pass
            else:
                soln_converged = True  # Treat as converged if a merge point was found
        elif not np.isnan(rmax_r0_final) and not np.isnan(rmerge_r0_final):
            # Bisection converged, and we stored a valid result
            soln_converged = True
        else:
            # Bisection finished, but no valid intersection was stored (shouldn't happen if bounds were ok)
            warnings.warn(
                f"rmax/r0 bisection finished but no valid merge point found for Ck/Cd = {current_ck_cd:.2f}."
            )
            # Try increasing CkCd
            pass

        # --- If converged, break CkCd loop ---
        if soln_converged:
            break

        # --- If not converged, try adjusting CkCd ---
        else:
            if ckcd_attempt < max_ckcd_adjust_attempts - 1:
                current_ck_cd += ckcd_adjust_step
                current_ck_cd = min(current_ck_cd, 2.5)  # Add upper limit to adjustment
                warnings.warn(
                    f"Solution did not converge. Adjusting Ck/Cd to {current_ck_cd:.2f} and retrying."
                )
                # Reset state for next attempt
                soln_converged = False
                rmax_r0_final = np.nan
                rmerge_r0_final = np.nan
                mmerge_m0_final = np.nan
                rmax_final = np.nan
                # v_er11_final = None # Reset stored profile too
                # rr_er11_final = None
            else:
                warnings.warn(
                    f"Solution failed to converge even after adjusting Ck/Cd {max_ckcd_adjust_attempts} times."
                )
                return None  # Return None indicating failure

    # --- Step 3: Construct Final Merged Profile ---
    if not soln_converged or np.isnan(rmax_r0_final) or np.isnan(rmerge_r0_final):
        warnings.warn("Failed to find a converged merge solution.")
        return None

    # We have rmax_r0_final, rmerge_r0_final, mmerge_m0_final
    # Need the corresponding final ER11 profile (dimensional V vs r)
    rmax_final = rmax_r0_final * r0_target
    # Define final grid for ER11 calculation
    drfracrm = 0.01
    if rmax_final > 100e3:
        drfracrm /= 10.0
    rfracrm_min = 0.0
    rfracrm_max = max(50.0, 1.1 / rmax_r0_final if rmax_r0_final > 0 else 50.0)
    rfracrm_max = min(
        rfracrm_max, 200.0 / rmax_r0_final if rmax_r0_final > 0.01 else 20000
    )
    rr_er11_eval_final = (
        np.arange(rfracrm_min, rfracrm_max + drfracrm / 2.0, drfracrm) * rmax_final
    )
    if not np.isclose(rr_er11_eval_final[0], 0.0):
        rr_er11_eval_final = np.insert(rr_er11_eval_final, 0, 0.0)

    v_er11_final, _ = er11_radprof(
        vmax_target, rmax_final, "rmax", fcor, current_ck_cd, rr_er11_eval_final
    )

    if np.all(np.isnan(v_er11_final)):
        warnings.warn("Final ER11 profile calculation failed.")
        return None

    # Convert final ER11 V(r) to M/M0 vs r/r0
    valid_er11 = ~np.isnan(v_er11_final)
    rr_er11 = rr_er11_eval_final[valid_er11]
    vv_er11 = v_er11_final[valid_er11]
    if len(rr_er11) < 2:
        warnings.warn("Final ER11 profile calculation yielded insufficient points.")
        return None

    mm_er11 = rr_er11 * vv_er11 + 0.5 * fcor * rr_er11**2
    rr_frac_r0_er11 = rr_er11 / r0_target
    mm_frac_m0_er11 = mm_er11 / m0_e04
    valid = ~np.isnan(rr_frac_r0_er11) & ~np.isnan(mm_frac_m0_er11)
    rr_frac_r0_er11 = rr_frac_r0_er11[valid]
    mm_frac_m0_er11 = mm_frac_m0_er11[valid]

    if len(rr_frac_r0_er11) < 2:
        warnings.warn("Final ER11 profile conversion failed.")
        return None

    # Define points for interpolation based on merge radius
    # Use ER11 inside rmerge, E04 outside rmerge

    # Ensure profiles are sorted by radius
    sort_idx_er11 = np.argsort(rr_frac_r0_er11)
    rr_frac_r0_er11_s = rr_frac_r0_er11[sort_idx_er11]
    mm_frac_m0_er11_s = mm_frac_m0_er11[sort_idx_er11]

    sort_idx_e04 = np.argsort(rr_frac_r0_e04)
    rr_frac_r0_e04_s = rr_frac_r0_e04[sort_idx_e04]
    mm_frac_m0_e04_s = mm_frac_m0_e04[sort_idx_e04]

    # Select points for the merged profile
    ii_er11 = rr_frac_r0_er11_s < rmerge_r0_final
    ii_e04 = rr_frac_r0_e04_s >= rmerge_r0_final

    # Combine points, adding the merge point itself
    rr_frac_r0_merged = np.concatenate(
        (rr_frac_r0_er11_s[ii_er11], [rmerge_r0_final], rr_frac_r0_e04_s[ii_e04])
    )
    mm_frac_m0_merged = np.concatenate(
        (mm_frac_m0_er11_s[ii_er11], [mmerge_m0_final], mm_frac_m0_e04_s[ii_e04])
    )

    # Remove duplicates that might arise at the merge point boundary and sort
    uniq_indices = np.unique(rr_frac_r0_merged, return_index=True)[1]
    rr_frac_r0_merged = rr_frac_r0_merged[np.sort(uniq_indices)]
    mm_frac_m0_merged = mm_frac_m0_merged[np.sort(uniq_indices)]

    # Interpolate onto a regular grid using M/Mm vs r/rm (like MATLAB)
    # Convert merged profile to M/Mm vs r/rm space
    mm_final = vmax_target * rmax_final + 0.5 * fcor * rmax_final**2
    m0_final = 0.5 * fcor * r0_target**2

    rr_frac_rm_merged = rr_frac_r0_merged * (r0_target / rmax_final)
    mm_frac_mm_merged = mm_frac_m0_merged * (m0_final / mm_final)

    # Create interpolator M/Mm(r/rm)
    # Ensure sorted and unique for interpolation
    sort_idx = np.argsort(rr_frac_rm_merged)
    rr_frac_rm_merged_s = rr_frac_rm_merged[sort_idx]
    mm_frac_mm_merged_s = mm_frac_mm_merged[sort_idx]
    uniq_idx = np.unique(rr_frac_rm_merged_s, return_index=True)[1]
    rr_frac_rm_interp = rr_frac_rm_merged_s[uniq_idx]
    mm_frac_mm_interp = mm_frac_mm_merged_s[uniq_idx]

    if len(rr_frac_rm_interp) < 2:
        warnings.warn("Cannot create final interpolator, insufficient points.")
        return None

    # Pchip needs at least 2 points, prefers more for non-linear
    interp_mm_frac_mm = None
    interp_args = ()
    if len(rr_frac_rm_interp) < 3:
        # Use linear interpolation as fallback if only 2 points
        interp_mm_frac_mm = np.interp
        interp_args = (rr_frac_rm_interp, mm_frac_mm_interp)
        warnings.warn("Using linear interpolation for final profile (few points).")
    else:
        try:
            interp_obj = PchipInterpolator(
                rr_frac_rm_interp, mm_frac_mm_interp, extrapolate=False
            )
            interp_mm_frac_mm = interp_obj  # Use object directly
            interp_args = ()  # Arguments passed directly to the object call
        except ValueError as e:
            warnings.warn(f"Final Pchip interpolation failed: {e}. Using linear.")
            interp_mm_frac_mm = np.interp
            interp_args = (rr_frac_rm_interp, mm_frac_mm_interp)

    # Define final output grid (dimensional radius)
    # Use a grid relative to rmax_final for good resolution near peak
    drfracrm_out = 0.005  # Finer resolution for output
    rfracrm_min_out = 0.0
    # Ensure grid extends slightly beyond r0
    rfracrm_max_out = (r0_target / rmax_final) * 1.01

    rr_frac_rm_out = np.arange(
        rfracrm_min_out, rfracrm_max_out + drfracrm_out / 2.0, drfracrm_out
    )
    rr_out = rr_frac_rm_out * rmax_final

    # Add r=0 if not present
    if not np.isclose(rr_out[0], 0.0):
        rr_out = np.insert(rr_out, 0, 0.0)
        rr_frac_rm_out = np.insert(rr_frac_rm_out, 0, 0.0)

    # Ensure grid does not exceed r0 significantly (causes issues with V calc?)
    # Truncate at r0? Let's keep points up to r0.
    max_r_out = r0_target * 1.001  # Allow slight overshoot for interpolation near r0
    rr_out_mask = rr_out <= max_r_out
    rr_out = rr_out[rr_out_mask]
    rr_frac_rm_out = rr_frac_rm_out[rr_out_mask]
    # Ensure r0 is the last point if we are close
    if not np.isclose(rr_out[-1], r0_target):
        rr_out = np.append(rr_out, r0_target)
        rr_frac_rm_out = np.append(rr_frac_rm_out, r0_target / rmax_final)

    # Interpolate M/Mm onto the output grid
    if interp_mm_frac_mm == np.interp:  # Check if it's the function object
        mm_frac_mm_out = interp_mm_frac_mm(rr_frac_rm_out, *interp_args)
    else:  # Pchip object call
        mm_frac_mm_out = interp_mm_frac_mm(rr_frac_rm_out, *interp_args)

    # Calculate dimensional wind speed V from M/Mm
    # V = (M / r) - 0.5 * f * r
    # V = (Mm * (M/Mm) / (rm * (r/rm))) - 0.5 * f * rm * (r/rm)
    # V = (Mm / rm) * (M/Mm / (r/rm)) - 0.5 * f * rm * (r/rm)

    vv_out = np.full_like(rr_out, np.nan)

    # Avoid division by zero for r/rm at r=0
    non_zero_idx = rr_frac_rm_out > 1e-12

    term1 = np.full_like(rr_out, 0.0)
    term1[non_zero_idx] = (mm_final / rmax_final) * (
        mm_frac_mm_out[non_zero_idx] / rr_frac_rm_out[non_zero_idx]
    )

    term2 = 0.5 * fcor * rr_out  # Use dimensional r here

    vv_out = term1 - term2

    # Handle r=0: V(0)=0
    vv_out[np.isclose(rr_out, 0.0)] = 0.0

    # Handle potential NaNs from interpolation failure (extrapolation)
    vv_out[np.isnan(vv_out)] = 0.0  # Set V=0 if M/Mm was NaN

    # Ensure V >= 0
    vv_out = np.maximum(0.0, vv_out)

    # Ensure V=0 at r0 if r0 is included in the grid
    r0_idx = np.isclose(rr_out, r0_target)
    vv_out[r0_idx] = 0.0

    # --- Apply Eye Adjustment if requested ---
    if eye_adj:
        vv_out = radprof_eyeadj(
            rr_out, vv_out, alpha_eye, r_eye_outer=rmax_final, v_eye_outer=vmax_target
        )

    # --- Calculate other outputs ---
    rmerge_final = rmerge_r0_final * r0_target
    # Calculate V at rmerge using the final profile
    # Need to interpolate vv_out at rmerge_final
    if len(rr_out) > 1:
        vmerge_final = float(np.interp(rmerge_final, rr_out, vv_out))
    else:
        vmerge_final = 0.0

    # Calculate M/M0 at rmax_final
    mm_m0_at_rmax_final = mm_final / m0_final

    # Final non-dimensional profile (M/M0 vs r/r0)
    rr_frac_r0_out = rr_out / r0_target
    mm_out = rr_out * vv_out + 0.5 * fcor * rr_out**2
    mm_frac_m0_out = mm_out / m0_final

    results = {
        "rr": rr_out,
        "vv": vv_out,
        "rmax": rmax_final,
        "rmerge": rmerge_final,
        "vmerge": vmerge_final,
        "rr_frac_r0": rr_frac_r0_out,
        "mm_frac_m0": mm_frac_m0_out,
        "rmax_r0": rmax_r0_final,
        "mm_m0_at_rmax": mm_m0_at_rmax_final,
        "rmerge_r0": rmerge_r0_final,
        "mmerge_m0": mmerge_m0_final,
        "final_ck_cd": current_ck_cd,  # Include the Ck/Cd value used
    }

    return results


# --- Main Execution ---
if __name__ == "__main__":

    # --- Read Configuration ---
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    # Allow specifying config file via command line argument?
    # For now, use the hardcoded name.
    if len(sys.argv) > 1:
        config_file_to_use = sys.argv[1]
        print(
            f"Using configuration file specified on command line: {config_file_to_use}"
        )
    else:
        config_file_to_use = CONFIG_FILE
        print(f"Using default configuration file: {config_file_to_use}")

    try:
        if not config.read(config_file_to_use):
            raise FileNotFoundError(
                f"Config file '{config_file_to_use}' not found or empty."
            )

        # Read parameters with fallbacks
        params = config["TCProfileParameters"]

        vmax = params.getfloat("Vmax", 50.0)  # m/s
        r0 = params.getfloat("r0", 900e3)  # m
        fcor = params.getfloat("fcor", 5e-5)  # s^-1

        cd_vary = params.getboolean("Cdvary", True)
        c_d_const = params.getfloat("Cd_const", 1.5e-3)  # if Cdvary=False
        w_cool = params.getfloat("w_cool", 2e-3)  # m/s

        ck_cd_vary = params.getboolean("CkCdvary", True)
        ck_cd_const = params.getfloat("CkCd_const", 1.0)  # if CkCdvary=False

        eye_adj = params.getboolean("eye_adj", True)
        alpha_eye = params.getfloat("alpha_eye", 0.15)  # if eye_adj=True

        # Check section existence before accessing
        if "Output" in config:
            output_section = config["Output"]
            output_file = output_section.get("OutputFile", "tc_profile_output.csv")
            plot_profile = output_section.getboolean("PlotProfile", True)
        else:
            output_file = "tc_profile_output.csv"
            plot_profile = True
            warnings.warn("Config file missing [Output] section, using defaults.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please create the config file or specify a valid one.")
        # Example config generation could be added here
        import sys

        sys.exit(1)
    except (KeyError, ValueError, configparser.Error) as e:
        print(f"ERROR reading configuration file '{config_file_to_use}': {e}")
        print("Please check the file format and parameter values.")
        import sys

        sys.exit(1)

    print("--- Running TC Profile Model ---")
    print(f"Parameters from {config_file_to_use}:")
    print(f"  Vmax = {vmax} m/s")
    print(f"  r0 = {r0/1000} km")
    print(f"  fcor = {fcor} s^-1")
    print(
        f"  Outer Cd: {'Varying (Donelan)' if cd_vary else f'Constant ({c_d_const})'}"
    )
    print(f"  w_cool = {w_cool} m/s")
    print(
        f"  Inner Ck/Cd: {'Varying (Vmax fit)' if ck_cd_vary else f'Constant ({ck_cd_const})'}"
    )
    print(
        f"  Eye Adjustment: {'Enabled' if eye_adj else 'Disabled'} (alpha={alpha_eye if eye_adj else 'N/A'})"
    )
    print("---------------------------------")

    # --- Run the model ---
    results = er11_e04_merge(
        vmax_target=vmax,
        r0_target=r0,
        fcor=fcor,
        cd_vary=cd_vary,
        c_d_const=c_d_const,
        w_cool=w_cool,
        ck_cd_vary=ck_cd_vary,
        ck_cd_const=ck_cd_const,
        eye_adj=eye_adj,
        alpha_eye=alpha_eye,
    )

    # --- Process Results ---
    if results is None:
        print("\nModel calculation failed to converge or produce results.")
    else:
        print("\nModel calculation successful.")
        print(
            f"  Calculated rmax = {results['rmax']/1000:.2f} km (rmax/r0 = {results['rmax_r0']:.4f})"
        )
        print(
            f"  Merge radius rmerge = {results['rmerge']/1000:.2f} km (rmerge_r0 = {results['rmerge_r0']:.4f})"
        )
        print(f"  Merge wind speed Vmerge = {results['vmerge']:.2f} m/s")
        print(f"  Final Ck/Cd used = {results['final_ck_cd']:.3f}")

        # Save results to CSV
        try:
            # Create a structured array or pandas DataFrame for saving
            output_data = np.column_stack(
                (
                    results["rr"],
                    results["vv"],
                    results["rr_frac_r0"],
                    results["mm_frac_m0"],
                )
            )
            header = "Radius (m),Wind Speed (m/s),r/r0,M/M0"
            np.savetxt(
                output_file, output_data, delimiter=",", header=header, comments=""
            )
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"\nError saving results to {output_file}: {e}")

        # Plot results if requested
        if plot_profile:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot wind profile
                ax.plot(
                    results["rr"] / 1000.0,
                    results["vv"],
                    "b-",
                    linewidth=2,
                    label="Merged Profile",
                )

                # Mark key points
                ax.plot(
                    results["rmax"] / 1000.0,
                    vmax,
                    "rx",
                    markersize=10,
                    mew=2,
                    label=f"Input $V_{{max}}$ ({vmax} m/s)",
                )
                ax.plot(
                    results["rmerge"] / 1000.0,
                    results["vmerge"],
                    "go",
                    markersize=8,
                    label=f"$r_{{merge}}$",
                )
                ax.plot(
                    r0 / 1000.0,
                    0,
                    "k.",
                    markersize=15,
                    label=f"Input $r_0$ ({r0/1000} km)",
                )

                # Find and mark actual max wind speed from the profile
                vv_prof = results["vv"]
                rr_prof = results["rr"]
                if np.any(~np.isnan(vv_prof)):
                    vmax_prof_idx = np.nanargmax(vv_prof)
                    if not np.isnan(vmax_prof_idx):
                        vmax_prof_actual = vv_prof[vmax_prof_idx]
                        rmax_prof_actual = rr_prof[vmax_prof_idx]
                        ax.plot(
                            rmax_prof_actual / 1000.0,
                            vmax_prof_actual,
                            "b+",
                            markersize=12,
                            mew=2,
                            label=f"Profile $V_{{max}}$ ({vmax_prof_actual:.1f} m/s)",
                        )

                ax.set_xlabel("Radius (km)")
                ax.set_ylabel("Azimuthal Wind Speed (m/s)")
                ax.set_title("Chavas et al. (2015) TC Radial Profile")
                ax.grid(True, linestyle=":", alpha=0.7)
                ax.legend(fontsize=10)
                ax.set_xlim(
                    left=0, right=(r0 / 1000.0) * 1.05
                )  # Adjust xlim based on r0
                ax.set_ylim(bottom=0)

                plt.tight_layout()
                # Construct plot filename based on output CSV name
                plot_filename = output_file.rsplit(".", 1)[0] + ".png"
                plt.savefig(plot_filename, dpi=150)
                print(f"Plot saved to {plot_filename}")
                # plt.show() # Comment out if running non-interactively

            except ImportError:
                print(
                    "\nMatplotlib not found. Cannot generate plot. Install with: pip install matplotlib"
                )
            except Exception as e:
                print(f"\nError generating plot: {e}")
