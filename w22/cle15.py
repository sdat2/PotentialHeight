"""
Python translation of MATLAB code for calculating the Chavas et al. (2015)
tropical cyclone wind profile, merging Emanuel & Rotunno (2011) inner
profile and Emanuel (2004) outer profile, using r0 as input.

Translated by Gemini-2.5-pro (experimental) on 2025-05-05.

Based on MATLAB scripts:
- ER11E04_nondim_r0input.m
- E04_outerwind_r0input_nondim_MM0.m
- ER11_radprof.m
- ER11_radprof_raw.m
- curveintersect.m (simplified implementation)
- radprof_eyeadj.m
- CLE15_plot_r0input.m (for example usage)

References:
  - Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for tropical
    cyclone wind speed and rainfall profiles with physical interpretations.
    Journal of the Atmospheric Sciences, 72(9), 3403-3428.
  - Emanuel, K., & Rotunno, R. (2011). Self-Stratification of Tropical
    Cyclone Outflow. Part I: Implications for Storm Structure. Journal of
    the Atmospheric Sciences, 68(10), 2236-2249.
  - Emanuel, K. (2004). Tropical Cyclone Energetics and Structure. In
    Atmospheric Turbulence and Mesoscale Meteorology (pp. 165-191).
    Cambridge University Press.
"""

from typing import Tuple, Union, Optional, Dict
import os
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import warnings
import matplotlib.pyplot as plt  # Optional: For plotting example
from sithom.time import timeit
from sithom.plot import plot_defaults
from .constants import (
    FIGURE_PATH,
    BACKGROUND_PRESSURE,
    W_COOL_DEFAULT,
    CK_CD_DEFAULT,
    CD_DEFAULT,
    RHO_AIR_DEFAULT,
)
from .utils import pressure_from_wind

# --- Constants ---
# Coefficients for Ck/Cd quadratic fit to Vmax (Chavas et al. 2015)
CKCD_COEFQUAD = 5.5041e-04
CKCD_COEFLIN = -0.0259
CKCD_COEFCNST = 0.7627

# Parameters for variable Cd fit (Donelan et al. 2004)
CD_LOWV = 6.2e-4
V_THRESH1 = 6.0  # m/s; transition from constant to linear increasing
V_THRESH2 = 35.4  # m/s; transition from linear increasing to constant
CD_HIGHV = 2.35e-3
# Avoid division by zero if thresholds are equal
if V_THRESH2 > V_THRESH1:
    LINEAR_SLOPE = (CD_HIGHV - CD_LOWV) / (V_THRESH2 - V_THRESH1)
else:
    LINEAR_SLOPE = 0  # Or handle as an error/warning

# --- Helper Functions ---


def _calculate_cd(V: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Calculates the drag coefficient (Cd) based on wind speed (V) using a
    piecewise linear fit to Donelan et al. (2004).

    Args:
        V (float or np.ndarray): Wind speed (m/s).

    Returns:
        float or np.ndarray: Drag coefficient (dimensionless).
    """
    V = np.asarray(V)
    Cd = np.zeros_like(V, dtype=float)

    low_mask = V <= V_THRESH1
    high_mask = V > V_THRESH2
    mid_mask = ~low_mask & ~high_mask

    Cd[low_mask] = CD_LOWV
    Cd[high_mask] = CD_HIGHV
    if V_THRESH2 > V_THRESH1:  # Avoid issues if thresholds are equal
        Cd[mid_mask] = CD_LOWV + LINEAR_SLOPE * (V[mid_mask] - V_THRESH1)
    else:  # If thresholds are equal, use low value up to threshold
        Cd[mid_mask] = CD_LOWV

    # Handle case where V might be exactly V_THRESH2 if thresholds are equal
    if V_THRESH1 == V_THRESH2:
        Cd[V == V_THRESH1] = CD_LOWV

    return Cd


def _e04_outerwind_r0input_nondim_mm0(
    r0: float,
    fcor: float,
    Cdvary: int,
    C_d_input: float,
    w_cool: float,
    Nr: int = 100000,
) -> Tuple[NDArray, NDArray]:
    """
    Calculates the Emanuel (2004) non-convecting outer wind profile,
    represented as M/M0 vs. r/r0, given the outer radius r0.

    Args:
        r0 (float): Outer radius where V=0 (m).
        fcor (float): Coriolis parameter (s^-1).
        Cdvary (int): 0 for constant C_d, 1 for variable C_d(V).
        C_d_input (float): Constant drag coefficient (used if Cdvary=0).
        w_cool (float): Radiative subsidence rate (m/s, positive downwards).
        Nr (int): Number of radial nodes inward from r0.

    Returns:
        tuple: (rrfracr0_E04, MMfracM0_E04)
            - rrfracr0_E04 (np.ndarray): Vector of r/r0.
            - MMfracM0_E04 (np.ndarray): Vector of M/M0 at rrfracr0.
    """
    fcor = abs(fcor)
    M0 = 0.5 * fcor * r0**2  # M at outer radius

    # Determine radial step size, ensuring precision for large/small storms
    drfracr0 = 0.001
    if r0 > 2500 * 1000 or r0 < 200 * 1000:
        drfracr0 = drfracr0 / 10.0

    # Ensure Nr doesn't lead to non-positive radii
    max_Nr = int(1.0 / drfracr0)
    if Nr > max_Nr:
        Nr = max_Nr
    if Nr < 2:
        Nr = 2  # Need at least 2 points

    # Set up radial grid (integrating inwards)
    rfracr0_max = 1.0
    rfracr0_min = rfracr0_max - (Nr - 1) * drfracr0
    # Ensure rfracr0_min is slightly > 0 if Nr is large
    if rfracr0_min <= 0:
        rfracr0_min = drfracr0 / 10.0  # Small positive number
        Nr = int((rfracr0_max - rfracr0_min) / drfracr0) + 1
        warnings.warn(f"Adjusted Nr to {Nr} to avoid non-positive radii in E04 calc.")

    rrfracr0 = np.linspace(rfracr0_min, rfracr0_max, Nr)
    MMfracM0 = np.full_like(rrfracr0, np.nan)
    MMfracM0[-1] = 1.0  # M/M0 = 1 at r/r0 = 1

    # --- Numerical Integration (Euler method, backwards from r0) ---
    # Initialize values at the second-to-last point (index Nr-2)
    rfracr0_temp = rrfracr0[-2]
    MfracM0_temp = MMfracM0[-1]  # d(M/M0)/d(r/r0) = 0 at r/r0 = 1
    MMfracM0[-2] = MfracM0_temp

    # Loop from the third-to-last point backwards to the first point
    for ii in range(Nr - 3, -1, -1):  # Corresponds to MATLAB ii=1:Nr-2
        # Current radius and M/M0 (at index ii+1)
        rfracr0_curr = rrfracr0[ii + 1]
        MfracM0_curr = MMfracM0[ii + 1]

        # Calculate C_d at the current radius if varying
        C_d = C_d_input
        if Cdvary == 1:
            # Need V at the current radius to calculate Cd
            # Avoid division by zero if rfracr0_curr is very small
            if rfracr0_curr > 1e-9:
                V_temp = (M0 / r0) * ((MfracM0_curr / rfracr0_curr) - rfracr0_curr)
                # Ensure V is not negative (can happen with numerical errors near r0)
                V_temp = max(0, V_temp)
                C_d = _calculate_cd(V_temp)
            else:
                C_d = CD_LOWV  # Assume low wind speed near center

        # Calculate gamma parameter
        # Avoid division by zero if w_cool is zero
        if abs(w_cool) < 1e-15:
            warnings.warn("w_cool is near zero, E04 gamma parameter will be infinite.")
            # Set derivative to a large number or handle appropriately
            dMfracM0_drfracr0_curr = np.inf  # Or some large number
        else:
            gamma = C_d * fcor * r0 / w_cool

            # Calculate derivative d(M/M0)/d(r/r0) at the current point (ii+1)
            # Avoid division by zero near r/r0 = 1 and potential negative inside sqrt
            M_term = MfracM0_curr - rfracr0_curr**2
            denominator = 1.0 - rfracr0_curr**2
            if denominator <= 1e-9:  # Near r/r0 = 1
                dMfracM0_drfracr0_curr = 0.0  # Derivative is zero at r0
            elif M_term < 0:
                # This shouldn't happen physically in the outer region,
                # indicates numerical issue or profile breakdown.
                warnings.warn(
                    f"E04 profile: M/M0 < (r/r0)^2 at r/r0={rfracr0_curr:.4f}. Setting derivative to 0."
                )
                dMfracM0_drfracr0_curr = 0.0
            else:
                # Original E04 uses M, not V, so it's (M/r - f*r/2)^2 which is (M0/r0 * (M/M0 / (r/r0) - r/r0))^2
                # The MATLAB code uses (M-f*r^2/2)^2 which corresponds to (M0*(M/M0 - (r/r0)^2))^2
                # Let's stick to the MATLAB implementation's formula:
                dMfracM0_drfracr0_curr = gamma * (M_term**2) / denominator

        # Update M/M0 at the next point inwards (index ii) using Euler step
        # Step is negative drfracr0 because we integrate backwards
        MMfracM0[ii] = MfracM0_curr - dMfracM0_drfracr0_curr * drfracr0

        # Check for unrealistic values (M/M0 should decrease inwards from 1)
        if MMfracM0[ii] > MMfracM0[ii + 1] + 1e-9:  # Add tolerance for float errors
            warnings.warn(
                f"E04 profile: M/M0 increased inwards at r/r0={rrfracr0[ii]:.4f}. Stopping E04 integration early."
            )
            MMfracM0[: ii + 1] = np.nan  # Mark remaining points as invalid
            break
        if MMfracM0[ii] < 0:
            warnings.warn(
                f"E04 profile: M/M0 became negative at r/r0={rrfracr0[ii]:.4f}. Stopping E04 integration early."
            )
            MMfracM0[: ii + 1] = np.nan  # Mark remaining points as invalid
            break

    # Clean up potential NaNs at the beginning if integration stopped early
    valid_mask = ~np.isnan(MMfracM0)
    return rrfracr0[valid_mask], MMfracM0[valid_mask]


def _er11_rmax_r0_relation(
    rmax_var: float, Vmax: float, r0: float, fcor: float, CkCd: float
) -> float:
    """
    Function representing the implicit relationship between rmax and r0
    in the ER11 model (Eq. 37, rearranged to be = 0).
    Used for root finding. Finds rmax given r0.

    Args:
        rmax_var (float): Variable radius (rmax) to find.
        Vmax (float): Maximum wind speed (m/s).
        r0 (float): Fixed outer radius (m).
        fcor (float): Coriolis parameter (s^-1).
        CkCd (float): Ratio of surface exchange coefficients Ck/Cd.

    Returns:
        float: Value of the implicit function (should be zero at solution).
    """
    if rmax_var <= 0 or rmax_var >= r0:  # Constraint for physical solution
        return np.inf  # Return large value outside valid range

    M0 = 0.5 * fcor * r0**2
    Mm = Vmax * rmax_var + 0.5 * fcor * rmax_var**2

    if Mm <= 0:  # Avoid issues with non-physical Mm
        return np.inf

    # Avoid division by zero or invalid exponentiation
    if 2 - CkCd == 0:
        # Handle CkCd = 2 case (logarithmic relationship, not covered here)
        warnings.warn("CkCd = 2 case in ER11 rmax/r0 relation not implemented.")
        return np.inf
    if CkCd < 0:  # CkCd is typically >= 0.5
        warnings.warn(f"Unphysical CkCd < 0 ({CkCd}) encountered.")
        # May need different handling or bounds

    ratio_M = M0 / Mm
    ratio_r_sq = (r0 / rmax_var) ** 2

    # Handle potential complex numbers if ratio_M is negative (shouldn't be)
    if ratio_M < 0:
        lhs = -np.inf  # Or handle error appropriately
    else:
        try:
            lhs = ratio_M ** (2.0 - CkCd)
        except ValueError:  # Handle potential complex result if base is negative
            warnings.warn(
                f"Potential complex number in ER11 rmax/r0 LHS (M0/Mm={ratio_M:.2e})"
            )
            return np.inf

    denominator_rhs = 2.0 - CkCd + CkCd * ratio_r_sq
    if denominator_rhs <= 0:  # Avoid division by zero or log(neg) if CkCd=2
        return np.inf  # Or handle error
    rhs = (2.0 * ratio_r_sq) / denominator_rhs

    return lhs - rhs


def _er11_r0_rmax_relation(
    r0_var: float, Vmax: float, rmax: float, fcor: float, CkCd: float
) -> float:
    """
    Function representing the implicit relationship between rmax and r0
    in the ER11 model (Eq. 37, rearranged to be = 0).
    Used for root finding. Finds r0 given rmax.

    Args:
        r0_var (float): Variable radius (r0) to find.
        Vmax (float): Maximum wind speed (m/s).
        rmax (float): Fixed outer radius (m).
        fcor (float): Coriolis parameter (s^-1).
        CkCd (float): Ratio of surface exchange coefficients Ck/Cd.

    Returns:
        float: Value of the implicit function (should be zero at solution).
    """
    if r0_var <= rmax:  # Constraint for physical solution
        return np.inf

    M0 = 0.5 * fcor * r0_var**2
    Mm = Vmax * rmax + 0.5 * fcor * rmax**2

    if Mm <= 0:
        return np.inf

    # Avoid division by zero or invalid exponentiation
    if 2 - CkCd == 0:
        warnings.warn("CkCd = 2 case in ER11 r0/rmax relation not implemented.")
        return np.inf
    if CkCd < 0:
        warnings.warn(f"Unphysical CkCd < 0 ({CkCd}) encountered.")

    ratio_M = M0 / Mm
    ratio_r_sq = (r0_var / rmax) ** 2

    if ratio_M < 0:
        lhs = -np.inf
    else:
        try:
            lhs = ratio_M ** (2.0 - CkCd)
        except ValueError:
            warnings.warn(
                f"Potential complex number in ER11 r0/rmax LHS (M0/Mm={ratio_M:.2e})"
            )
            return np.inf

    denominator_rhs = 2.0 - CkCd + CkCd * ratio_r_sq
    if denominator_rhs <= 0:
        return np.inf
    rhs = (2.0 * ratio_r_sq) / denominator_rhs

    return lhs - rhs


def _er11_radprof_raw(
    Vmax: float,
    r_in: float,
    rmax_or_r0: str,
    fcor: float,
    CkCd: float,
    rr_er11: np.ndarray,
) -> Tuple[NDArray, float]:
    """
    Calculates the raw Emanuel and Rotunno (2011) theoretical wind profile
    without iterative convergence for Vmax/rmax. Determines rmax from r0
    or vice-versa using numerical root finding.

    Args:
        Vmax (float): Maximum wind speed (m/s).
        r_in (float): Input radius (either rmax or r0) (m).
        rmax_or_r0 (str): 'rmax' if r_in is rmax, 'r0' if r_in is r0.
        fcor (float): Coriolis parameter (s^-1).
        CkCd (float): Ratio of surface exchange coefficients Ck/Cd.
        rr_er11 (np.ndarray): Vector of radii (m) for calculation.

    Returns:
        tuple: (V_ER11, r_out)
            - V_ER11 (np.ndarray): Vector of wind speeds (m/s) at rr_er11.
            - r_out (float): The calculated radius corresponding to the
                             *other* input (rmax if r0 was input, r0 if rmax).
                             Returns NaN if calculation fails.
    """
    fcor = abs(fcor)
    rmax = np.nan
    r0 = np.nan
    r_out = np.nan

    # --- Determine rmax and r0 using numerical root finding ---
    try:
        if rmax_or_r0.lower() == "r0":
            r0 = r_in
            # Find rmax that satisfies the relation for the given r0
            # Bracket: rmax must be between 0 and r0
            sol = root_scalar(
                _er11_rmax_r0_relation,
                args=(Vmax, r0, fcor, CkCd),
                bracket=[1e-3, r0 * 0.999],  # Avoid edges
                method="brentq",
            )  # Brentq is robust for brackets
            if sol.converged:
                rmax = sol.root
            else:
                warnings.warn(
                    f"ER11 root finding for rmax (given r0={r0/1000:.1f} km) failed: {sol.flag}"
                )
                return np.full_like(rr_er11, np.nan), np.nan

        elif rmax_or_r0.lower() == "rmax":
            rmax = r_in
            # Find r0 that satisfies the relation for the given rmax
            # Bracket: r0 must be greater than rmax. Need an upper bound guess.
            # Let's guess r0 is not excessively larger than rmax initially.
            r0_guess_low = rmax * 1.001
            r0_guess_high = rmax * 100  # Arbitrary large factor, might need adjustment
            sol = root_scalar(
                _er11_r0_rmax_relation,
                args=(Vmax, rmax, fcor, CkCd),
                bracket=[r0_guess_low, r0_guess_high],
                method="brentq",
            )
            # If initial bracket fails, try expanding the upper bound
            if not sol.converged:
                warnings.warn(
                    f"ER11 root finding for r0 (given rmax={rmax/1000:.1f} km) failed with initial bracket. Trying larger bracket."
                )
                r0_guess_high = rmax * 1000
                sol = root_scalar(
                    _er11_r0_rmax_relation,
                    args=(Vmax, rmax, fcor, CkCd),
                    bracket=[r0_guess_low, r0_guess_high],
                    method="brentq",
                )

            if sol.converged:
                r0 = sol.root
            else:
                warnings.warn(
                    f"ER11 root finding for r0 (given rmax={rmax/1000:.1f} km) failed: {sol.flag}"
                )
                return np.full_like(rr_er11, np.nan), np.nan
        else:
            raise ValueError("rmax_or_r0 must be 'rmax' or 'r0'")

    except ValueError as e:
        # Catch errors from root_scalar (e.g., bracket issues)
        warnings.warn(f"ER11 root finding error: {e}")
        return np.full_like(rr_er11, np.nan), np.nan

    # --- Calculate ER11 Profile ---
    if np.isnan(rmax):
        warnings.warn("ER11 rmax could not be determined.")
        return np.full_like(rr_er11, np.nan), np.nan

    # Angular momentum at rmax
    Mm = Vmax * rmax + 0.5 * fcor * rmax**2

    # Calculate profile using ER11 formula (Eq. 36 rearranged for V)
    # V(r) = (Mm / r) * [ (2*(r/rmax)^2) / (2 - CkCd + CkCd*(r/rmax)^2) ]^(1 / (2 - CkCd)) - 0.5*fcor*r
    V_ER11 = np.full_like(rr_er11, np.nan, dtype=float)
    rmax_sq = rmax**2

    # Avoid division by zero at r=0
    valid_r_mask = rr_er11 > 1e-9

    r_valid = rr_er11[valid_r_mask]
    r_over_rmax_sq = (r_valid / rmax) ** 2

    denominator_term = 2.0 - CkCd + CkCd * r_over_rmax_sq

    # Check for potential issues in the formula terms
    if abs(2.0 - CkCd) < 1e-9:
        warnings.warn(
            "CkCd is close to 2, ER11 formula approaches logarithmic form (not implemented)."
        )
        # Handle CkCd = 2 case separately if needed
        return np.full_like(rr_er11, np.nan), np.nan
    if np.any(denominator_term <= 0):
        warnings.warn("ER11 profile calculation: Denominator term <= 0 encountered.")
        # This might indicate an issue with CkCd or the rmax/r0 relation
        # Mark problematic points as NaN
        denominator_term[denominator_term <= 0] = np.nan

    try:
        # Power term calculation
        base = (2.0 * r_over_rmax_sq) / denominator_term
        # Handle base <= 0 before exponentiation
        if np.any(base <= 0):
            warnings.warn(
                "ER11 profile calculation: Base of power term <= 0 encountered."
            )
            base[base <= 0] = np.nan  # Mark invalid points

        exponent = 1.0 / (2.0 - CkCd)
        power_term = base**exponent

        # Final V calculation for valid radii
        V_ER11[valid_r_mask] = (Mm / r_valid) * power_term - 0.5 * fcor * r_valid

    except (ValueError, FloatingPointError) as e:
        warnings.warn(f"Numerical error during ER11 profile calculation: {e}")
        # V_ER11 will remain NaN for problematic points

    # Set V=0 at r=0
    zero_r_mask = rr_er11 <= 1e-9
    V_ER11[zero_r_mask] = 0.0

    # --- Determine r_out from the calculated profile ---
    # Clean profile: remove NaNs and negative winds for interpolation/max finding
    clean_mask = ~np.isnan(V_ER11) & (V_ER11 >= 0)
    rr_clean = rr_er11[clean_mask]
    V_clean = V_ER11[clean_mask]

    if len(rr_clean) < 2:  # Not enough points to determine r_out
        warnings.warn("ER11 raw profile calculation resulted in < 2 valid points.")
        return V_ER11, np.nan

    if rmax_or_r0.lower() == "r0":
        # Find rmax from the profile
        if len(V_clean) > 0:
            imax = np.argmax(V_clean)
            rmax_profile = rr_clean[imax]
            r_out = rmax_profile
        else:
            r_out = np.nan  # No valid Vmax found
            warnings.warn("Could not find Vmax in calculated ER11 profile (r0 input).")

    elif rmax_or_r0.lower() == "rmax":
        # Find r0 from the profile using interpolation
        imax = np.argmax(V_clean)
        # Interpolate V vs r for r > rmax_profile
        r_outer = rr_clean[imax:]
        V_outer = V_clean[imax:]

        if len(r_outer) >= 2 and V_outer[0] > 1e-6:  # Need points and Vmax > 0
            try:
                # Ensure V is monotonically decreasing for interpolation to find V=0
                # Find the first point where V starts increasing again (if any)
                dV = np.diff(V_outer)
                first_increase_idx = np.where(dV >= 0)[0]
                if len(first_increase_idx) > 0:
                    end_idx = (
                        first_increase_idx[0] + 1
                    )  # Include the peak, stop before increase
                    if end_idx < 2:
                        end_idx = 2  # Need at least 2 points
                    r_outer_interp = r_outer[:end_idx]
                    V_outer_interp = V_outer[:end_idx]
                else:
                    r_outer_interp = r_outer
                    V_outer_interp = V_outer

                if len(r_outer_interp) >= 2 and V_outer_interp[-1] < V_outer_interp[0]:
                    # Interpolate r as a function of V
                    interp_func = interp1d(
                        V_outer_interp,
                        r_outer_interp,
                        kind="linear",  # MATLAB default is linear for interp1
                        bounds_error=False,  # Allow extrapolation slightly
                        fill_value=np.nan,
                    )  # Return NaN if outside range
                    r0_profile = interp_func(0.0)
                    # Check if interpolation yielded a valid result > rmax
                    if np.isnan(r0_profile) or r0_profile <= rr_clean[imax]:
                        # Try PCHIP if linear failed or gave bad result, like MATLAB fallback
                        if len(r_outer_interp) >= 3:  # PCHIP needs at least 3 points
                            try:
                                interp_func_pchip = interp1d(
                                    V_outer_interp,
                                    r_outer_interp,
                                    # kind="pchip",
                                    bounds_error=False,
                                    fill_value=np.nan,
                                )
                                r0_profile = interp_func_pchip(0.0)
                                if np.isnan(r0_profile) or r0_profile <= rr_clean[imax]:
                                    warnings.warn(
                                        "ER11 r0 interpolation (PCHIP) failed or result <= rmax."
                                    )
                                    r0_profile = np.nan
                            except ValueError:
                                warnings.warn("ER11 r0 interpolation (PCHIP) failed.")
                                r0_profile = np.nan
                        else:
                            warnings.warn(
                                "ER11 r0 interpolation failed (linear) and not enough points for PCHIP."
                            )
                            r0_profile = np.nan
                    r_out = r0_profile
                else:
                    warnings.warn(
                        "Could not interpolate r0: V not decreasing beyond rmax or too few points."
                    )
                    r_out = np.nan
            except ValueError as e:
                warnings.warn(f"Error during ER11 r0 interpolation: {e}")
                r_out = np.nan
        else:
            warnings.warn(
                "Could not find Vmax > 0 or not enough points beyond rmax for r0 interpolation."
            )
            r_out = np.nan

    return V_ER11, r_out


def _er11_radprof(
    Vmax_target: float,
    r_in_target: float,
    rmax_or_r0: str,
    fcor: float,
    CkCd: float,
    rr_ER11: NDArray,
) -> Tuple[NDArray, float]:
    """
    Calculates the ER11 profile, iteratively adjusting internal Vmax and r_in
    parameters until the resulting profile matches the target Vmax_target
    and r_in_target (either rmax or r0).

    Args:
        Vmax_target (float): Target maximum wind speed (m/s).
        r_in_target (float): Target input radius (rmax or r0) (m).
        rmax_or_r0 (str): 'rmax' if r_in_target is rmax, 'r0' if r_in_target is r0.
        fcor (float): Coriolis parameter (s^-1).
        CkCd (float): Ratio of surface exchange coefficients Ck/Cd.
        rr_ER11 (np.ndarray): Vector of radii (m) for calculation.

    Returns:
        tuple: (V_ER11, r_out_prof)
            - V_ER11 (np.ndarray): Converged wind profile (m/s). Returns NaNs
                                   if convergence fails.
            - r_out_prof (float): The other radius (r0 or rmax) obtained from
                                  the converged profile. Returns NaN if fails.
    """
    if len(rr_ER11) < 2:
        warnings.warn("ER11 convergence: Input radius vector too short.")
        return np.full_like(rr_ER11, np.nan), np.nan
    dr = rr_ER11[1] - rr_ER11[0]  # Assumes uniform spacing

    # Initial guess for internal parameters = target values
    Vmax_current = Vmax_target
    r_in_current = r_in_target

    max_iter = 25  # Max iterations to prevent infinite loops
    n_iter = 0
    converged = False

    V_ER11 = np.full_like(rr_ER11, np.nan)  # Initialize output
    r_out_prof = np.nan

    while n_iter < max_iter:
        n_iter += 1

        # Calculate raw profile with current internal parameters
        V_ER11_raw, r_out_raw = _er11_radprof_raw(
            Vmax_current, r_in_current, rmax_or_r0, fcor, CkCd, rr_ER11
        )

        # Check if raw calculation failed
        if np.all(np.isnan(V_ER11_raw)):
            warnings.warn(
                f"ER11 convergence failed: Raw profile calculation returned NaNs (iter {n_iter})."
            )
            V_ER11.fill(np.nan)  # Ensure output is NaN
            r_out_prof = np.nan
            break  # Exit loop

        # --- Calculate profile properties (Vmax_prof, r_in_prof) ---
        # Use cleaned profile (non-NaN, non-negative V)
        clean_mask = ~np.isnan(V_ER11_raw) & (V_ER11_raw >= 0)
        rr_clean = rr_ER11[clean_mask]
        V_clean = V_ER11_raw[clean_mask]

        if len(rr_clean) < 2:
            warnings.warn(
                f"ER11 convergence failed: Raw profile had < 2 valid points (iter {n_iter})."
            )
            V_ER11.fill(np.nan)
            r_out_prof = np.nan
            break

        Vmax_prof = np.max(V_clean) if len(V_clean) > 0 else 0.0
        imax_prof = np.argmax(V_clean) if Vmax_prof > 0 else -1

        r_in_prof = np.nan
        if rmax_or_r0.lower() == "rmax":
            if Vmax_prof > 0:
                rmax_prof = rr_clean[imax_prof]
                r_in_prof = rmax_prof
            else:
                warnings.warn(
                    f"ER11 convergence: Could not find Vmax > 0 in profile (iter {n_iter})."
                )
                # Keep r_in_prof as NaN, likely loop will break or adjust badly

        elif rmax_or_r0.lower() == "r0":
            # Interpolate for r0 (similar logic as in _er11_radprof_raw)
            if imax_prof != -1:
                r_outer = rr_clean[imax_prof:]
                V_outer = V_clean[imax_prof:]
                if len(r_outer) >= 2 and V_outer[0] > 1e-6:
                    try:
                        # Find monotonic decreasing part
                        dV = np.diff(V_outer)
                        first_increase_idx = np.where(dV >= 0)[0]
                        end_idx = (
                            first_increase_idx[0] + 1
                            if len(first_increase_idx) > 0
                            else len(r_outer)
                        )
                        if end_idx < 2:
                            end_idx = 2
                        r_outer_interp = r_outer[:end_idx]
                        V_outer_interp = V_outer[:end_idx]

                        if (
                            len(r_outer_interp) >= 2
                            and V_outer_interp[-1] < V_outer_interp[0]
                        ):
                            # Try linear interp first
                            interp_func = interp1d(
                                V_outer_interp,
                                r_outer_interp,
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            r0_prof_interp = interp_func(0.0)
                            if (
                                np.isnan(r0_prof_interp)
                                or r0_prof_interp <= rr_clean[imax_prof]
                            ):
                                # Try PCHIP
                                if len(r_outer_interp) >= 3:
                                    try:
                                        interp_func_pchip = interp1d(
                                            V_outer_interp,
                                            r_outer_interp,
                                            kind="pchip",
                                            bounds_error=False,
                                            fill_value=np.nan,
                                        )
                                        r0_prof_interp = interp_func_pchip(0.0)
                                        if (
                                            np.isnan(r0_prof_interp)
                                            or r0_prof_interp <= rr_clean[imax_prof]
                                        ):
                                            r0_prof_interp = np.nan  # Mark as failed
                                    except ValueError:
                                        r0_prof_interp = np.nan
                                else:
                                    r0_prof_interp = np.nan
                            r_in_prof = (
                                r0_prof_interp  # Assign result (or NaN if failed)
                            )
                        else:
                            r_in_prof = np.nan  # Not decreasing or too few points
                    except ValueError:
                        r_in_prof = np.nan  # Interpolation error
                else:
                    r_in_prof = np.nan  # Vmax=0 or too few points
            else:
                r_in_prof = np.nan  # No Vmax found

        # --- Check for convergence ---
        # Need valid profile values to check
        valid_check = Vmax_prof > 0 and not np.isnan(r_in_prof)

        if valid_check:
            # Calculate errors
            # Use relative error for Vmax, absolute error for r_in (like MATLAB)
            dVmax_err = Vmax_target - Vmax_prof
            drin_err = r_in_target - r_in_prof

            # Convergence criteria (similar to MATLAB)
            vmax_conv_thresh = 1e-2  # Relative error for Vmax
            # Use half the grid spacing for r_in error (absolute)
            # Handle potential zero dr case
            rin_conv_thresh = dr / 2.0 if dr > 0 else 1.0  # Use 1m if dr=0

            if (
                abs(dVmax_err / Vmax_target) < vmax_conv_thresh
                and abs(drin_err) < rin_conv_thresh
            ):
                converged = True
                V_ER11 = V_ER11_raw  # Store the converged profile
                r_out_prof = r_out_raw  # Store the corresponding r_out
                break  # Exit loop

            # --- Adjust internal parameters for next iteration ---
            # Simple additive adjustment based on error (like MATLAB)
            # Adjust r_in first, then Vmax
            r_in_current = r_in_current + drin_err
            Vmax_current = Vmax_current + dVmax_err

            # Add basic checks to prevent parameters becoming non-physical
            if r_in_current <= 0:
                warnings.warn(
                    f"ER11 convergence: r_in_current became non-positive ({r_in_current:.2f}). Resetting slightly positive."
                )
                r_in_current = 1.0  # Reset to small positive value
            if Vmax_current <= 0:
                warnings.warn(
                    f"ER11 convergence: Vmax_current became non-positive ({Vmax_current:.2f}). Resetting slightly positive."
                )
                Vmax_current = 1.0  # Reset to small positive value

        else:
            # Profile calculation failed to produce valid Vmax/r_in_prof
            warnings.warn(
                f"ER11 convergence: Profile Vmax or r_in calculation failed (iter {n_iter}). Stopping."
            )
            V_ER11.fill(np.nan)
            r_out_prof = np.nan
            break

    # End of while loop
    if not converged:
        warnings.warn(
            f"ER11 profile did not converge within {max_iter} iterations "
            f"for target Vmax={Vmax_target:.1f}, r_in={r_in_target/1000:.1f} km ({rmax_or_r0}), Ck/Cd={CkCd:.2f}."
        )
        V_ER11.fill(np.nan)  # Ensure output is NaN if not converged
        r_out_prof = np.nan

    return V_ER11, r_out_prof


def _curve_intersect(
    x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray, min_points: int = 10
) -> Tuple[NDArray, NDArray]:
    """
    Finds intersection points of two curves defined by (x1, y1) and (x2, y2).

    This is a simplified version inspired by MATLAB's curveintersect.m,
    focusing on finding where the difference between the two curves,
    interpolated onto a common grid, changes sign.

    Args:
        x1, y1 (np.ndarray): Data points for curve 1.
        x2, y2 (np.ndarray): Data points for curve 2.
        min_points (int): Minimum number of points for interpolation grid.

    Returns:
        tuple: (x_intersect, y_intersect)
            - x_intersect (np.ndarray): x-coordinates of intersection points.
            - y_intersect (np.ndarray): y-coordinates of intersection points.
            Returns empty arrays if no intersections found or input is invalid.
    """
    # Basic input validation
    if any(arr is None or len(arr) < 2 for arr in [x1, y1, x2, y2]):
        # warnings.warn("Curve intersection input requires at least 2 points per curve.")
        return np.array([]), np.array([])
    if len(x1) != len(y1) or len(x2) != len(y2):
        warnings.warn(
            "Curve intersection input arrays must have matching lengths (x1,y1) and (x2,y2)."
        )
        return np.array([]), np.array([])

    # Ensure data is sorted by x for interpolation
    sort1 = np.argsort(x1)
    x1, y1 = x1[sort1], y1[sort1]
    sort2 = np.argsort(x2)
    x2, y2 = x2[sort2], y2[sort2]

    # Determine overlapping x-range
    x_min = max(np.min(x1), np.min(x2))
    x_max = min(np.max(x1), np.max(x2))

    if x_min >= x_max:  # No overlap
        return np.array([]), np.array([])

    # Create a common, fine x-grid for interpolation
    # Use number of points proportional to the density of the input data
    num_points = max(min_points, len(x1), len(x2)) * 2  # Heuristic factor
    x_common = np.linspace(x_min, x_max, num_points)

    try:
        # Interpolate both curves onto the common grid
        # Use linear interpolation, similar to MATLAB's default interp1
        # Handle cases where x_common might be outside the bounds of x1 or x2
        interp1_func = interp1d(
            x1, y1, kind="linear", bounds_error=False, fill_value=np.nan
        )
        interp2_func = interp1d(
            x2, y2, kind="linear", bounds_error=False, fill_value=np.nan
        )

        y1_common = interp1_func(x_common)
        y2_common = interp2_func(x_common)

        # Calculate the difference between the curves
        y_diff = y1_common - y2_common

        # Find where the difference crosses zero
        # 1. Identify points where y_diff is exactly zero (within tolerance)
        zero_cross_indices = np.where(np.abs(y_diff) < 1e-10)[0]

        # 2. Identify sign changes between adjacent points
        sign_diff = np.sign(y_diff)
        sign_change_indices = np.where(np.diff(sign_diff) != 0)[
            0
        ]  # Index before change

        # Combine indices, removing duplicates
        intersect_indices_approx = np.unique(
            np.concatenate((zero_cross_indices, sign_change_indices))
        )

        x_intersect = []
        y_intersect = []

        # Refine intersection points by linear interpolation between points where sign changes
        for idx in intersect_indices_approx:
            if idx + 1 < len(x_common):  # Ensure we have the next point
                # Check if this index was due to sign change or exact zero
                is_sign_change = idx in sign_change_indices
                is_zero_cross = idx in zero_cross_indices

                if is_zero_cross and not is_sign_change:
                    # If it's an exact zero but not a sign change, it might be a tangent point
                    # or numerical artifact. Include it for now.
                    x_int = x_common[idx]
                    y_int = y1_common[idx]  # or y2_common[idx]
                elif is_sign_change:
                    # Linear interpolation to find the zero crossing between idx and idx+1
                    x_a, x_b = x_common[idx], x_common[idx + 1]
                    y_diff_a, y_diff_b = y_diff[idx], y_diff[idx + 1]

                    # Check if valid for interpolation (avoid division by zero if diffs are same)
                    if (
                        abs(y_diff_b - y_diff_a) > 1e-12
                        and not np.isnan(y_diff_a)
                        and not np.isnan(y_diff_b)
                    ):
                        # Formula for x where line segment crosses zero
                        x_int = x_a - y_diff_a * (x_b - x_a) / (y_diff_b - y_diff_a)
                        # Interpolate the y-value on one of the original curves at x_int
                        y_int = interp1_func(x_int)
                        # Check if y_int is valid (interpolation might fail at exact bounds)
                        if np.isnan(y_int):
                            y_int = interp2_func(x_int)  # Try the other curve

                        if np.isnan(y_int):
                            continue  # Skip if both interpolations fail

                    else:
                        continue  # Skip if cannot interpolate

                else:
                    continue  # Should not happen if logic is correct

                # Add the refined intersection point
                x_intersect.append(x_int)
                y_intersect.append(y_int)

        return np.array(x_intersect), np.array(y_intersect)

    except ValueError as e:
        warnings.warn(f"Curve intersection failed during interpolation: {e}")
        return np.array([]), np.array([])


def _radprof_eyeadj(
    rr_in: np.ndarray,
    VV_in: np.ndarray,
    alpha: float,
    r_eye_outer: float = None,
    V_eye_outer: float = None,
) -> np.ndarray:
    """
    Applies an empirical adjustment to the wind profile within the eye,
    multiplying the wind speed by (r/r_eye_outer)^alpha for r <= r_eye_outer.

    Args:
        rr_in (np.ndarray): Input radius vector (m).
        VV_in (np.ndarray): Input wind speed vector (m/s).
        alpha (float): Exponent for eye wind profile adjustment.
        r_eye_outer (float, optional): Outer radius for eye modification (m).
                                       Defaults to rmax from the profile.
        V_eye_outer (float, optional): Wind speed at r_eye_outer (m/s).
                                       Defaults to Vmax from the profile.

    Returns:
        np.ndarray: Adjusted wind speed vector (m/s).
    """
    VV_out = VV_in.copy()  # Start with a copy of the input data

    # Determine default eye boundary if not provided
    if r_eye_outer is None or V_eye_outer is None:
        # Find Vmax and rmax from the input profile
        valid_mask = ~np.isnan(VV_in) & (VV_in >= 0)
        if np.any(valid_mask):
            Vmax_prof = np.max(VV_in[valid_mask])
            rmax_prof = rr_in[valid_mask][np.argmax(VV_in[valid_mask])]
            if r_eye_outer is None:
                r_eye_outer = rmax_prof
            if V_eye_outer is None:
                V_eye_outer = Vmax_prof
        else:
            warnings.warn(
                "Eye adjustment: Cannot determine default rmax/Vmax from input profile."
            )
            return VV_out  # Return original profile if defaults cannot be found

    # Check if eye boundary is valid
    if r_eye_outer <= 0 or V_eye_outer <= 0:
        warnings.warn(
            f"Eye adjustment: Invalid eye boundary r={r_eye_outer}, V={V_eye_outer}. Skipping adjustment."
        )
        return VV_out

    # Identify eye region
    indices_eye = rr_in <= r_eye_outer
    if not np.any(indices_eye):
        return VV_out  # No points within the eye radius

    rr_eye = rr_in[indices_eye]
    VV_eye = VV_out[indices_eye]  # Get current wind speeds in the eye

    # Calculate normalized radius (handle r_eye_outer=0 case implicitly checked above)
    rr_eye_norm = rr_eye / r_eye_outer

    # Define multiplicative scaling factor (avoid issues if alpha is large/neg)
    try:
        # Ensure base is non-negative for power, though rr_eye_norm should be
        rr_eye_norm_safe = np.maximum(0, rr_eye_norm)
        eye_factor = rr_eye_norm_safe**alpha
    except FloatingPointError:
        warnings.warn(
            f"Eye adjustment: Floating point error calculating eye factor (alpha={alpha}). Skipping."
        )
        return VV_out

    # Multiply original wind data in the eye by the scaling factor
    VV_eye_adjusted = VV_eye * eye_factor

    # Replace eye values in the output array
    VV_out[indices_eye] = VV_eye_adjusted

    # Ensure V(r=0) = 0
    VV_out[rr_in <= 1e-9] = 0.0

    return VV_out


# --- Main Function ---


@timeit
def chavas_et_al_2015_profile(
    Vmax: float,
    r0: float,
    fcor: float,
    Cdvary: int,
    C_d: float,
    w_cool: float,
    CkCdvary: int,
    CkCd_input: float,
    eye_adj: int,
    alpha_eye: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """
    Calculates the Chavas et al. (2015) merged tropical cyclone wind profile.

    Merges the Emanuel (2004) outer profile and Emanuel & Rotunno (2011)
    inner profile by finding a tangent point in M/M0 vs r/r0 space.

    Args:
        Vmax (float): Target maximum wind speed (m/s).
        r0 (float): Outer radius where V=0 (m).
        fcor (float): Coriolis parameter (s^-1).
        Cdvary (int): 0: Outer region Cd = C_d; 1: Outer region Cd = f(V).
        C_d (float): Surface drag coefficient in outer region (if Cdvary=0).
        w_cool (float): Radiative-subsidence rate (m/s).
        CkCdvary (int): 0: Inner region Ck/Cd = CkCd_input;
                         1: Inner region Ck/Cd = f(Vmax).
        CkCd_input (float): Ratio Ck/Cd in inner region (if CkCdvary=0).
        eye_adj (int): 0: Use raw ER11 profile in eye; 1: Apply adjustment.
        alpha_eye (float): Exponent for eye adjustment (if eye_adj=1).

    Returns:
        tuple: (rr, VV, rmax, rmerge, Vmerge, rrfracr0, MMfracM0, rmaxr0,
                MmM0, rmerger0, MmergeM0)
            - rr (np.ndarray): Radius vector (m).
            - VV (np.ndarray): Wind speed vector (m/s).
            - rmax (float): Radius of maximum wind (m).
            - rmerge (float): Radius of merge point (m).
            - Vmerge (float): Wind speed at merge point (m/s).
            - rrfracr0 (np.ndarray): Non-dim radius vector r/r0.
            - MMfracM0 (np.ndarray): Non-dim angular momentum M/M0.
            - rmaxr0 (float): Non-dim rmax/r0.
            - MmM0 (float): Non-dim M/M0 at rmax.
            - rmerger0 (float): Non-dim rmerge/r0.
            - MmergeM0 (float): Non-dim M/M0 at rmerge.
            Returns NaNs for values if calculation fails.
    """
    fcor = abs(fcor)

    # --- Determine Ck/Cd ---
    CkCd = CkCd_input
    if CkCdvary == 1:
        CkCd = CKCD_COEFQUAD * Vmax**2 + CKCD_COEFLIN * Vmax + CKCD_COEFCNST
        # Apply caps as in MATLAB code
        if CkCd < 0.5:
            # warnings.warn("Ck/Cd calculated < 0.5, setting to 0.5.")
            CkCd = 0.5
        if CkCd > 1.9:
            warnings.warn(
                "Ck/Cd calculated > 1.9, capping at 1.9. "
                "Vmax may be outside range for Ck/Cd fit."
            )
            CkCd = 1.9

    # --- Step 1: Calculate E04 Outer Profile (M/M0 vs r/r0) ---
    Nr_e04 = 200000  # Increase points for better resolution near center
    try:
        rrfracr0_E04, MMfracM0_E04 = _e04_outerwind_r0input_nondim_mm0(
            r0, fcor, Cdvary, C_d, w_cool, Nr=Nr_e04
        )
        if len(rrfracr0_E04) < 2:
            raise ValueError("E04 profile calculation yielded < 2 points.")
    except Exception as e:
        warnings.warn(f"Failed to calculate E04 outer profile: {e}")
        nan_arr = np.array([np.nan])
        return (
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    M0_E04 = 0.5 * fcor * r0**2

    # --- Step 2: Find Tangent Point (rmaxr0) via Bisection/Iteration ---
    # Search for rmaxr0 where ER11 M/M0 curve becomes tangent to E04 curve.
    # We iterate rmaxr0 and check for intersections between the profiles.
    # Tangency occurs when the number of intersections transitions from >=1 to 0.

    rmaxr0_min = 0.001
    rmaxr0_max = 0.75  # Initial search range from MATLAB
    rmaxr0_final = np.nan
    rmerger0 = np.nan
    MmergeM0 = np.nan
    VV_ER11_final = None  # Store the profile associated with this
    rr_ER11_final = None  # Store the matching radius vector for VV_ER11_final

    max_iter_merge = 50  # Max iterations for rmaxr0 search
    drmaxr0_thresh = 1e-6  # Convergence threshold for rmaxr0
    iter_merge = 0
    soln_converged = False
    last_valid_rmaxr0 = None  # Keep track of last rmaxr0 that gave intersections

    # Store results from last successful intersection check
    rmerger0_last = np.nan
    MmergeM0_last = np.nan

    # Bisection search loop
    rmaxr0_low = rmaxr0_min
    rmaxr0_high = rmaxr0_max

    while iter_merge < max_iter_merge:
        iter_merge += 1
        rmaxr0_guess = (rmaxr0_low + rmaxr0_high) / 2.0

        # Calculate ER11 profile for this rmaxr0_guess
        rmax_guess = rmaxr0_guess * r0
        drfracrm = 0.01  # Grid spacing relative to rmax for ER11 calc
        if rmax_guess > 100 * 1000:
            drfracrm /= 10.0
        rfracrm_max_er11 = 50.0  # Extend ER11 profile far out
        # Ensure enough points, especially if drfracrm is small
        num_pts_er11 = max(500, int(rfracrm_max_er11 / drfracrm) + 1)
        rrfracrm_ER11 = np.linspace(0, rfracrm_max_er11, num_pts_er11)
        rr_ER11 = rrfracrm_ER11 * rmax_guess

        # Use the converging ER11 function
        V_ER11, _ = _er11_radprof(Vmax, rmax_guess, "rmax", fcor, CkCd, rr_ER11)

        # Check if ER11 calculation succeeded
        if np.all(np.isnan(V_ER11)):
            warnings.warn(
                f"Merge search: ER11 profile failed for rmaxr0_guess={rmaxr0_guess:.4f}. "
                f"Assuming rmaxr0 is too low (reducing Ro). Adjusting search range."
            )
            # If ER11 fails, it often means Ro=Vmax/(f*rmax) is too high,
            # which implies rmax (and rmaxr0) is too small.
            # Treat as if no intersection found, increase lower bound.
            rmaxr0_low = rmaxr0_guess
            num_intersections = 0
        else:
            # Convert ER11 profile to M/M0 vs r/r0 space
            # Ensure rr_ER11 > 0 for M calculation where V is defined
            valid_er11_mask = ~np.isnan(V_ER11) & (rr_ER11 > 1e-9)
            rr_ER11_valid = rr_ER11[valid_er11_mask]
            V_ER11_valid = V_ER11[valid_er11_mask]

            if len(rr_ER11_valid) < 2:
                warnings.warn(
                    f"Merge search: ER11 profile valid points < 2 for rmaxr0_guess={rmaxr0_guess:.4f}."
                )
                # Treat as failure, adjust search range (assume too low)
                rmaxr0_low = rmaxr0_guess
                num_intersections = 0
            else:
                MM_ER11 = rr_ER11_valid * V_ER11_valid + 0.5 * fcor * rr_ER11_valid**2
                rrfracr0_ER11 = rr_ER11_valid / r0
                MMfracM0_ER11 = MM_ER11 / M0_E04

                # Find intersections between E04 and ER11 curves
                x_intersect, y_intersect = _curve_intersect(
                    rrfracr0_E04, MMfracM0_E04, rrfracr0_ER11, MMfracM0_ER11
                )
                num_intersections = len(x_intersect)

        # Adjust search range based on intersections
        if num_intersections == 0:
            # No intersections -> rmaxr0 too small (ER11 curve below E04)
            # Increase the lower bound
            rmaxr0_low = rmaxr0_guess
        else:
            # Intersection(s) found -> rmaxr0 too large (ER11 curve crosses E04)
            # Decrease the upper bound
            rmaxr0_high = rmaxr0_guess
            # Store this as a potential candidate for the merge point
            last_valid_rmaxr0 = rmaxr0_guess
            # Use the mean of intersection points (like MATLAB) as estimate
            rmerger0_last = np.mean(x_intersect)
            MmergeM0_last = np.mean(y_intersect)
            VV_ER11_final = V_ER11  # Store the profile associated with this
            rr_ER11_final = rr_ER11  # Keep the radius vector in sync

        # Check for convergence
        if (rmaxr0_high - rmaxr0_low) < drmaxr0_thresh:
            soln_converged = True
            # The final rmaxr0 should be the highest value that still had
            # intersections (or very close to it). Use last_valid_rmaxr0.
            if last_valid_rmaxr0 is not None:
                rmaxr0_final = last_valid_rmaxr0
                rmerger0 = rmerger0_last
                MmergeM0 = MmergeM0_last
            else:
                # This happens if no intersections were ever found (E04 always above ER11)
                # Could indicate parameter issue or model limitation.
                # Use the upper bound of the search as a best guess? Or fail.
                warnings.warn(
                    "Merge search converged, but no intersections ever found. Using upper bound."
                )
                rmaxr0_final = rmaxr0_high  # Or potentially fail here
                # Need to recalculate ER11 and find closest point if we proceed
                soln_converged = False  # Mark as not truly converged

            break  # Exit loop

    # End of merge search loop

    if not soln_converged or np.isnan(rmaxr0_final):
        # Try adjusting CkCd slightly if convergence failed (like MATLAB fallback)
        warnings.warn(
            f"Merge search failed to converge or find valid rmaxr0. Trying CkCd adjustment (current={CkCd:.3f})."
        )
        # This part is complex to replicate exactly without deeper analysis.
        # For now, we will just fail if the primary search fails.
        # A more robust implementation might try small CkCd adjustments.
        warnings.warn("Merge process failed. Returning NaNs.")
        nan_arr = np.array([np.nan])
        return (
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # Ensure the radius vector matches the stored profile
    if rr_ER11_final is None:
        rr_ER11_final = rr_ER11

    # --- Final Profile Calculation & Merging ---
    rmax = rmaxr0_final * r0
    Mm = Vmax * rmax + 0.5 * fcor * rmax**2
    MmM0 = Mm / M0_E04  # Non-dim M at rmax

    # Recalculate the final ER11 profile corresponding to rmaxr0_final
    # (VV_ER11_final should hold the profile from the last valid intersection check)
    if VV_ER11_final is None or np.all(np.isnan(VV_ER11_final)):
        # If VV_ER11_final wasn't stored correctly, recalculate
        warnings.warn("Recalculating final ER11 profile for merging.")
        drfracrm = 0.01
        if rmax > 100 * 1000:
            drfracrm /= 10.0
        rfracrm_max_er11 = 50.0
        num_pts_er11 = max(500, int(rfracrm_max_er11 / drfracrm) + 1)
        rrfracrm_ER11 = np.linspace(0, rfracrm_max_er11, num_pts_er11)
        rr_ER11 = rrfracrm_ER11 * rmax
        VV_ER11_final, _ = _er11_radprof(Vmax, rmax, "rmax", fcor, CkCd, rr_ER11)
        rr_ER11_final = rr_ER11
        if np.all(np.isnan(VV_ER11_final)):
            warnings.warn("Failed to recalculate final ER11 profile. Returning NaNs.")
            nan_arr = np.array([np.nan])
            return (
                nan_arr,
                nan_arr,
                np.nan,
                np.nan,
                np.nan,
                nan_arr,
                nan_arr,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

    # Define the final radial grid - use relative steps near rmax for precision
    drfracrm_final = 0.01  # Final grid spacing relative to rmax
    rfracrm_max_final = r0 / rmax  # Extend grid to r0
    # Ensure enough points, especially if r0 >> rmax
    num_pts_final = max(1000, int(rfracrm_max_final / drfracrm_final) * 2 + 1)
    rrfracrm_final = np.linspace(0, rfracrm_max_final, num_pts_final)
    rr_final = rrfracrm_final * rmax

    # Prepare ER11 and E04 data for interpolation onto the final grid
    # ER11: M/Mm vs r/rm
    valid_er11_mask = ~np.isnan(VV_ER11_final) & (rr_ER11_final > 1e-9)
    rr_ER11_valid = rr_ER11_final[valid_er11_mask]
    V_ER11_valid = VV_ER11_final[valid_er11_mask]
    MM_ER11 = rr_ER11_valid * V_ER11_valid + 0.5 * fcor * rr_ER11_valid**2
    rrfracrm_ER11_interp = rr_ER11_valid / rmax
    MMfracMm_ER11_interp = MM_ER11 / Mm

    # E04: M/Mm vs r/rm
    rr_E04 = rrfracr0_E04 * r0
    MM_E04 = MMfracM0_E04 * M0_E04
    rrfracrm_E04_interp = rr_E04 / rmax
    MMfracMm_E04_interp = MM_E04 / Mm

    # Create interpolation functions
    # Ensure data is sorted and unique for interp1d
    sort_er11 = np.argsort(rrfracrm_ER11_interp)
    rrfracrm_ER11_interp = rrfracrm_ER11_interp[sort_er11]
    MMfracMm_ER11_interp = MMfracMm_ER11_interp[sort_er11]
    _, unique_er11_idx = np.unique(rrfracrm_ER11_interp, return_index=True)

    sort_e04 = np.argsort(rrfracrm_E04_interp)
    rrfracrm_E04_interp = rrfracrm_E04_interp[sort_e04]
    MMfracMm_E04_interp = MMfracMm_E04_interp[sort_e04]
    _, unique_e04_idx = np.unique(rrfracrm_E04_interp, return_index=True)

    if len(unique_er11_idx) < 2 or len(unique_e04_idx) < 2:
        warnings.warn(
            "Not enough unique points for final interpolation. Returning NaNs."
        )
        nan_arr = np.array([np.nan])
        return (
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            nan_arr,
            nan_arr,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    interp_ER11 = interp1d(
        rrfracrm_ER11_interp[unique_er11_idx],
        MMfracMm_ER11_interp[unique_er11_idx],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_E04 = interp1d(
        rrfracrm_E04_interp[unique_e04_idx],
        MMfracMm_E04_interp[unique_e04_idx],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Merge profiles based on rmerger0
    rmergerfracrm = rmerger0 * r0 / rmax
    MMfracMm_merged = np.full_like(rrfracrm_final, np.nan)

    inner_mask = rrfracrm_final < rmergerfracrm
    outer_mask = rrfracrm_final >= rmergerfracrm

    MMfracMm_merged[inner_mask] = interp_ER11(rrfracrm_final[inner_mask])
    MMfracMm_merged[outer_mask] = interp_E04(rrfracrm_final[outer_mask])

    # Handle potential NaNs from interpolation near boundaries
    # Fill NaNs in the inner part with ER11, outer part with E04 if possible
    nan_mask = np.isnan(MMfracMm_merged)
    if np.any(nan_mask & inner_mask):
        MMfracMm_merged[nan_mask & inner_mask] = interp_ER11(
            rrfracrm_final[nan_mask & inner_mask]
        )
    if np.any(nan_mask & outer_mask):
        MMfracMm_merged[nan_mask & outer_mask] = interp_E04(
            rrfracrm_final[nan_mask & outer_mask]
        )

    # Final check for NaNs
    if np.any(np.isnan(MMfracMm_merged)):
        warnings.warn("NaNs found in merged M/Mm profile after interpolation.")
        # Could try PCHIP interpolation as a fallback here if needed

    # --- Calculate Dimensional Wind Speed ---
    # VV = (Mm / rmax) * (MMfracMm / rrfracrm) - 0.5 * fcor * rmax * rrfracrm
    # Avoid division by zero at r=0
    VV_final = np.full_like(rr_final, np.nan)
    valid_r_mask_final = rrfracrm_final > 1e-9
    r_valid_final = rr_final[valid_r_mask_final]
    rrfracrm_valid_final = rrfracrm_final[valid_r_mask_final]
    MMfracMm_valid_final = MMfracMm_merged[valid_r_mask_final]

    VV_final[valid_r_mask_final] = (Mm / rmax) * (
        MMfracMm_valid_final / rrfracrm_valid_final
    ) - 0.5 * fcor * rmax * rrfracrm_valid_final

    # Set V=0 at r=0
    VV_final[rr_final <= 1e-9] = 0.0

    # Remove any negative wind speeds resulting from numerical issues near r0
    VV_final[VV_final < 0] = 0.0

    # --- Calculate Merge Point Variables ---
    rmerge = rmerger0 * r0
    # Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0) # From M definition
    # Calculate Vmerge by interpolating the final profile
    try:
        interp_V_final = interp1d(
            rr_final, VV_final, kind="linear", bounds_error=False, fill_value=np.nan
        )
        Vmerge = interp_V_final(rmerge)
        if np.isnan(Vmerge):
            # Fallback using M definition if interpolation fails
            if abs(rmerger0) > 1e-9:
                Vmerge = (M0_E04 / r0) * ((MmergeM0 / rmerger0) - rmerger0)
            else:
                Vmerge = 0.0  # Assume V=0 if rmerge is near 0
            Vmerge = max(0, Vmerge)  # Ensure non-negative
    except ValueError:
        warnings.warn("Could not interpolate Vmerge from final profile.")
        Vmerge = np.nan

    # --- Apply Eye Adjustment if requested ---
    if eye_adj == 1:
        # Use Vmax (target) and rmax (calculated) as eye boundary
        VV_final = _radprof_eyeadj(rr_final, VV_final, alpha_eye, rmax, Vmax)

    # --- Final Non-dimensional outputs ---
    rrfracr0_final = rr_final / r0
    MMfracM0_final = MMfracMm_merged * (Mm / M0_E04)

    # --- Return Results ---
    # Ensure rmax corresponds to the peak of the *final* profile
    final_valid_mask = ~np.isnan(VV_final) & (VV_final >= 0)
    if np.any(final_valid_mask):
        Vmax_out = np.max(VV_final[final_valid_mask])
        rmax_out = rr_final[final_valid_mask][np.argmax(VV_final[final_valid_mask])]
    else:
        Vmax_out = np.nan
        rmax_out = np.nan
        warnings.warn("Final profile contains no valid wind speeds.")

    # Return dimensional and key non-dimensional results
    return (
        rr_final,
        VV_final,
        rmax_out,
        rmerge,
        Vmerge,
        rrfracr0_final,
        MMfracM0_final,
        rmaxr0_final,
        MmM0,
        rmerger0,
        MmergeM0,
    )


def process_inputs(inputs: dict) -> dict:
    """Process the input parameters for the CLE15 model.

    Args:
        inputs (dict): Input parameters.

    Returns:
        dict: Processed input parameters.
    """
    # load default inputs
    # ins = read_json(os.path.join(DATA_PATH, "inputs.json"))
    from .constants import (
        CDVARY_DEFAULT,
        CKCDVARY_DEFAULT,
        CK_CD_DEFAULT,
        CD_DEFAULT,
        VMAX_DEFAULT,
        RA_DEFAULT,
        F_COR_DEFAULT,
        EYE_ADJ_DEFAULT,
        ALPHA_EYE_DEFAULT,
        BACKGROUND_PRESSURE,
    )

    ins = {}
    # ins["Vmax"] = VMAX_DEFAULT
    ins["w_cool"] = W_COOL_DEFAULT
    ins["p0"] = BACKGROUND_PRESSURE / 100  # in hPa instead
    ins["CkCd"] = CK_CD_DEFAULT
    ins["Cd"] = CD_DEFAULT
    ins["Cdvary"] = CDVARY_DEFAULT
    ins["CkCdvary"] = CKCDVARY_DEFAULT
    ins["Vmax"] = VMAX_DEFAULT
    ins["r0"] = RA_DEFAULT
    ins["fcor"] = F_COR_DEFAULT
    ins["eye_adj"] = EYE_ADJ_DEFAULT
    ins["alpha_eye"] = ALPHA_EYE_DEFAULT

    if "p0" in inputs:
        assert inputs["p0"] > 900 and inputs["p0"] < 1100  # between 900 and 1100 hPa
    # ins["CkCd"]

    if inputs is not None:
        for key in inputs:
            if key in ins:
                ins[key] = inputs[key]

    # print("inputs", inputs)
    return ins


def run_cle15(
    plot: bool = False,
    inputs: Optional[Dict[str, any]] = None,
    rho0=RHO_AIR_DEFAULT,  # [kg m-3]
    pressure_assumption: str = "isopycnal",
) -> Tuple[float, float, float]:  # pm, rmax, vmax, pc
    """
    Run the CLE15 model.

    Args:
        plot (bool, optional): Plot the output. Defaults to False.
        inputs (Optional[Dict[str, any]], optional): Input parameters. Defaults to None.

    Returns:
        Tuple[float, float, float]: pm [Pa], rmax [m], pc [Pa]
    """
    ins = process_inputs(inputs)  # find old data.
    o = chavas_et_al_2015_profile(
        ins["Vmax"],
        ins["r0"],
        ins["fcor"],
        ins["Cdvary"],
        ins["Cd"],
        ins["w_cool"],
        ins["CkCdvary"],
        ins["CkCd"],
        ins["eye_adj"],
        ins["alpha_eye"],
    )
    ou = {"rr": o[0], "VV": o[1], "rmax": o[2], "rmerge": o[3], "Vmerge": o[4]}
    # print("ou", ou)

    if plot:
        # print(ou)
        # plot the output
        rr = np.array(ou["rr"]) / 1000
        rmerge = ou["rmerge"] / 1000
        vv = np.array(ou["VV"])
        plt.plot(rr[rr < rmerge], vv[rr < rmerge], "g", label="ER11 inner profile")
        plt.plot(rr[rr > rmerge], vv[rr > rmerge], "orange", label="E04 outer profile")
        plt.plot(
            ou["rmax"] / 1000, ins["Vmax"], "b.", label="$r_{\mathrm{max}}$ (output)"
        )
        plt.plot(
            ou["rmerge"] / 1000,
            ou["Vmerge"],
            "kx",
            label="$r_{\mathrm{merge}}$ (input)",
        )
        plt.plot(ins["r0"] / 1000, 0, "r.", label="$r_a$ (input)")

        def _f(x: any) -> float:
            if isinstance(x, float):
                return x
            else:
                return np.nan

        plt.ylim(
            [0, np.nanmax([_f(v) for v in vv]) * 1.10]
        )  # np.nanmax(out["VV"]) * 1.05])
        plt.legend()
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Rotating wind speed, $V$, [m s$^{-1}$]")
        plt.title("CLE15 Wind Profile")
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pm.pdf"), format="pdf")
        plt.clf()

    # integrate the wind profile to get the pressure profile
    # assume wind-pressure gradient balance
    p0 = ins["p0"] * 100  # [Pa] [originally in hPa]
    ou["VV"][-1] = 0  # get rid of None at end of
    ou["VV"][np.isnan(ou["VV"])] = 0  # get rid of None at end of
    assert None not in ou["VV"]
    rr = np.array(ou["rr"], dtype="float32")  # [m]
    vv = np.array(ou["VV"], dtype="float32")  # [m/s]
    # print("rr", rr[:10], rr[-10:])
    # print("vv", vv[:10], vv[-10:])
    p = pressure_from_wind(
        rr,
        vv,
        p0=p0,
        rho0=rho0,
        fcor=ins["fcor"],
        assumption=pressure_assumption,
    )  # [Pa]
    ou["p"] = p.tolist()

    if plot:
        plt.plot(rr / 1000, p / 100, "k")
        plt.xlabel("Radius, $r$, [km]")
        plt.ylabel("Pressure, $p$, [hPa]")
        plt.title("CLE15 Pressure Profile")
        plt.ylim([np.min(p) / 100, np.max(p) * 1.0005 / 100])
        plt.xlim([0, rr[-1] / 1000])
        plt.savefig(os.path.join(FIGURE_PATH, "r0_pmp.pdf"), format="pdf")
        plt.clf()

    return (
        float(
            interp1d(rr, p)(ou["rmax"])
        ),  # find the pressure at the maximum wind speed radius [Pa]
        ou["rmax"],  # rmax radius [m]
        p[0],
    )  # p[0]  # central pressure [Pa]


def profile_from_stats(
    vmax: float, fcor: float, r0: float, p0: float, pressure_assumption="isothermal"
) -> dict:
    """
    Run the CLE15 model with given parameters and return the wind profile.
    This function is a wrapper around the _run_cle15_octave function.

    Args:
        vmax (float): Maximum wind speed [m/s].
        fcor (float): Coriolis parameter [s-1].
        r0 (float): Radius of the storm [m].
        p0 (float): Background pressure [hPa].

    Returns:
        dict: Dictionary containing the wind profile and pressure profile.
    """
    ins = process_inputs({"Vmax": vmax, "fcor": fcor, "r0": r0, "p0": p0})
    o = chavas_et_al_2015_profile(
        ins["Vmax"],
        ins["r0"],
        ins["fcor"],
        ins["Cdvary"],
        ins["Cd"],
        ins["w_cool"],
        ins["CkCdvary"],
        ins["CkCd"],
        ins["eye_adj"],
        ins["alpha_eye"],
    )
    out = {"rr": o[0], "VV": o[1], "rmax": o[2], "rmerge": o[3], "Vmerge": o[4]}
    out["VV"][-1] = 0
    out["p"] = (
        pressure_from_wind(
            out["rr"], out["VV"], p0=p0 * 100, fcor=fcor, assumption=pressure_assumption
        )
        / 100
    )
    return out


# --- Example Usage ---
if __name__ == "__main__":
    # python -m w22.cle15
    print("Running Chavas et al. (2015) TC Profile Calculation Example...")

    # Parameters from CLE15_plot_r0input.m example
    Vmax_in = 50.0  # Target Vmax [m/s]
    r0_in = 900 * 1000.0  # Input r0 [m]
    fcor_in = 5e-5  # Coriolis parameter [s^-1]
    Cdvary_in = 0  # Use constant outer Cd? (0=No, 1=Yes)
    Cd_in = 1.5e-3  # Outer Cd value (if Cdvary_in=0)
    w_cool_in = 2 / 1000.0  # Radiative subsidence [m/s]
    CkCdvary_in = 0  # Use constant inner Ck/Cd? (0=No, 1=Yes)
    CkCd_in = 1.0  # Inner Ck/Cd value (if CkCdvary_in=0)
    eye_adj_in = 0  # Apply eye adjustment? (0=No, 1=Yes)
    alpha_eye_in = 0.15  # Eye adjustment exponent (if eye_adj_in=1)

    # Calculate the profile
    results = chavas_et_al_2015_profile(
        Vmax_in,
        r0_in,
        fcor_in,
        Cdvary_in,
        Cd_in,
        w_cool_in,
        CkCdvary_in,
        CkCd_in,
        eye_adj_in,
        alpha_eye_in,
    )

    # Unpack results
    (
        rr,
        VV,
        rmax,
        rmerge,
        Vmerge,
        rrfracr0,
        MMfracM0,
        rmaxr0,
        MmM0,
        rmerger0,
        MmergeM0,
    ) = results

    # Print key results if calculation succeeded
    if rr is not None and not np.all(np.isnan(rr)):
        print("\n--- Calculation Results ---")
        print(f"Input Vmax:       {Vmax_in:.1f} m/s")
        print(f"Input r0:         {r0_in/1000:.0f} km")
        print(f"Calculated rmax:  {rmax/1000:.1f} km (Non-dim: {rmaxr0:.3f})")
        # Find Vmax from the *output* profile
        Vmax_prof = np.nanmax(VV) if not np.all(np.isnan(VV)) else np.nan
        print(f"Profile Vmax:     {Vmax_prof:.1f} m/s")
        print(f"Merge radius:     {rmerge/1000:.1f} km (Non-dim: {rmerger0:.3f})")
        print(f"Merge wind speed: {Vmerge:.1f} m/s (Non-dim M/M0: {MmergeM0:.3f})")
        print(f"M/M0 at rmax:     {MmM0:.3f}")

        # --- Optional Plotting ---
        try:
            plt.figure(figsize=(10, 6))
            # Plot dimensional profile
            plt.plot(rr / 1000, VV, "b-", linewidth=2, label="Merged Profile (V vs r)")
            plt.plot(
                rmax / 1000,
                Vmax_prof,
                "bx",
                markersize=10,
                mew=2,
                label=f"Rmax ({rmax/1000:.1f} km)",
            )
            plt.plot(
                r0_in / 1000, 0, "r.", markersize=15, label=f"R0 ({r0_in/1000:.0f} km)"
            )
            plt.plot(
                rmerge / 1000,
                Vmerge,
                ".",
                color="grey",
                markersize=12,
                label=f"Merge ({rmerge/1000:.1f} km)",
            )

            plt.xlabel("Radius (km)")
            plt.ylabel("Wind Speed (m/s)")
            plt.title("Chavas et al. (2015) TC Wind Profile")
            plt.legend(loc="best")
            plt.grid(True)
            plt.xlim(
                0, max(1.1 * r0_in / 1000, 5 * rmax / 1000)
            )  # Adjust xlim dynamically
            plt.ylim(
                0,
                (
                    max(Vmax_in * 1.1, Vmax_prof * 1.1)
                    if not np.isnan(Vmax_prof)
                    else Vmax_in * 1.1
                ),
            )
            plt.tight_layout()
            plt.show()

            # Plot non-dimensional profile
            plt.figure(figsize=(10, 6))
            plt.plot(
                rrfracr0,
                MMfracM0,
                "g-",
                linewidth=2,
                label="Merged Profile (M/M0 vs r/r0)",
            )
            plt.plot(
                rmaxr0,
                MmM0,
                "gx",
                markersize=10,
                mew=2,
                label=f"Rmax/R0 ({rmaxr0:.3f})",
            )
            plt.plot(1, 1, "r.", markersize=15, label="R0/R0 (1,1)")  # E04 boundary
            plt.plot(
                rmerger0,
                MmergeM0,
                ".",
                color="grey",
                markersize=12,
                label=f"Merge/R0 ({rmerger0:.3f})",
            )

            # Optionally overlay the individual E04 and ER11 profiles used in merge
            # (Requires recalculating them for plotting)

            plt.xlabel("r / r0")
            plt.ylabel("M / M0")
            plt.title("Non-Dimensional Angular Momentum Profile")
            plt.legend(loc="best")
            plt.grid(True)
            plt.xlim(0, 1.1)
            plt.ylim(0, max(1.1, MmM0 * 1.1) if not np.isnan(MmM0) else 1.1)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nMatplotlib not found. Skipping plots.")
        except Exception as e:
            print(f"\nError during plotting: {e}")

    else:
        print("\nProfile calculation failed. Cannot display results or plots.")
