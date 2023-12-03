import numpy as np
from scipy.interpolate import interp1d


def ER11E04_nondim_r0input(
    Vmax, r0, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye
):
    fcor = abs(fcor)

    # Overwrite CkCd if want varying (quadratic fit to Vmax from Chavas et al. 2015)
    if CkCdvary == 1:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst

    if CkCd > 1.9:
        CkCd = 1.9
        print("Ck/Cd is capped at 1.9 and has been set to this value.")

    # Step 1: Calculate E04 M/M0 vs. r/r0
    Nr = 100000
    # Define the function E04_outerwind_r0input_nondim_MM0
    rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(
        r0, fcor, Cdvary, C_d, w_cool, Nr
    )

    M0_E04 = 0.5 * fcor * r0**2

    # Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point with E04 M/M0
    # This step involves several iterations and calls to other functions like ER11_radprof
    # The detailed implementation of this step will depend on the logic and calculation specifics of the original MATLAB code

    # Final Step: Implement interpolation and calculation of final outputs
    # This will involve using the interp1d function from SciPy and the results from the previous steps

    # Example of interpolation (details depend on the actual data and logic)
    # interpolator = interp1d(rrfracr0_temp, MMfracM0_temp, kind='cubic')
    # MMfracM0 = interpolator(rrfracr0)

    # Return the calculated values
    # return rr, VV, rmax, rmerge, Vmerge, rrfracr0, MMfracM0, rmaxr0, MmM0, rmerger0, MmergeM0


# Placeholder for the sub-function, needs to be defined
def E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr):
    # Implement the logic for this function
    pass


# Placeholder for the sub-function, needs to be defined
def ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr):
    # Implement the logic for this function
    pass


# Example usage
# rr, VV, rmax, rmerge, Vmerge, rrfracr0, MMfracM0, rmaxr0, MmM0, rmerger0, MmergeM0 = ER11E04_nondim_r0input(Vmax, r0, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye)
