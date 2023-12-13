import numpy as np
from scipy.interpolate import interp1d
from sithom.time import timeit
from chavas15.e04.outerwind_r0input_MM0 import E04_outerwind_r0input_nondim_MM0
from chavas15.er11.radprof import ER11_radprof
from chavas15.intersect import curveintersect


@timeit
def ER11E04_nondim_r0input(
    Vmax: float,
    r0: float,
    fcor: float,
    Cdvary: float,
    C_d: float,
    w_cool: float,
    CkCdvary: bool,
    CkCd: float,
    eye_adj: bool,
    alpha_eye: bool,
    Nr: int = 100000,
):
    fcor = abs(fcor)

    # Overwrite CkCd if want varying (quadratic fit to Vmax from Chavas et al. 2015)
    if CkCdvary:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst

    if CkCd > 1.9:
        CkCd = 1.9
        print("Ck/Cd is capped at 1.9 and has been set to this value.")

    # Step 1: Calculate E04 M/M0 vs. r/r0
    # Define the function E04_outerwind_r0input_nondim_MM0
    rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(
        r0, fcor, Cdvary, C_d, w_cool, Nr
    )
    from matplotlib import pyplot as plt

    plt.plot(rrfracr0_E04, MMfracM0_E04, "blue", label="E04")
    plt.xlabel("$r$/$r_0$ [dimensionless]")
    plt.ylabel("$M$/$M_0$ [dimensionless]")
    plt.legend()
    plt.title("Initial E04 outer wind model, r0={:.0f} km".format(r0 / 1000))
    plt.savefig("test/e04_r0input_nondim.png")
    plt.close()

    M0_E04 = 0.5 * fcor * r0**2

    # Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point with E04 M/M0

    rmerger0 = None

    soln_converged = False
    while not soln_converged:
        # break up into 3 points, take 2 between which intersection vanishes, repat until this converges.
        rmaxr0_min = 0.001
        rmaxr0_max = 0.75
        rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 1  # guess middle
        rmaxr0 = rmaxr0_new
        drmaxr0 = rmaxr0_max - rmaxr0
        drmaxr0_thresh = 1e-6
        i = 0
        rfracrm_min = 0  # [dimensionless] # start at r=0
        rfracrm_max = 50  # [dimensionless] # many rmaxs away
        while abs(drmaxr0) >= drmaxr0_thresh:
            i += 1
            # dimensionalize rmax
            rmax = rmaxr0_new * r0
            drfracrm = 0.01
            if rmax > 100 * 1000:  # large storm > 100km to rmax
                print("Large storm detected. Increasing drfracrm.")
                drfracrm = drfracrm / 10  # extra precision for large storm
            rrfracrm_ER11 = np.linspace(
                rfracrm_min,
                rfracrm_max,
                num=int((rfracrm_max - rfracrm_min) // drfracrm),
            )
            rr_ER11 = rrfracrm_ER11 * rmax
            rmax_or_r0 = "rmax"
            VV_ER11, _ = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)

            # what does this mean in matlab?
            if ~np.isnan(np.nanmax(VV_ER11)):  # ER11_radprof converged.
                # nondimensionalize?
                rrfracr0_ER11 = rr_ER11 / r0  # divide by outer radius
                MMfracM0_ER11 = (rr_ER11 * VV_ER11 + 0.5 * fcor * rr_ER11) / M0_E04

                # Testing: Plot radial profile, mark rrad, and plot E04 model fits and rmaxs
                from matplotlib import pyplot as plt

                plt.plot(rrfracr0_ER11, MMfracM0_ER11, "black", label="ER11")
                plt.plot(rrfracr0_E04, MMfracM0_E04, "blue", label="E04")
                plt.xlabel("$r$/$r_0$ [dimensionless]")
                plt.ylabel("$M$/$M_0$ [dimensionless]")
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.legend()
                plt.title("Matching up ER11 and E04 absolute angular momentum profiles")
                plt.savefig("test/tester11e041.png")
                plt.close()

                plt.plot(rr_ER11 / 1000, VV_ER11, "k", label="ER11")
                rr_temp = rrfracr0_E04 * r0
                # print("rr_temp.shape", rr_temp.shape)
                # print("MMfracM0_E04.shape", MMfracM0_E04.shape)
                VV_temp = MMfracM0_E04 * M0_E04 / rr_temp - 0.5 * fcor * rr_temp
                plt.plot(rr_temp / 1000, VV_temp, "blue", label="E04")
                plt.xlabel("$r$ [km]")
                plt.ylabel("$V$ [m/s]")
                plt.legend()
                plt.title(
                    "$r_0$={:.0f} km, $r_{{max}}$={:.0f} km".format(
                        r0 / 1000, rmax / 1000
                    )
                )
                plt.savefig("test/tester11e0412.png")
                plt.close()

                x0, y0 = curveintersect(
                    rrfracr0_E04,
                    MMfracM0_E04,
                    rrfracr0_ER11[rrfracr0_ER11 < 1],
                    MMfracM0_ER11[rrfracr0_ER11 < 1],
                )
                if len(x0) == 0:
                    drmaxr0 = abs(drmaxr0) / 2
                else:
                    drmaxr0 = -abs(drmaxr0) / 2
                    rmerger0 = np.mean(x0)
                    MmergeM0 = np.mean(y0)
            else:
                drmaxr0 = -abs(drmaxr0) / 2

            rmaxr0 = rmaxr0_new
            rmaxr0_new = rmaxr0_new + drmaxr0

        if np.isnan(np.nanmax(VV_ER11)) and rmerger0 is not None:
            soln_converged = True
            print("Convergence achieved.")
        else:
            soln_converged = False
            CkCd = CkCd + 0.1
            print("Adjusting CkCd to find convergence.")

    M0 = 0.5 * fcor * r0**2
    Mm = 0.5 * fcor * rmax**2 + rmax * Vmax
    MmM0 = Mm / M0
    ii_ER11 = rrfracr0_ER11 < rmerger0 and MMfracM0_ER11 < MmergeM0
    ii_E04 = rrfracr0_E04 >= rmerger0 and MMfracM0_E04 > MmergeM0
    rrfracr0_temp = np.concat([rrfracr0_ER11(ii_ER11), rrfracr0_E04(ii_E04)])
    MMfracM0_temp = np.concat([MMfracM0_ER11(ii_ER11), MMfracM0_E04(ii_E04)])
    del ii_E04
    del ii_ER11

    # Final Step: Implement interpolation and calculation of final outputs
    # This will involve using the interp1d function from SciPy and the results from the previous steps

    drfracrm = 0.01
    rfracrm_min = 0
    rfracrm_max = r0 / rmax
    rrfracrm = np.linspace(
        rfracrm_min,
        rfracrm_max + drfracrm,
        num=int((rfracrm_max, +drfracrm - rfracrm_min) // drfracrm),
    )

    MMfracMm = interp1d(
        rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm), rrfracrm, "pchip"
    )

    rrfracr0 = rrfracrm * rmax / r0
    MMfracM0 = MMfracMm * Mm / M0

    # Example of interpolation (details depend on the actual data and logic)
    # interpolator = interp1d(rrfracr0_temp, MMfracM0_temp, kind='cubic')
    # MMfracM0 = interpolator(rrfracr0)

    VV = (Mm / rmax) * (MMfracMm / rrfracrm) - 0.5 * fcor * rmax * rrfracrm
    rr = rrfracrm * rmax

    # Make sure V-0 at r=0
    VV[rr == 0] = 0

    rmerge = rmerger0 * r0  # [m]
    Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [m s-1]

    if True:
        from matplotlib import pyplot as plt

        plt.plot(rr / 1000, VV, "blue")
        plt.plot(rmerge / 1000, Vmerge, "r*")
        plt.xlabel("$r$ [km]")
        plt.ylabel("$V$ [m/s]")
        plt.title("ER11E04")
        plt.savefig("test/er11e04test.png")

    # Return the calculated values
    return (
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
    )


# Example usage
# rr, VV, rmax, rmerge, Vmerge, rrfracr0, MMfracM0, rmaxr0, MmM0, rmerger0, MmergeM0 = ER11E04_nondim_r0input(Vmax, r0, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye)
