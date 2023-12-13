import numpy as np
import matplotlib.pyplot as plt
from er11e04.r0input import ER11E04_nondim_r0input
from sithom.plot import plot_defaults

plot_defaults()

# Assuming the function ER11E04_nondim_r0input is already defined in Python

# Storm parameters
Vmax = 50  # [ms-1]
r0 = 900 * 1000  # [m]
fcor = 5e-5  # [s-1]

# Environmental parameters
# Outer region
Cdvary = 0
Cd = 1.5e-3
w_cool = 2 / 1000  # [ms-1]

# Inner region
CkCdvary = 0
CkCd = 1

# Eye adjustment
eye_adj = 0
alpha_eye = 0.15

# Get profile
rr, VV, rmax, rmerge, Vmerge = ER11E04_nondim_r0input(
    Vmax, r0, fcor, Cdvary, Cd, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye
)

# Plotting
plt.figure(figsize=(12, 12))
plt.plot(rr / 1000, VV, "b", linewidth=3, label="Model")
plt.plot(rmax / 1000, Vmax, "bx", markersize=20, label="r_max")
plt.plot(r0 / 1000, 0, "r.", markersize=40, label="r_0 (input)")
plt.plot(
    rmerge / 1000, Vmerge, ".", color=[0.5, 0.5, 0.5], markersize=30, label="r_merge"
)

xmax_pl = np.ceil((1.01 * r0 / 1000) / 100) * 100
ymax_pl = np.ceil(1.01 * Vmax / 5) * 5
plt.axis([0, xmax_pl, 0, ymax_pl])
plt.xlabel("radius [km]")
plt.ylabel("rotating wind speed [ms^{-1}]")
plt.legend(loc="northeast")
plt.grid(True)
plt.title("CLE15 Wind Profile Plot")

# Save plot
plt.savefig("CLE15_plot_r0input.pdf", format="pdf")
plt.show()
