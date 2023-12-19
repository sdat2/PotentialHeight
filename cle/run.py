"""Run the CLE15 model with json files."""

import numpy as np
from matplotlib import pyplot as plt
from sithom.io import read_json
from sithom.plot import plot_defaults

plot_defaults()

ins = read_json("inputs.json")
print(ins)

# Storm parameters

# run octave file r0_pm.m
# os.system("octave r0_pm.m")

# read in the output from r0_pm.m
ou = read_json("outputs.json")
print(ou)

# plot the output
rr = np.array(ou["rr"]) / 1000
rmerge = ou["rmerge"] / 1000
vv = np.array(ou["VV"])
plt.plot(rr[rr < rmerge], vv[rr < rmerge], "g", label="ER11 inner profile")
plt.plot(rr[rr > rmerge], vv[rr > rmerge], "orange", label="E04 outer profile")
plt.plot(ou["rmax"] / 1000, ins["Vmax"], "b.", label="$r_{\mathrm{max}}$ (output)")
plt.plot(ou["rmerge"] / 1000, ou["Vmerge"], "kx", label="$r_{\mathrm{merge}}$ (input)")
plt.plot(ins["r0"] / 1000, 0, "r.", label="$r_a$ (input)")
plt.ylim([0, 55])  # np.nanmax(out["VV"]) * 1.05])
plt.legend()
plt.xlabel("Radius, $r$, [km]")
plt.ylabel("Rotating wind speed, $V$, [m s$^{-1}$]")
plt.title("CLE15 Wind Profile")
plt.savefig("r0_pm.pdf", format="pdf")
