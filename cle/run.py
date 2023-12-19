"""Run the CLE15 model with json files."""

import numpy as np
from matplotlib import pyplot as plt
from sithom.io import read_json
from sithom.plot import plot_defaults

plot_defaults()

print(read_json("inputs.json"))

# run octave file r0_pm.m
# os.system("octave r0_pm.m")

# read in the output from r0_pm.m
out = read_json("outputs.json")
print(out)

# plot the output
plt.plot(np.array(out["rr"]) / 1000, out["VV"], "k")
# plt.plot(out["rmax"] / 1000, out["Vmax"], "kx")
# plt.plot(out["r0"] / 1000, 0, "r.")
plt.plot(out["rmerge"] / 1000, out["Vmerge"], "k.")
plt.ylim([0, 60])  # np.nanmax(out["VV"]) * 1.05])
plt.xlabel("Radius [km]")
plt.ylabel("Rotating wind speed, $V$, [m s$^{-1}$]")
plt.title("CLE15 Wind Profile Plot")
plt.savefig("r0_pm.pdf", format="pdf")
