"""A driver for calculating potential sizes using CMIP6 data.

There are two types of potential size we should calculate:
(1) The potential size corresponding to the storm at its potential intensity, and (2) the potential size corresponding to the storm at the lower bound of 33 m/s for a category 1 tropical cyclone.

Each of these types of potential size requires a calculation with a different "vmax" input. They will produce a different
output for rmax and r0 (the radius of maximum wind and the radius of vanishing winds, respectively).
"""
