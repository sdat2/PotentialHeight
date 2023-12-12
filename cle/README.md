# CLE15

CLE_plot_r0input.m calls 
E1104_nondim_r0input with dimensional inputs

E1104_nondim_r0input calls E04_outerwind_r0input_nondim_MM0, again with still dimensional inputs.

E1104_nondim_r0input also calls ER11_radprof a large number of times.

curveintersect is called a huge number of times after nondimensionalising.
