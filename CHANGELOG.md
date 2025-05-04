v0.0.4:
- Added `tcpips/ibtracs.py` script to compare observations with potential and intensity calculated from `era5` monthly averages (all post 1980).
- Added more dual graph functionality in `adforce/dual_graph.py` so that we can train different ML algorithms with ADCIRC output, fixing bugs in imp lementation. Is pretty well tested, but takes around a minute to convert each run of ADCIRC to a dual graph (5GB of data).
- `pyproj` now an option for all distance calculations. It is a bit slow though, and now takes 6x longer to make the `fort.22.nc` input file for ADCIRC. Fairly well tested, but could be improved. Adds about 2 minutes to the run time of each adforce call on archer2 (to 5 minutes from 3 minutes).
- Also added a `sphere` option, which will is much faster. Choice is controlled in `adforce/config/grid/grid_fort22.yaml` by `geoid` option. Now defaults to `sphere` option due to speed.
- Added ability to choose between different GP kernels and data acquisition functions in `adbo/exp.py`.
- Explored using different GP kernels in `adbo/gp_exp`.
- Added 1D Bayesian optimization expeirment to `adbo/exp_1d.py` and `adbo/gp_exp.py`.
- Added `lc12` asymmetry option to `adforce/fort22.py`.
- Added Ide et al. 2022 curved parabolic tropical cyclone tracks to `adforce/fort22.py` and `adforce/geo.py`.

v0.0.3:
- Improved `era5` data download script.
- Labels for `figure_two.pdf` now in correct locations.
