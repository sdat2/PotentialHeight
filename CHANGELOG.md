v0.1.3:
- Created ability to transform data to required dual graph format for SurgeNet training in `adforce/mesh.py`.
- Created training datasets needed to create the SurgeNet model, by forcing the ADCIRC model with IBTrACS storms from 1980-2024 in `adforce/generate_training_data.py`.
- Created the potential height test set for SurgeNet training in `adbo/create_test_set.py`.


v0.1.2:
- Improved logic for filling in temperature and humidity profile.
- Actually managed to calculate potential sizes on ERA5 Aug/Feb data 1980-2024 for the whole grid 40S to 40N by burning 2000 node hours (128 CPUs per node) on Archer2. This shows parallelization works, but still very strong constraints. 99% of compute is still on the CLE15 profile calculation.
- Improved the potential size calculation in `w22.ps` so that both potential sizes can be calculated together.
- Improved the `w22.stats2` module to generate lots of new tables to describe the CMIP6 results.
- Added additional filters for IBTrACS data in `tcpips.ibtracs` to remove storms that are undergoing extratropical transition. Seems to support hypothesis that most supersize storms are either undergoing ET or have data quality issues.

v0.1.1:
- Added `tcpips/ibtracs.py` script to compare observations with potential sizes and potential intensity calculated from `era5` monthly averages (all post 1980).
- Added more dual graph functionality in `adforce/dual_graph.py` so that we can train different ML algorithms with ADCIRC output, fixing bugs in imp lementation. Is pretty well tested, but takes around a minute to convert each run of ADCIRC to a dual graph (5GB of data).
- `pyproj` now an option for all distance calculations. It is a bit slow though, and now takes 6x longer to make the `fort.22.nc` input file for ADCIRC. Fairly well tested, but could be improved. Adds about 2 minutes to the run time of each adforce call on archer2 (to 5 minutes from 3 minutes).
- Also added a `sphere` option, which will is much faster. Choice is controlled in `adforce/config/grid/grid_fort22.yaml` by `geoid` option. Now defaults to `sphere` option due to speed.
- Added ability to choose between different GP kernels and data acquisition functions in `adbo/exp.py`.
- Explored using different GP kernels in `adbo/gp_exp`.
- Added 1D Bayesian optimization expeirment to `adbo/exp_1d.py` and `adbo/gp_exp.py`.
- Added `lc12` asymmetry option to `adforce/fort22.py`.
- Added Ide et al. 2022 curved parabolic tropical cyclone tracks to `adforce/fort22.py` and `adforce/geo.py`.
- Added new python implementation of calculating the CLE15 profile at `w22/cle15.py`, which is at least 10x faster, has trivial parallelization, has less artifacts, but is less numerically stable than original matlab implementation. In practice on archer2, it turns 
out to be about 10x faster than the matlab implementation, and is now the default.
- Improved figure 2 in `w22/plot.py` to use `figure_two()` function, which is now more flexible and can be used for different places, and plots using ERA5 data as well as CMIP6 data.
- Toy example of pathological extrapolation for surgenet.

v0.0.3:
- Improved `era5` data download script.
- Labels for `figure_two.pdf` now in correct locations.
