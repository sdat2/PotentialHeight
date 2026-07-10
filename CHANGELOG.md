v0.1.4 (unreleased):
- REFACTORED: `adforce/generate_training_data.py` (1088-line hurried single file)
  split into the documented `adforce/training/` package (`storms`, `atcf`, `cfl`,
  `inputs`, `driver`); the old module remains as a behavior-identical CLI shim
  (`python -m adforce.generate_training_data` unchanged, --help byte-identical).
  Function bodies moved unchanged (AST-verified); 29 characterization tests added
  (`tests/test_training_data.py`), captured against the original module first.
  BEHAVIOR CHANGE (intentional): the driver now composes its config via
  `adforce.wrap.get_default_config()`, so the `ADCIRC_EXE_PATH`/`ADCIRC_NP`/
  `WORSTSURGE_MODULES` machine-local env overrides apply to training-data
  generation too (previously bypassed via a bare `hydra.compose`).
- FIXED (correctness): the Wang-2022 y -> p_m Carnot back-conversion in `w22/ps.py`
  dropped the ambient relative-humidity factor from its numerator at all three
  solver entry points, so solved potential sizes were effectively the rh = 1
  answer regardless of the input humidity (canonical point: r0 -2.7%, rmax -4.5%
  at rh = 0.9; expected -4-8% at typical CMIP6 Gulf humidities). Single shared
  implementation now in `w22.w22_carnot.carnot_pm_from_y`; regression tests in
  `tests/test_ps_units.py`; golden pins regenerated. See REPRODUCE.md for the
  list of artifacts that predate the fix.
- FIXED: `point_solution_ps` silently ignored the `env_humidity` and `cd_ck`
  input keys (canonical keys `rh`/`ck_cd`; aliases now accepted). The canonical
  Wang test had been running at CkCd = 0.9 instead of the intended 1.
- FIXED: missing `return` in `adforce.profile.pressures_profile` (fallback path
  always returned None); `read_profile` now rejects profiles whose far-field
  pressure is not in hPa (guards against the legacy Pa-unit JSONs).
- FIXED: `w22.plot.timeseries_plot` mutated the dataset in place (km conversion)
  and the profile writer silently depended on it; `qair2rh` produced rh clipped
  to 1.0 for plain-float pressure inputs in Pa.
- Unit-annotation sweep of the potential-size path: bisection tolerance renamed
  `R0_BISECTION_TOLERANCE` (it is 1 m of r0 bracket, not "1 mbar"/"1 Pa");
  corrected unit comments (F_COR s-1, R_v J/kg/K, L_v provenance, Buck Pa);
  cle15n run_cle15 (Pa) vs profile_from_stats ('p' in hPa) documented.

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
