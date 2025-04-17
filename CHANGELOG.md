v0.0.4:
- Added `ibtracs` scripts to compare observations with potential and intensity calculated from `era5` monthly averages (all post 1980).
- Added more `dual_graph` functionality so that we can train different ML algorithms with ADCIRC output, fixing bugs in implementation.
- `pyproj` now used for all distance calculations so no more flat earth approximations (it is a bit slow though)!

v0.0.3:
- Improved `era5` data download script.
- Labels for figure_two.pdf now in correct locations.
