"""A driver to run the whole pipeline for CMIP6 data.


Steps:
1. Download data,
2. Regrid data,
3. Calculate potential intensity,
4. Calculate potential sizes
5. Calculate biases (possibly after regridding on to either ERA5 or the regridded CMIP6 grid).

"""
