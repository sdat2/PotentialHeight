# TC potential intensity and potential size

CMIP6/{STAGE}/{EXP}/{TYPE}/{MODEL}/{MEMBER}.nc


## Workflow:

- pangeo -- download data
- regrid -- regrid data
- pi/convert -- process data for potential intensity
- ps -- process data for potential size


stage = {RAW, REGRIDDED, BIAS_CORRECTED, PI}
EXP = {"ssp585", "historical"}
model =  {any possible cmip6 model}
member = {any possible cmip6 ensemble member}

CMIP6/{STAGE}/{EXP}/{TYPE}/{MODEL}/{MEMBER}.nc
locks are created in CMIP6/{stage}/{exp}.{typ}.{model}.{member}.lock
