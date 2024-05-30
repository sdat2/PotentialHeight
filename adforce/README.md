# ADFORCE

Force ADCIRC with an idealized azimuthally symetric straight-track tropical cyclone (using NWS=13).

## Run an idealized storm surge

```bash
python -m adforce.wrap --exp_name notide-ex-NEW_ORLEANS --profile_name 2025.json --stationid 3 --resolution mid-notide

python -m adforce.wrap --exp_name notide-5sec --profile_name 2025.json  --resolution mid-notide


python -m adforce.wrap --exp_name notide-20sec --profile_name 2025.json  --resolution mid-notide
```

## Animate a run:

```bash
python -m adforce.ani --path_in . --step_size 1
```

### Problems with fort.15 file/owi_netCDF.nc

```txt
Error termination. Backtrace:
At line 294 of file /work/n02/n02/sdat2/adcirc-swan/adcirc/src/owiwind_netcdf.F
Fortran runtime error: Bad value during integer read
```