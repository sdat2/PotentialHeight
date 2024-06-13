# ADFORCE

Force ADCIRC with an idealized azimuthally symetric straight-track tropical cyclone using netcdf (NWS=13, ADCIRC>=55.02).

At the moment it takes the time and space coordinates from an existing netcdf in `fort22.py` to create the forcing file.

## Organization

Some functions are organized around the file from ADCIRC they read/write.

- fort15 - write example fort.15 namelists using `adcircpy`.
- fort22 - write wind/pressure fort.22.nc files to force ADCIRC.
- fort61 - Read the tide gauge file fort.61.nc.
- fort63 - Read the ssh/wind output timeseries fort.63.nc file.

Then there are some general utility files

- mesh - generic fast mesh reading/processing utilities.
- profile - read the profiles created by `cle`.
- ani - animate ARCHER2 runs winds/ssh.
- plot - produce paper figures.
- constants - some geographic constants etc. for ADFORCE.

And the main logic to run the ADCIRC model is given in:

- wrap - run the ADCIRC model in parallel for nodes on archer2 supercomputer.

When changing to new machines, it is likely that `wrap` is the main file that needs to be edited.


## TC characteristics:

  - profile (assumed not to change, kept at PI\&PS limit calculated in `tcpips`)
  - track characteristics (varied BayesOpt loop `adbo`):
    - angle (bearing from due north)
    - trans_speed (translation speed of tropical cyclone)
    - displacement (E/W, relative to observation location)
  - observation location.
  - resolution of the model.
  - [ADCIRC settings (e.g. run up, model settings, tides)]
  - [impact time of tropical cyclone (only important relative to tides, and possibly ramp)]


Possible new TC characteristics:

 - Curve / alow angle/trakc displacement to be specified in another way.
 - Profile varying over time
 



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

