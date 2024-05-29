# ADFORCE

Force ADCIRC with an idealized azimuthally symetric straight-track tropical cyclone (using NWS=13).

```bash
python -m adforce.wrap --exp_name notide-ex-NEW_ORLEANS --profile_name 2025.json --stationid 3 --resolution mid-notide
```

## Animate a run:

```bash
python -m adforce.ani --path_in . --step_size 1
```

