# comp — historical surge validation against tide gauges

Validates the historical ADCIRC storm-surge simulations (the SurgeNet training set,
228 IBTrACS North-Atlantic landfalling TCs on the EC95d mesh, published on Hugging
Face as [`sdat2/surgenet-train`](https://huggingface.co/datasets/sdat2/surgenet-train))
against de-tided NOAA CO-OPS tide-gauge observations.

The historical ADCIRC runs use realistic NWS=20 GAHM forcing from IBTrACS but
**exclude tides**, so the simulated sea-surface height is the storm-surge component
only. We therefore compare it against the **de-tided observed residual**, not total
water level.

## Method

For each storm:

1. download the storm netCDF from Hugging Face;
2. extract simulated surge `SSH = WD + DEM` at the nearest *wet* mesh node to each
   NOAA CO-OPS water-level gauge in a NW-Gulf box (Texas → Florida panhandle);
3. de-tide the gauge record with a robust [`utide`](https://github.com/wesleybowman/UTide)
   harmonic fit on the storm's calendar year (falls back to CO-OPS `predictions`);
4. compare peak surge and peak timing; score "clean" pairs (complete record,
   simultaneous peak, meaningful surge, not a documented gauge failure).

## Run

```bash
python -m comp.validate                       # full 14-storm sweep
python -m comp.validate --storms "Ida 2021"   # one storm
```

Outputs: `img/comp/val_scatter.png` and `data/comp/out/val_summary.csv`.
Downloads/cache live under `data/comp/` (git-ignored).

## Configuration

All knobs are in [`constants.py`](constants.py): the `STORMS` list, `GAUGE_BOX`,
`KNOWN_FAILED` gauge failures, node-selection thresholds, and the "clean" filter.

## Result (14 storms, 2005–2023)

152 clean gauge-storm pairs: **r = 0.89, RMSE = 0.54 m, bias = −0.30 m**, median
peak-timing error 2.0 h. The slight low bias is consistent with the omitted wave
setup/runup and medium mesh resolution; over-predictions concentrate at shallow
semi-enclosed bay/pass gauges during direct landfalls.

## Dependencies

`huggingface_hub`, `utide`, `xarray`, `scipy`, `pandas`, `requests`, `matplotlib`.
