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
4. score skill on three axes:
   - **peak** surge bias / RMSE / across-gauge correlation (with 5–95% bootstrap CIs),
     plus a within-storm *spatial* correlation that removes between-storm magnitude spread;
   - **time series** — the simulated hydrograph is interpolated onto the hourly observed
     residual and scored by temporal correlation and RMSE (`ts_r`, `ts_rmse`);
   - **timing** — peak-time difference, reported over *valid* pairs (meaningful surge, not
     gauge-failed) so it is **not** conditioned on the simultaneous-peak filter.
5. tag "clean" pairs (meaningful surge, not a documented gauge failure, simultaneous peak)
   used for the peak-skill scatter and table.

## Run

```bash
python -m comp.validate                       # full 14-storm sweep
python -m comp.validate --storms "Ida 2021"   # one storm (skips example panels + table)
```

A full sweep regenerates **everything the paper uses**, in one step, so the figures, the
table and the prose cannot drift apart:

| output | path |
| --- | --- |
| scatter (quick-look / paper) | `img/comp/val_scatter.png` · `<thesis>/img/comp_val_scatter.pdf` |
| example time series          | `img/comp/val_examples.png` · `<thesis>/img/comp_val_examples.pdf` |
| per-storm LaTeX table        | `<thesis>/paper/comp_val_table.tex` (`\input` by the appendix) |
| summary table                | `data/comp/out/val_summary.csv` |

The thesis tree is located by searching for `paper/appendix.tex`; override with the
`WORSTSURGE_PAPER_ROOT` env var. Downloads/cache live under `data/comp/` (git-ignored).

## Is the signal real? (negative controls + sensitivity)

`val_summary.csv` is enough to falsify the result without re-running ADCIRC:

```bash
python -m comp.nulltest            # permutation, cross-storm, and temporal-lag nulls
python -m comp.nulltest --no-lag   # peak-level nulls only (no netCDF)
python -m comp.sensitivity         # threshold + node-selection robustness
python -m comp.sensitivity --no-node
```

`comp.nulltest` writes `img/comp_val_nulltests.pdf` and reports: a label-permutation null
(observed r=0.89 vs null max 0.31 over 5000 shuffles, p<1e-3); a within-storm permutation
(observed spatial r=0.86 vs null max 0.61) showing real *spatial* skill; a cross-storm
same-gauge null (collapses to r~0.1–0.2); and a temporal-lag curve (time-series r peaks
sharply at lag 0 and decays to zero within two days). `comp.sensitivity` shows the headline
is stable across the clean-filter cut-offs (r 0.81–0.89) and the node-selection knobs
(r changes <0.001 with the wet-depth threshold).

Unit + regression tests live in [`../tests/test_comp.py`](../tests/test_comp.py) (skill
metrics, time-series alignment, the valid/clean gating, the LaTeX table, and pinned
headline numbers + negative-control separations):

```bash
python -m pytest tests/test_comp.py -o addopts=""
```

## Configuration

All knobs are in [`constants.py`](constants.py): the `STORMS` list, `GAUGE_BOX`,
`KNOWN_FAILED` gauge failures, node-selection thresholds, and the "clean" filter.

Note: the archived per-storm netCDF stores fields at mesh **element centroids** (the
mSWE-GNN dual graph), so the "nearest wet node" sampled here is an element centroid, not a
primal mesh node; the triangle averaging mildly smooths the simulated peak.

## Result (14 storms, 2005–2023)

152 clean gauge-storm pairs:

- **peak** r = 0.89 (5–95% CI 0.84–0.92), RMSE = 0.54 m, bias = −0.30 m; within-storm
  spatial r = 0.86;
- **time series** median temporal r = 0.65, median RMSE = 0.25 m;
- **timing** is magnitude-dependent: for surges ≥ 1 m the peaks agree to a median 2.1 h
  (69% within 6 h); over all meaningful-surge pairs the median is 7.9 h, because the
  de-tided peak of a small far-field residual is noise (the "2.0 h" of the clean set is
  conditioned on the within-6 h gate and is not an independent skill measure).

The slight low bias is consistent with the omitted wave setup/runup and medium mesh
resolution (datum and node-sampling effects checked and found minor); over-predictions
concentrate at shallow semi-enclosed bay/pass gauges during direct landfalls.

## Dependencies

`huggingface_hub`, `utide`, `xarray`, `scipy`, `pandas`, `requests`, `matplotlib`,
`sithom` (paper figure style: `plot_defaults`, `get_dim`, `label_subplots`).
