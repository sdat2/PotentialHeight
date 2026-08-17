# adforce.eval — ADCIRC configuration evaluation against tide gauges

(Formerly the top-level `comp/` package; on-disk caches stay under
`data/comp/` because the raw CO-OPS responses are not reproducibly
re-downloadable and the utide fits are expensive.)

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
   NOAA CO-OPS water-level gauge in the storm's region box — NW Gulf (Texas → Florida
   panhandle; the New Orleans and Galveston study regions) for Gulf storms, or the
   Florida peninsula (Key West → Fernandina Beach; the Miami study region) for the
   Atlantic/Florida storms (`FLORIDA_STORMS` in [`constants.py`](constants.py));
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
python -m adforce.eval.validate                       # full 19-storm sweep (Gulf + Florida)
python -m adforce.eval.validate 'storms=["Ida 2021"]'   # one storm (skips example panels + table)
python -m adforce.eval.validate validate.examples_only=true   # example figure, from cache (fast)
python -m adforce.eval.validate validate.examples_only=true validate.refresh=true   # recompute first
```

A full sweep regenerates **everything the paper uses**, in one step, so the figures, the
table and the prose cannot drift apart:

| output | path |
| --- | --- |
| scatter (quick-look / paper) | `img/comp/val_scatter.png` · `<thesis>/img/comp_val_scatter.pdf` |
| example time series          | `img/comp/val_examples.png` · `<thesis>/img/comp_val_examples.pdf` |
| per-storm LaTeX table        | `<thesis>/paper/comp_val_table.tex` (`\input` by the appendix) |
| summary table                | `data/comp/out/val_summary.csv` |

The slow step is the per-gauge `utide` de-tiding. A full sweep **caches** each storm's
de-tided `(sim, obs)` series as Parquet under `data/comp/ts_cache/` (write-through), keyed by
the node-selection + de-tiding parameters so the cache self-invalidates if any of those change.
`validate.examples_only=true` then re-renders `comp_val_examples.pdf` from that cache in seconds (vs minutes)
— use it to iterate on the figure's layout without re-detiding; `validate.refresh=true` forces a recompute.

The thesis tree is located by searching for `paper/appendix.tex`; override with the
`WORSTSURGE_PAPER_ROOT` env var. Downloads/caches live under `data/comp/` (git-ignored).

## Is the signal real? (negative controls + sensitivity)

`val_summary.csv` is enough to falsify the result without re-running ADCIRC:

```bash
python -m adforce.eval.nulltest            # permutation, cross-storm, and temporal-lag nulls
python -m adforce.eval.nulltest nulltest.lag=false   # peak-level nulls only (no netCDF)
python -m adforce.eval.sensitivity         # threshold + node-selection robustness
python -m adforce.eval.sensitivity sensitivity.node=false
```

`adforce.eval.nulltest` writes `img/comp_val_nulltests.pdf` and reports: a label-permutation null
(observed r=0.89 vs null max 0.31 over 5000 shuffles, p<1e-3); a within-storm permutation
(observed spatial r=0.86 vs null max 0.61) showing real *spatial* skill; a cross-storm
same-gauge null (collapses to r~0.1–0.2); and a temporal-lag curve (time-series r peaks
sharply at lag 0 and decays to zero within two days). `adforce.eval.sensitivity` shows the headline
is stable across the clean-filter cut-offs (r 0.81–0.89) and the node-selection knobs
(r changes <0.001 with the wet-depth threshold).

Unit + regression tests live in [`../../tests/test_eval.py`](../../tests/test_eval.py) (skill
metrics, time-series alignment, the valid/clean gating, the LaTeX table, and pinned
headline numbers + negative-control separations):

```bash
python -m pytest tests/test_eval.py -o addopts=""
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

## Comparing adforce configurations (general framework)

Beyond the historical HF-archive validation above, the submodule compares
**any** adforce configurations (resolution low/mid/high, tides on/off, SWAN)
against gauges and against each other, driven entirely by YAML under
[`config/`](config/):

- **matrix** ([`config/matrix/`](config/matrix/)) — the comparison grid:
  cartesian `axes` over wrap-config paths (`adcirc.resolution.value`,
  `adcirc.tide.value`, ...) minus `exclude`, plus explicit `cells`, with a
  designated `baseline` for model-vs-model.
- **launch** (`python -m adforce.eval.launch`) — expands the matrix into
  per-(cell, storm) runs under `<runs_root>/<study>/<cell>/<slug>/`, routes
  every historical run through the training driver's per-storm input
  generation (correct tidal windows — never the Katrina-pinned static
  decks), skips completed runs, refuses FOREIGN directories, and records a
  full-config sha256 per run in `eval_manifest.json`. `dry_run=true` is the
  default: it prints the plan table and fires nothing. `controls=true` adds
  the tide-only runs of the tide-surge-interaction triple.
- **extract** (`python -m adforce.eval.extract`) — remote-side reducer:
  `fort.63.nc` (5–8 GB) → `gauge_ts.parquet` per run (long format
  `storm, sid, gauge, time, zeta`), same node selection as the validation.
- **harvest** (`python -m adforce.eval.harvest`) — rsyncs the minimal
  artifact set (`config.yaml`, `gauge_ts.parquet`, `fort.61.nc`,
  `maxele.63.nc`, `slurm.out`, manifest) to the laptop.
- **pairs** (`python -m adforce.eval.pairs`) — model-vs-model:
  `resolution_bias_table` (reproduces `rerun/results/resolution_bias.csv`
  exactly) and `tide_surge_interaction`
  (`peak(both) − peak(storm) − peak(tide)`; reproduces the published
  low-res rows of `rerun/results/tide_surge_interaction.csv` 683/683).
  Both reproductions are pinned by tests.

Example — low vs mid × tide on/off for two storms:

```bash
# remote (GCP spot VM): preview, then launch
python -m adforce.eval.launch study=kat-ida matrix=res_x_tide \
    'storms=["Katrina 2005","Ida 2021"]'                    # dry run (default)
python -m adforce.eval.launch study=kat-ida matrix=res_x_tide \
    'storms=["Katrina 2005","Ida 2021"]' dry_run=false controls=true
# laptop:
python -m adforce.eval.harvest remote=gcp-vm:/work/exp/eval study=kat-ida dry_run=false
python -m adforce.eval.pairs action=interaction res=mid \
    tide_series=... both_series=... out=data/comp/out/kat-ida_tsi.csv
```

Caveat (tide-on scoring): comparing a tide-on run against raw gauge water
level needs the datum/steric offset handled (pre-storm-mean alignment) and
skew surge as the headline metric; that obs-side path is staged follow-up
work — do not publish tide-on skill numbers before it lands.

## Dependencies

`huggingface_hub`, `utide`, `xarray`, `scipy`, `pandas`, `pyarrow`, `requests`, `matplotlib`,
`sithom` (paper figure style: `plot_defaults`, `get_dim`, `label_subplots`).
