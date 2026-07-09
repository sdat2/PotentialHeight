# rerun/ MANIFEST — reproducibility index

Every script that regenerates a paper/thesis number, figure, table, or dataset
with the **fixed** potential-size solver (2026-07 humidity/units fix,
`w22.w22_carnot.carnot_pm_from_y`). Recovered from the session transcript after
the scratchpad was wiped, then polished to run from durable paths. The
computation in each polished script was **adversarially verified byte-faithful**
to the verbatim original in [`_extracted/`](_extracted/) (13/14 CONFIRMED; the
one flag was a file-naming artifact, not a logic change — see notes).

Run everything under the guard + `caffeinate` so RAM is capped and the machine
won't sleep. Reads come from the SSD (`/Volumes/s/tcpips/…`); the repo `data/`
symlinks are only needed by the figure scripts (noted per-row).

```bash
export HDF5_USE_FILE_LOCKING=FALSE
caffeinate -i -s python rerun/mem_guard.py <script.py> <cap_mb> <timeout_s> [-- args]
```

## ERA5 gridded PS (top level)

| script | produces | inputs | outputs | runtime |
|--------|----------|--------|---------|---------|
| `era5_ps_local.py` | ERA5 gridded fixed-PS product (feeds ERA5 snapshot maps + global trends) | `data/era5` (→ SSD) | `/Volumes/s/tcpips/data/era5/ps_fixed/era5_ps_fixed_{year}.nc` | ~27 h/decade |
| `mem_guard.py` | RSS-cap + timeout watchdog (wraps every job) | — | — | — |
| `RESUME_era5_1980s.sh` | resume the 1980–1989 decade (skips finished years) | — | — | — |

## IBTrACS observational validation (`ibtracs/`)

| script | produces | inputs (SSD `data/ibtracs/`) | outputs | runtime |
|--------|----------|------------------------------|---------|---------|
| **`resolve_exceedance.py`** ← *authoritative* | Exact fit-free re-solve of every Cat-1 point for r1/r2/r3 → survival-at-1 exceedance table + top-6 supersize + `exact_{r1,r2,r3}.npz` | `.cps`, `.pi_ps`, `.ps_cat1`, `.normalized` | `…/fixed_ibtracs/exact_resolve/exact_{r1,r2,r3}.npz` + stdout table | ~72 min (n_jobs=4) |
| `resolve_r2.py` | Fit-based **cross-check** of r2 (validated the exact solve, <1% held-out). Numbers cite `resolve_exceedance.py`, not this. | `.cps`, `.normalized` | stdout only | ~few min |
| `patch_track_files.py` | Patches fixed rmax into fresh copies of the 3 track files so the repo plotters draw fixed figures | `.pi_ps/.cps/.ps_cat1` + `exact_resolve/exact_{r1,r2,r3}.npz` | `…/fixed_ibtracs/*.nc` (+ sibling symlinks) | ~1 min |
| `figures_normalized_stage1.py` | Survival (Image-8), histograms/quad (Image-9), exceedance table, top tables — via repo `ibtracs` plotters on patched data | patched `fixed_ibtracs/` (via `data/ibtracs` symlink) | `…/fixed_figures/*.pdf` | ~min |
| `figure_v_tradeoff.py` | v-tradeoff / Katrina figure (Image-1) via `vary_v_cps` | patched `fixed_ibtracs/` | `…/fixed_figures/Image-1_*.pdf` | ~min |
| `figures_tracks.py` | 16 per-storm track figures via `plot_tc_example`; self-recomputes `track_{r1,r2,r3}.npz` if absent | `exact_resolve/` + track files | `…/fixed_figures/tracks/*_track.pdf` | ~min–tens |

**Expected fixed exceedance** (buggy→fixed): r3 40.8→46.1 (all) / 35.7→42.4 (|lat|≤30);
r2 17.06→19.9 (all) / 7.94→10.2 (sst≥26.5) / 5.89→7.94 (AF); r1 9.85→13.5 / 2.39→4.2.
Top-6 supersize (r2, AF): Ellen 3.59→3.86, Man-Yi 2.86→3.11.

**Symlink needed by the figure scripts:** `ln -sfn /Volumes/s/tcpips/fixed_ibtracs worstsurge/data/ibtracs`.

**Order to reproduce from scratch:** `resolve_exceedance.py` → `patch_track_files.py` →
(`figures_normalized_stage1.py`, `figure_v_tradeoff.py`, `figures_tracks.py`).

## CMIP6 / ERA5 trends at 28.75°N (`cmip6/`)

| script | produces | notes |
|--------|----------|-------|
| `regen_point_timeseries.py` | NO/HK CMIP6+ERA5 point timeseries at 28.75°N, fixed solver | the data step for the trend figures/tables |
| `job_cities.py`, `job_no_r4.py` | point/spatial job drivers (cities; NO r4) | fixed-solver PS |
| `figures_multipanel.py` | `{new_orleans,hong_kong}_multipanel.pdf` (Image-16/17) + spatial; **recomputes PI** (`recalculate_pi=True`) before plotting | superset of the archived `plot_only.py` |
| `tables_trends.py` | `{place}_temporal_{correlation,fit}_pi4.tex` trend tables | slopes ≈ unchanged by the fix (cancels in gradients) |

## Thesis figure 4 (`thesis_figs/`)

| script | produces |
|--------|----------|
| `figure_4.py` | `figure_4a/4b` + `4a/4b-*-ours.csv` (CLE15 comparison) |
| `validate_fig4.py` | validation companion for the fig-4 regen |

## Verbatim archive & throwaway probes

- [`_extracted/`](_extracted/) — all 30 session scripts, **verbatim** from the
  transcript. The ground-truth record; the polished scripts above are faithful
  re-homings of the reproducibility-critical ones. `extract_scripts.py` here is
  the harvester that built the archive (provenance).
- **Archive-only** (superseded by a polished version): `ibtracs_r2.py`,
  `ibtracs_r2b.py`, `ibtracs_r13.py` (ratio-fit, superseded by the exact solve),
  `tracks.py`, `plot_tradeoff.py`, `plot_only.py`, `plot_cmip6.py`,
  `era5_decade.py`, `guarded_run.py`.
- **Pure probes** (no artifact): `count_time.py`, `era5_probe.py`, `job_probe.py`,
  `smoke_joblib.py`, `one.py`, `analyze.py` (bug-magnitude diagnostic — the
  canonical deltas are pinned in `tests/test_ps_units.py`; 2 of its 3 city inputs
  are gone, so it can't fully re-run).

## Known gaps

- The `exact_{r1,r2,r3}.npz` intermediates were lost with the scratchpad but are
  **deterministically regenerable** by `resolve_exceedance.py` (fixed n_jobs,
  seeded). The *published numbers and figures already exist* in `fixed_ibtracs/*.nc`
  and `fixed_figures/`, so the npz are only needed to re-run patch/figure stages.
- `/Volumes/s/tcpips/data/era5/ps_fixed/` is being filled now (1980–1989 running);
  ERA5 snapshot maps / global trend figures are not yet regenerated.
