# rerun/ — local post-bug-fix recomputation

Laptop/AWS-safe recomputation scripts for regenerating the potential-size (PS)
products with the **fixed** solver after the 2026-07 humidity/units bug
(`w22.w22_carnot.carnot_pm_from_y`; see `../unit_bug_fix.md`). These replace the
ARCHER2 dask-mpi jobs now that ARCHER2 access is gone, and are written to run
unattended on a single machine without crashing it.

**[MANIFEST.md](MANIFEST.md) is the index** — every script, what published
number/figure/table/dataset it regenerates, its durable inputs/outputs, expected
values, and run order. Start there. This README covers the shared operating
discipline and the ERA5 job in detail.

## Layout

| path | what it does |
|------|--------------|
| `mem_guard.py` | Runs any script under a hard RSS cap + wall-clock timeout (its own process group, killed if it exceeds either). The external safety net — dask's own memory controls are not trustworthy here. |
| `era5/era5_ps_local.py` | ERA5 gridded PS re-solve, one year at a time, checkpointed. Parameterised by `--start-year/--end-year`, so the same script does every decade. |
| `era5/` | The rest of the ERA5 job: RESUME scripts, plus the GCP scaffolding (config, provision, VM startup, per-decade cloud runner). |
| `adcirc/` | ADCIRC + BO on GCP: Docker build (proven), srun/module shims, smoke test, materiality profiles, SWAN coupling kit (`swan/`). |
| `results/` | Archived experiment outputs: BO/sweep ledgers, materiality maxele tarballs. |
| `ibtracs/` | IBTrACS observational validation: exact exceedance re-solve, rmax patching, and the survival/histogram/track/v-tradeoff figures. |
| `cmip6/` | CMIP6+ERA5 point timeseries at 28.75°N, the trend multipanels, and the trend `.tex` tables. |
| `thesis_figs/` | Thesis figure 4 (CLE15 comparison) + its validation. |
| `_extracted/` | **Verbatim archive** of all 30 session scripts, recovered from the transcript. Ground-truth record; the polished scripts above are faithful re-homings of the critical ones. |

Every polished script's computation was adversarially verified byte-faithful to
its `_extracted/` original, so re-running reproduces the exact published numbers.

## Running it

Always go through the guard, and `caffeinate` so the machine won't sleep:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
caffeinate -i -s python rerun/mem_guard.py \
    rerun/era5/era5_ps_local.py 7168 259200 -- --start-year 1980 --end-year 1989
```

- `7168` = 7 GB RSS cap, `259200` = 3-day timeout. Steady-state RSS is ~1 GB at
  `n_jobs=6`; the cap only ever fires on a runaway.
- Output: `/Volumes/s/tcpips/data/era5/ps_fixed/era5_ps_fixed_{year}.nc`
  (SSD, not the near-full internal disk).
- **Resumable:** re-running the identical command skips any year whose `.nc`
  already exists. Writes are atomic (`.tmp` then `os.replace`), so a kill mid-year
  never leaves a corrupt file.
- ~2.7 h/year, ~27 h/decade at 6 cores.

Combine when a decade is done:

```python
import xarray as xr
ds = xr.open_mfdataset(
    "/Volumes/s/tcpips/data/era5/ps_fixed/era5_ps_fixed_*.nc",
    combine="nested", concat_dim="year",
)
```

The five decades 1980–2024 feed the ERA5 snapshot maps and global PS trend
figures. Longer term this belongs on AWS (better for reproducibility too).

## Hard-won gotchas (don't relearn these)

- **Use joblib, not dask `processes`.** On macOS the `processes`/dask-mpi
  scheduler `BrokenProcessPool`s during the netCDF merge (spawn + netCDF-backed
  dask worker dies — not a memory issue). Load lazily with the **synchronous**
  scheduler, then solve with joblib(loky).
- **exFAT SSD:** `HDF5_USE_FILE_LOCKING=FALSE` is mandatory or reads/writes fail
  with an HDF error. netCDF4 also can't append while an xarray handle is open —
  close handles first (not relevant here since we write fresh files).
- **Cap the workers.** Some library defaults use `n_jobs=-1` (all 10 cores ≈ 5 GB).
  Keep `--n-jobs` ≤ 6 and BLAS single-threaded (the script sets the env vars).
- **Scripts live here, in the repo** — the repo is on the internal disk and
  survives reboots. The session scratchpad and even the SSD do not always; keep
  the canonical copy version-controlled here.
