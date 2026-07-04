# SLURM scripts for the paper's experiments

All of the paper's production experiments were run on [ARCHER2](https://www.archer2.ac.uk/)
with the SLURM scheduler. Each script activates a Python environment, then runs one or
more `python -m <module>` commands from this repository. They are grouped below by
pipeline stage (see [../REPRODUCE.md](../REPRODUCE.md) for the stage ordering).

## What a re-user must change (ARCHER2-specifics)

These scripts are tied to the author's ARCHER2 allocations and will not run elsewhere
unedited:

- **Account codes**: `#SBATCH --account=n02-bas` or `n01-biopole` — replace with your
  own budget code. The `n01_*` scripts are duplicates of experiments run under the
  second (n01) allocation.
- **Environment activation**: most scripts `source /work/n02/n02/sdat2/.bashrc` and
  `micromamba activate t1`; the n01 scripts instead `source /work/n01/n01/sithom/.bashrc`
  and `conda activate base`. Point these at your own shell init and environment
  (created from `../env.yml`, see `create_env.slurm`).
- **Hard-coded `/work` paths**: e.g.
  `/work/n02/n02/sdat2/adcirc-swan/worstsurge/...`, the `run_5sec`/`swegnn_5sec`
  training-data directories, and the `MPLCONFIGDIR` matplotlib cache paths.
- **Partition/QoS names** (`standard`, `short`, `gpu`/`gpu-shd`) and the
  `module load` lines (`cdo`, `nco`, `rocm`, `PrgEnv-*`) are ARCHER2-specific.
- **Email**: `--mail-user=sdat2@cam.ac.uk`.
- A `logs/` directory must exist next to the submission directory
  (`#SBATCH --output=logs/%x-%j.out`).
- ADCIRC-driving jobs (`adforce`/`adbo`) additionally need compiled `padcirc` and
  `adcprep` executables in `adforce/config/files/` (see the main README).

Typical resources: 1 node × 128 tasks, `--time` 12–24 h unless noted.

## Environment setup

| Script | Runs | Produces |
| --- | --- | --- |
| `create_env.slurm` | `micromamba create -n t1 -f env.yml`; `pip install ../sithom`; then two `adforce.wrap` smoke runs (`notide`, 2025/2097 profiles, `mid-notide` mesh) | the `t1` environment + test ADCIRC runs (20 min, `qos=short`) |
| `gpu.slurm` | creates a `gpuenv` env, checks TensorFlow sees the AMD GPU (`rocm`), then `python -m worst.vary_noise` | GPU trial of the TensorFlow EVT fit (single GPU, 12 h) |

## `tcpips` — CMIP6/ERA5 download, regridding, potential intensity

| Script | Runs | Produces |
| --- | --- | --- |
| `regrid.slurm` | `tcpips.regrid_cdo.regrid_cmip6_part` (historical, CESM2 `r10i1p1f1`, ocean + atmos) | regridded CMIP6 under `data/cmip6/regridded` |
| `cdo.slurm` | `module load cdo nco`; `regrid_cmip6_part` (historical, CESM2 `r4i1p1f1`) then `tcpips.pi_driver.pi_cmip6_part` for the same member (many other members present but commented out) | regridded CMIP6 + potential intensity netCDFs |
| `pi.slurm` | `tcpips.pi_driver.pi_cmip6_part` for CESM2 `ssp585` (`r11`, `r10`, `r4`) and `historical` (`r11`) | potential intensity netCDFs |
| `era5.slurm` | `tcpips.era5.calculate_potential_sizes` via `dask_cluster_wrapper`, 1980–1989 (later decades commented out) | ERA5 monthly potential-size products |
| `ps_era5.slurm` | `srun python tcpips/era5_ps_calc.py --start_year 1980 --end_year 1989` (dask-mpi, **20 nodes**) | ERA5 potential-size sweep (large parallel version) |
| `dask.slurm` | `python tcpips/run_dask_calculation.py` — a controller that itself submits dask `SLURMCluster` workers | dask-driven PS calculation |
| `ibtracs.slurm` | `python -m tcpips.ibtracs` | IBTrACS vs ERA5 PI/PS observational figures (`img/ibtracs/`); needs `CARTOPY_DIR` |

## `w22` — potential size

| Script | Runs | Produces |
| --- | --- | --- |
| `ps.slurm` | `w22.ps_runs`: `trimmed_cmip6_example()` (default + `pi_version=3,4`), `global_cmip6()`, `point_timeseries` for New Orleans/Galveston/Miami (members 4, 10, 11) | CMIP6 potential-size products + the point time series behind `figure_two` |
| `test-ps.slurm` | `python -m w22.ps_runs` | quick PS smoke test (20 min, `qos=short`) |

## `adforce` — ADCIRC runs and SurgeNet training data

| Script | Runs | Produces |
| --- | --- | --- |
| `wrap.slurm` | `python -m adforce.wrap name=test2` | single test ADCIRC run (20 min, `qos=short`) |
| `n01_wrap.slurm` | `python -m adforce.wrap --name=test-2 --use_slurm=false --animate` | single test run + animation (n01 variant) |
| `gen_tcs.slurm` | active line: `adforce.check_training_runs --runs-parent-name run_5sec --save-to swegnn_5sec` (the generating command `adforce.generate_training_data` is commented out above it) | checks/collects the historical-TC ADCIRC runs used as SurgeNet training data |
| `process_array.sbatch` | SLURM **array job** (`0-32%10`, 8 tasks/node) packing `adforce.process_single --run-path ... --save-path ...` over run directories | one dual-graph netCDF per historical run in `swegnn_5sec/` |

## `adbo` — Bayesian-optimization experiments

| Script | Runs | Produces |
| --- | --- | --- |
| `bo_1d.slurm` | `adbo.exp_1d --exp_name=1d-ani --resolution=mid --seed=21` (n01, conda base) | 1D BO experiment/animation inputs |
| `bo_2d.slurm` | `adbo.exp_2d --exp_name=2d-ani-ei --resolution=mid --seed=22 --daf=ei` | 2D BO experiment (GP animation source) |
| `n01_bo_2d.slurm` | `adbo.exp_2d --exp_name 2d-ani-subprocess --resolution mid` | 2D BO experiment (n01 variant) |
| `exp.slurm` | `adbo.exp_3d` with `exp_name=i{init}b{daf}[t{trial}]` (shown: `i3b47`, seed derived from split+trial) | LHS-vs-BO split ensembles in `exp/` — inputs to the convergence/regret figures (`adbo.plot.plot_bo_comp`) |
| `new-orleans.slurm` | `adbo.exp_3d --exp_name=new-orleans-2015/-2100` (one pair) | New Orleans 2015-vs-2100 BO runs |
| `n01_new-orleans.slurm` | `adbo.exp_3d` New Orleans 2015/2100, three trials (`exp` 0–2, seeds via variables) | New Orleans ensemble (n01 variant) |
| `galverston.slurm` | `adbo.exp_3d` Galveston 2015/2100 × 3 trials, `daf=ei` | Galveston 2015-vs-2100 BO runs |
| `miami.slurm` | `adbo.exp_3d` Miami 2015/2100 × trials 2–4, `daf=ei` | Miami 2015-vs-2100 BO runs |
| `n01_bo_first.slurm` → `n01_bo_fourth.slurm` | sequential batches of `adbo.exp_3d` over 7 NOAA station IDs (8729840…8764044) × {2015, 2100}: *first* = 2100 batch 1, *second* = 2100 batch 2 + first 2015 run, *third*/*fourth* = remaining 2015 runs | the along-coast experiment set (`exp/{stationid}-{year}`) behind the along-coast/parallel-coordinates figures |
| `bo_along_coast.slurm` | same along-coast runs on the n02 account (currently batch "3" active, others commented) | along-coast reruns |
| `n01_bo_m32_ei.slurm` | `adbo.exp_3d --exp_name=bo-m32-ei --kernel=Matern52 --daf=ei` | GP-kernel/acquisition comparison run |
| `n01_create_test_set.slurm` | `python -m adbo.create_test_set` | potential-height test set (best BO run per location) for SurgeNet |
| `sweep_v_no.slurm` | curve gen (`w22.tradeoff`) + `adbo.sweep_vmax` New Orleans {2025, 2097}, fixed track (**edit to the 3D optimum**) | 1D intensity sweep along the size-intensity tradeoff curve r(V) + `vmax_sweep.pdf` |
| `bo_4d_no.slurm` | curve gen (`w22.tradeoff`) + `adbo.exp_4d` New Orleans {2025, 2097}, 35+35 samples | 4D BO (track + on-curve intensity) + `bo_4d_samples.pdf`; resumable on rerun |

## `worst` — EVT fits

| Script | Runs | Produces |
| --- | --- | --- |
| `worst_fit.slurm` | `python -m worst.vary_noise` | synthetic EVT fit sweep (CPU; TensorFlow) |
| `gpu.slurm` | `python -m worst.vary_noise` (GPU; see Environment setup above) | same, on one GPU |

The remaining `worst` figure scripts (`vary_gamma_beta`, `vary_samples_ns`,
`vary_nonstationary`, `sigma_robustness`, `ns_evt_figs`) are light enough to run
locally and have no SLURM scripts.
