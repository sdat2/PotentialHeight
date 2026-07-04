# Reproducing the paper

This file maps each figure/table of *"Finding the potential height of tropical
cyclone storm surges in a changing climate using Bayesian optimization"*
(EarthArXiv <https://doi.org/10.31223/X57T5R>, under revision at Environmental
Data Science) to the command that produces it, its key inputs, and its output
path.

Note on years: the paper's BO comparison (Fig. 7) is for **2025 vs 2097**
(CESM2 SSP5-8.5 August monthly means), but the repository's experiment
directories, figure filenames, and slurm variables use the legacy **2015/2100**
naming (e.g. `2015-vs-2100-new-orleans.pdf`, `$p25`/`$p97` in
`new-orleans.slurm`). They are the same experiments.

See the [README](README.md#getting-started) for environment setup
(`env.yml` portable spec; `environment.lock.yml`/`requirements.lock.txt` exact
validated solve) and `pip install -e .[bo]` etc. for the optional dependency
groups needed by each stage. [`slurm/README.md`](slurm/README.md) documents the
ARCHER2 job scripts referenced below.

## Full pipeline

The end-to-end chain is:

**CMIP6/ERA5 download (`tcpips`) → regrid (`tcpips`) → potential intensity
(`tcpips.pi`) → potential size (`w22.ps`) → ADCIRC forcing (`adforce`) →
Bayesian optimization (`adbo.exp*`) → EVT with an upper bound (`worst`)**

| Stage | Module command | Where it runs | Notes |
| --- | --- | --- | --- |
| 1. CMIP6 download | `python -m tcpips.pangeo` | local or ARCHER2 (needs internet + `[cmip]` extra) | intake-esm/gcsfs Pangeo catalogue → `data/cmip6/raw` |
| 2. Regrid | `python -c "from tcpips.regrid_cdo import regrid_cmip6_part; ..."` (or `tcpips.regrid` for xESMF) | ARCHER2 (`regrid.slurm`, `cdo.slurm`) or local | `cdo`/`nco` modules or `[cmip]` extra |
| 3. Potential intensity | `python -c "from tcpips.pi_driver import pi_cmip6_part; ..."` | ARCHER2 (`pi.slurm`, `cdo.slurm`) or local | uses `tcpypi` |
| 4. ERA5 download + PI/PS | `python -m tcpips.era5` (download needs `cdsapi` credentials); `tcpips/era5_ps_calc.py` for the big parallel PS sweep | ARCHER2 (`era5.slurm`, `ps_era5.slurm`, `dask.slurm`) | `[mpi]` extra for the dask-mpi sweep |
| 5. Potential size | `python -m w22.ps_runs` (quick test); driver functions `trimmed_cmip6_example`, `global_cmip6`, `point_timeseries` | ARCHER2 (`ps.slurm`, `test-ps.slurm`) or local for small cases | CLE15 profile via `cle15` (Numba default) |
| 6. ADCIRC forcing | `python -m adforce.wrap` | ARCHER2 only (needs `padcirc`/`adcprep` binaries, see `adforce/config/files/`) | `wrap.slurm`, `n01_wrap.slurm` |
| 7. Bayesian optimization | `python -m adbo.exp_1d/_2d/_3d ...` | ARCHER2 only (drives ADCIRC via `adforce`) | `[bo]` extra; `bo_*.slurm`, `exp.slurm`, per-city scripts, `n01_bo_first..fourth` |
| 8. EVT with upper bound | `python -m worst.vary_samples_ns`, `worst.vary_noise`, `worst.vary_nonstationary`, `worst.sigma_robustness`, `worst.ns_evt_figs` | local (CPU; `[bo]` extra for TensorFlow fits) | synthetic experiments — no ADCIRC needed |
| Validation (side chain) | `python -m comp.validate` / `comp.nulltest` / `comp.detide_sensitivity` / `comp.sensitivity` | local (needs internet + `[comp]` extra) | Hugging Face archived runs vs NOAA gauges |
| Observations (side chain, thesis) | `python -m tcpips.ibtracs` | local or ARCHER2 (`ibtracs.slurm`) | IBTrACS vs ERA5-derived PI/PS |

Stages 6–7 require ARCHER2 (or another HPC machine with a ported
`adforce/wrap.py` and compiled ADCIRC ≥ 55.02); everything else can run
locally. The archived outputs of stages 1–7 shipped in `data/` and the
Hugging Face datasets let the figure scripts below run without re-doing the
HPC stages.

## Paper figures and tables

Some plotting entry points are driven from commented-out calls in the module
`__main__` blocks (noted below); uncomment the relevant call and run
`python -m <module>`.

Figures 1 (methodology schematic) and 4 (Bayesian-optimization flow chart) are
TikZ diagrams drawn directly in the paper source — no repository script.
Tables 1 (abbreviations) and 2 (framework parameters) are hand-written in the
paper source.

### Main text

| Paper item | Command | Key inputs | Output | SLURM job |
| --- | --- | --- | --- | --- |
| **Fig. 2** — GOM potential size/intensity maps + New Orleans CMIP6 time series (`worst:fig:PIPS`) | `python -m w22.plot` (enable `figure_two(place="new_orleans")` in `__main__`) | CMIP6 PI/PS products from `w22.ps_runs` (`point_timeseries`, `trimmed_cmip6_example`) | `w22/img/figure_two.pdf` | upstream: `ps.slurm` |
| **Fig. 3** — ADCIRC snapshot with CLE15 wind quivers, New Orleans "worst case" (`worst:fig:adcirc`) | `python -m adforce.ani` (snapshot plotting, `figure_name=*snapshot.pdf`; edit `__main__` for the run directory) | a completed ADCIRC run directory | `img/adforce/*_snapshot.pdf` (e.g. `2025_mid_notide_snapshot.pdf`) | `wrap.slurm` (upstream run) |
| **Fig. 5** — simple-regret comparison of strategies A (50 LHS) vs B (25+25 MES), 11 trials (`worst:fig:bo_regret`) | `python -m adbo.plot` (enable `plot_bo_comp()`) | `exp/i{i}b{b}t{t}/experiments.json` from repeated 50-sample BO runs | `img/2015vs2100/bo_regret_2525vs50.pdf` (companions: `bo_exp_2525vs50_trials.pdf`, `bo_comp_2525vs50.pdf`) | `exp.slurm` |
| **Fig. 6** — active-learning example: GP mean/std + MES acquisition in 2D track-parameter space (`worst:fig:bo`) | `python -m adbo.ani` (edit `path_in` to the experiment directory) | a 2D BO experiment directory (e.g. `2d-ani*`) | `<exp>/img/gp_{i}.pdf` (animation: `<exp>/gif/gp.gif`) | `bo_2d.slurm`, `n01_bo_2d.slurm` |
| **Fig. 7** — potential height in 2025 vs 2097 for New Orleans, samples + best-so-far (`worst:fig:bo-2025-2097`) | `python -m adbo.plot` (enable the `plot_diff` loop) | `exp/new-orleans-2015-*`, `exp/new-orleans-2100-*` BO runs (legacy dir naming; profiles are 2025/2097) | `img/2015vs2100/2015-vs-2100-new-orleans.pdf` | `new-orleans.slurm`, `n01_new-orleans.slurm` |
| **Fig. 8** — reference points / tide-gauge station map on the simplified mesh (`worst:fig:places-plot`) | `python -m adbo.plot` (enable `plot_places()`) | `exp/{stationid}-{year}` BO runs for 7 NOAA station IDs | `img/2015vs2100/stationid_map.pdf` | `n01_bo_first..fourth.slurm` (sequence) |
| **Fig. 9** — parallel-coordinates plot of optimal SSH and TC parameters (`worst:fig:parallel_coordinates`) | `python -m adbo.plot` (enable `plot_multi_argmax()`) | same along-coast BO runs | `img/2015vs2100/parallel_coordinates_all.pdf` | as above |
| **Fig. 10** — AEP with known vs unknown upper bound, GEV resampling (Nr=600) (`worst:fig:evt`) | `python -m worst.vary_samples_ns` (hydra config `worst/config/`) | synthetic GEV draws (TensorFlow fit) | `img/worst/evt_fig_tens_*.pdf` | — (local) |
| **Fig. 11** — effect of upper-bound uncertainty σ_ẑ* on return-value estimates (`worst:fig:vary_z_star_sigma`) | `python -m worst.vary_noise` | synthetic GEV draws; cache `data/worst/vary_z_star_*.nc` | `img/worst/vary_z_star_sigma.pdf` | — (local) |

### Appendix A — historical surge validation (`comp`; see [`comp/README.md`](comp/README.md))

| Paper item | Command | Key inputs | Output | SLURM job |
| --- | --- | --- | --- | --- |
| **Table 3** — de-tiding-method sensitivity (`tab:detide-robustness`) | `python -m comp.detide_sensitivity` | de-tided gauge cache (`data/comp/ts_cache/`) | `<thesis>/paper/comp_detide_table.tex` (`\input` by the appendix) | — (local) |
| **Table 4** — per-storm skill vs de-tided NOAA gauges (`tab:gauge-validation`) | `python -m comp.validate` | Hugging Face [`sdat2/surgenet-train`](https://huggingface.co/datasets/sdat2/surgenet-train); NOAA CO-OPS gauges (de-tided with `utide`) | `<thesis>/paper/comp_val_table.tex` | — (local) |
| **Fig. 12** — peak-skill scatter (`fig:gauge-validation-scatter`) | `python -m comp.validate` (full 14-storm sweep) | as above | `img/comp/val_scatter.png` · `<thesis>/img/comp_val_scatter.pdf` | — (local) |
| **Fig. 13** — example de-tided surge time series (`fig:gauge-validation-examples`) | `python -m comp.validate` (or `--examples-only` from cache) | as above (cache: `data/comp/ts_cache/`) | `img/comp/val_examples.png` · `<thesis>/img/comp_val_examples.pdf` | — (local) |
| **Fig. 14** — negative-control null tests (`fig:gauge-validation-null`) | `python -m comp.nulltest` | `data/comp/out/val_summary.csv` (+ netCDFs for the lag null) | `<thesis>/img/comp_val_nulltests.pdf` · `img/comp/val_nulltests.png` | — (local) |
| Summary metrics (support; pinned by `tests/test_comp.py`) | `python -m comp.validate` | as above | `data/comp/out/val_summary.csv` | — (local) |
| Threshold/node-selection sensitivity (text numbers) | `python -m comp.sensitivity` | `val_summary.csv` | console report (headline r stability) | — (local) |

### Appendix B — non-stationary EVT

| Paper item | Command | Key inputs | Output | SLURM job |
| --- | --- | --- | --- | --- |
| **Fig. 15** — estimator bias/range vs upper-bound trend (`fig:ns-evt-bias-range`) | `python -m worst.vary_nonstationary` (writes cache) then `python -m worst.ns_evt_figs` | cached sweep `data/worst/vary_nonstationary_*.nc` | `<thesis>/img/evt_ns_bias_range.pdf` (sweep previews under `img/worst/`) | — (local) |
| **Fig. 16** — robustness to bound-level uncertainty σ (`fig:ns-evt-sigma`) | `python -m worst.sigma_robustness` (cache + quick-look) then `python -m worst.ns_evt_figs` | cached sweep `data/worst/sigma_robustness_*.nc` | `<thesis>/img/evt_ns_sigma.pdf`; quick-look `img/worst/sigma_robustness.pdf` | — (local) |

### Appendices C & D

| Paper item | Command | Key inputs | Output | SLURM job |
| --- | --- | --- | --- | --- |
| **Fig. 17** — BO convergence across random seeds, 2015 & 2100 (`worst:fig:bo-convergence`) — TODO(verify producer: embedded as `<thesis>/img/worstpp/Image-4.pdf`; candidates are `plot_bo_exp()`/`plot_bo_comp()` in `adbo/plot.py`) | `python -m adbo.plot` | repeated-seed BO runs | `img/2015vs2100/bo_exp_2525vs50_trials.pdf` (candidate) | `exp.slurm` |
| **Table 5** — GP kernel comparison for the surge surrogate (`tab:bo-kernel`) | `python -m adbo.gp_exp` (`save_results_tex(n_train=25)`) | 787 high-resolution ADCIRC samples (`gather_data`) | LaTeX table | — |
| Appendix D — supplementary bounded-EVT theory (no figures in the current revision) | theory notebook-style module: `python -m worst.evt_theory` regenerates `evt_saturation.pdf` / `evt_closedform.pdf` if wanted | synthetic | `img/worst/` (not embedded in the paper) | — (local) |

## Thesis / supporting material (not in the current paper)

These are produced by the repository and appear in the thesis or support the
analysis, but have **no figure/table number in `worst_new.tex`**:

| Item | Command | Output |
| --- | --- | --- |
| Along-coast potential height 2015/2100 panels | `python -m adbo.plot` (enable `plot_places()`) | `img/2015vs2100/along-coast-2015.pdf`, `along-coast-2100.pdf` |
| Per-city GP comparison, Miami & Galveston | `python -m adbo.plot` (`plot_diff` loop) | `img/2015vs2100/2015-vs-2100-{miami,galverston}.pdf` |
| Argmax input/output table (EI and MES) — commented out of the paper (`worst_new.tex` ~591) | `python -m adbo.plot` (default `__main__`: `make_argmax_table("ei"/"mes")`) | LaTeX table on stdout |
| EVT sensitivity to γ, β | `python -m worst.vary_gamma_beta` | `img/worst/vary_gamma_beta_*.pdf` |
| IBTrACS observed vs potential (dual quad + full plot set) | `python -m tcpips.ibtracs` (default: `plot_normalized_quad_dual()`; `run_all_plots()` for the full set) | `img/ibtracs/*.pdf` |
| ERA5 PI climatology/trend panels — unclear — check (likely `era5_pi_trends`/`era5_ps_trends` in `tcpips/era5.py`) | see `tcpips.era5` | `img/era5/*.pdf` |
| Reproduction of Wang et al. (2022) Figs. 4a/4b/5a (PS solution check) | `python -m w22.test_figures` (also run by pytest) | `w22/img/w22/figure_{4a,4b,5a}.pdf` |
| Octave (original MATLAB) vs Python CLE15 profile check | `python -m w22.test_figures` (needs Octave for `cle15.cle15m`) | `w22/img/cle15/octave-vs-python*.pdf` |
| CMIP6 results tables (HadGEM3-GC31-MM, MIROC6, CESM2) | `python -m w22.stats2` | LaTeX tables |
| ReLU-extrapolation toy example (SurgeNet motivation) | `python -m surgenet.toy_example` | `surgenet/toy_out/img/*.pdf` |
