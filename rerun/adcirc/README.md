# rerun/adcirc/ — ADCIRC + BO on Google Cloud (Part B, DRAFT)

**Status: untested draft scaffolding.** This is the harder half — ADCIRC is Fortran/MPI and
must be *compiled*. The plan: bake ADCIRC v55.02 + the BO driver into one Docker image, run
it on a single GCP VM, and **lead with a cheap 2-run materiality check** before committing to
any full Bayesian-optimization re-run. Nothing here has been executed; treat the Dockerfile
(especially the ADCIRC build) as a starting point to iterate on.

## The strategy: materiality check *first*
The whole reason to touch ADCIRC is the ~8% potential-size fix. Before rebuilding cloud HPC
and re-running ~50-simulation BO experiments, answer the cheap question: **does the fix move
the surge enough to matter?** Run the *same* idealized New Orleans TC once with the old
(pre-fix) CLE15 profile and once with the fixed profile, and compare max water level. Two
ADCIRC runs (~a couple of hours on one VM) can tell you whether the full re-run is even
needed. `run_on_gcp.sh` step 3 does exactly this.

## How ADCIRC runs here (the facts that shaped this)
- Driven entirely from Python: `adbo/exp_{1,2,3}d` (trieste GP Bayesian opt, ~50 evals =
  `init_steps + daf_steps`) → `adforce/wrap.py::idealized_tc_observe` → per evaluation copies
  `fort.14/13/15` from `adforce/setup/`, writes the `fort.22.nc` wind forcing in Python
  (NWS=13 OWI-NWS13 netCDF), runs **`adcprep` ×2 then `padcirc`**, and reads back one scalar:
  max water level at New Orleans from `maxele.63.nc`.
- **Small mesh**: `fort.14.mid` = 31,435 nodes / 58,369 elements, DT=5 s, 7-day run, no tides,
  no waves. A single-VM job — **not** HPC-scale. ARCHER2 used 108 ranks; a VM needs far fewer.
- **Sequential**: one MPI ADCIRC run per BO evaluation, one at a time. So the VM only needs to
  run *one* ADCIRC job well — no cluster, no multi-node, no low-latency interconnect.

## The two ARCHER2-isms (handled by shims, so repo code runs unchanged)
`adforce/subprocess.py` — even on the default `use_slurm=false` path — hard-codes two things
that don't exist on a plain VM. Rather than fork the code, the image neutralises both:
1. **`srun ... --ntasks=N padcirc`** → the `shims/srun` script rewrites it to `mpirun -np N padcirc`.
2. **`module load <Cray modules>`** (raises `RuntimeError` if it fails) → the `shims/module`
   no-op returns success. Cleaner still: also pass `slurm.modules=''` so the block is skipped.

Everything else is config: point `files.exe_path` at the in-image binaries (`/opt/adcirc/work`)
and set the rank count via `slurm.tasks_per_node=<vCPU>`, `slurm.reserved_cpus=0` (the ARCHER2
`np = (128−20)×nodes = 108` formula then yields your VM's core count).

## Files
| file | role |
|------|------|
| `Dockerfile` | builds ADCIRC v55.02 (conda-forge gfortran + MPICH + parallel netCDF-Fortran) + the BO Python env + the shims |
| `shims/srun`, `shims/module` | translate the two ARCHER2-isms |
| `vm_files_config.yaml` | drop-in `adforce/config/files/vm.yaml` pointing at in-image paths |
| `run_on_gcp.sh` | provision a VM, build the image, run the materiality check, sync to GCS |

## Building ADCIRC (the hard part) — debugging playbook
The Dockerfile's ADCIRC build mirrors the fork's own `compile.sh` (its known-good recipe).
Only **three** things change from ARCHER2, and the rest must stay identical:

| ARCHER2 | GCP / conda | why |
|---|---|---|
| `ftn` / `cc` (Cray wrappers) | `mpif90` / `mpicc` | conda MPICH wrappers |
| `cray-netcdf-hdf5parallel` module | `netcdf-fortran=*=mpi_mpich*` | conda parallel netCDF |
| `module load PrgEnv-gnu …` | *(dropped)* | no Cray modules |

The Fortran flags are unchanged and **load-bearing**: `-fallow-argument-mismatch` (gfortran ≥10
turns arg mismatches into hard errors) and `-ffixed/free-line-length-132` (the long-lines issue
the fork README calls out). Strip either and the build fails.

**Debug loop — do NOT iterate via `docker build` on a cloud VM** (slowest possible cycle):
1. Get the toolchain layer, then an interactive shell: `docker run -it --rm adcirc-build-env bash`, and run cmake/make **by hand** — seconds per cycle. Debug **locally** with `docker build --platform=linux/amd64` (matches the GCP x86 target); you only need a VM to run at scale. If debugging on GCP, use a cheap **on-demand** (not spot) box.
2. **Prove the toolchain before ADCIRC** — half of failures are a broken netCDF/MPI stack: `micromamba list | grep -E "hdf5|netcdf|mpich"` (all must be `mpi_mpich_*`), `nf-config --all`, `nc-config --has-parallel`, and a 2-rank `mpirun` hello.
3. Configure, then inspect what CMake resolved: `cmake -LAH build | grep -iE "netcdf|fortran_flags|mpi"` — confirm your flags survived and `libnetcdff` was found.
4. Build **verbose, `-j1`, one target** (`adcprep` first — serial, simpler), and read the **first** error, not the last.

**Failure signatures → fix (top rows hit for real during local build-testing, 2026-07-10):**
| error | cause / fix |
|---|---|
| MPI ranks segfault (signal 11) at launch inside the container | Docker's default **64 MB `/dev/shm`** — MPICH's shared-memory transport exhausts it → `docker run --shm-size=8g` |
| `process_vm_readv failed (errno 1)` from MPICH | Docker's seccomp blocks CMA cross-process reads → `docker run --cap-add=SYS_PTRACE`. **Every MPI `docker run` needs BOTH flags** (all invocations in these scripts carry them) |
| `set_target_properties called with incorrect number of arguments` (macros.cmake, at configure) | the fork's macros expand `Fortran_LINELENGTH_FLAG`/`ADDITIONAL_FLAGS_*` **unquoted**; unset ⇒ the arg vanishes. Pass `-DFortran_LINELENGTH_FLAG="-ffixed-line-length-132 -ffree-line-length-132"` (+ `-DADDITIONAL_FLAGS_ADCPREP`) exactly like `compile.sh` — do NOT fold line-length into `CMAKE_Fortran_FLAGS` |
| CMake configure rejects `cmake_minimum_required(2.8.12)` | conda `cmake` resolved to v4 → pin `"cmake<4"` |
| `could not create work tree dir … Permission denied` (git clone) | mambaorg image default user can't write `/opt` → `USER root` |
| apt `500 Internal Server Error` from deb.debian.org | some networks break plain-HTTP to Fastly (host-level!) → image is apt-free, everything from conda-forge |
| `Type mismatch between actual and dummy argument` | `-fallow-argument-mismatch` didn't reach the compile line |
| line truncated at col 132 / `Unterminated … constant` | the line-length flags didn't reach that target (see first row) |
| `undefined reference to nf90_*` (link) | netCDF-Fortran not linked → fix `NETCDF_F90_LIBRARY` / `NETCDFHOME` |
| `undefined reference to H5*` | netCDF built against a different HDF5 → `ldd $CONDA_PREFIX/lib/libnetcdf.so \| grep hdf5` |

Also: Dockerfiles do **not** support inline `#` comments on instruction lines, and never judge a
`docker build` by piped output — capture the full log and check the exit code (`| tail` eats it).

**"Compiles" ≠ "works".** Validate end-to-end with the smoke test (baked onto PATH as
`adcirc-smoke-test`), which runs toolchain → binaries → `ldd` → a real `adforce.wrap` run
that must produce a finite `maxele.63.nc`:
```bash
docker run --rm -v ~/work:/work adcirc-ws adcirc-smoke-test 4
```
Only once that passes: freeze the working commands, **pin the fork commit** (`git checkout <sha>`),
and rebuild the image clean. Then run the materiality check (`run_on_gcp.sh` step 3).

## Machine sizing & cost
- **Materiality check** (2 runs): `c2d-standard-16`, ~a few hours, a couple of dollars.
- **Full BO experiment** (~50 sequential runs): one ADCIRC run is ~20 min on ARCHER2's 108
  cores; on ~16 VM cores expect ~1–1.5 h, so ~50–75 h/experiment. Use a bigger box
  (`c3-standard-44/88`) to get back toward ARCHER2 wall-times (~25–35 h/experiment). On spot
  (~⅓ price) a full New-Orleans experiment is roughly $20–50. The paper had several
  experiments (NO 2015/2100, Galveston, Miami, 7-station along-coast × 2 years, convergence
  ensembles) — decide from the materiality result which are actually worth redoing.

## BO path wiring (IMPLEMENTED + tested)
The pipeline is **argparse → adbo → adforce+Hydra**: `python -m adforce.wrap` takes Hydra
dot-overrides directly (the materiality check uses those), but the `adbo.exp_*`/`sweep_vmax`
drivers are argparse and compose their config internally via
`adforce.wrap.get_default_config()` — dot-overrides can't reach them, and they would silently
compose the ARCHER2 defaults (108 ranks, Cray modules, `/work/...` exe path) on a VM.

`get_default_config()` now honours three env vars (applied only when set, so ARCHER2
behaviour is unchanged; pinned by `tests/test_wrap_config_env.py`, 3/3):
| var | effect |
|---|---|
| `ADCIRC_EXE_PATH` | `files.exe_path` (dir with adcprep/padcirc) — baked into the image |
| `ADCIRC_NP` | `slurm.tasks_per_node=N, reserved_cpus=0` → N MPI ranks on the mid mesh (the only resolution the observe loop accepts — it asserts `resolution=='mid'`, so the `high`/8-node multiplier is unreachable). Image bakes a safe `16`; match your VM with `docker run -e ADCIRC_NP=<n>` |
| `WORSTSURGE_MODULES` | `slurm.modules`; `""` (baked into the image) disables the `module load` step |

Integration-proven: `python -m adbo.sweep_vmax --test` with these set emits `wrap_cfg.yaml`
with `exe_path: /opt/adcirc/work, tasks_per_node: 16, reserved_cpus: 0, modules: ''`.
(`vm_files_config.yaml` remains as an optional Hydra-group alternative for `adforce.wrap` runs.)

## Reproducibility
The **image is the reproducible artifact** — pin the ADCIRC fork commit and the env, build
once, push to Artifact Registry, and every run (and the paper) references the same container.
This finally gives ADCIRC the reproducibility the ERA5/PS side is getting via `rerun/`.

## Open iteration points (be honest — this is a draft)
- **The ADCIRC build is the risk**, but we now start from the fork's own `compile.sh` (CMake;
  the README says CMake "much more reliable than Makefile") — the Dockerfile mirrors its flags.
  The fork also has a `requirements/` folder worth reading for its non-Cray dependency set. The
  `find … -name padcirc` copy may still need its path adjusted to wherever the build tree emits
  the binaries; the smoke test's step-2 check will tell you immediately.
- **Rank count vs mesh size**: 31k nodes over-partitioned into too many subdomains gets
  inefficient/unstable — keep `NP` modest (16–32), not 108.
- **Profiles for the materiality check** come from `w22/data` (the pre-fix JSONs need the fixed
  one regenerated first — a cheap CLE15/w22 step; see `../../unit_bug_fix.md`).
