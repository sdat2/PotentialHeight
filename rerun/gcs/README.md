# rerun/gcs/ — ERA5 potential-size re-solve on Google Cloud (DRAFT)

**Status: untested draft scaffolding** (reviewed: a 27-finding adversarial pass has been
applied). Fill in `config.sh`, read each script before running, and refine. This moves the
remaining ERA5 decades (2000s, 2010s, 2020–2024) off the laptop's flaky external SSD onto
Google Cloud — reliable local NVMe on the VM, durable object storage in GCS, and cheap
preemptible compute. This directory covers **Part A** (the embarrassingly-parallel Python
job); **Part B (ADCIRC + BO) lives in [`adcirc/`](adcirc/)** — Dockerfile, shims, smoke
test, materiality-check profiles — see its README. The ADCIRC compile has been proven
locally (linux/amd64, BUILD TEST OK).

## Why this shape
The job is pure Python + data + CPU, RAM-light, and **already checkpointed per year**.
That makes it ideal for a **Spot VM** (~60–70% cheaper, can be reclaimed): if preempted,
just re-run the same command and it skips finished years. GCS is the durable store (the
ARCHER2/SSD replacement); the VM is disposable.

```
   your SSD ──upload once──▶ gs://BUCKET/data/era5/{raw,pi_og}   (inputs)
                                     │
                          spot VM pulls to local NVMe
                                     │  era5_ps_local.py under mem_guard.py (joblib, N vCPU)
                                     ▼
   gs://BUCKET/ps_fixed/era5_ps_fixed_{year}.nc  ◀──rsync every 2 min──  local ps_fixed/
```

## Files
| file | role |
|------|------|
| `config.sh` | **edit this** — project, region, bucket, machine type, job params |
| `environment.yml` | compute-subset conda env (validated by an import check on the VM) |
| `provision.sh` | create bucket, upload a decade's inputs, create the spot VM |
| `startup.sh` | VM bootstrap: micromamba + env + clone repo (metadata startup-script or run by hand) |
| `run_decade_gcs.sh` | on the VM: pull inputs+checkpoints, run one decade, sync outputs back |
| `adcirc/` | **Part B**: ADCIRC v55.02 Docker build (locally proven), srun/module shims, smoke test, materiality profiles + wiring tests — see `adcirc/README.md` |

## Prerequisites
- A GCP project with billing enabled; `gcloud` CLI installed and authed (`gcloud auth login`, `gcloud config set project <id>`).
- The local ERA5 decade files on the SSD (or a `~/.cdsapirc` to re-download from Copernicus on the VM instead of uploading).

## Steps
```bash
cd rerun/gcs
$EDITOR config.sh                 # fill in PROJECT / BUCKET / ZONE
./provision.sh                    # bucket + upload 2000s inputs + create VM (read it first!)
gcloud compute ssh era5-worker --zone=europe-west2-c
#   on the VM:
cd ~/worstsurge/rerun/gcs && ./run_decade_gcs.sh 2000 2009
#   repeat for 2010 2019, then 2020 2024 (upload each decade's inputs first)
gcloud compute instances delete era5-worker --zone=europe-west2-c   # tear down when done
```
Combine, once all decades are in the bucket (pull local first — xarray can't glob a
`gs://` URL without extra gcsfs/fsspec plumbing):
```bash
gcloud storage cp 'gs://BUCKET/ps_fixed/era5_ps_fixed_*.nc' .
```
```python
import xarray as xr
ds = xr.open_mfdataset("era5_ps_fixed_*.nc", combine="nested", concat_dim="year")
```

## Considerations
- **Spot vs on-demand.** Spot is ~⅓ the price and preemption is a non-event here (per-year
  checkpoints + the 2-min output sync). Keep `SPOT=true`. On-demand only if you want zero interruptions.
- **x86, not ARM.** Use `c2d`/`c3`/`n2` so numba/scipy wheels are frictionless. (ARM would work but why fight it.)
- **Stage data to local disk — don't `gcsfuse`-mount it.** netCDF does many small random reads; a
  mounted bucket is slow/flaky for that. `run_decade_gcs.sh` copies inputs down to local NVMe first.
- **Cost:** ~$5–20 total for the 3.5 remaining decades on spot, plus ~$1/mo storage. Negligible.
  A decade is ~12 h at 16 vCPU.
- **Region:** keep bucket + VM in the same region (europe-west2 London for you); outputs are tiny so egress is trivial.
- **Back up the outputs.** GCS is durable, but consider also pulling each finished decade to your
  internal disk (as we did for the 1980s) so there's an off-cloud copy.

## Reproducibility hardening (after it works)
`environment.yml` is the compute *subset* of the local `t1` env and is not version-locked. Once the
import check passes on the VM, freeze it — either `conda-lock -f environment.yml` for a cross-platform
lockfile, or bake env + repo into a **Docker image** and run that. Either makes the whole pipeline
re-runnable years from now and is a clean artifact to cite in the paper. (This also finally closes the
"no lockfile" gap flagged in the repo audit.)
