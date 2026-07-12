# SWAN wave-setup materiality (DRAFT — in progress)

Goal: quantify how much **wave setup** adds to the potential height (R1's
"omitted physics"), by A/B-ing bare `padcirc` vs coupled `padcswan` on the
same storm. Expected cost ~5–15× per run (≈1–1.5 h on 16 vCPU vs 7.5 min),
so a 2-site A/B is a few dollars.

## Pieces
| piece | status |
|---|---|
| `padcswan` binary | building on adcirc-worker-3 (`swan_vm_startup.sh`; fork vendors SWAN, `compile.sh` built it on ARCHER2) |
| `fort.26` (this dir) | adapted from the OFFICIAL adcirc-testsuite coupled case (`adcirc_swan_apes_irene-parallel`); only the time window changed (2005-08-25 → 09-01, matching adforce's fort.15) |
| fort.15 changes | `NWS: 13 → 313` (+300 = SWAN-coupled, per the testsuite's NWS=320=20+300) and WTIMINC line `900` → `900 1200` (RSTIMINC = fort.26 COMPUTE/coupling interval) |
| run recipe | copy a completed padcirc run dir (fort.13/14/22.nc reusable), patch fort.15, drop in fort.26, `adcprep --np 16 --partmesh && --prepall`, `mpirun -np 16 padcswan` (same `--shm-size=8g --cap-add=SYS_PTRACE`) |

## A/B design
Control runs already exist (the humidity-materiality runs): `materiality_fixed_2015`
(NO surge 6.403 m). Wave run = same inputs + coupling. Δ(maxele at NO) = wave setup
at the observation point. Repeat at Miami (steep shelf — setup expected LARGER there;
control 4.98-m-class run exists from the 4D campaign's best).

## Caveats for interpretation
- Same mesh resolution for waves as surge (mid, 31k nodes) — coarse for surf-zone
  setup; treat the result as an order-of-magnitude materiality bound, not a
  production wave hindcast.
- SWAN physics = official test-case settings (KOMEN + AGROW, MADSEN friction,
  BSBT), not tuned.
