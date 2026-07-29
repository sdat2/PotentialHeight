# Resolution sensitivity and tide-surge interaction: consolidated verdict

*2026-07-29. The two model-fidelity experiments requested for the three-city
paper: (A) the full historical validation sweep re-run on the low-resolution
mesh (8,303 nodes) with identical GAHM/IBTrACS forcing, plus low-resolution
potential-height experiments; (B) the storm/tide/storm+tide interaction
triple at both resolutions (6-day tidal spinup, HAMTIDE constituents, TAU-
renumbered GAHM padding). Total ~150 ADCIRC runs across 3 spot VMs, ~$15.*

## A. Does the resolution error extrapolate linearly to the potential height?

**No — it shrinks, and the optimized ceiling is nearly mesh-invariant.**

| quantity | mid (31k) | low (8k) | low/mid |
|---|---|---|---|
| Shell Beach validation events (2-4 m) | ×1.24 vs obs | ×0.83 vs obs | **0.67** |
| mid argmax s2 re-run (−80°, 1.52 m/s) | 16.41 m | 15.05 m | **0.92** |
| mid argmax s1 re-run (−56°, 2.88 m/s) | 16.15 m | 13.88 m | **0.86** |
| own optimum (50-eval GIBBON BO) | 16.41 m | **16.63 m** | **1.01** |

Linear extrapolation of the validation-amplitude ratio predicted ~11 m at
low resolution; the actual fixed-storm values are 13.9-15.0 m and the
re-optimized potential height fully recovers (via a different optimum:
angle −59°, ts 0.70 m/s — BO adapts the storm to the mesh). Interpretation:
the coarse-mesh error at validation amplitudes is dominated by unresolved
marsh/funnel friction, which drowns out at ~16 m amplitudes. The New Orleans
potential height carries a **≲10% mesh-amplitude uncertainty (optimized:
~1%)**, far below the γ_sg envelope (−47%/+43%). Obs point pinned at the mid
node's coordinates (−89.8204, 29.9595; low mesh's nearest wet node is in the
same Lake Borgne basin, 8.9 km away). Ledgers: `rerun/results/lowres_ph/`.

## B. Nonlinear tide-surge interaction (both resolutions, 19 storms)

Peak-based interaction = peak(storm+tide) − peak(storm) − peak(tide),
`rerun/results/tide_surge_interaction.csv` (pairs with storm peak ≥ 0.5 m):

| res | n | median interaction | median as % of storm peak |
|---|---|---|---|
| low | 134 | −0.33 m | −24% |
| mid | 121 | −0.34 m | −38% |

**Destructive, resolution-robust, and site-dependent:**

* **New Orleans funnel (Shell Beach)**: −0.6 to −0.7 m (Gustav/Ike/Isaac) —
  the tide's contribution to the *peak* is almost fully cancelled
  (Isaac: 4.01 storm + 0.50 tide → 3.89 combined). Ignoring tides costs
  ~0.1 m; adding them linearly overstates by ~0.6 m.
* **Galveston Pier 21 (open coast)**: Ike interaction −0.03 m — near-linear
  addition (2.49 + 0.60 → 3.07).
* **Miami (Virginia Key)**: Irma 1.09 + 0.51 → 1.48 (interaction −0.12) —
  most of the (larger, semidiurnal) tide survives into the peak, confirming
  tides matter more at Miami both in amplitude and in linearity; but at
  Fernandina/NE-Florida (1.3-1.5 m tides) the interaction is strongly
  destructive again (−0.9 to −1.0 m).

Caveats: mid storm-only peaks for 13/19 storms come from the EC95d archive
(same mesh + forcing pipeline, slightly different dt/adcircpy vintage;
`storm_src` column flags them); tide phase is the historical calendar (one
phase draw per storm), so these are typical-phase interactions, not
worst-case alignment (the planned τ-BO addresses that); low-resolution tidal
amplitudes are within ~×1.5 of observed constituents (8 constituents,
HAMTIDE boundary).

## Paper implications

1. NO's mid-mesh ×1.24 validation bias should be quoted with the amplitude
   caveat: at the potential-height regime the mesh sensitivity is ≲10%
   (optimized ~1%) — γ_sg dominates everything.
2. "Storm tide = surge + tide" double-counts at shallow Gulf sites by
   0.3-0.7 m; the no-tide simplification is accurate to ~0.1 m at the NO
   peak. At Miami, tides deserve inclusion (or the τ-BO variable).
3. The three-city validation + these two sensitivity studies together give
   each city's potential height a signed, quantified fidelity statement
   (NO: mesh-hot but amplitude-forgiven; Galveston: near-neutral; Miami:
   wave-cold, tide-relevant).
