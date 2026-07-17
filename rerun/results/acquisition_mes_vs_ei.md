# MES vs EI acquisition comparison — INVALIDATED (data corruption in the MES arm)

*2026-07-14, updated same day. An earlier version of this note concluded "EI beat
MES by 14%". That conclusion is **retracted**: the MES run's data was corrupted
by infrastructure failures, discovered by auditing the ledgers. Kept as a
methods post-mortem + protocol for the clean redo.*

## Design (was sound)

New Orleans 2015 fixed profile (`2015_new_orleans_profile_r4i1p1f1`, vmax = Vp),
3D track search (angle ∈ [−80°, 80°], trans_speed ∈ [0, 15] m/s,
displacement ∈ [−2, 2]°), seed 42, 25-point LHS init + 25 acquisition evals,
trieste 4.4.0 (`MinValueEntropySearch` / `ExpectedImprovement` on the negated
objective, inputs rescaled to the unit box — scaling audited, no issue).
Pairing verified: the two ledgers are **point-identical for all 25 init evals**.
Ledgers: `no3d-{mes,ei}-s42_ledger.json`.

## Why it's invalid (mechanism, corrected 2026-07-14 late)

The direct killer was a **"janitor" cleanup loop racing the live run**, not
disk-full per se (disk pressure was why the janitor was added). The campaign
launch scripts ran `rm -rf exp_00*/PE0*` on a timer (600 s in `ei.sh` /
`ctl_then_mes.sh`, 480 s in `mes3.sh`) over *all* eval dirs including the
active one. Deleting the already-open fort.6x outputs of a running eval is
harmless on Linux; deleting the PE partition dirs in the ~1-min
adcprep→padcirc startup window kills every rank and leaves an all-fill
maxele. The PE dirs (partitioned fort.14/15/18, a few MB) were never the
storage problem — the ~1 GB/eval was fort.63/64/73/74 — so the fatal
`PE0*` clause bought nothing.

Whether a kill corrupted data depended on the error handling in play:

* `ei.sh` and `ctl_then_mes.sh` ran the **unpatched** exp.py (fail-loudly):
  janitor kills crashed the campaign visibly (EI needed several relaunches —
  ei.log/ei2.log/eirun*.log) and the interrupted point simply re-ran. Wall
  time lost, **no data corruption** — EI's 50 evals contain no zeros.
* `mes3.sh` (the eval-36 resume, 2026-07-14 05:59) was the only script that
  combined the janitor with the then-current 0.0-scoring patch: kills became
  silent observations. MES ledger evals **36, 37, 39, 40, 45, 49, 50 are
  0.0** — all at points adjacent to MES's own 11.7 m incumbent (e.g. angle
  +13.6°, ts 1.13, disp 0.40 vs incumbent +10.1°, 1.24, 0.45), physically
  impossible as true zeros. They poisoned the GP exactly where MES was
  exploiting, explaining its flat best-so-far after eval 35.

Numbers for the record: EI best 13.670 m, MES best 11.698 m — **not
comparable**.

**Blast radius audit (all ledgers scanned for res == 0.0):** the janitor
first appears 2026-07-13 13:14 (`ei.sh`); everything earlier (materiality,
tide decks, both 70-eval 4D campaigns, all 1D sweeps) ran without one and
contains zero 0.0-scored evals. Deck-based results are additionally
cross-validated (sg-1.2 profile 6.403 m vs storm_ctl_g2 6.402 m on
independent workers/images). The kill mode is binary — a hit either crashes
the run (all-fill maxele) or misses entirely — so there is no channel for
plausible-but-wrong values. Total silent damage: the 7 MES-s42 zeros,
i.e. this one comparison.

## What the pre-existing evidence actually says

1. **Thesis appendix (worstpp `tab:all-params-ei` / `tab:all-params-mes`;
   4 EI + 12 MES trials × 6 site-years, 50 evals each, old solver):** EI's
   best-found is higher in effectively every block — NO-2015 EI 16.20–16.65
   (4/4) vs MES 13.21–16.42 (median ≈ 16.2); Miami-2015 EI ≈ 5.87 vs MES ≈ 5.49
   mean; Galveston-2015 EI ≈ 8.76 vs MES ≈ 7.87. MES also shows occasional
   catastrophic stalls (NO-2100 trials finishing at 7.07 / 9.21 m with the best
   found at eval ≤ 3). So the recollection "MES won" is not supported by these
   tables either — it may predate them or refer to a different criterion
   (consistency of optimum *location*, not value). The thesis itself flags the
   comparison as preparatory ("a more thorough comparison is needed").
2. **Seed variance dominates acquisition choice at this budget.** Thesis trials
   routinely reached 15.5–16.6 m at NO-2015 in 50 evals, and today's 4D run
   found 16.52 m at the angle = −80° boundary (inside the 3D slice) — yet
   *both* of today's arms, sharing one seed-42 LHS init that left the oblique
   corner uncovered, plateaued 3–5 m below that. Single-seed comparisons of
   acquisition functions are noise.

## Fixes applied (before any redo)

- `adforce/wrap.py`: the maxele guard now separates **`DryObservationPoint`**
  (obs node invalid while the mesh is ≥ 50% valid — healthy runs measure 100%)
  from **`AdcircRunFailure`** (mesh mostly invalid; deliberately *not* a
  ValueError). `adbo/exp.py` scores only the former as 0.0; failures again kill
  the loop loudly. Regression tests in `tests/test_wrap_guard.py` (12 pass).
- Remaining hazard to close at source: extend `low_storage` cleanup to delete
  fort.63/64/73/74 after each eval so disk-full cannot recur (janitor loops are
  a workaround, and forgetting one is what corrupted this run).

## Clean redo campaign — RESULTS (2026-07-15)

Paired design executed: seeds 1–5 × {MES, EI}, shared per-seed 25-pt LHS init
(verified identical within every pair), 25 acquisition evals, NO-2015 fixed
profile, patched wrap/exp bind-mounted, sequential cleanup only (no janitor),
`redo3d-*` ledgers in `rerun/results/redo3d/`. Zero failure-scored evals in
all 500 evaluations; two spot preemptions self-healed via ledger resume.

| seed | init-25 best | MES best@50 (eval, angle) | EI best@50 (eval, angle) | EI−MES |
|---|---|---|---|---|
| 1 | 13.808 | 16.147 (44, −56°) | 16.189 (50, −57°) | +0.042 |
| 2 | 8.019 | **16.407** (31, **−80°**) | 16.314 (44, −67°) | −0.092 |
| 3 | 14.455 | 16.313 (27, −68°) | 16.307 (49, −64°) | −0.005 |
| 4 | 15.641 | 16.402 (49, −80°) | 16.401 (38, −80°) | −0.001 |
| 5 | 12.371 | 14.771 (47, −32°) | 15.990 (48, −56°) | +1.219 |

- **No significant acquisition effect**: paired mean EI−MES = +0.233 m
  (sd 0.554, t = 0.94, n = 5, n.s.). Four of five pairs agree within 0.1 m.
- Convergence speed is identical: evals to 95% of the pooled best —
  MES [28, 29, 27, 13, >50], EI [28, 30, 27, 13, 42].
- The one real difference is a **left-tail stall** (MES, seed 5, stuck at
  −32°): consistent with the thesis-era observation that MES occasionally
  stalls, but 1-in-5 here vs ties elsewhere.
- Every clean run (16.0–16.4 m; pooled best 16.407 at the −80° corner,
  ts 1.5) sits where the thesis-era trials (15.5–16.6) and the 4D corner
  (16.52) sit — confirming the corrupted seed-42 pair (11.70 / 13.67) was
  artifact, not acquisition physics. The potential height at this
  site/profile is ≥ 16.4 m, and 50-eval BO reaches within ~1% of it on
  4 of 5 seeds with either acquisition.
- Paper-facing conclusion: the choice of MES vs EI does not materially
  affect the potential height found; robustness comes from seeds and
  verification sweeps, not the acquisition function.

### Bonus pair: seed 42 re-run clean (both arms)

| | corrupted/janitor era | clean re-run |
|---|---|---|
| MES-42 | 11.698 (GP poisoned by 7 false zeros) | **13.093** (eval 43, angle −4°, ts 0.31) |
| EI-42 | 13.670 | **13.670** (eval 50, angle −12°, ts 1.28) |

Two closures: (i) the clean EI-42 reproduces the janitor-era EI-42 **exactly**
(13.670, same optimum, same eval) — the original EI number was legitimate; its
crashes cost only wall time. (ii) Seed 42 is simply a bad seed: its init
(best 10.694, worst of all six) steers *both* acquisitions into the weak
near-perpendicular basin, and neither finds the −56…−80° oblique basin in
25 acquisition evals — both arms land ~3 m below seeds 1–4. Seed/init
variance (±3 m) dwarfs acquisition choice (±0.3 m).

### Final statistics (n = 6 paired seeds)

Paired diffs EI−MES: [+0.042, −0.092, −0.005, −0.001, +1.219, +0.577] →
mean **+0.290 m, t = 1.38, not significant** (crit 2.57). MES mean 15.52
(range 13.09–16.41), EI mean 15.81 (range 13.67–16.40), pooled best 16.407.
EI never lost materially; MES stalled in a weaker basin on 2 of 6 seeds.
Fair summary for the paper: *no significant acquisition effect; the dominant
uncertainty in a 50-eval search is the initial design, which argues for
multi-seed potential-height estimates and 1D verification sweeps rather than
acquisition-function tuning.*

## GIBBON batch acquisition on the trap seed (2026-07-16)

Follow-up to the whole-seed-failure diagnosis: `adbo` now supports
**GIBBON** (batch max-value entropy search, Moss et al. 2021) via
`--daf gibbon --batch N` (batches chosen jointly with a repulsion term;
`bo-config.json` records `daf`/`batch_size`). Validation run
`redo3d-gibbon-s42`: the seed-42 trap init, batch = 3, same 25+25 budget,
ledger `l_gibbon_s42.json`.

| run (identical init) | best@50 | at eval | best angle | behaviour |
|---|---|---|---|---|
| MES (sequential) | 13.093 | 43 | −4° | plateaued in decoy basin |
| EI (sequential) | 13.670 | 50 | −12° | plateaued in decoy basin |
| **GIBBON batch-3** | **14.924** | 50 | −35° | still climbing at cutoff |

GIBBON's diverse batches never hard-locked: after four batches probing the
decoy region it walked obliqueward batch-by-batch (−4° → −9° → −16° → −22°
→ −29° → −35°), gaining +1.25 m over EI and +1.83 m over MES on the same
init — but 25 acquisition evals ran out before it reached the −56…−80°
basin (regret ≈ 1.6 m vs 16.5). **Conclusion: batch diversity prevents the
plateau failure mode but does not replace basin knowledge — the scalable
design for higher dimensions is physics-informed init points + GIBBON
batches (+ a stagnation kill-rule), with a 1D sweep as the unbiased check.**
Note: within the current driver the batch still *evaluates* sequentially;
true wall-clock parallelism needs a multi-worker dispatcher in `obj()`.

## Original protocol (as designed)

N ≥ 5 paired seeds × {MES, EI}, shared init, halt-and-inspect on ledger
zeros, per-seed paired best@50 + regret curves. Actual cost ≈ $25 (3 spot
c2d-standard-16, ~30 h wall including one aborted launch).
