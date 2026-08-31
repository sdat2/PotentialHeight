# Tidal-forcing diagnosis (2026-08-16)

Why the tide-on programme is gated: the EC95d tide-only runs over-predict
coastal tidal amplitude at the NOAA Gulf/Florida gauge panel, and the error
is **resolution-independent** — so it is the tidal setup, not the mesh.
Produced by `adforce.eval.tidecheck` / `adforce.eval.tideconst` from the
2026-07 GCP sweeps (`data/comp/{lowres,midres}` extracts).

## Evidence

**Bulk (tidecheck; tide-only runs vs CO-OPS predictions):**

| panel | pairs | amp ratio (median) | lag (median) | r | rmse | \|datum offset\| |
| --- | --- | --- | --- | --- | --- | --- |
| low  | 1019 | **1.45** | −66 min | 0.82 | 0.13 m | 0.09 m |
| mid  | 969  | **1.41** | −42 min | 0.86 | 0.13 m | 0.09 m |

Splitting by gauge class (paired low/mid): the TX semi-enclosed bays
(Seadrift, Aransas, Rockport, Copano, Matagorda, Packery) are structurally
broken at **both** resolutions (amp ≈ 5.8×, r ≈ 0.1 — the narrow inlets that
choke tidal energy are unresolved at either mesh); the rest of the panel
sits at amp ≈ 1.39–1.41 with good phase correlation (r ≈ 0.85–0.87) at both.

**Per constituent (tideconst; 6 storms, low panel, fixed 5-constituent utide
fits on both sides; 13–15-day windows — S2 carries M2-leakage caveat):**

| constituent | n | pred amp (med) | amp ratio (q25–med–q75) | Δphase (med) |
| --- | --- | --- | --- | --- |
| M2 | 272 | 0.04 m | 0.80 – **1.05** – 1.52 | −34° |
| S2 | 227 | 0.03 m | 1.13 – **1.52** – 2.08 | −33° |
| N2 | 139 | 0.03 m | 0.77 – **1.08** – 1.70 | −17° |
| K1 | 279 | 0.10 m | 1.03 – **1.18** – 1.42 | −26° |
| O1 | 279 | 0.12 m | 1.19 – **1.34** – 1.62 | −20° |

The Gulf panel is diurnal-dominated (O1/K1 ≈ 10–12 cm vs M2 ≈ 4 cm), and it
is precisely the diurnal band that is inflated (O1 +34%, K1 +18%), with a
uniform ~20–35° phase **lead** across all bands.

**Tide-on consequence (twl; low storm+tide runs vs raw gauges):** aligned
instantaneous TWL peak bias is deceptively small (−0.02 m median) because
the over-amplified tide compensates the under-predicted surge; the skew
surge exposes it (model 0.07 m vs observed 0.36 m median). Median aligned
datum offset 0.20 m (storm+tide) / 0.09 m (tide-only).

## Where the setup comes from

Historical tide runs route through `adforce.training.inputs.
generate_adcirc_inputs` (`adforce/training/inputs.py:244-256`):
adcircpy `Tides(tidal_source=HAMTIDE)` with 8 constituents
(M2 S2 N2 K2 K1 O1 P1 Q1), nodal factors per run window; **no explicit
friction settings** — bottom friction is fort.13
`mannings_n_at_sea_floor` with a uniform default **n = 0.022**
(`adforce/setup/fort.13.mid`). 6-day tidal spinup; NRAMP via the custom
Fort15 subclass.

## Interpretation

Diurnal-selective amplification + an all-band phase lead is the signature of
an **under-damped basin**: the Gulf of Mexico is near-resonant for the
diurnal band (Helmholtz-like response through the Yucatan/Florida straits),
so insufficient dissipation inflates O1/K1 specifically and speeds up
propagation everywhere. A uniform Manning's n of 0.022 is at the low end
for Gulf-shelf ADCIRC setups. Boundary-forcing (HAMTIDE) error cannot be
excluded, but it would not naturally produce band-selective amplification
with this phase structure; M2 sitting near 1.0 argues the boundary input is
roughly right.

## Update (2026-08-17): the Manning's-n sweep was a verified NO-OP — NWP=0

The 9-run GCP sweep (n ∈ {0.022, 0.028, 0.035} × {Katrina, Ida, Matthew})
completed 9/9 and produced **byte-identical `gauge_ts.parquet`s across the
three friction cells** (identical md5s). Root cause, read off the harvested
decks: the generated fort.15 (and the static idealized decks alike) carry
**`NWP = 0`** — ADCIRC is told there are no nodal attributes, so the copied
fort.13 (and its Manning field) has NEVER been read by any run in this
project. The live friction is fort.15's `NOLIBF=2` hybrid line:

```
CF HBREAK FTHETA FGAMMA = 0.0025 1 10 0.333333
```

i.e. a **uniform quadratic Cd = 0.0025 over the whole basin**, with the
Manning-like depth scaling only engaging below HBREAK = 1 m — the Gulf
shelf effectively runs at the deep-water default. This *strengthens* the
under-damped-basin interpretation and redirects the experiment to fort.15's
CF (`friction_cf` axis; the `mannings_n` axis is now blocked at plan time
with a pointer here). Consolation prize: three byte-identical runs from
three independent launches is a clean end-to-end determinism proof of the
pipeline.

## The experiment, take 2: friction_cf (fort.15 CF) — run on GCP

The friction axis is wired end-to-end: `friction_cf` is an eval-side
pseudo-axis (with `forcing`) in matrix cells; launch passes it through
`drive_storm` → `generate_adcirc_inputs`, which rewrites the generated
fort.15's `CF HBREAK FTHETA FGAMMA` line in place (CF only, the other three
kept — `adforce/fort15.py`, verified single-line diff on the harvested
decks); provenance lands in the run's `config.yaml` under `eval_axes`, so a
friction cell can never be confused with a control. `mannings_n` cells are
refused at plan time AND at input generation (defense in depth).

On the GCP VM (worstsurge container; ADCIRC-only allocation — ARCHER2 is
retired):

```bash
python -m adforce.eval.launch study=friction matrix=friction_tide \
    'storms=["Katrina 2005","Ida 2021","Matthew 2016"]'              # plan (9 runs)
python -m adforce.eval.launch study=friction matrix=friction_tide \
    'storms=["Katrina 2005","Ida 2021","Matthew 2016"]' dry_run=false
# each run auto-reduces to gauge_ts.parquet and strips fort.63 afterwards
```

Locally, after `python -m adforce.eval.harvest remote=gcp-vm:... study=friction dry_run=false`:

```bash
for n in 0.0025 0.005 0.0075; do
  python -m adforce.eval.tidecheck "series=data/comp/runs/friction/tide-cf$n/*/gauge_ts.parquet" label=cf$n
  python -m adforce.eval.tideconst "series=data/comp/runs/friction/tide-cf$n/*/gauge_ts.parquet" label=cf$n
done
```

Success = O1/K1 amp ratios → 1 and the phase lead shrinking without
destroying M2 (Matthew guards the Florida semidiurnal side).

## Take-2 results (2026-08-18): friction is the semidiurnal knob, not the diurnal one

9/9 runs (CF verified in each harvested deck; the cf0.0025 control's Katrina
parquet is byte-identical to the mannings-sweep runs — a fourth independent
determinism check). 153 scored pairs per cell:

| CF | bulk amp | bulk lag | M2 | S2 | N2 | K1 | O1 | r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0025 (control) | 1.43 | −36 min | 1.15 | 1.44 | 1.60 | 1.20 | 1.39 | 0.884 |
| 0.005 | 1.32 | −24 min | **1.05** | 1.34 | 1.59 | 1.13 | 1.34 | 0.880 |
| 0.0075 | 1.30 | −18 min | **0.98** | 1.24 | 1.56 | 1.11 | **1.34** | 0.872 |

Friction acts exactly as the under-damped hypothesis predicted — amplitudes
down, phase leads shrink, correlation unharmed — **but band-selectively**:

* the **semidiurnal band is friction-controlled**: M2 is fully corrected at
  3×CF (1.15 → 0.98; q25 already 0.77 — over-damping begins), S2 −0.20,
  all phase leads roughly halved;
* the **diurnal band saturates**: O1 drops 1.39 → 1.34 on the first doubling
  and then stops; K1 similarly (1.20 → 1.11). The residual ≈ +30% O1 /
  +11% K1 is **friction-insensitive** — consistent with the GoM's diurnal
  amplitude being set by the basin's admittance through the Yucatan/Florida
  straits (Helmholtz-like) and/or HAMTIDE's diurnal boundary amplitudes,
  not by shelf dissipation.

**Verdict.** (1) CF ≈ 0.005 is the tide-side sweet spot (M2 ≈ 1.05, leads
halved, nothing over-damped). (2) The remaining diurnal bias needs a
boundary-side check — compare HAMTIDE O1/K1 boundary amplitudes against
TPXO/observations, or run one cell with an alternative tidal database —
and until then should be treated as a known, quantified bias (~+34% O1).
(3) **Do not change CF for surge runs without re-validating surge**: the
published surge skill (bias −0.22 m, r 0.871) was obtained at CF = 0.0025;
any friction change perturbs it, so a surge cell (storm-only, cf0.005) must
go through `adforce.eval.validate`-equivalent scoring before adoption.

Also: exclude the TX-bay gauges from any tide-on skill panel (mesh, not
physics — unfixable without inlet refinement), and treat `datum_offset_m`
as a reported result everywhere (never absorb it silently).

## Final pieces (2026-08-18): boundary exonerated; surge vetoes the CF change

**Surge sensitivity (friction_storm sweep; Laura + Ida paired, 22
meaningful-surge gauges — Katrina storm-mode runs failed on a GAHM
">4 isotachs per cycle" error from prep-image environment drift, a known
issue):** doubling CF to 0.005 damps surge peaks by a **median −24%
(−0.19 m)**, worsening peak bias from −0.09 to −0.20 m on this subset and
degrading hydrograph skill (r 0.50 → 0.43). **The tide-side sweet spot is
surge-side harmful: keep CF = 0.0025 for the surge/potential-height
configuration.** Reconciling tides and surge in one deck needs
spatially-varying friction — i.e. finally implementing NWP = 1 — or
accepting the documented tide bias. (`data/comp/out/friction_surge_sensitivity.csv`)

**Boundary check (`adforce.eval.tidedb`; model vs HAMTIDE vs NOAA harcon at
39–49 gauges):** HAMTIDE's own Gulf field sits at **~+8% vs NOAA for every
constituent** — the forcing database is exonerated. The model amplifies O1
by +25% *relative to the HAMTIDE field that forces it* (model/ham: M2 1.00,
K1 0.91, O1 1.25), and that amplification is friction-insensitive ⇒ it is
the **EC95d basin response** (coarse Yucatan/Florida straits setting the
diurnal Helmholtz admittance), fixable only by mesh/domain work. Two
metric caveats sharpened here: harcon-K1 shows the model K1 is actually
fine (≈0.95) — tideconst's K1 ≈ 1.2 was partly P1 leakage in 13-day fits —
and S2 comparisons carry NOAA's radiational-S2 component.
(`data/comp/out/tidedb_threeway.csv`)

**Closing verdict.** The tidal error decomposes into: (a) a
friction-controlled semidiurnal part — correctable (CF≈0.005) but vetoed by
surge skill; (b) a friction-insensitive O1 basin response from the EC95d
domain (~+25%) — a mesh property, to be reported as a known bias; (c) TX-bay
gauges structurally unusable. For the paper's tide-excluded surge
methodology none of this changes the published configuration — it
*justifies* it: tides-off + de-tided-residual comparison sidesteps a tidal
model whose basin response cannot be made accurate without mesh work.
