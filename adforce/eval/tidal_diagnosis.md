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

## The experiment (plumbed and ready — run on GCP)

The friction axis is wired end-to-end: `mannings_n` is an eval-side
pseudo-axis (with `forcing`) in matrix cells; launch passes it through
`drive_storm` → `generate_adcirc_inputs`, which writes a fort.13 whose
uniform default is rewritten in place (per-node overrides untouched —
verified: the n=0.028 variant differs from the shipped deck by exactly one
line); provenance lands in the run's `config.yaml` under `eval_axes`, so a
friction cell can never be confused with a control.

On the GCP VM (worstsurge container; ADCIRC-only allocation — ARCHER2 is
retired):

```bash
python -m adforce.eval.launch study=mannings matrix=mannings_tide \
    'storms=["Katrina 2005","Ida 2021","Matthew 2016"]'              # plan (9 runs)
python -m adforce.eval.launch study=mannings matrix=mannings_tide \
    'storms=["Katrina 2005","Ida 2021","Matthew 2016"]' dry_run=false
# each run auto-reduces to gauge_ts.parquet and strips fort.63 afterwards
```

Locally, after `python -m adforce.eval.harvest remote=gcp-vm:... study=mannings dry_run=false`:

```bash
for n in 0.022 0.028 0.035; do
  python -m adforce.eval.tidecheck "series=data/comp/runs/mannings/tide-n$n/*/gauge_ts.parquet" label=n$n
  python -m adforce.eval.tideconst "series=data/comp/runs/mannings/tide-n$n/*/gauge_ts.parquet" label=n$n
done
```

Success = O1/K1 amp ratios → 1 and the phase lead shrinking without
destroying M2 (Matthew guards the Florida semidiurnal side).

Also: exclude the TX-bay gauges from any tide-on skill panel (mesh, not
physics — unfixable without inlet refinement), and treat `datum_offset_m`
as a reported result everywhere (never absorb it silently).
