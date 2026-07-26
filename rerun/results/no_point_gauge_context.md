# Observational context for the New Orleans potential-height point

*2026-07-26. The paper's NO point (−90.0715, 29.9511) is observed by ADCIRC at
the nearest wet mid-mesh node, (−89.820, 29.960) — in the Lake Borgne funnel
east of the city, not Lake Pontchartrain. The most comparable NOAA gauge is
therefore **Shell Beach, LA (8761305)**: 17.5 km away, same basin. (Carrollton
6.5 km and New Canal 9.4 km are nearer the nominal city point but are a
Mississippi River stage gauge and a Pontchartrain gauge respectively.)*

## Largest surges at Shell Beach

| source | event | peak surge |
|---|---|---|
| Observed (de-tided residual, full record 2009–2025) | **Isaac 2012** | **3.11 m** |
| Observed, runner-up | Ida 2021 | 2.15 m |
| Observed, partial year 2008 | Gustav 2008 | 2.64 m |
| Largest non-TC observed | Nov 2009 (Ida'09 remnant) | 1.45 m |
| ADCIRC historical archive (EC95d, GAHM) at gauge node | **Katrina 2005** | **6.06 m** |
| — same, at the NO obs node | Katrina 2005 | 5.84 m |
| ADCIRC archive runner-up (obs node) | Isaac 4.56 m; Gustav 3.81 m | |
| Idealized Katrina-like control (paper mid mesh, obs node) | storm_ctl_g2 | 6.40 m |
| 2015 potential height (paper, obs node) | — | ≈ 16 m |

## Is there any comparable point for Katrina?

No surviving gauge. The CO-OPS network was censored exactly at the surge
maximum: Shell Beach was destroyed, Bay Waveland Yacht Club (63 km, MS Sound
ground zero) returned no usable 2005 data, and the nearest *clean* Katrina
pair in the validation set is **Grand Isle, 79 km away and west of the track**
— a different surge regime entirely (observed 1.40 m). The largest clean
gauge observation of Katrina anywhere in our set is Pilots Station East,
S.W. Pass (121 km): 1.87 m. Every instrument in the 5–6 m core region died.

The comparable Katrina observations for the obs node are therefore
**high-water marks**, not gauges: the USACE/IPET/FEMA collections (FEMA
Louisiana HWM report; St. Bernard polder studies) document surge exceeding
15 ft (≥4.6 m) through Lake Borgne and the IHNC, with marks in the
St. Bernard / MRGO funnel commonly ~4.6–5.8 m NAVD88. The historical-archive
ADCIRC value at the obs node (5.84 m) sits at the top of that range —
consistent, though this node's archive bias for its two biggest gauged events
(Isaac, Gustav) is ~+32%, so the true Katrina peak there was plausibly nearer
the low-to-middle of the HWM band.

Methodological point for the paper: gauge censoring at the extremes means the
observed record's 3.11 m ceiling understates even the *historical* maximum by
roughly a factor of two — a concrete argument for model-informed bounds over
purely observational extreme-value fits at this site.

Caveats: (i) the gauge record **misses Katrina** — see above. (ii) At this gauge the EC95d archive **over-predicts** its two
largest cleanly observed events by ~+32% (Isaac 4.13 vs 3.11; Gustav 3.48 vs
2.64) — opposite in sign to the Miami-region under-prediction; the coarse mesh
in a semi-enclosed funnel. Sim-vs-obs pairs from `comp.validate`
(`val_summary.csv`); annual maxima from `comp.annual_max.annual_maxima
("8761305", 1995, 2025)` (cached parquet); archive peaks recomputed from the
19 cached `sdat2/surgenet-train` storm files (nearest wet node, min WD > 0.3,
SSH = WD + DEM).
