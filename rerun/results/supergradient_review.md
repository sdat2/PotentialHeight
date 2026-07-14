# The supergradient factor γ_sg: plausible range and evidence

*Synthesized 2026-07-14 from a salvaged deep-research sweep (108 extracted claims,
30 sources; formal 3-vote verification stage not completed — claims below are
consistent with the primary literature but page-level quotes should be spot-checked
before citation). Context: worstsurge uses γ_sg = 1.2 (inherited from Wang, Lin &
Chavas 2022), entering ONLY the W22 Carnot/angular-momentum budget
(`maximum_wind_speed = Vp × γ_sg` in `w22/ps.py`); our end-to-end sensitivity:
sg 1.0/1.1/1.2/1.3 → NO potential height 3.42/4.66/6.40/9.17 m (−47/−27/0/+43%).*

## Evidence ladder

| class | value(s) | sources |
|---|---|---|
| Linear BL theory | 1.02–1.04 (known underestimate; neglects vertical advection) | Kepert (2001) |
| Nonlinear BL model, stationary storms | **1.10–1.25**, larger for intense storms & peaked profiles | Kepert & Wang (2001) |
| Nonlinear BL model, by structure | 1.07 (moderate/broad), 1.11 (intense), 1.22 (peaked/inertially neutral) | Kepert & Wang (2001) |
| Per-storm validated hindcasts | Danielle 1.03, Isabel 1.15 — model matched obs in both | Schwendike & Kepert (2008) |
| Dropsonde obs at the RMW | Georges ≈1.00 (even subgradient); Mitch ≈1.15; Isabel 1.12–1.15 | Kepert (2006a,b); Bell & Montgomery (2008); Schwendike & Kepert (2008) |
| Large composites (~3000 sondes, 30 TCs) | azimuthal-mean flow ≈ balance on average (≈1.0); supergradient likelihood rises for Cat 3–5 and peaked profiles | (Zhang-type composite studies, 1999–2012 data) |
| Historical aircraft | Gray & Shea: supergradient inside RMW; Willoughby (1990): none | — |
| Axisymmetric numerics (radially near-inviscid ceiling) | v_max/E-PI ≈ 1.5 (108/72 m/s); classical overshoot bound 1.27–1.6 | Bryan & Rotunno (2009); Rott & Lewellen (1966) |
| Obs relative to PI (the W22-relevant ratio) | Isabel v_max/E-PI ≈ 1.25–1.33 over three days (incl. eye-entropy "turbo boost"); unbalanced correction alone 1.05–1.3 | Bell & Montgomery (2008); Bryan & Rotunno (2009); Rotunno (2022) |
| Surge/risk-model practice | Vickery et al. (2009): fitted U10/Vg = 0.85 vs dropsonde U10/Umax ≈ 0.72 ⇒ implied jet/gradient ≈ **1.18** near the RMW; many risk models (e.g. Lin & Chavas 2012 lineage) use gradient winds + 0.85 reduction, i.e. effectively γ_sg = 1 | Vickery et al. (2009); Lin & Chavas (2012) |

## Reading of the evidence

1. **γ_sg is not a universal constant.** It varies with storm intensity, wind-profile
   peakedness, radius, quadrant (max left-of-track, NH), and height (the jet at
   ~400–800 m; at 10 m the flow is SUB-gradient, ~0.77–0.87 — that's `v_reduc`,
   a separate parameter). Storm-to-storm observed range at the RMW: **~1.00–1.15**,
   with composite *means* near 1.0 and intense/peaked storms at the high end.
2. **For the W22 use-case specifically** — an intense storm at potential intensity
   with a peaked profile — the relevant end of the distribution is the high one:
   nonlinear BL theory gives 1.10–1.25 (1.22 for peaked/inertially-neutral cores),
   Vickery's surge-practice calibration implies ≈1.18, and observed v_max/PI ratios
   in a Cat-5 reached 1.25–1.33. **1.2 is therefore a defensible central value for
   a storm at its thermodynamic limit**, though it would overstate the jet in
   moderate, broad storms (where ~1.05 is more typical).
3. **Counter-evidence for ≈1.0 exists and is real** (Willoughby 1990; Georges;
   composite means) but applies mostly to moderate storms and the azimuthal mean —
   not to the peaked, intense limit the potential-height concept targets.
4. **Calibration caveat (the strongest single defense):** in W22, γ_sg = 1.2 was
   adopted as part of the parameter set {γ_sg, β_l, η, w_cool} calibrated against
   their axisymmetric simulations. Varying γ_sg alone (as our sensitivity does)
   breaks that internal calibration, so the −47% at sg = 1.0 is a *bounding*
   exercise, not an equally-likely alternative model.

## Recommended uncertainty statement

- **Central: γ_sg = 1.2** (W22 calibration; consistent with nonlinear BL theory for
  intense/peaked storms, Vickery-implied 1.18, and Isabel-class observations).
- **Honest interval for sensitivity reporting: 1.1–1.3**, mapping to potential
  heights of 4.7–9.2 m at the NO test point (−27% / +43%).
- **Documented floor: 1.0** (Willoughby/Georges/composite-mean class of evidence),
  mapping to 3.4 m (−47%) — report as a bound while noting it contradicts the
  intense-storm evidence class and breaks the W22 calibration.

## Draft limitations/rebuttal text

> The supergradient factor is the dominant parametric sensitivity of the potential
> height: varying γ_sg by ±0.1 about the Wang et al. (2022) calibration value of
> 1.2 changes the New Orleans potential height by −27%/+43% (and setting
> γ_sg = 1.0 lowers it by 47%). We retain γ_sg = 1.2 because (i) it is part of the
> internally calibrated W22 parameter set, and (ii) for storms at their intensity
> limit with peaked wind profiles — the regime the potential height targets —
> nonlinear boundary-layer theory (Kepert & Wang 2001: 10–25%, 22% for peaked
> cores), surge-practice calibrations (Vickery et al. 2009: implied ≈18%), and
> Category-5 observations (Bell & Montgomery 2008: v_max/PI ≈ 1.25–1.33) all
> support an enhancement of this magnitude, whereas near-balanced flow
> (Willoughby 1990; Kepert 2006a) is characteristic of weaker, broader storms.
> The quoted range 1.1–1.3 should be interpreted as the parametric uncertainty
> on the potential height itself.
