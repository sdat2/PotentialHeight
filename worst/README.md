# Worst

Fit EVT distributions knowing (or not-knowing) the worst-case ahead of time.

- sci - scipy optimization.
- tens - tensorflow optimization.
- utils - shared functions for plotting etc.

## Parameters of GEV

- alpha -- loc parameter
- beta -- scale parameter (>0)
- gamma -- shape parameter (<0 if Weibull)

With this sign convention the upper bound of the Weibull distribution ends up being:

```text
z_star = alpha - beta / gamma
```

## Task

We seek to minimize the negative log-likelihood (equivalent to maximising the likelihood) in the case that we know the upper bound ahead of time and in the case we don't.

For our chosen set of parameters, we have shown large reductions in sampling uncertainty and bias from knowing the upper bound.

However:

 - We assume that we perfectly know the upper bound ahead of time (not considering uncertainty in upper bound, and that it changes with time of year).
 - We assume that the true distribution is exactly a GEV to start with.
    - Real observations may be from a mixture of different physical processes and thus not satisfy the i.i.d assumptions made when assuming the GEV.
- Further, physically reducing the uncertainty given some number of observations may be useful, but those observations are themselves nonstationary due to climate change, how should we take this into account.
- Potential size and intensity are themselves random variables, meaning that potential height is also a random variable. How could we get to potential height much more quickly?

## run using tensorflow optimizer

```bash
python -m worst.vary_gamma_beta

python -m worst.vary_ns

python -m worst.vary_noise

python -m worst.vary_nonstationary
```

## Non-stationary synthetic EVT experiment

`worst.vary_nonstationary` is a general statistical simulation for the rebuttal point
that observations are non-stationary under climate change. It generates annual
block maxima from a bounded GEV where the true upper bound evolves linearly,
then compares four fits:

1. `stationary_unbounded`
2. `stationary_bounded`
3. `nonstationary_unbounded`
4. `nonstationary_bounded`

The script writes:

- data to `data/worst/vary_nonstationary_*.nc`
- figure to `img/worst/vary_nonstationary_*.pdf`

You can tune parameters in `worst/config/vary_nonstationary.yaml`, especially
`z_star_trend`, `n_years`, and `z_star_assumed_sigma`.

Caching and replot workflow:

- Recompute simulations and redraw figure:

```bash
python -m worst.vary_nonstationary use_cache=false
```

- Replot only from cached data (no simulation rerun):

```bash
python -m worst.vary_nonstationary plot_only=true use_cache=true
```

```text
alpha0 = 0.0 m
beta0 = 1.0 m
gamma0 = -0.1 [dimensionless]

beta > 0
gamma < 0
```