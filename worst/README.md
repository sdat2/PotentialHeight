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

z_star = alpha - beta / gamma

## Task

We seek to minimize the negative log-likelihood (equivalent to maximising the likelihood) in the case that we know the upper bound ahead of time and in the case we don't.

For our chosen set of parameters we have shown large reductions in sampling uncertainty and bias from knowing the upper bound.

However:
 - We assume that we perfectly know the upper bound ahead of time (not considering uncertainty in upper bound, and that it changes with time of year).
 - We assume that the true distribution is exactly a GEV to start with.
    - Real observations may be from a mixture of different physical processes and thus not satisfy the i.i.d assumptions made when assuming the GEV.
- Further, phyically reducing the uncertainty given some number of observations may be useful, but those observations are themselves nonstationary due to climate change, how should we take this into account.
- Potential size and intensity are themselves random variables, meaning that potential height is also a random variable. How could we get to potential height much more quickly?
