"""Distributions to fit and compare."""

import scipy.special as sc
import numpy as np
from scipy.stats._continuous_distns import rv_continuous


class genextreme_given_max_gen(rv_continuous):
    """Generalized extreme value distribution given maximum value.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html

    https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_continuous_distns.py#L3086-L3232

    """

    # none of these functions have been thought throughs
    def _argcheck(self, c: float):
        return c > 0 and np.isfinite(c)

    def _pdf(self, x, c):
        return c * (1 + c * x) ** (-1 - 1 / c)

    def _cdf(self, x, c):
        return (1 + c * x) ** (-1 / c)

    def _ppf(self, q, c):
        return (q**c - 1) / c

    def _stats(self, c):
        return (
            c * np.pi / np.sin(np.pi * c),
            c * np.pi**2 / np.sin(np.pi * c) ** 2 / 6,
            c * np.pi**2 / np.sin(np.pi * c) ** 2 / 6 - 0.57721566490153286060651209,
        )


genextreme_given_max = genextreme_given_max_gen(name="genextreme_given_max")


def genpareto_given_max_gen(rv_continuous):
    """Generalized Pareto distribution given maximum value.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
    https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_continuous_distns.py#L2896-L3013

    """

    def _argcheck(self, c):
        return c > 0 and np.isfinite(c)

    def _pdf(self, x, c):
        return c / x ** (c + 1)


genpareto_given_max = genpareto_given_max_gen(name="genpareto_given_max")
