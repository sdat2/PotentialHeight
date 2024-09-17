"""Solve module for numerical methods."""

import warnings
from typing import Callable
import numpy as np


def bisection(f: Callable, left: float, right: float, tol: float) -> float:
    """
    Bisection numerical method.

    https://en.wikipedia.org/wiki/Root-finding_algorithms#Bisection_method

    Args:
        f (Callable): Function to find root of.
        left (float): Left boundary.
        right (float): Right boundary.
        tol (float): tolerance for convergence.

    Returns:
        float: x such that |f(x)| < tol.

    Example::
        >>> f = lambda x: x - 2
        >>> np.isclose(bisection(f, 1, 3, 1e-6), 2, rtol=1e-2, atol=1e-2)
        True
        >>> f = lambda x: x**2 - 4
        >>> np.isclose(bisection(f, 0, 3, 1e-3), 2, rtol=1e-3, atol=1e-3)
        True
    """
    fleft = f(left)
    fright = f(right)
    # print(fleft, fright)

    if fleft * fright > 0:
        # print("Error: f(left) and f(right) must have opposite signs.")
        warnings.warn("Warning: f(left) and f(right) must have opposite signs.")
        return np.nan

    while fleft * fright < 0 and right - left > tol:
        mid = (left + right) / 2
        fmid = f(mid)
        if fmid == 0:
            return mid
        # print(left, mid, right)
        # print(fleft, fmid, fright)
        if fleft * fmid < 0:
            right = mid
            fright = fmid
        else:
            left = mid
            fleft = fmid
    return (left + right) / 2
