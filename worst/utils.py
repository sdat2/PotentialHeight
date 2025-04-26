"""Worst.utils.py"""

from typing import Optional
import numpy as np
from scipy.stats import genextreme
import matplotlib
import matplotlib.pyplot as plt


# assume gamma < 0 (Weibull class)
def z_star_from_alpha_beta_gamma(alpha: float, beta: float, gamma: float) -> float:
    """
    Calculate the upper bound of the Weibull class GEV distribution.

    Args:
        alpha (float): Location parameter of the GEV distribution.
        beta (float): Scale parameter of the GEV distribution (should be positive).
        gamma (float): Shape parameter of the GEV distribution (should be negative).

    Returns:
        float: The upper bound of the GEV distribution, z_star.
    """
    return alpha - beta / gamma


def alpha_from_z_star_beta_gamma(z_star: float, beta: float, gamma: float) -> float:
    """
    Calculate the location parameter of the Weibull class GEV distribution.

    Args:
        z_star (float): Upper bound of the GEV distribution.
        beta (float): Scale parameter of the GEV distribution (should be positive).
        gamma (float): Shape parameter of the GEV distribution (should be negative).

    Returns:
        float: The location parameter of the GEV distribution, alpha.
    """
    return z_star + beta / gamma


def bg_cdf(z: np.ndarray, z_star: float, beta: float, gamma: float) -> np.ndarray:
    """
    CDF of the Weibull class GEV distribution with a known upper bound.

    Args:
        z (np.ndarray): Data points to find the CDF for.
        z_star (float): Upper bound of the GEV distribution.
        beta (float): Scale parameter of the GEV distribution.
        gamma (float): Shape parameter of the GEV distribution.

    Returns:
        np.ndarray: The CDF of the GEV distribution for different z values.
    """
    return np.exp(-((gamma / beta * (z - z_star)) ** (-1 / gamma)))


def plot_rp(
    alpha: float,
    beta: float,
    gamma: float,
    color: str = "blue",
    label: str = "",
    ax: Optional[matplotlib.axes.Axes] = None,
    plot_alpha: float = 0.8,
) -> None:
    """
    Plot the return period vs return value curve for the given parameters.

    Args:
        alpha (float): The location parameter.
        beta (float): The scale parameter.
        gamma (float): The shape parameter.
        color (str, optional): The color of the line. Defaults to "blue".
        label (str, optional): The line label. Defaults to "".
        ax (Optional[matplotlib.axes.Axes], optional): The axes to add the figure too. Defaults to None.
        plot_alpha (float, optional): Transparency parameter of line. Defaults to 0.8.
    """
    z1yr = genextreme.isf(0.8, c=-gamma, loc=alpha, scale=beta)
    z1myr = genextreme.isf(1 / 1_000_000, c=-gamma, loc=alpha, scale=beta)
    znew = np.linspace(z1yr, z1myr, num=100)

    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
        )

    print(label, "rp1yr", z1yr, "rv1Myr", z1myr)
    if gamma < 0:  # Weibull class have upper bound
        z_star = z_star_from_alpha_beta_gamma(alpha, beta, gamma)
        ax.hlines(z_star, 0.3, 1_000_000, color=color, linestyles="dashed")
        rp = 1 / (1 - bg_cdf(znew, z_star, beta, gamma))
    else:
        rp = 1 / genextreme.sf(znew, c=-gamma, loc=alpha, scale=beta)

    ax.semilogx(rp, znew, color=color, label=label, alpha=plot_alpha)
    ax.set_ylabel("Return Value [m]")
    ax.set_xlabel("Return Period [years]")
    ax.set_xlim(0.3, 1_000_000)


def plot_sample_points(
    data: np.ndarray,
    color: str = "black",
    ax: Optional[matplotlib.axes.Axes] = None,
    label: str = "Sampled data points",
) -> None:
    """
    Plot the sample points on the return period vs return value curve.

    Args:
        data (np.ndarray): The sample points to plot.
        color (str, optional): The color of the points. Defaults to "black".
        ax (Optional[matplotlib.axes.Axes], optional): The axes to add the figure too. Defaults to None.
        label (str, optional): The label of the points. Defaults to "Sampled data points".

    """
    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
        )
    sorted_zs = np.sort(data)
    empirical_rps = len(data) / np.arange(1, len(data) + 1)[::-1]

    ax.scatter(
        empirical_rps,
        sorted_zs,
        s=3,
        alpha=0.8,
        color=color,
        label=label,
    )
    ax.set_ylabel("Return Value (RV) $v$ [m]")
    ax.set_xlabel("Return Period (RP) $p$ [years]")


def retry_wrapper(max_retries: int = 10) -> callable:
    """Retry wrapper. If a function fails flakeily, then wrap with this to retry it a few times.

    Puts the function in a try except block and retries it a `max_retries` times if unsuccesful.

    Args:
        max_retries (int, optional): Number of retries. Defaults to 10.

    Returns:
        callable: Wrapper function to recall the function in case of failure.

    Example::
        >>> @retry_wrapper(max_retries=5)
        ... def my_function():
        ...     # Some code that will fail
        ...     assert False
        >>> try:
        ...    my_function()
        ... except Exception as e:
        ...    print(e)
            Exception:
            Retrying my_function 1/5
            Exception:
            Retrying my_function 2/5
            Exception:
            Retrying my_function 3/5
            Exception:
            Retrying my_function 4/5
            Exception:
            Retrying my_function 5/5
            Max retries exceeded for function my_function

    """

    def retry_decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs) -> any:
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print("Exception: ", e)
                    print(f"Retrying {func.__name__} {i+1}/{max_retries}")
            raise Exception(f"Max retries exceeded for function {func.__name__}")

        return wrapper

    return retry_decorator
