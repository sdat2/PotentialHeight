"""Worst.utils.py"""

import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt


# assume gamma < 0 (Weibull class)
def z_star_from_alpha_beta_gamma(alpha: float, beta: float, gamma: float) -> float:
    return alpha - beta / gamma


def alpha_from_z_star_beta_gamma(z_star: float, beta: float, gamma: float) -> float:
    return z_star + beta / gamma


def bg_cdf(z: np.ndarray, z_star: float, beta: float, gamma: float) -> np.ndarray:
    return np.exp(-((gamma / beta * (z - z_star)) ** (-1 / gamma)))


def plot_rp(
    alpha: float,
    beta: float,
    gamma: float,
    color: str = "blue",
    label: str = "",
    ax=None,
    plot_alpha: float = 0.8,
):
    z1yr = genextreme.isf(0.8, c=-gamma, loc=alpha, scale=beta)
    z1myr = genextreme.isf(1 / 1_000_000, c=-gamma, loc=alpha, scale=beta)
    znew = np.linspace(z1yr, z1myr, num=100)

    print(label, "rp1yr", z1yr, "rv1Myr", z1myr)
    if gamma < 0:  # Weibull class have upper bound
        z_star = z_star_from_alpha_beta_gamma(alpha, beta, gamma)
        if ax is None:
            plt.hlines(z_star, 2, 1_000_000, color=color, linestyles="dashed")
        else:
            ax.hlines(z_star, 2, 1_000_000, color=color, linestyles="dashed")
        rp = 1 / (1 - bg_cdf(znew, z_star, beta, gamma))
    else:
        rp = 1 / genextreme.sf(znew, c=-gamma, loc=alpha, scale=beta)
    if ax is None:
        plt.semilogx(rp, znew, color=color, label=label, alpha=plot_alpha)
        plt.ylabel("Return Value [m]")
        plt.xlabel("Return Period [years]")
    else:
        ax.semilogx(rp, znew, color=color, label=label, alpha=plot_alpha)
        ax.set_ylabel("Return Value [m]")
        ax.set_xlabel("Return Period [years]")
