"""Fit GEV using scipy and compare to known upper bound."""

from typing import Callable, Tuple, List
import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
import xarray as xr
import matplotlib.pyplot as plt
from sithom.plot import get_dim, label_subplots, plot_defaults
from sithom.time import timeit
from tcpips.constants import FIGURE_PATH
import os


def bg_cdf(z: np.ndarray, z_star: float, beta: float, gamma: float) -> np.ndarray:
    return np.exp(-((gamma / beta * (z - z_star)) ** (-1 / gamma)))


def bg_pdf(z: np.ndarray, z_star: float, beta: float, gamma: float) -> np.ndarray:
    return (
        1
        / beta
        * (gamma / beta * (z - z_star)) ** (-1 / gamma - 1)
        * np.exp(-((gamma / beta * (z - z_star)) ** (-1 / gamma)))
    )


def gev_pdf(z: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    return (
        1
        / beta
        * (1 + gamma * (z - alpha) / beta) ** (-1 / gamma - 1)
        * np.exp(-((1 + gamma * (z - alpha) / beta) ** (-1 / gamma)))
    )


# assume gamma < 0
def z_star_from_alpha_beta_gamma(alpha: float, beta: float, gamma: float) -> float:
    return alpha - beta / gamma


def alpha_from_z_star_beta_gamma(z_star: float, beta: float, gamma: float) -> float:
    return z_star + beta / gamma


def return_ll_beta_gamma(
    z: np.ndarray, z_star: float
) -> Callable[[float, float], float]:
    @np.vectorize
    def ll_beta_gamma(beta: float = 1, gamma: float = -0.5) -> float:
        return np.sum(np.log(bg_pdf(z, z_star, beta, gamma)))

    return ll_beta_gamma


def min_ll_bg(z: np.ndarray, z_star: float) -> Tuple[float, float]:
    mins = minimize(
        lambda x: -return_ll_beta_gamma(z, z_star)(x[0], x[1]),
        x0=[1, -0.5],
        bounds=[(0.01, None), (None, -0.01)],
    )
    return mins.x[0], mins.x[1]


def fit_gev(z: np.ndarray) -> Tuple[float, float, float]:
    c, mu, sigma = genextreme.fit(z)
    return mu, sigma, -c


def plot_rp(
    alpha: float, beta: float, gamma: float, color: str = "blue", label="", ax=None
):
    z1yr = genextreme.isf(0.8, c=-gamma, loc=alpha, scale=beta)
    z1myr = genextreme.isf(1 / 1_000_000, c=-gamma, loc=alpha, scale=beta)
    znew = np.linspace(z1yr, z1myr, num=100)

    print(z1yr, z1myr)
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
        plt.semilogx(rp, znew, color=color, label=label, alpha=0.5)
        plt.ylabel("Return Value [m]")
        plt.xlabel("Return Period [years]")
    else:
        ax.semilogx(rp, znew, color=color, label=label, alpha=0.5)
        ax.set_ylabel("Return Value [m]")
        ax.set_xlabel("Return Period [years]")


def gen_samples_from_gev(
    z_star: float, beta: float, gamma: float, n: int
) -> np.ndarray:
    # generate samples from the gev distribution
    return genextreme.rvs(
        loc=alpha_from_z_star_beta_gamma(z_star, beta, gamma),
        scale=beta,
        c=-gamma,
        size=n,
    )


def try_fit(
    z_star: float = 7,
    beta: float = 4,
    gamma: float = -0.1,
    n: int = 40,
    seed: int = 42,
    quantiles=[1 / 100, 1 / 200],
) -> None:
    np.random.seed(seed)
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_samples_from_gev(z_star, beta, gamma, n)
    bg_beta, bg_gamma = min_ll_bg(zs, z_star)
    bg_alpha = alpha_from_z_star_beta_gamma(z_star, bg_beta, bg_gamma)
    s_alpha, s_beta, s_gamma = fit_gev(zs)

    return (
        genextreme.isf(quantiles, c=-gamma, loc=alpha, scale=beta),
        genextreme.isf(quantiles, c=-bg_gamma, loc=bg_alpha, scale=bg_beta),
        genextreme.isf(quantiles, c=-s_gamma, loc=s_alpha, scale=s_beta),
    )


@timeit
def try_fits(
    z_star: float = 7,
    beta: float = 1,
    gamma: float = -0.3,
    seed: float = 100,
    nums: List[int] = [20, 22, 25, 27, 30, 33, 35, 40, 50, 60, 75, 100, 200, 500, 1000],
):
    results = []
    for n in nums:
        results.append(
            try_fit(
                z_star=z_star,
                beta=beta,
                gamma=gamma,
                n=int(n),
                quantiles=[1 / 100, 1 / 500],
                seed=seed,
            )
        )
    return xr.Dataset(
        data_vars={
            "rv": (
                ("number", "fit", "rp", "seed"),
                np.expand_dims(np.array(results), -1),
                {"units": "m"},
            )
        },
        coords={
            "number": nums,
            "fit": ["true", "max_known", "scipy"],
            "rp": (("rp"), [100, 500], {"units": "years"}),
            "seed": (("seed"), [seed]),
        },
    )["rv"]


@timeit
def fit_seeds(
    z_star: float,
    beta: float,
    gamma: float,
    seeds=np.linspace(0, 10000, num=1000).astype(int),
    nums=np.logspace(np.log(20), np.log(1000), num=50, dtype="int16"),
):
    print("nums", nums, type(nums))
    results = []
    for seed in seeds:
        results.append(
            try_fits(
                z_star=z_star,
                beta=beta,
                gamma=gamma,
                seed=seed,
                nums=nums,
            )
        )
    return xr.concat(results, dim="seed")


@timeit
def evt_fig_scipy(
    z_star: float = 7,
    beta: float = 1,
    gamma: float = -0.3,
    ex_seed: int = 42,
    ex_num: int = 50,
    min_samp: int = 20,
    max_samp: int = 1000,
    samp_steps: int = 50,
    seed_steps: int = 1000,
    save_fig_path: str = os.path.join(FIGURE_PATH, "evt_fig_scipy.pdf"),
    color_true: str = "black",
    color_max_known: str = "purple",
    color_max_unknown: str = "blue",
):

    plot_defaults()
    res_ds = fit_seeds(
        z_star,
        beta,
        gamma,
        seeds=np.linspace(0, 10000, num=seed_steps).astype(int),
        nums=np.logspace(
            np.log(min_samp), np.log(max_samp), num=samp_steps, base=np.e, dtype="int32"
        ),
    )
    mn = res_ds.mean(dim="seed")
    std = res_ds.std(dim="seed")
    _, axs = plt.subplots(
        3, 1, height_ratios=[2, 1, 1], figsize=get_dim(ratio=0.6180339887498949 * 2)
    )
    np.random.seed(ex_seed)
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_samples_from_gev(z_star, beta, gamma, ex_num)
    bg_beta, bg_gamma = min_ll_bg(zs, z_star)
    bg_alpha = alpha_from_z_star_beta_gamma(z_star, bg_beta, bg_gamma)
    s_alpha, s_beta, s_gamma = fit_gev(zs)
    plot_rp(alpha, beta, gamma, color=color_true, label="Original GEV", ax=axs[0])
    sorted_zs = np.sort(zs)
    empirical_rps = len(zs) / np.arange(1, len(zs) + 1)[::-1]
    axs[0].scatter(
        empirical_rps,
        sorted_zs,
        s=3,
        alpha=0.8,
        color=color_true,
        label="Sampled data points",
    )
    plot_rp(
        bg_alpha,
        bg_beta,
        bg_gamma,
        color=color_max_known,
        label="Known upper bound GEV fit",
        ax=axs[0],
    )
    plot_rp(
        s_alpha,
        s_beta,
        s_gamma,
        color=color_max_unknown,
        label="Unbounded GEV fit",
        ax=axs[0],
    )

    axs[0].legend()
    axs[0].set_xlim([0.6, 1e6])

    numbers = res_ds.number.values

    axs[1].fill_between(
        numbers,
        mn.isel(rp=0, fit=0).values - std.isel(rp=0, fit=0).values,
        mn.isel(rp=0, fit=0).values + std.isel(rp=0, fit=0).values,
        alpha=0.2,
        color=color_true,
    )
    axs[2].fill_between(
        numbers,
        mn.isel(rp=1, fit=0).values - std.isel(rp=1, fit=0).values,
        mn.isel(rp=1, fit=0).values + std.isel(rp=1, fit=0).values,
        alpha=0.2,
        color=color_true,
        linestyle="-",
    )
    axs[1].fill_between(
        numbers,
        mn.isel(rp=0, fit=1).values - std.isel(rp=0, fit=1).values,
        mn.isel(rp=0, fit=1).values + std.isel(rp=0, fit=1).values,
        alpha=0.2,
        color=color_max_known,
        linestyle="-",
    )

    axs[2].fill_between(
        numbers,
        mn.isel(rp=1, fit=1).values - std.isel(rp=1, fit=1).values,
        mn.isel(rp=1, fit=1).values + std.isel(rp=1, fit=1).values,
        alpha=0.2,
        color=color_max_known,
        linestyle="-",
    )
    axs[1].fill_between(
        numbers,
        mn.isel(rp=0, fit=2).values - std.isel(rp=0, fit=2).values,
        mn.isel(rp=0, fit=2).values + std.isel(rp=0, fit=2).values,
        alpha=0.2,
        color=color_max_unknown,
        linestyle="-",
    )
    axs[2].fill_between(
        numbers,
        mn.isel(rp=1, fit=2).values - std.isel(rp=1, fit=2).values,
        mn.isel(rp=1, fit=2).values + std.isel(rp=1, fit=2).values,
        alpha=0.2,
        color=color_max_unknown,
        linestyle="-",
    )

    axs[1].plot(numbers, res_ds.isel(rp=0, fit=0).values, color=color_true, linewidth=1)
    axs[2].plot(
        numbers,
        mn.isel(rp=1, fit=0).values,
        color=color_true,
        linewidth=1,
        label="True GEV",
    )
    axs[1].plot(
        numbers, mn.isel(rp=0, fit=1).values, color=color_max_known, linewidth=1
    )
    axs[2].plot(
        numbers,
        mn.isel(rp=1, fit=1).values,
        color=color_max_known,
        linewidth=1,
        label="Max known GEV",
    )
    axs[1].plot(
        numbers, mn.isel(rp=0, fit=2).values, color=color_max_unknown, linewidth=1
    )
    axs[2].plot(
        numbers,
        mn.isel(rp=1, fit=2).values,
        color=color_max_unknown,
        linewidth=1,
        label="Unbounded GEV",
    )
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("Number of samples")
    axs[1].set_ylabel("1 in 100 year RV [m]")
    axs[2].set_ylabel("1 in 500 year RV [m]")
    axs[1].set_xlim([min_samp, max_samp])
    axs[2].set_xlim([min_samp, max_samp])
    label_subplots(axs)
    # plt.legend()
    # plt.show()
    # plt.clf()
    plt.savefig(save_fig_path)
    plt.clf()


# plot_vals(mn)
if __name__ == "__main__":
    # python -m worst.sci
    evt_fig_scipy()
