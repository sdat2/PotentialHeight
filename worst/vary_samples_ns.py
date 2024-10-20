"""Make figure showing the effect of the number of samples on the EVT fits."""

from typing import List, Tuple, Optional
import os
import numpy as np
import xarray as xr
import hydra
from omegaconf import DictConfig
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from tcpips.constants import DATA_PATH, FIGURE_PATH
from tqdm import tqdm
from sithom.plot import plot_defaults, label_subplots, get_dim
from sithom.time import timeit
from .utils import alpha_from_z_star_beta_gamma, plot_rp, plot_sample_points
from .tens import (
    seed_all,
    gen_data,
    fit_gev_upper_bound_not_known,
    fit_gev_upper_bound_known,
    tfd,
)
from .constants import CONFIG_PATH


def try_fit(
    z_star: float = 7,
    beta: float = 4,
    gamma: float = -0.1,
    ns: int = 40,
    seed: int = 42,
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try fitting the GEV distributions to some data, and calculate the return values for some quantiles.

    Args:
        z_star (float, optional): Upper bound of GEV. Defaults to 7.
        beta (float, optional): Scale. Defaults to 4.
        gamma (float, optional): Concentration. Defaults to -0.1.
        ns (int, optional): Number of data samples. Defaults to 40.
        seed (int, optional): Random seed. Defaults to 42.
        quantiles (List[float], optional): Quantiles to get. Defaults to [1 / 100, 1 / 500].

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Return values for the quantiles for the true GEV,
            the GEV with the known upper bound, and the GEV with no upper bound.
    """
    seed_all(seed)
    print(
        "Fitting for z_star",
        z_star,
        "beta",
        beta,
        "gamma",
        gamma,
        "ns",
        ns,
        "seed",
        seed,
    )
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_data(alpha, beta, gamma, ns)
    bg_alpha, bg_beta, bg_gamma = fit_gev_upper_bound_known(zs, z_star)
    s_alpha, s_beta, s_gamma = fit_gev_upper_bound_not_known(zs)

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
) -> xr.DataArray:
    """
    Try fitting the GEV distributions to some data, and calculate the return values for some quantiles.

    Args:
        z_star (float, optional): Upper bound. Defaults to 7.
        beta (float, optional): Scale. Defaults to 1.
        gamma (float, optional): Shape. Defaults to -0.3.
        seed (float, optional): Seed. Defaults to 100.
        nums (List[int], optional): Number of samples. Defaults to [20, 22, 25, 27, 30, 33, 35, 40, 50, 60, 75, 100, 200, 500, 1000].

    Returns:
        xr.DataArray: Return values for the quantiles for the true GEV,
    """
    results = []

    for ns in nums:

        def _retry(max_retries: int = 10) -> None:
            # sometimes the optimization randomly fails, so we retry
            try:
                return try_fit(
                    z_star=z_star,
                    beta=beta,
                    gamma=gamma,
                    ns=int(ns),
                    quantiles=[1 / 100, 1 / 500],
                    seed=seed,
                )
            except Exception as e:
                print("Exception", e)
                print("Failed for ns", ns)
                print("Seed", seed)
                print("z_star", z_star)
                print("beta", beta)
                print("gamma", gamma)
                if max_retries == 0:
                    raise e
                return _retry(max_retries - 1)

        results.append(_retry())

    return xr.Dataset(
        data_vars={
            "rv": (
                ("ns", "fit", "rp", "seed"),
                np.expand_dims(np.array(results), -1),
                {"units": "m"},
            )
        },
        coords={
            "ns": nums,
            "fit": ["true", "max_known", "scipy"],
            "rp": (("rp"), [100, 500], {"units": "years"}),
            "seed": (("seed"), [seed]),
        },
    )["rv"]


@timeit
def fit_for_seeds(
    z_star: float,
    beta: float,
    gamma: float,
    seeds=np.linspace(0, 10000, num=1000, dtype="int16"),
    nums=np.logspace(np.log(20), np.log(1000), num=50, dtype="int16"),
) -> xr.DataArray:
    """
    Fit for different seeds.

    Args:
        z_star (float): Upper bound.
        beta (float): Scale parameter.
        gamma (float): Shape parameter.
        seeds (np.ndarray): Seeds.
        nums (np.ndarray): Number of samples.

    Returns:
        xr.DataArray: Return values for the quantiles for the true GEV,
        the GEV with the known upper bound, and the GEV with no upper bound.
    """
    print("nums", nums, type(nums))
    results = []
    for seed in tqdm(
        seeds,
        desc="Seeds",
        total=len(seeds),
    ):
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


def get_evt_fit_data(
    z_star: float,
    beta: float,
    gamma: float,
    seeds: np.ndarray,
    nums: np.ndarray,
    load: bool = True,
) -> xr.DataArray:
    """
    Either load or calculate the EVT fit 1 in 100 year and 1 in 500 year events.

    Args:
        z_star (float): Upper bound.
        beta (float): Scale parameter.
        gamma (float): Shape parameter.
        seeds (np.ndarray): Seeds.
        nums (np.ndarray): Number of samples.
        load (bool, optional): Load or calculate. Defaults to True.

    Returns:
        xr.DataArray: Return values for the quantiles for the true GEV,
        the GEV with the known upper bound, and the GEV with no upper bound.
    """
    data_name = os.path.join(
        DATA_PATH,
        f"evt_fig_tens_{z_star:.2f}_{beta:.2f}_{gamma:.2f}_{len(nums)}_{len(seeds)}.nc",
    )
    print(f"data_name = {data_name}")
    if load and os.path.exists(data_name):
        return xr.open_dataarray(data_name)
    else:
        data = fit_for_seeds(z_star, beta, gamma, seeds, nums)
        data.to_netcdf(data_name)
        return data


def plot_ex(
    z_star: float,
    beta: float,
    gamma: float,
    ex_seed: int,
    ex_num: int,
    color_true: str,
    color_max_known: str,
    color_max_unknown: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    fig_path: str = os.path.join(FIGURE_PATH, "evt_fit_ex_tens.pdf"),
) -> None:
    """
    Plot single example fit.

    Args:
        z_star (float): Upper bound.
        beta (float): Scale parameter.
        gamma (float): Shape parameter.
        ex_seed (int): Example seed.
        ex_num (int): Example number of samples.
        color_true (str): Color of original GEV.
        color_max_known (str): Color of known upper bound GEV.
        color_max_unknown (str): Color of unknown upper bound GEV.
        ax (_type_, optional): Axes. Defaults to None.
        fig_path (str, optional): Figure path to save to. Defaults to os.path.join(FIGURE_PATH, "evt_fit_ex_tens.pdf").
    """
    plot_individually = False
    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
        )
        plot_individually = True
    # plot an example fit
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    np.random.seed(ex_seed)
    zs = gen_data(alpha, beta, gamma, ex_num)
    ## fit the known upper bound and the unbounded case
    bg_alpha, bg_beta, bg_gamma = fit_gev_upper_bound_known(zs, z_star)
    bg_alpha = alpha_from_z_star_beta_gamma(z_star, bg_beta, bg_gamma)
    s_alpha, s_beta, s_gamma = fit_gev_upper_bound_not_known(zs)
    # plot the original data
    plot_rp(alpha, beta, gamma, color=color_true, label="Original GEV", ax=ax)
    plot_sample_points(zs, color="black", ax=ax, label="Samples")

    plot_rp(
        bg_alpha,
        bg_beta,
        bg_gamma,
        color=color_max_known,
        label="I: Known upper bound GEV fit",
        ax=ax,
    )
    plot_rp(
        s_alpha,
        s_beta,
        s_gamma,
        color=color_max_unknown,
        label="II: Unbounded GEV fit",
        ax=ax,
    )

    ax.legend()
    ax.set_xlim([0.6, 1e6])  # up to 1 in 1million year return period
    if plot_individually:
        plt.savefig(fig_path)


@timeit
def evt_fig_tens(
    z_star: float = 7,  # m
    beta: float = 1,  # m
    gamma: float = -0.2,  # [dimensionless]
    ex_seed: int = 57,
    ex_num: int = 50,
    min_samp: int = 20,
    max_samp: int = 1000,
    samp_steps: int = 26,
    seed_steps: int = 600,
    color_true: str = "black",
    color_max_known: str = "#1b9e77",
    color_max_unknown: str = "#d95f02",
    load: bool = True,
) -> None:
    """
    Plot the EVT fits for the known upper bound and the unbounded case.

    Args:
        z_star (float, optional): Upper bound. Defaults to 7m.
        beta (float, optional): Scale parameter. Defaults to 1m.
        gamma (float, optional): Shape parameter. Defaults to -0.2.
        ex_seed (int, optional): Example seed. Defaults to 57.
        ex_num (int, optional): Example number of samples. Defaults to 50.
        min_samp (int, optional): Minimum number of samples. Defaults to 20.
        max_samp (int, optional): Maximum number of samples. Defaults to 1000.
        samp_steps (int, optional): Number of different samples. Defaults to 500.
        seed_steps (int, optional): Number of different seeds. Defaults to 20.
        color_true (str, optional): Color or original GEV. Defaults to "black".
        color_max_known (str, optional): Color of GEV with max known. Defaults to "#1b9e77".
        color_max_unknown (str, optional): Color of GEV with unknown max. Defaults to "#d95f02".
        load (bool, optional): reload if data exists or recalculate. Defaults to True for reload.
    """
    save_fig_path: str = os.path.join(
        FIGURE_PATH,
        f"evt_fig_tens_{z_star:.2f}_{beta:.2f}_{gamma:.2f}_{samp_steps}_{seed_steps}.pdf",
    )
    print(f"save_fig_path = {save_fig_path}")

    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)

    plot_defaults()
    res_ds = get_evt_fit_data(
        z_star,
        beta,
        gamma,
        seeds=np.linspace(0, 10000, num=seed_steps, dtype="int16"),
        nums=np.logspace(
            np.log(min_samp), np.log(max_samp), num=samp_steps, base=np.e, dtype="int16"
        ),
        load=load,
    )

    # setup figure
    _, axs = plt.subplots(
        3, 1, height_ratios=[2, 1, 1], figsize=get_dim(ratio=0.6180339887498949 * 2)
    )
    # plot example fit
    plot_ex(
        z_star,
        beta,
        gamma,
        ex_seed,
        ex_num,
        color_true,
        color_max_known,
        color_max_unknown,
        ax=axs[0],
    )

    # plot the systematic fits for the known upper bound and the unbounded case

    numbers = res_ds.ns.values

    # calculate statistics to work out sampling error
    mn = res_ds.mean(dim="seed")
    std = res_ds.std(dim="seed")

    for lp, up in [(0.05, 0.95)]:  # , (0.25, 0.75)]:
        if seed_steps > 10:
            # get 5th and 95th percentiles
            lower_p = res_ds.quantile(lp, dim="seed")
            upper_p = res_ds.quantile(up, dim="seed")
            range_p = upper_p - lower_p
            print("for", lp, "to", up)
            print("ranges at 50 samples", range_p.sel(ns=50, method="nearest"))
            ratio_change = range_p.isel(fit=1) / range_p.isel(fit=2)
            print(
                "knowing maxima makes difference",
                ratio_change.sel(ns=50, method="nearest"),
            )
        else:
            lower_p = mn - std
            upper_p = mn + std

        for rp in [0, 1]:
            for fit, color in [
                (0, color_true),
                (1, color_max_known),
                (2, color_max_unknown),
            ]:
                axs[rp + 1].fill_between(
                    numbers,
                    lower_p.isel(rp=rp, fit=fit).values,
                    upper_p.isel(rp=rp, fit=fit).values,
                    alpha=0.2,
                    color=color,
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
    axs[2].set_xlabel("Number of samples, $N_s$")
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
    print(("alpha", alpha, "beta", beta, "gamma", gamma, "z_star", z_star))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_ns")
def vary_samples_ns(config: DictConfig) -> None:
    print(config)
    evt_fig_tens(
        z_star=config.z_star,
        beta=config.beta,
        gamma=config.gamma,
        ex_seed=config.ex_seed,
        ex_num=config.ex_num,
        min_samp=config.min_samp,
        max_samp=config.max_samp,
        samp_steps=config.samp_steps,
        seed_steps=config.seed_steps,
        color_true=config.color_true,
        color_max_known=config.color_max_known,
        color_max_unknown=config.color_max_unknown,
        load=True,
    )


if __name__ == "__main__":
    vary_samples_ns()
