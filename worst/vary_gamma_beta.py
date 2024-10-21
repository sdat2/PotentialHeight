import os
from typing import List
import hydra
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import genextreme
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults, feature_grid
from .constants import CONFIG_PATH, FIGURE_PATH, DATA_PATH
from .tens import (
    seed_all,
    gen_data,
    fit_gev_upper_bound_not_known,
    fit_gev_upper_bound_known,
)
from .utils import alpha_from_z_star_beta_gamma

# @timeit
# def try_fit():
#    def fit():


def retry_wrapper(max_retries: int = 10) -> callable:
    """Retry wrapper.

    Args:
        max_retries (int, optional): Number of retries. Defaults to 10.

    Returns:
        callable: Wrapper function to recall the function in case of failure.
    """

    def retry_decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print("Exception: ", e)
                    print(f"Retrying {i+1}/{max_retries}")
            raise Exception("Max retries exceeded")

        return wrapper

    return retry_decorator


# @retry_wrapper(max_retries=10)
def try_fit(
    z_star: float,
    beta: float,
    gamma: float,
    ns: int,
    seed: int,
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> xr.DataArray:
    """Try fitting the data.

    Args:
        z_star (float): z_star.
        beta (float): beta.
        gamma (float): gamma.
        ns (int): Number of samples.
        seed (int): Seed.
        quantiles (List[float], optional): Quantiles. Defaults to [1/100, 1/500].

    Returns:
        xr.DataArray: DataArray with the return values.
    """
    seed_all(seed)
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_data(alpha, beta, gamma, ns)
    bg_alpha, bg_beta, bg_gamma = fit_gev_upper_bound_known(zs, z_star)
    s_alpha, s_beta, s_gamma = fit_gev_upper_bound_not_known(zs)
    true_q = genextreme.isf(quantiles, c=-gamma, loc=alpha, scale=beta)
    low_q = genextreme.isf(quantiles, c=-bg_gamma, loc=bg_alpha, scale=bg_beta)
    up_known_q = genextreme.isf(quantiles, c=-s_gamma, loc=s_alpha, scale=s_beta)
    results = np.array([true_q, low_q, up_known_q])
    return xr.Dataset(
        data_vars={
            "rv": (
                ("ns", "fit", "rp", "seed", "beta", "gamma"),
                np.expand_dims(
                    np.expand_dims(np.expand_dims(np.expand_dims(results, 0), -1), -1),
                    -1,
                ),
                {"units": "m"},
            )
        },
        coords={
            "ns": (("ns"), [ns], {"long_name": "Number of samples"}),
            "fit": (("fit"), ["true", "max_known", "max_unknown"]),
            "rp": (
                ("rp"),
                [int(1 / q) for q in quantiles],
                {"units": "years", "long_name": "Return period"},
            ),
            "seed": (
                ("seed"),
                [seed],
                {
                    "long_name": "Seed",
                    "units": "none",
                    "description": "Seed for the random number generator",
                },
            ),
            "gamma": (("gamma"), [gamma]),
            "beta": (("beta"), [beta]),
        },
    )["rv"]


def _name_base(config: DictConfig) -> str:
    """
    Name base.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        str: Unique name based on the configuration.
    """
    return f"z_star_{config.z_star:.2f}_ns_{config.ns}_Nr_{config.seed_steps_Nr}_gamma_{config.gamma_range.min:.2f}_{config.gamma_range.max:.2f}_{config.gamma_range.steps}_beta_{config.beta_range.min:.2f}_{config.beta_range.max:.2f}_{config.beta_range.steps}"


def get_fit_da(config: DictConfig) -> xr.DataArray:
    """Get fit xr.DataArray.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        xr.DataArray: DataArray with the fit data.
    """
    data_name = os.path.join(DATA_PATH, f"vary_{_name_base(config)}.nc")
    if not os.path.exists(data_name) or config.reload:
        quantiles = list(config.quantiles)
        betas = np.linspace(
            config.beta_range.min, config.beta_range.max, config.beta_range.steps
        )
        gammas = np.linspace(
            config.gamma_range.min, config.gamma_range.max, config.gamma_range.steps
        )
        seed_offsets = np.linspace(
            0, config.seed_steps_Nr, config.seed_steps_Nr, dtype=int
        )
        da = xr.concat(
            [
                xr.concat(
                    [
                        xr.concat(
                            [
                                try_fit(
                                    config.z_star,
                                    beta,
                                    gamma,
                                    config.ns,
                                    seed,
                                    quantiles,
                                )
                                for beta in betas
                            ],
                            dim="beta",
                        )
                        for gamma in gammas
                    ],
                    dim="gamma",
                )
                for seed in tqdm(seed_offsets)
            ],
            dim="seed",
        )
        da.to_netcdf(data_name)
    else:
        da = xr.open_dataarray(data_name)
    return da


def plot_fit_da(config: DictConfig, da: xr.DataArray) -> None:
    """Plot fit xr.DataArray.

    Args:
        config (DictConfig): Hydra config object.
        da (xr.DataArray): DataArray with the fit data.
    """
    figure_name = os.path.join(FIGURE_PATH, f"vary_{_name_base(config)}.pdf")
    means = da.mean("seed")
    std = da.std("seed")
    lower_p = da.quantile(config.figure.lp, dim="seed")
    upper_p = da.quantile(config.figure.up, dim="seed")
    range_p = upper_p - lower_p
    plot_defaults()
    ds = xr.merge(
        {
            "mean": means,
            "std": std,
            "lower_p": lower_p,
            "upper_p": upper_p,
            "range_p": range_p,
        }
    )
    feature_grid(
        ds,
        [
            ["mean", "std"],
            ["lower_p", "upper_p"],
        ],
        [["m", "m"], ["m", "m"]],
        [r"$\mu$", r"$\sigma$", r"$q_{0.05}$", r"$q_{0.95}$"],
        [[None, None], [None, None]],
        [None, None],
        xy=(
            (
                "beta",
                r"Scale, $\beta$",
                "m",
            ),
            ("gamma", r"Shape, $\gamma$", ""),
        ),
    )
    plt.savefig(figure_name)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_gamma_beta")
def run_vary_gamma_beta(config: DictConfig) -> None:
    """
    Run vary gamma beta experiment.

    Keep z_star constant and vary beta and gamma, and see how the fits change.

    Args:
        config (DictConfig): Hydra config object.
    """
    da = get_fit_da(config)
    plot_fit_da(config, da)


if __name__ == "__main__":
    # python -m worst.vary_gamma_beta
    run_vary_gamma_beta()
