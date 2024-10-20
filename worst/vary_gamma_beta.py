import os
from typing import List
import hydra
import numpy as np
import xarray as xr
from scipy.stats import genextreme
from omegaconf import DictConfig
from .constants import CONFIG_PATH
from .tens import (
    seed_all,
    gen_data,
    fit_gev_upper_bound_not_known,
    fit_gev_upper_bound_known,
)
from .utils import alpha_from_z_star_beta_gamma
from .constants import DATA_PATH

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
            ),
            "gamma": (("gamma"), [gamma]),
            "beta": (("beta"), [beta]),
        },
    )["rv"]


def make_fit_da(config: DictConfig) -> xr.DataArray:
    quantiles = list(config.quantiles)

    betas = np.linspace(
        config.beta_range.min, config.beta_range.max, config.beta_range.steps
    )
    gammas = np.linspace(
        config.gamma_range.min, config.gamma_range.max, config.gamma_range.steps
    )
    seed_offsets = np.linspace(0, config.seed_steps_Nr, config.seed_steps_Nr)
    da = xr.concat(
        [
            xr.concat(
                [
                    xr.concat(
                        [
                            try_fit(
                                config.z_star, beta, gamma, config.ns, seed, quantiles
                            )
                            for beta in betas
                        ],
                        dim="beta",
                    )
                    for gamma in gammas
                ],
                dim="gamma",
            )
            for seed in seed_offsets
        ],
        dim="seed",
    )
    da.to_netcdf(os.path.join(DATA_PATH, "vary_gamma_beta.nc"))
    return da


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_gamma_beta")
def run_vary_gamma_beta(config: DictConfig) -> None:
    """
    Run vary gamma beta experiment.

    Keep z_star constant and vary beta and gamma, and see how the fits change.

    Args:
        config (DictConfig): Hydra config object.
    """
    make_fit_da(config)


if __name__ == "__main__":
    # python -m worst.vary_gamma_beta
    run_vary_gamma_beta()
