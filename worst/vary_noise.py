"""Vary noise of the upper bound, see how big the noise has to get before its no longer valuable."""

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
from .utils import (
    alpha_from_z_star_beta_gamma,
    plot_rp,
    plot_sample_points,
    retry_wrapper,
)
from .tens import (
    seed_all,
    gen_data,
    fit_gev_upper_bound_not_known,
    fit_gev_upper_bound_known,
)
from .constants import CONFIG_PATH


@retry_wrapper(max_retries=10)
def try_fit(
    z_star_mu: float,
    z_star_sigma: float,
    beta: float,
    gamma: float,
    ns: int,
    seed: int,
    config: DictConfig,
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> xr.DataArray:
    """Try fitting the data for the upperbound known with error.

    Presumably at higher noises its better to not know the upper bound that it is to know it.

    Args:
        z_star_mu (float): z_star mean (true value).
        z_star_sigma (float): z_star noise Gaussian std. dev.
        beta (float): beta.
        gamma (float): gamma.
        ns (int): Number of samples.
        seed (int): Seed.
        quantiles (List[float], optional): Quantiles. Defaults to [1/100, 1/500].

    Returns:
        xr.DataArray: DataArray with the return values.
    """
    seed_all(seed)
    alpha = alpha_from_z_star_beta_gamma(z_star_mu, beta, gamma)
    zs = gen_data(alpha, beta, gamma, ns)
    fit = config.fit
    z_star_with_noise = float(np.random.normal(z_star_mu, z_star_sigma))
    z_star_assumed = max(
        z_star_with_noise, np.max(zs) + 0.005
    )  # if we see a larger z in the data, default to that as the upper bound, otherwise fit impossible
    bg_alpha, bg_beta, bg_gamma = fit_gev_upper_bound_known(
        zs,
        z_star_assumed,
        opt_steps=fit.steps,
        lr=fit.lr,
        beta_guess=fit.beta_guess,
        gamma_guess=fit.gamma_guess,
        verbose=config.verbose,
    )
    ubk_q = genextreme.isf(quantiles, c=-bg_gamma, loc=bg_alpha, scale=bg_beta)
    results = np.array(ubk_q)
    return xr.Dataset(
        data_vars={
            "rv": (
                ("ns", "rp", "seed", "beta", "gamma", "z_star_sigma"),
                np.expand_dims(
                    np.expand_dims(
                        np.expand_dims(
                            np.expand_dims(np.expand_dims(results, 0), -1)  # seed
                            - 1  # beta,
                        ),
                        -1,  # gamma
                    ),
                    -1,  # z_star_sigma
                ),
                {"units": "m"},
            )
        },
        coords={
            "ns": (("ns"), [ns], {"long_name": "Number of samples"}),
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
            "z_star_sigma": (("z_star_sigma"), [z_star_sigma]),
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
    return f"z_star_{config.z_star:.2f}_ns_{config.ns}_Nr_{config.seed_steps_Nr}_gamma_{config.gamma:.2f}_beta_{config.beta:.2f}_z_star_sigma_{config.z_star_sigma.min:.2f}_{config.z_star_sigma.max:.2f}_{config.z_star_sigma.steps}"


def get_fit_da(config: DictConfig) -> xr.DataArray:
    """Get fit xr.DataArray.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        xr.DataArray: DataArray with the fit data.
    """
    data_name = os.path.join(DATA_PATH, f"vary_{_name_base(config)}.nc")
    if not os.path.exists(data_name) or not config.reload:
        quantiles = list(config.quantiles)
        seed_offsets = np.linspace(
            0, config.seed_steps_Nr, config.seed_steps_Nr, dtype=int
        )
        sigmas = np.linspace(
            config.z_star_sigma.min, config.z_star_sigma.max, config.z_star_sigma.steps
        )
        da = xr.concat(
            [
                xr.concat(
                    [
                        try_fit(
                            config.z_star,
                            z_star_sigma,
                            config.beta,
                            config.gamma,
                            config.ns,
                            seed,
                            config,
                            quantiles,
                        )
                        for z_star_sigma in sigmas
                    ],
                    dim="z_star_sigma",
                )
                for seed in tqdm(seed_offsets)
            ],
            dim="seed",
        )
        da.to_netcdf(data_name)
    else:
        da = xr.open_dataarray(data_name)
    return da


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_noise")
def run_vary_noise(config: DictConfig) -> None:
    """
    Run vary gamma beta experiment.

    Keep z_star constant and vary beta and gamma, and see how the fits change.

    Args:
        config (DictConfig): Hydra config object.
    """
    da = get_fit_da(config)
    print(da)
    # plot_fit_da(config, da)


if __name__ == "__main__":
    # python -m worst.vary_noise
    # _name_base(config)
    run_vary_noise()
