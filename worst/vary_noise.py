"""Vary noise of the upper bound, see how big the noise has to get before its no longer valuable.

We want to see what would happen if our method of determining the potential height is inaccurate, and so there is some random noise. Surely the information about the upper bound becomes less valuable as it becomes less certain. We choose one value of scale (beta), shape (gamma) and assymptote (z_star) for the GEV we are sampling, with a fixed sample size, n_s. Then we follow two ways of fitting the GEV b: one where we directly fit beta, gamma, and z_star by minimizing the negative log likelihood (upper bound unknown). The other by fitting beta and gamma while assuming a value of the upper bound (upper bound known). In this second fitting method we assume that our calculation of the upper bound is the true value with some Gaussian error added on whose standard deviation sigma_{z_star} is varied. For each resample we select a sample of n_s, and we also select a new calculated value of \hat{z_star} to assume for the second fitting method. As before we resample the true GEV distribution N_r=600 times, repeating this for each value of sigma_{z_star}.

"""

from typing import List
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
def try_fit_upperbound(
    z_star_mu: float,
    z_star_sigma: float,
    beta: float,
    gamma: float,
    ns: int,
    seed: int,
    config: DictConfig,
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> xr.Dataset:
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
        xr.Dataset: Dataset with the return values.
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
    return xr.Dataset(
        data_vars={
            "ubk": (
                ("ns", "rp", "seed", "beta", "gamma", "z_star_sigma"),
                np.expand_dims(
                    np.expand_dims(
                        np.expand_dims(
                            np.expand_dims(
                                np.expand_dims(np.array(ubk_q), 0), -1  # ns
                            ),  # seed
                            -1,  # beta,
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
    )


@timeit
def try_fit_no_upperbound(
    z_star: float,
    beta: float,
    gamma: float,
    ns: int,
    seed: int,
    config: DictConfig,
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> xr.Dataset:
    """
    Try fit no upperbound.

    Args:
        z_star (float): z_star mean (true value).
        beta (float): beta.
        gamma (float): gamma.
        ns (int): Number of samples.
        seed (int): Seed.
        quantiles (List[float], optional): Quantiles. Defaults to [1/100, 1/500].

    Returns:
        xr.Dataset: Dataset with the return values.
    """
    seed_all(seed)
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_data(alpha, beta, gamma, ns)
    fit = config.fit

    upnk_alpha, upnk_beta, upnk_gamma = fit_gev_upper_bound_not_known(
        zs,
        opt_steps=fit.steps,
        lr=fit.lr,
        alpha_guess=fit.alpha_guess,
        beta_guess=fit.beta_guess,
        gamma_guess=fit.gamma_guess,
        verbose=config.verbose,
    )
    up_not_known_q = genextreme.isf(
        quantiles, c=-upnk_gamma, loc=upnk_alpha, scale=upnk_beta
    )
    return xr.Dataset(
        data_vars={
            "ubu": (
                ("ns", "rp", "seed", "beta", "gamma"),
                np.expand_dims(
                    np.expand_dims(
                        np.expand_dims(
                            np.expand_dims(np.array(up_not_known_q), 0), -1  # ns
                        ),  # seed
                        -1,  # beta,
                    ),
                    -1,  # gamma
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
        },
    )


def _name_base(config: DictConfig) -> str:
    """
    Name base.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        str: Unique name based on the configuration.
    """
    return f"z_star_{config.z_star:.2f}_ns_{config.ns}_Nr_{config.seed_steps_Nr}_gamma_{config.gamma:.2f}_beta_{config.beta:.2f}_z_star_sigma_{config.z_star_sigma.min:.2f}_{config.z_star_sigma.max:.2f}_{config.z_star_sigma.steps}"


@timeit
def get_fit_ds(config: DictConfig) -> xr.Dataset:
    """Get fit xr.Dataset.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        xr.Dataset: Dataset with the fit data.
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
        ds_ubk = xr.concat(
            [
                xr.concat(
                    [
                        try_fit_upperbound(
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
        ds_ubu = xr.concat(
            [
                try_fit_no_upperbound(
                    config.z_star,
                    config.beta,
                    config.gamma,
                    config.ns,
                    seed,
                    config,
                    quantiles,
                )
                for seed in tqdm(seed_offsets)
            ],
            dim="seed",
        )
        ds = xr.merge([ds_ubu, ds_ubk])
        ds.to_netcdf(data_name)
    else:
        ds = xr.open_dataset(data_name)
    return ds


def process_noise(config: DictConfig, da: xr.DataArray) -> xr.Dataset:
    """Process noise.

    Args:
        config (DictConfig): config.
        da (xr.DataArray): dataarray

    Returns:
        xr.Dataset:
    """
    lower_p = da.quantile(config.figure.lp, dim="seed")
    upper_p = da.quantile(config.figure.up, dim="seed")
    lower_p = lower_p.drop_vars("quantile")
    upper_p = upper_p.drop_vars("quantile")
    range_p = upper_p - lower_p
    da_d = {
        "mn": da.mean("seed"),
        "sd": da.std("seed"),
        "lower_p": lower_p,
        "upper_p": upper_p,
        "range_p": range_p,
    }
    da_l = []
    for k, da in da_d.items():
        da = da.rename(k)
        da_l.append(da)
    return xr.merge(da_l)


def true_ds(config: DictConfig) -> xr.Dataset:
    """True dataset.

    Args:
        config (DictConfig): includes parameters of GEV.

    Returns:
        xr.Dataset
    """
    return xr.Dataset(
        data_vars={
            "true": (
                ("rp", "beta", "gamma"),
                np.expand_dims(
                    np.expand_dims(
                        genextreme.isf(
                            list(config.quantiles),
                            c=-config.gamma,
                            loc=alpha_from_z_star_beta_gamma(
                                config.z_star, config.beta, config.gamma
                            ),
                            scale=config.beta,
                        ),
                        -1,
                    ),
                    -1,
                ),
            ),
        },
        coords={
            "rp": (
                ("rp"),
                [int(1 / q) for q in config.quantiles],
                {"units": "years", "long_name": "Return period"},
            ),
            "gamma": (("gamma"), [config.gamma]),
            "beta": (("beta"), [config.beta]),
        },
    )


def plot_fit_ds(config: DictConfig, ds: xr.Dataset) -> None:
    """
    Plot fit ds.

    Args:
        config (DictConfig): Config.
        ds (xr.Dataset): Dataset.
    """
    # work out true return periods
    true_rp = true_ds(config)

    print(config, ds)
    print("ubu", ds["ubu"])
    ubu = process_noise(config, ds["ubu"])
    print("ubk", ds["ubk"])
    ubk = process_noise(config, ds["ubk"])
    print("ubu", ubu, "\n\nubk", ubk)
    ratio = ubu["range_p"] / ubk["range_p"]
    plot_defaults()

    # setup figure
    _, axs = plt.subplots(
        3,
        1,
        height_ratios=[1, 1, 1],
        figsize=get_dim(ratio=0.6180339887498949 * 2),
        sharex=True,
    )

    # intitial plots
    for irp in (0, 1):
        axs[irp].set_ylabel("1 in " + str(int(1 / config.quantiles[irp])) + " years")
        axs[irp].hlines(
            true_rp.isel(rp=irp).values,
            config.z_star_sigma.min,
            config.z_star_sigma.max,
            color=config.color.true,
        )
        iubu = ubu.isel(rp=irp)
        iubk = ubk.isel(rp=irp)
        axs[irp].fill_between(
            ratio.z_star_sigma,
            iubu.lower_p.values.ravel(),
            iubu.upper_p.values.ravel(),
            alpha=0.2,
            color=config.color.max_unknown,
            linestyle="-",
        )
        axs[irp].fill_between(
            ratio.z_star_sigma,
            iubk.lower_p.values.ravel(),
            iubk.upper_p.values.ravel(),
            alpha=0.2,
            color=config.color.max_known,
            linestyle="-",
        )
        axs[irp].plot(
            ratio.z_star_sigma,
            ubk.mn.values,
            color=config.color.max_known,
            linewidth=1,
            label="Max known GEV",
        )
        axs[irp].plot(
            ratio.z_star_sigma,
            ubu.mn.values,
            color=config.color.max_known,
            linewidth=1,
            label="Unbounded GEV",
        )

    axs[0].legend()

    axs[2].plot(
        ratio.z_star_sigma,
        ratio.isel(rp=0).values.ravel(),
        color="blue",
        label="1 in " + str(int(1 / config.quantiles[0])) + " years",
    )
    axs[2].plot(
        ratio.z_star_sigma,
        ratio.isel(rp=1).values.ravel(),
        color="red",
        label="1 in " + str(int(1 / config.quantiles[1])) + " year RV",
    )
    axs[2].legend()
    axs[2].set_xlabel(
        r"Standard deviation of calculated upper bound, $\sigma_{\hat{z}*}$ [m]"
    )
    axs[2].set_ylabel(
        r"Ratio of 5%-95% envelopes, $\frac{r_{\mathrm{upper\; bound\; unknown}}}{r_{\mathrm{upper\; bound\; known}}}$"  # [dimensionless]"
    )
    axs[2].hlines(
        1,
        config.z_star_sigma.min,
        config.z_star_sigma.max,
        color="grey",
        linestyles="dashed",
    )
    plt.xlim(config.z_star_sigma.min, config.z_star_sigma.max)
    label_subplots(axs)
    plt.savefig(os.path.join(FIGURE_PATH, "vary_" + _name_base(config) + ".pdf"))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_noise")
def run_vary_noise(config: DictConfig) -> None:
    """
    Run vary gamma beta experiment.

    Keep z_star constant and vary beta and gamma, and see how the fits change.

    Args:
        config (DictConfig): Hydra config object.
    """
    ds = get_fit_ds(config)
    print(ds)
    plot_fit_ds(config, ds)


if __name__ == "__main__":
    # python -m worst.vary_noise
    # _name_base(config)
    run_vary_noise()
