import os
from typing import List
import hydra
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import genextreme
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults, feature_grid, label_subplots
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
    """Retry wrapper. If a function fails flakeily, then wrap with this to retry it a few times.

    Puts the function in a try except block and retries it a `max_retries` times if unsuccesful.

    Args:
        max_retries (int, optional): Number of retries. Defaults to 10.

    Returns:
        callable: Wrapper function to recall the function in case of failure.
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


@retry_wrapper(max_retries=10)
def try_fit(
    z_star: float,
    beta: float,
    gamma: float,
    ns: int,
    seed: int,
    config: DictConfig,
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
    fit = config.fit
    bg_alpha, bg_beta, bg_gamma = fit_gev_upper_bound_known(
        zs,
        z_star,
        opt_steps=fit.steps,
        lr=fit.lr,
        beta_guess=fit.beta_guess,
        gamma_guess=fit.gamma_guess,
        verbose=config.verbose,
    )
    upnk_alpha, upnk_beta, upnk_gamma = fit_gev_upper_bound_not_known(
        zs,
        opt_steps=fit.steps,
        lr=fit.lr,
        alpha_guess=fit.alpha_guess,
        beta_guess=fit.beta_guess,
        gamma_guess=fit.gamma_guess,
        verbose=config.verbose,
    )
    true_q = genextreme.isf(quantiles, c=-gamma, loc=alpha, scale=beta)
    ubk_q = genextreme.isf(quantiles, c=-bg_gamma, loc=bg_alpha, scale=bg_beta)
    up_not_known_q = genextreme.isf(
        quantiles, c=-upnk_gamma, loc=upnk_alpha, scale=upnk_beta
    )
    results = np.array([true_q, ubk_q, up_not_known_q])
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
    if not os.path.exists(data_name) or not config.reload:
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
                                    config,
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
    means = da.mean("seed")
    std = da.std("seed")
    lower_p = da.quantile(config.figure.lp, dim="seed")
    upper_p = da.quantile(config.figure.up, dim="seed")
    range_p = upper_p - lower_p

    # remove quantile coordinate from lower_p and upper_p
    lower_p = lower_p.drop_vars("quantile")
    upper_p = upper_p.drop_vars("quantile")

    # get ready for plotting
    plot_defaults()

    # I want to put all of these xr.DataArrays together in a single xr.Dataset and then plot them.

    da_d = {
        "mean": means,
        "std": std,
        "lower_p": lower_p,
        "upper_p": upper_p,
        "range_p": range_p,
    }
    da_l = []
    for k, da in da_d.items():
        da = da.rename(k)
        da_d[k] = da
        da_l.append(da)
        assert ("ns", "fit", "rp", "beta", "gamma") == da.dims
        assert isinstance(da, xr.DataArray)

    ds = xr.merge(da_l)
    print(ds)

    plot_fit_primary_var(
        ds.isel(fit=1, rp=0),
        config,
        os.path.join(FIGURE_PATH, f"max_known_rv100_{_name_base(config)}.pdf"),
    )
    plot_fit_primary_var(
        ds.isel(fit=2, rp=0),
        config,
        os.path.join(FIGURE_PATH, f"max_unknown_rv100_{_name_base(config)}.pdf"),
    )
    plot_fit_primary_var(
        ds.isel(fit=1, rp=1),
        config,
        os.path.join(FIGURE_PATH, f"max_known_rv500_{_name_base(config)}.pdf"),
    )
    plot_fit_primary_var(
        ds.isel(fit=2, rp=1),
        config,
        os.path.join(FIGURE_PATH, f"max_unknown_rv500_{_name_base(config)}.pdf"),
    )
    bias_max_known = ds["mean"].sel(fit="max_known") - ds["mean"].sel(fit="true")
    bias_max_unknown = ds["mean"].sel(fit="max_unknown") - ds["mean"].sel(fit="true")

    relative_std = ds["std"].isel(fit=2) / ds["std"].isel(fit=1)
    relative_range = ds["range_p"].isel(fit=2) / ds["range_p"].isel(fit=1)

    da_d = {
        "bias_max_known": bias_max_known,
        "bias_max_unknown": bias_max_unknown,
        "relative_range": relative_range,
        "relative_std": relative_std,
    }
    da_l = []
    for k, da in da_d.items():
        da = da.rename(k)
        da_d[k] = da
        da_l.append(da)
        print(k, da.dims, da.shape)
    ds = xr.merge(da_l)
    print("ds2", ds)
    figure_name = os.path.join(
        FIGURE_PATH, f"bias_relative_range_RV100_{_name_base(config)}.pdf"
    )
    plot_diff(ds.isel(rp=0), config, figure_name)

    plot_diff(
        ds.isel(rp=1),
        config,
        os.path.join(
            FIGURE_PATH, f"bias_relative_range_RV500_{_name_base(config)}.pdf"
        ),
    )


def plot_diff(ds: xr.Dataset, config: DictConfig, figure_name: str) -> None:
    """Plot difference.

    Args:
        ds (xr.Dataset): Dataset with the fit data.
        config (DictConfig): Hydra config object.
        figure_name (str): Figure name.
    """
    _, axs = feature_grid(
        ds,
        [["bias_max_known", "bias_max_unknown"], ["relative_range", "relative_std"]],
        [["m", "m"], [None, None]],
        [["Bias max known", "Bias max unknown"], ["Relative range", "Relative std"]],
        [[None, None], [None, None]],
        [None, None],
        xy=(
            ("beta", r"Scale, $\beta$", "m"),
            ("gamma", r"Shape, $\gamma$", "dimensionless"),
        ),
    )
    label_subplots(axs)
    plt.savefig(figure_name)
    plt.close()


def plot_fit_primary_var(ds: xr.Dataset, config: DictConfig, figure_name: str) -> None:
    """Plot fit primary variable.

    Args:
        ds (xr.Dataset): Dataset with the fit data.
        config (DictConfig): Hydra config object.
        figure_name (str): Figure name.
    """
    _, axs = feature_grid(
        ds,
        [
            ["mean", "std"],
            ["lower_p", "upper_p"],
        ],
        [["m", "m"], ["m", "m"]],
        [
            [
                "Mean estimate RV" + str(ds.rp.values) + r", $\mu$",
                "Standard deviation in estimates of RV"
                + str(ds.rp.values)
                + r", $\sigma$",
            ],
            [
                "RV"
                + str(ds.rp.values)
                + f" {100*config.figure.lp:.0f}"
                + "th percentile estimate, $q_{"
                + f"{config.figure.lp:.2f}"
                + "}$",
                "RV"
                + str(ds.rp.values)
                + f" {100*config.figure.up:.0f}"
                + "th percentile estimate, $q_{"
                + f"{config.figure.up:.2f}"
                + "}$",
            ],
        ],
        [[None, None], [None, None]],
        [None, None],
        xy=(
            (
                "beta",
                r"Scale, $\beta$",
                "m",
            ),
            ("gamma", r"Shape, $\gamma$", "dimensionless"),
        ),
    )
    label_subplots(axs)
    plt.savefig(figure_name)
    plt.close()


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
