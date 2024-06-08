"""Tensorflow optimization for the worst-case analysis of random variables."""

from typing import List
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from .utils import z_star_from_alpha_beta_gamma, alpha_from_z_star_beta_gamma, plot_rp
from scipy.stats import genextreme
from sithom.plot import plot_defaults, label_subplots, get_dim
from tcpips.constants import DATA_PATH, FIGURE_PATH
from sithom.time import timeit


def gen_data(alpha: float, beta: float, gamma: float, n: int = 1000):
    gev = tfd.GeneralizedExtremeValue(loc=alpha, scale=beta, concentration=gamma)
    return gev.sample(n).numpy()


def fit_gev(data):
    # Define the parameters of the model
    loc = tf.Variable(0.0, dtype=tf.float32)
    scale = tf.Variable(1.0, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg())
    concentration = tf.Variable(0.0, dtype=tf.float32)  # Start with zero for stability

    # Define the log likelihood function
    def log_likelihood(loc, scale, concentration, data):
        dist = tfd.GeneralizedExtremeValue(
            loc=loc, scale=scale, concentration=concentration
        )
        log_likelihoods = dist.log_prob(data)
        return tf.reduce_sum(log_likelihoods)

    # Define the loss function (negative log likelihood)
    def neg_log_likelihood():
        return -log_likelihood(loc, scale, concentration, data)

    # Set up the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Define the training step
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = neg_log_likelihood()
        gradients = tape.gradient(loss, [loc, scale, concentration])
        optimizer.apply_gradients(zip(gradients, [loc, scale, concentration]))
        return loss

    # Training loop
    for step in range(2000):
        loss = train_step()
        if step % 100 == 0:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Loc: {loc.numpy()}, Scale: {scale.numpy()}, Concentration: {concentration.numpy()}"
            )

    print(
        f"Estimated Loc: {loc.numpy()}, Estimated Scale: {scale.numpy()}, Estimated Concentration: {concentration.numpy()}"
    )
    return loc.numpy(), scale.numpy(), concentration.numpy()


def fit_upknown(data, z_star, steps=5000):
    # Initial guess for sigma and xi
    initial_params = [1.0, -0.1]
    beta = tf.Variable(
        initial_params[0], dtype=tf.float32, constraint=tf.keras.constraints.NonNeg()
    )
    gamma = tf.Variable(
        initial_params[1],
        dtype=tf.float32,  # , constraint =tf.keras.constraints.NonPos()
    )

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    def log_likelihood(beta, gamma, data):
        dist = tfd.GeneralizedExtremeValue(
            loc=alpha_from_z_star_beta_gamma(z_star, beta, gamma),
            scale=beta,
            concentration=gamma,
        )
        log_likelihoods = dist.log_prob(data)
        return tf.reduce_sum(log_likelihoods)

    # Optimization step function
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = -log_likelihood(beta, gamma, data)
        grads = tape.gradient(loss, [beta, gamma])
        optimizer.apply_gradients(zip(grads, [beta, gamma]))
        return loss

    # Training loop
    for step in range(steps):
        loss = train_step()
        if step % 500 == 0:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Beta: {beta.numpy()}, Gamma: {gamma.numpy()}"
            )

    # Extract the fitted parameters
    beta = beta.numpy()
    gamma = gamma.numpy()
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    print(f"Estimated Alpha: {alpha}, Estimated Beta: {beta}, Estimated Gamma: {gamma}")
    return alpha, beta, gamma


def plot_ex_fits(
    z_star: float = 7.0,
    beta: float = 1.0,
    gamma: float = -0.2,
    seed: int = 42,
    n: int = 50,
):
    plot_defaults()
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # alpha = 0.0
    # z_star = z_star_from_alpha_beta_gamma(alpha, beta, gamma)
    data = gen_data(alpha, beta, gamma, n=n)
    alpha_unb, beta_unb, gamma_unb = fit_gev(data)
    alpha_bound, beta_bound, gamma_bound = fit_upknown(data, z_star)
    plt.hist(data, bins=50, density=True, alpha=0.5)
    x = np.linspace(np.min(data), np.max(data), 1000)
    gev = tfd.GeneralizedExtremeValue(loc=alpha, scale=beta, concentration=gamma)
    plt.plot(x, gev.prob(x).numpy(), color="black")
    gev = tfd.GeneralizedExtremeValue(
        loc=alpha_unb, scale=beta_unb, concentration=gamma_unb
    )
    plt.plot(x, gev.prob(x).numpy(), color="orange")
    gev_bound = tfd.GeneralizedExtremeValue(
        loc=float(alpha_bound),
        scale=float(beta_bound),
        concentration=float(gamma_bound),
    )
    plt.plot(x, gev_bound.prob(x).numpy(), color="purple")
    plt.show()
    fig, ax = plt.subplots()
    sorted_zs = np.sort(data)
    empirical_rps = len(data) / np.arange(1, len(data) + 1)[::-1]

    ax.scatter(
        empirical_rps,
        sorted_zs,
        s=3,
        alpha=0.8,
        color="black",
        label="Sampled data points",
    )
    plot_rp(alpha, beta, gamma, color="black", ax=ax, label="Original GEV")
    plot_rp(
        alpha_bound,
        beta_bound,
        gamma_bound,
        color="purple",
        ax=ax,
        label="I: GEV upper bound known",
    )
    plot_rp(
        alpha_unb, beta_unb, gamma_unb, color="orange", label="II: GEV no bound", ax=ax
    )
    plt.legend()

    plt.show()


def try_fit(
    z_star: float = 7,
    beta: float = 4,
    gamma: float = -0.1,
    n: int = 40,
    seed: int = 42,
    quantiles: List[float] = [1 / 100, 1 / 200],
) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_data(alpha, beta, gamma, n)
    bg_alpha, bg_beta, bg_gamma = fit_upknown(zs, z_star)
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
) -> xr.DataArray:
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
) -> xr.DataArray:
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


def get_evt_fit_data(
    z_star: float,
    beta: float,
    gamma: float,
    seeds: np.ndarray,
    nums: np.ndarray,
    load: bool = True,
) -> xr.DataArray:
    data_name = os.path.join(
        DATA_PATH, f"evt_fig_tens_{z_star:.2f}_{beta:.2f}_{gamma:.2f}.nc"
    )
    print(data_name)
    if load and os.path.exists(data_name):
        return xr.open_dataarray(data_name)
    else:
        data = fit_seeds(z_star, beta, gamma, seeds, nums)
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
    ax=None,
):

    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
        )
    # plot an example fit
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    np.random.seed(ex_seed)
    zs = gen_data(alpha, beta, gamma, ex_num)
    ## fit the known upper bound and the unbounded case
    bg_alpha, bg_beta, bg_gamma = fit_upknown(zs, z_star)
    bg_alpha = alpha_from_z_star_beta_gamma(z_star, bg_beta, bg_gamma)
    s_alpha, s_beta, s_gamma = fit_gev(zs)
    # plot the original data
    plot_rp(alpha, beta, gamma, color=color_true, label="Original GEV", ax=ax)
    sorted_zs = np.sort(zs)
    empirical_rps = len(zs) / np.arange(1, len(zs) + 1)[::-1]

    ax.scatter(
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


@timeit
def evt_fig_tens(
    z_star: float = 7,
    beta: float = 1,
    gamma: float = -0.3,
    ex_seed: int = 42,
    ex_num: int = 50,
    min_samp: int = 20,
    max_samp: int = 1000,
    samp_steps: int = 100,
    seed_steps: int = 100,
    save_fig_path: str = os.path.join(FIGURE_PATH, "evt_fig_tens.pdf"),
    color_true: str = "black",
    color_max_known: str = "#1b9e77",
    color_max_unknown: str = "#d95f02",
):
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
        load=False,
    )
    # calculate statistics to work out sampling error
    mn = res_ds.mean(dim="seed")
    std = res_ds.std(dim="seed")

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
    print("alpha", alpha, "beta", beta, "gamma", gamma, "z_star", z_star)


if __name__ == "__main__":
    # python -m worst.tens
    evt_fig_tens()
