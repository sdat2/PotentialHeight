"""Tensorflow optimization for the worst-case analysis of random variables.

Sometimes breaks for no clear reason; in that case fit is retried 9 further times.
It is very unlikely that the optimization will fail 10 times in a row, so this should be sufficient.
"""

from typing import List, Tuple, Optional
import numpy as np
import os
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from scipy.stats import genextreme
from tcpips.constants import DATA_PATH, FIGURE_PATH
from tqdm import tqdm
from sithom.plot import plot_defaults, label_subplots, get_dim
from sithom.time import timeit
from .utils import alpha_from_z_star_beta_gamma, plot_rp, plot_sample_points

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename="log.log", level=logging.INFO)


def seed_all(seed: int) -> None:
    """Seed all random number generators.

    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)
    #  tf.set_random_seed(seed)
    tf.random.set_seed(seed)


def gen_data(alpha: float, beta: float, gamma: float, n: int = 1000) -> np.ndarray:
    """
    Generate data from a Generalized Extreme Value distribution.

    Args:
        alpha (float): Location parameter.
        beta (float): Scale parameter.
        gamma (float): Shape parameter.
        n (int, optional): Number of samples. Defaults to 1000.

    Returns:
        np.ndarray: Generated sample data.
    """
    gev = tfd.GeneralizedExtremeValue(loc=alpha, scale=beta, concentration=gamma)
    return gev.sample(n).numpy()


def fit_gev_upper_bound_not_known(
    data: np.ndarray,
    opt_steps: int = 1000,
    lr: float = 0.01,
    alpha_guess: float = 0.0,
    beta_guess: float = 1.0,
    gamma_guess: float = -0.1,
    force_weibull: bool = False,
) -> Tuple[float, float, float]:
    """
    Fit a Generalized Extreme Value distribution to data when the upper bound is not known.

    Args:
        data (np.ndarray): Data to fit the GEV distribution to.
        opt_steps (int, optional): Optimization steps. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 0.01.
        alpha_guess (float, optional): Initial location. Defaults to 0.0.
        beta_guess (float, optional): Initial scale. Defaults to 1.0.
        gamma_guess (float, optional): Initial concentration. Defaults to -0.1.
        force_weibull (bool, optional): Force gamma<0. Defaults to False.

    Returns:
        Tuple[float, float, float]: Estimated alpha, beta, gamma.
    """
    print("Fitting upper bound not known, N=", len(data))

    # Define the parameters of the model
    alpha = tf.Variable(alpha_guess, dtype=tf.float32)
    beta = tf.Variable(
        beta_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg()
    )
    if force_weibull:  # add some constraints to make sure gamma is negative
        neg_gamma = tf.Variable(
            -gamma_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg()
        )  # Start with zero for stability
    else:
        neg_gamma = tf.Variable(
            -gamma_guess, dtype=tf.float32
        )  # Start with zero for stability

    # Define the log likelihood function
    def neg_log_likelihood(
        alpha: tf.Variable, beta: tf.Variable, neg_gamma: tf.Variable, data: np.ndarray
    ) -> tf.Tensor:
        """
        Negative log likelihood function for the Generalized Extreme Value distribution.

        Args:
            alpha (tf.Variable): Location parameter.
            beta (tf.Variable): Scale parameter.
            neg_gamma (tf.Variable): Shape parameter.
            data (np.ndarray): Data to fit the GEV distribution to.

        Returns:
            tf.Tensor: Negative log likelihood loss.
        """
        dist = tfd.GeneralizedExtremeValue(
            loc=alpha, scale=beta, concentration=-neg_gamma
        )
        log_likelihoods = dist.log_prob(data)
        return -tf.reduce_sum(log_likelihoods)

    # Define the loss function (negative log likelihood)
    # def neg_log_likelihood() -> tf.Tensor:
    #    return -log_likelihood(loc, scale, concentration, data)

    # Set up the optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    # Define the training step
    @tf.function
    def train_step() -> tf.Tensor:
        """
        One optimization step for tensorflow.

        Returns:
            tf.Tensor: return the loss.
        """
        with tf.GradientTape() as tape:
            loss = neg_log_likelihood(alpha, beta, neg_gamma, data)
        gradients = tape.gradient(loss, [alpha, beta, neg_gamma])
        optimizer.apply_gradients(zip(gradients, [alpha, beta, neg_gamma]))
        return loss

    # Training loop
    for step in range(opt_steps):
        loss = train_step()
        if step % 100 == 0:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Alpha: {alpha.numpy()}, Beta: {beta.numpy()}, Gamma: {-neg_gamma.numpy()}"
            )

    print(
        f"Estimated Alpha: {alpha.numpy()}, Estimated Beta: {beta.numpy()}, Estimated Gamma: {-neg_gamma.numpy()}"
    )
    return alpha.numpy(), beta.numpy(), -neg_gamma.numpy()


def fit_gev_upper_bound_known(
    data: np.ndarray,
    z_star: float,
    opt_steps: int = 1000,
    lr: float = 0.01,
    beta_guess=1.0,
    gamma_guess=-0.1,
) -> Tuple[float, float, float]:
    """
    Fit a Generalized Extreme Value distribution to data when the upper bound is known.

    Args:
        data (np.ndarray): Data to fit the GEV distribution to.
        z_star (float): Known upper bound.
        opt_steps (int, optional): Optimization steps. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 0.01.
        beta_guess (float, optional): Initial scale. Defaults to 1.0.
        gamma_guess (float, optional): Initial concentration. Defaults to -0.1.

    Returns:
        Tuple[float, float, float]: Estimated alpha, beta, gamma.
    """
    print("Fitting upper bound known, N=", len(data))
    # Initial guess for sigma and xi
    beta = tf.Variable(
        beta_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg()
    )
    neg_gamma = tf.Variable(
        -gamma_guess,
        dtype=tf.float32,  # , constraint =tf.keras.constraints.NonPos()
        constraint=tf.keras.constraints.NonNeg(),
    )

    # Define the optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    def neg_log_likelihood(
        beta: tf.Variable, neg_gamma: tf.Variable, data: np.ndarray
    ) -> tf.Tensor:
        """
        Negative log likelihood function for the Generalized Extreme Value distribution
        with a known upper bound.

        Args:
            beta (tf.Variable): Scale parameter.
            neg_gamma (tf.Variable): Shape parameter.
            data (np.ndarray): Data to fit the GEV distribution to.

        Returns:
            tf.Tensor: Negative log likelihood loss.
        """
        dist = tfd.GeneralizedExtremeValue(
            loc=alpha_from_z_star_beta_gamma(z_star, beta, -neg_gamma),
            scale=beta,
            concentration=-neg_gamma,
        )
        log_likelihoods = dist.log_prob(data)
        return -tf.reduce_sum(log_likelihoods)

    # Define the loss function (negative log likelihood)
    # def neg_log_likelihood() -> tf.Tensor:
    #    return -log_likelihood(beta, gamma, data)

    # Optimization step function
    @tf.function
    def train_step() -> tf.Tensor:
        """
        Train the model for one step.

        Returns:
            tf.Tensor: Loss.
        """
        with tf.GradientTape() as tape:
            loss = neg_log_likelihood(beta, neg_gamma, data)
        grads = tape.gradient(loss, [beta, neg_gamma])
        optimizer.apply_gradients(zip(grads, [beta, neg_gamma]))
        return loss

    # Training loop
    for step in range(opt_steps):
        loss = train_step()
        if step % 100 == 0:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Alpha: "
                + f"{alpha_from_z_star_beta_gamma(z_star, beta.numpy(), -neg_gamma.numpy())},"
                + f" Beta: {beta.numpy()}, Gamma: {-neg_gamma.numpy()}"
            )

    # Extract the fitted parameters
    beta = beta.numpy()
    gamma = -neg_gamma.numpy()
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    print(f"Estimated Alpha: {alpha}, Estimated Beta: {beta}, Estimated Gamma: {gamma}")
    return alpha, beta, gamma


def plot_ex_fits(
    z_star: float = 7.0,
    beta: float = 1.0,
    gamma: float = -0.2,
    seed: int = 42,
    n: int = 50,
) -> None:
    """
    Show some example fits.

    Args:
        z_star (float, optional): Upper bound. Defaults to 7.0.
        beta (float, optional): Scale parameter. Defaults to 1.0.
        gamma (float, optional): Shape parameter. Defaults to -0.2.
        seed (int, optional): Seed. Defaults to 42.
        n (int, optional): Number of samples. Defaults to 50.
    """
    plot_defaults()
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    seed_all(seed)
    # alpha = 0.0
    # z_star = z_star_from_alpha_beta_gamma(alpha, beta, gamma)
    data = gen_data(alpha, beta, gamma, n=n)
    alpha_unb, beta_unb, gamma_unb = fit_gev_upper_bound_not_known(data)
    alpha_bound, beta_bound, gamma_bound = fit_gev_upper_bound_known(data, z_star)
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
    _, ax = plt.subplots()
    plot_sample_points(data, color="black", ax=ax, label="Samples")
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
    quantiles: List[float] = [1 / 100, 1 / 500],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try fitting the GEV distributions to some data, and calculate the return values for some quantiles.

    Args:
        z_star (float, optional): Upper bound of GEV. Defaults to 7.
        beta (float, optional): Scale. Defaults to 4.
        gamma (float, optional): Concentration. Defaults to -0.1.
        n (int, optional): Number of data samples. Defaults to 40.
        seed (int, optional): Random seed. Defaults to 42.
        quantiles (List[float], optional): Quantiles to get. Defaults to [1 / 100, 1 / 500].

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Return values for the quantiles for the true GEV,
            the GEV with the known upper bound, and the GEV with no upper bound.
    """
    seed_all(seed)
    print(
        "Fitting for z_star", z_star, "beta", beta, "gamma", gamma, "n", n, "seed", seed
    )
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    zs = gen_data(alpha, beta, gamma, n)
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

    for n in nums:

        def _retry(max_retries: int = 10) -> None:
            # sometimes the optimization randomly fails, so we retry
            try:
                return try_fit(
                    z_star=z_star,
                    beta=beta,
                    gamma=gamma,
                    n=int(n),
                    quantiles=[1 / 100, 1 / 500],
                    seed=seed,
                )
            except Exception as e:
                print("Exception", e)
                print("Failed for n", n)
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
        DATA_PATH, f"evt_fig_tens_{z_star:.2f}_{beta:.2f}_{gamma:.2f}_{len(nums)}.nc"
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
        z_star (float, optional): Upper bound. Defaults to 7.
        beta (float, optional): Scale parameter. Defaults to 1.
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

    numbers = res_ds.number.values

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
            print("ranges at 50 samples", range_p.sel(number=50, method="nearest"))
            ratio_change = range_p.isel(fit=1) / range_p.isel(fit=2)
            print(
                "knowing maxima makes difference",
                ratio_change.sel(number=50, method="nearest"),
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
    print(("alpha", alpha, "beta", beta, "gamma", gamma, "z_star", z_star))


if __name__ == "__main__":
    # python -m worst.tens
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--z_star", type=float, default=7)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=-0.2)
    parser.add_argument("--ex_seed", type=int, default=42)
    parser.add_argument("--ex_num", type=int, default=50)
    parser.add_argument("--min_samp", type=int, default=20)
    parser.add_argument("--max_samp", type=int, default=1000)
    parser.add_argument("--samp_steps", type=int, default=26)
    parser.add_argument("--seed_steps", type=int, default=600)
    parser.add_argument("--color_true", type=str, default="black")
    parser.add_argument("--color_max_known", type=str, default="#1b9e77")
    parser.add_argument("--color_max_unknown", type=str, default="#d95f02")
    args = parser.parse_args()
    evt_fig_tens(
        z_star=args.z_star,
        beta=args.beta,
        gamma=args.gamma,
        ex_seed=args.ex_seed,
        ex_num=args.ex_num,
        min_samp=args.min_samp,
        max_samp=args.max_samp,
        samp_steps=args.samp_steps,
        seed_steps=args.seed_steps,
        color_true=args.color_true,
        color_max_known=args.color_max_known,
        color_max_unknown=args.color_max_unknown,
        load=True,
    )
    # plot_defaults()
    # plot_ex(7, 1, -0.3, 42, 50, "black", "#1b9e77", "#d95f02")
