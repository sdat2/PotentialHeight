"""Tensorflow optimization for the worst-case analysis of random variables.

Sometimes breaks for no clear reason; in that case fit is retried 9 further times.
It is very unlikely that the optimization will fail 10 times in a row, so this should be sufficient.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from .utils import alpha_from_z_star_beta_gamma

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
    verbose: bool = False,
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
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        Tuple[float, float, float]: Estimated alpha, beta, gamma.
    """
    if verbose:
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
        if step % 100 == 0 and verbose:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Alpha: {alpha.numpy()}, Beta: {beta.numpy()}, Gamma: {-neg_gamma.numpy()}"
            )
    if verbose:
        print(
            f"Estimated Alpha: {alpha.numpy()}, Estimated Beta: {beta.numpy()}, Estimated Gamma: {-neg_gamma.numpy()}"
        )
    return alpha.numpy(), beta.numpy(), -neg_gamma.numpy()


def fit_gev_upper_bound_known(
    data: np.ndarray,
    z_star: float,
    opt_steps: int = 1000,
    lr: float = 0.01,
    beta_guess: float = 1.0,
    gamma_guess: float = -0.1,
    verbose: bool = False,
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
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        Tuple[float, float, float]: Estimated alpha, beta, gamma.
    """
    if verbose:
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
        if step % 100 == 0 and verbose:
            print(
                f"Step {step}, Loss: {loss.numpy()}, Alpha: "
                + f"{alpha_from_z_star_beta_gamma(z_star, beta.numpy(), -neg_gamma.numpy())},"
                + f" Beta: {beta.numpy()}, Gamma: {-neg_gamma.numpy()}"
            )

    # Extract the fitted parameters
    beta = beta.numpy()
    gamma = -neg_gamma.numpy()
    alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
    if verbose:
        print(
            f"Estimated Alpha: {alpha}, Estimated Beta: {beta}, Estimated Gamma: {gamma}"
        )
    return alpha, beta, gamma
