"""Tensorflow optimization for the worst-case analysis of random variables."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from .utils import z_star_from_alpha_beta_gamma, alpha_from_z_star_beta_gamma, plot_rp
from sithom.utils import plot_defaults, plot_rp


def gen_data(alpha, beta, gamma, n=1000):
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


if __name__ == "__main__":
    plot_ex_fits(seed=42)
