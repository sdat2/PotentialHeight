"""
Toy Example: Extrapolation Behaviour of ReLU NN Emulators

The Gaussian PDF of x, relates x to y as y = -PDF(x).
This script compares the empirical exceedance probability of an NN emulator
to the true analytical exceedance probability derived from the known input PDF.

Equation:
    y = -N(mu, sigma^2)(x)
    and PDF of x:
    f_x = N(mu, sigma^2)(x)

True analytical exceedance probability P(Y > t) is derived based on X ~ N(mu, sigma^2) and Y = -f_x(X).
For t >= 0, P(Y > t) = 0.
For -f_x(mu) < t < 0, P(Y > t) = 2 * (1 - Phi(sigma * sqrt(-2 * ln(-t * sqrt(2*pi*sigma^2)))) / sigma)
                     = 2 * (1 - Phi(sqrt(-2 * ln(-t * sqrt(2*pi*sigma^2)))))
where Phi is the standard normal CDF.
For t <= -f_x(mu), P(Y > t) = 1.
f_x(mu) = 1 / sqrt(2*pi*sigma^2) is the maximum value of the PDF, and -f_x(mu) is the minimum value of Y.
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from sithom.plot import plot_defaults, label_subplots

plot_defaults()

# --- Configuration Parameters ---
# Gaussian PDF parameters for X
MU = 0.0  # Mean of the Gaussian x distribution
SIGMA = 1.0  # Standard deviation of the Gaussian x distribution
# Sample and Training parameters
N_SAMPLES = 1000  # Sample size for each run
N_RUNS = 5  # Number of times to repeat the experiment (for variability analysis)
# Neural Network Architecture
N_HIDDEN_LAYERS = 2
N_NEURONS_PER_LAYER = 32
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
# Output directory
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_FILE_PATH, "toy_out")
DATA_PATH = os.path.join(OUTPUT_DIR, "data")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "img")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)


def plot_problem_setup_explanation(mu, sigma, save_path=""):
    """
    Plots the true input PDF f_x(x) and the true target function y = -f_x(x)
    to explain the experiment setup, sharing the x-axis.

    Args:
        mu (float): Mean of the Gaussian distribution for X.
        sigma (float): Standard deviation of the Gaussian distribution for X.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    x_dense = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    f_x_dense = ss.norm.pdf(x_dense, loc=mu, scale=sigma)
    y_dense_true = -f_x_dense

    fig, axes = plt.subplots(2, 1, sharex=True)  # , figsize=(8, 8))

    # Plot the Input PDF f_x(x)
    axes[0].plot(x_dense, f_x_dense, label=r"True PDF $f_X(x) = N(\mu, \sigma^2)(x)$")
    axes[0].set_ylabel(r"$f_X(x)$")
    axes[0].set_title(r"Probability Density Function of Input $f_X(x)$")
    axes[0].grid(True)
    # axes[0].legend()

    # Plot the Target Function y = -f_x(x)
    axes[1].plot(
        x_dense, y_dense_true, "r-", label=r"True Target Function $y = -f_X(x)$"
    )
    axes[1].set_xlabel(r"Input $x$")
    axes[1].set_ylabel(r"$y = -f_X(x)$")
    axes[1].set_title(r"True Target Function $y = -f_X(x)$")
    axes[1].grid(True)
    # axes[1].legend()

    # Overall title
    # fig.suptitle(
    #    "Experiment Problem Setup: Input Distribution and Target Function", y=1.02
    # )  # Adjust y for title position

    plt.tight_layout()  # Adjust layout to prevent overlap

    label_subplots(axes)

    if save_path:
        plt.savefig(
            save_path, bbox_inches="tight"
        )  # bbox_inches='tight' ensures suptitle is included
        print(f"Saved problem setup explanation plot to {save_path}")
    plt.close(fig)  # Close the figure


def generate_data(mu, sigma, n_samples, random_seed, run_id, save_path_prefix=""):
    """Generates input samples x from N(mu, sigma^2) and true outputs y = -PDF(x).

    Args:
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        n_samples (int): Number of samples to generate.
        random_seed (int): Seed for reproducibility.
        run_id (int): Identifier for the current experimental run.
        save_path_prefix (str, optional): Prefix for saving output files. Defaults to "".

    Returns:
        tuple: Contains:
            - x_samples (np.ndarray): 1D array of input samples.
            - y_true (np.ndarray): 1D array of true function outputs.
            - x_samples_nn (np.ndarray): 2D array of input samples (for NN).
            - y_true_nn (np.ndarray): 2D array of true outputs (for NN).
    """
    np.random.seed(random_seed)
    x_samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    x_samples_nn = x_samples.reshape(-1, 1)
    # True function is y = -PDF(x)
    y_true = -ss.norm.pdf(x_samples, loc=mu, scale=sigma)
    y_true_nn = y_true.reshape(-1, 1)

    if save_path_prefix:
        np.save(f"{save_path_prefix}_x_samples_run{run_id}.npy", x_samples_nn)
        np.save(f"{save_path_prefix}_y_true_run{run_id}.npy", y_true_nn)
        print(f"Saved x_samples and y_true for run {run_id}")

    return x_samples, y_true, x_samples_nn, y_true_nn


def plot_initial_data(x_samples, y_true, mu, sigma, n_samples, run_id, save_path=""):
    """Plots the generated data and the true function y = -Gaussian PDF(x).

    Args:
        x_samples (np.ndarray): 1D array of input samples.
        y_true (np.ndarray): 1D array of true function outputs.
        mu (float): Mean of the Gaussian PDF.
        sigma (float): Standard deviation of the Gaussian PDF.
        n_samples (int): Number of samples.
        run_id (int): Identifier for the current experimental run.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    # plt.figure(figsize=(8, 5))
    plt.scatter(
        x_samples, y_true, alpha=0.5, label=f"Sampled (x, y=f(x)), N={n_samples}"
    )
    x_dense = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    # Plot the true function y = -PDF(x)
    y_dense_true = -ss.norm.pdf(x_dense, loc=mu, scale=sigma)
    plt.plot(x_dense, y_dense_true, "r-", label=f"True Function $y = -f_X(x)$")
    plt.xlabel("Input x")
    plt.ylabel("True Output y = f(x)")
    plt.title(f"Run {run_id}: Ground Truth Function and Training Data")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved initial data plot to {save_path}")
    plt.close()


def build_relu_nn_model(input_shape, n_hidden_layers, n_neurons):
    """Builds a Keras Sequential model with ReLU hidden layers and a linear output layer."""
    model = keras.Sequential(name="ReLU_Emulator")
    model.add(layers.InputLayer(shape=input_shape))
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))  # Linear output for regression
    return model


def train_model(
    model,
    x_train,
    y_train,
    learning_rate,
    epochs,
    batch_size,
    validation_split,
    random_seed_nn,
    run_id,
    save_path_prefix="",
):
    """Compiles and trains the Keras model."""
    tf.random.set_seed(random_seed_nn)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = "mean_squared_error"
    model.compile(
        loss=loss_function, optimizer=optimizer, metrics=["mean_absolute_error"]
    )

    print(f"Run {run_id}: Starting model training...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=0,  # Suppress verbose output during training
    )
    print(f"Run {run_id}: Training complete.")

    if save_path_prefix:
        # Save in Keras format
        model.save(f"{save_path_prefix}_run{run_id}.keras")
        print(f"Saved trained model for run {run_id}")

        # Plot training history
        # plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training Loss (MSE)")
        plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
        plt.title(f"Run {run_id}: Model Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path_prefix}_training_history_run{run_id}.pdf")
        print(f"Saved training history plot for run {run_id}")
        plt.close()

    return model, history


def predict_with_model(model, x_data, run_id, save_path_prefix=""):
    """Generates predictions using the trained model."""
    y_pred_nn = model.predict(x_data, verbose=0)
    y_pred = y_pred_nn.flatten()

    if save_path_prefix:
        np.save(f"{save_path_prefix}_y_pred_run{run_id}.npy", y_pred)
        print(f"Saved y_pred for run {run_id}")
    return y_pred


def plot_emulator_fit(
    x_samples, y_true, y_pred, mu, sigma, n_samples, run_id, save_path=""
):
    """Plots the true function, sample data, and emulator predictions."""
    # plt.figure(figsize=(10, 6))
    plt.scatter(
        x_samples,
        y_true,
        alpha=0.3,
        label=f"Sampled (x, y=f(x)), N={n_samples}",
        color="blue",
        s=10,
    )

    # Sort for plotting lines
    sort_indices = np.argsort(x_samples)
    x_sorted = x_samples[sort_indices]
    y_pred_sorted = y_pred[sort_indices]

    x_dense = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    # Plot the true function y = -PDF(x)
    y_dense_true = -ss.norm.pdf(x_dense, loc=mu, scale=sigma)

    plt.plot(
        x_dense, y_dense_true, "b-", label=f"True Function $y = -f_X(x)$", linewidth=1
    )
    plt.plot(
        x_sorted, y_pred_sorted, "r--", label=f"Emulator $\hat{{y}}(x)$", linewidth=1
    )

    plt.xlabel("Input x")
    plt.ylabel("Output y or $\hat{y}$")
    plt.title(f"Run {run_id}: Emulator Fit vs. True Function")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved emulator fit plot to {save_path}")
    plt.close()


def calculate_true_exceedance(t_values, mu, sigma):
    """
    Calculates the true analytical exceedance probability P(Y > t) for Y = -N(mu, sigma^2)(X), X ~ N(mu, sigma^2).

    Args:
        t_values (np.ndarray): Array of threshold values t.
        mu (float): Mean of the Gaussian distribution for X.
        sigma (float): Standard deviation of the Gaussian distribution for X.

    Returns:
        np.ndarray: Array of true exceedance probabilities P(Y > t).
    """
    true_probs = np.zeros_like(t_values, dtype=float)
    max_pdf_val = 1.0 / (sigma * np.sqrt(2 * np.pi))  # f_x(mu)
    min_y_val = -max_pdf_val  # Minimum value of Y

    for i, t in enumerate(t_values):
        if t >= 0:
            true_probs[i] = 0.0
        elif t < min_y_val:
            # Should theoretically be 1, but due to float precision, check against boundary
            # If t is significantly below the minimum, prob is 1
            # We handle this boundary implicitly by solving for delta.
            # If -t * sqrt(2*pi*sigma^2) >= 1, log is >= 0, sqrt is complex/nan.
            # The formula derived applies for min_y_val < t < 0.
            # For t <= min_y_val, P(Y > t) = 1
            true_probs[i] = 1.0
        else:  # This is the range min_y_val < t < 0
            c = -t  # c is positive
            arg_ln = c * sigma * np.sqrt(2 * np.pi)
            # Ensure arg_ln is within (0, 1) for ln to be negative and sqrt argument positive
            if arg_ln >= 1.0:
                # This threshold t is <= min_y_val, probability is 1
                # Due to floating point, arg_ln might be slightly > 1 near boundary
                # Clamp to 1 if it's very close to 1 but should be >= 1
                if np.isclose(arg_ln, 1.0):
                    true_probs[i] = 1.0
                else:
                    # This case ideally shouldn't happen for t > min_y_val
                    # but could indicate numerical instability or edge case.
                    # Based on derivation, arg_ln >= 1 implies t <= min_y_val.
                    true_probs[i] = 1.0
            elif arg_ln <= 0:
                # Should not happen for t < 0
                true_probs[i] = 0.0  # Or handle error
            else:
                delta_over_sigma_squared = -2.0 * np.log(arg_ln)
                if delta_over_sigma_squared < 0:
                    # Should not happen for 0 < arg_ln < 1
                    true_probs[i] = 0.0  # Or handle error
                else:
                    delta_over_sigma = np.sqrt(delta_over_sigma_squared)
                    # P(Z > delta/sigma) = 1 - Phi(delta/sigma)
                    true_probs[i] = 2.0 * (1.0 - ss.norm.cdf(delta_over_sigma))

    return true_probs


def manual_ecdf_exceedance(data_array):
    """Calculates empirical exceedance probability (1 - ECDF)."""
    # Sort the data
    x = np.sort(data_array)
    # Calculate ECDF values (proportion of data points <= x)
    y_cdf = np.arange(1, len(x) + 1) / len(x)
    # Exceedance probability is 1 - CDF
    y_exceedance = 1 - y_cdf
    return x, y_exceedance


def plot_exceedance_probabilities(y_true, y_pred, mu, sigma, run_id, save_path=""):
    """
    Calculates and plots empirical and true analytical exceedance probabilities.
    """
    # plt.figure(figsize=(10, 6))

    # Determine a relevant range for t (threshold values)
    # The relevant range for Y is [min_y, 0)
    # We should plot t values that cover this range and potentially go slightly lower for tails
    max_pdf_val = 1.0 / (sigma * np.sqrt(2 * np.pi))
    min_y_val = -max_pdf_val
    # Use the minimum observed y_pred or y_true as a lower bound, or a bit lower than true min_y_val
    t_min_plot = min(np.min(y_true), np.min(y_pred), min_y_val * 1.1)
    t_max_plot = max(
        np.max(y_true), np.max(y_pred), 0.05
    )  # Go slightly above 0 if needed
    t_values = np.linspace(t_min_plot, t_max_plot, 500)

    # Plot True Analytical Exceedance Probability
    true_analytic_exceedance = calculate_true_exceedance(t_values, mu, sigma)
    plt.plot(
        t_values,
        true_analytic_exceedance,
        "k-",
        linewidth=0.5,
        label="True Analytical $P(Y > t)$",
    )

    # Plot Empirical Exceedance Probability for True Samples (Y)
    y_true_sorted, y_true_exceedance_emp = manual_ecdf_exceedance(y_true)
    plt.step(
        y_true_sorted,
        y_true_exceedance_emp,
        where="post",
        label="Empirical True $P(Y > t)$ (Samples)",
        marker=".",
        linestyle="-",
        color="blue",
        alpha=0.7,
    )

    # Plot Empirical Exceedance Probability for Emulator Predictions (Y_hat)
    y_pred_sorted, y_pred_exceedance_emp = manual_ecdf_exceedance(y_pred)
    plt.step(
        y_pred_sorted,
        y_pred_exceedance_emp,
        where="post",
        label="Empirical Emulator $\hat{P}(\hat{Y} > t)$ (Predictions)",
        marker="x",
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    plt.xlabel("Threshold $t$")
    plt.ylabel(r"Exceedance Probability $P(\text{Value} > t)$")
    plt.title(f"Run {run_id}: Exceedance Probabilities - Analytical vs. Empirical")
    plt.yscale("log")
    plt.ylim(1e-5, 1.1)  # Set reasonable limits for log scale
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved exceedance probability plot to {save_path}")
    plt.close()


def plot_all_emulator_exceedance_curves(all_y_pred, mu, sigma, n_runs, save_path=""):
    """
    Plots the true analytical exceedance curve against all empirical emulator curves.
    """
    # plt.figure(figsize=(12, 7))

    # Determine a relevant range for t (threshold values) across all predictions
    all_pred_flat = np.concatenate(all_y_pred)
    max_pdf_val = 1.0 / (sigma * np.sqrt(2 * np.pi))
    min_y_val = -max_pdf_val
    t_min_plot = min(np.min(all_pred_flat), min_y_val * 1.1)
    t_max_plot = max(np.max(all_pred_flat), 0.05)
    t_values = np.linspace(t_min_plot, t_max_plot, 500)

    # Plot True Analytical Exceedance Probability (Reference)
    true_analytic_exceedance = calculate_true_exceedance(t_values, mu, sigma)
    plt.plot(
        t_values,
        true_analytic_exceedance,
        "k-",
        linewidth=0.5,  # Thicker line for reference
        label="True Analytical $P(Y > t)$",
    )

    # Plot all empirical emulator curves
    for i, y_pred_run in enumerate(all_y_pred):
        x_pred_sorted, y_pred_exceedance_emp = manual_ecdf_exceedance(y_pred_run)
        plt.step(
            x_pred_sorted,
            y_pred_exceedance_emp,
            where="post",
            alpha=0.5,
            linestyle="--",
            label=(
                f"Emulator Run {i+1} (Empirical)" if i < 5 else None
            ),  # Limit legend entries
        )

    plt.xlabel("Threshold $t$")
    plt.ylabel("Exceedance Probability P$(\\text{Y} > t)$")
    # plt.title(
    #    f"Variability of Empirical Emulator Exceedance Probabilities ({n_runs} Runs) vs. True Analytical"
    # )
    plt.yscale("log")
    plt.ylim(1e-5, 1.1)  # Set reasonable limits for log scale
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved aggregate exceedance plot to {save_path}")
    plt.close()


# --- Main Experiment Script ---
if __name__ == "__main__":
    # Use sithom plot style if available
    # try:
    #     from sithom.plot import plot_defaults
    #     plot_defaults()
    #     print("Using sithom plot defaults.")
    # except ImportError:
    #     print("sithom not found. Using default matplotlib styles.")
    #     # Set some basic defaults if sithom is not available
    #     plt.style.use('seaborn-v0_8-darkgrid')

    all_emulator_predictions = []

    for run in range(N_RUNS):
        print(f"\n--- Starting Run {run + 1} of {N_RUNS} ---")
        current_run_id = run + 1
        data_random_seed = 42 + run  # Vary seed for data generation per run
        nn_random_seed = 100 + run  # Vary seed for NN initialization/training per run

        # 1. Generate Data
        x_s, y_t, x_s_nn, y_t_nn = generate_data(
            MU,
            SIGMA,
            N_SAMPLES,
            data_random_seed,
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"data_run{current_run_id}"),
        )

        # Plot initial data for this run (plots y = -PDF(x))
        plot_initial_data(
            x_s,
            y_t,
            MU,
            SIGMA,
            N_SAMPLES,
            current_run_id,
            save_path=os.path.join(
                FIGURE_PATH, f"initial_data_plot_run{current_run_id}.pdf"
            ),
        )

        # 2. Build Model
        model = build_relu_nn_model(
            input_shape=(x_s_nn.shape[1],),  # Input shape is (1,) for 1D x
            n_hidden_layers=N_HIDDEN_LAYERS,
            n_neurons=N_NEURONS_PER_LAYER,
        )
        if run == 0:  # Print model summary only for the first run
            model.summary()

        # 3. Train Model
        trained_model, history = train_model(
            model,
            x_s_nn,
            y_t_nn,
            LEARNING_RATE,
            EPOCHS,
            BATCH_SIZE,
            VALIDATION_SPLIT,
            nn_random_seed,
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"model"),
        )

        # 4. Plot Emulator Fit for training data
        y_predicted = predict_with_model(
            trained_model,
            x_s_nn,
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"predictions"),
        )
        plot_emulator_fit(
            x_s,
            y_t,
            y_predicted,
            MU,
            SIGMA,
            N_SAMPLES,
            current_run_id,
            save_path=os.path.join(
                FIGURE_PATH, f"emulator_fit_plot_run{current_run_id}.pdf"
            ),
        )

        # 4. Generate Predictions (on new test data)

        x_s, y_t, x_s_nn, y_t_nn = generate_data(
            MU,
            SIGMA,
            N_SAMPLES * 10,
            data_random_seed + 1000,  # Different seed for prediction data
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"predictions"),
        )
        y_predicted = predict_with_model(
            trained_model,
            x_s_nn,
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"predictions"),
        )
        all_emulator_predictions.append(y_predicted)

        # 6. Plot Exceedance Probabilities for this run (comparing Analytical, Empirical True, Empirical Emulator)
        plot_exceedance_probabilities(
            y_t,
            y_predicted,
            MU,
            SIGMA,
            current_run_id,
            save_path=os.path.join(
                FIGURE_PATH, f"exceedance_plot_run{current_run_id}.pdf"
            ),
        )

        print(f"--- Completed Run {run + 1} ---")

    # 7. Plot all emulator empirical exceedance curves together against the True Analytical curve
    if N_RUNS > 0:
        plot_all_emulator_exceedance_curves(
            all_emulator_predictions,
            MU,
            SIGMA,
            N_RUNS,
            save_path=os.path.join(FIGURE_PATH, "all_emulator_exceedance_curves.pdf"),
        )

    print("\nExperiment finished. Outputs are saved in the specified directories.")

    plot_problem_setup_explanation(
        MU, SIGMA, save_path=os.path.join(FIGURE_PATH, "problem_setup_explanation.pdf")
    )
