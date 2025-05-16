"""
Toy Example: Extrapolation Behavior of ReLU NN Emulators

The Gaussian PDF of x, is the negative of the Gaussian transform that relates x to y.

Equation:
    y = -N(mu, sigma^2)(x)
    and PDF:
    f_x = N(mu, sigma^2)(x)

"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sithom.plot import plot_defaults

plot_defaults()

# --- Configuration Parameters ---
# Gaussian PDF parameters
MU = 0.0  # Mean of the Gaussian both of x and Y = func(x)
SIGMA = 1.0  # Standard deviation of the Gaussian, both of x and Y = func(x)
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

# --- Helper Functions ---


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)


def generate_data(mu, sigma, n_samples, random_seed, run_id, save_path_prefix=""):
    """Generates input samples x from N(mu, sigma^2) and true outputs y = PDF(x).

    Now changed to
    Equation:
    y = -N(mu, sigma^2)(x)
    and PDF:
    f_x = N(mu, sigma^2)(x)

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

    Examples:
        >>> x, y, x_nn, y_nn = generate_data(0.0, 1.0, 5, 42, 0)
        >>> x.shape
        (5,)
        >>> y.shape
        (5,)
        >>> x_nn.shape
        (5, 1)
        >>> y_nn.shape
        (5, 1)
        >>> np.allclose(y, ss.norm.pdf(x, 0.0, 1.0))
        True
    """
    np.random.seed(random_seed)
    x_samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)  # [2, 3]
    x_samples_nn = x_samples.reshape(-1, 1)
    y_true = -ss.norm.pdf(x_samples, loc=mu, scale=sigma)  # [4, 5]
    y_true_nn = y_true.reshape(-1, 1)

    if save_path_prefix:
        np.save(f"{save_path_prefix}_x_samples_run{run_id}.npy", x_samples_nn)
        np.save(f"{save_path_prefix}_y_true_run{run_id}.npy", y_true_nn)
        print(f"Saved x_samples and y_true for run {run_id}")

    return x_samples, y_true, x_samples_nn, y_true_nn


def plot_initial_data(x_samples, y_true, mu, sigma, n_samples, run_id, save_path=""):
    """Plots the generated data and the true Gaussian PDF.

    Args:
        x_samples (np.ndarray): 1D array of input samples.
        y_true (np.ndarray): 1D array of true function outputs.
        mu (float): Mean of the Gaussian PDF.
        sigma (float): Standard deviation of the Gaussian PDF.
        n_samples (int): Number of samples.
        run_id (int): Identifier for the current experimental run.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x_samples, y_true, alpha=0.5, label=f"Sampled (x, y=f(x)), N={n_samples}"
    )
    x_dense = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    y_dense = ss.norm.pdf(x_dense, loc=mu, scale=sigma)  # [4, 5]
    plt.plot(x_dense, y_dense, "r-", label=f"True Gaussian PDF f(x)")
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
    """Builds a Keras Sequential model with ReLU hidden layers and a linear output layer.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (1,) for 1D input).
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential(name="ReLU_Emulator")  # [6, 7]
    model.add(layers.InputLayer(input_shape=input_shape))
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation="relu"))  # [6, 7]
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
    """Compiles and trains the Keras model.

    Args:
        model (keras.Model): The Keras model to train.
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        random_seed_nn (int): Seed for TensorFlow operations.
        run_id (int): Identifier for the current experimental run.
        save_path_prefix (str, optional): Prefix for saving model and plots. Defaults to "".

    Returns:
        tuple: Contains:
            - model (keras.Model): The trained Keras model.
            - history (keras.callbacks.History): Training history object.
    """
    tf.random.set_seed(random_seed_nn)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = "mean_squared_error"  # [8]
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
        verbose=0,  # Suppress verbose output during training for cleaner script output
    )
    print(f"Run {run_id}: Training complete.")

    if save_path_prefix:
        model.save(f"{save_path_prefix}_model_run{run_id}.keras")
        print(f"Saved trained model for run {run_id}")

        # Plot training history
        plt.figure(figsize=(8, 5))
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
    """Generates predictions using the trained model.

    Args:
        model (keras.Model): The trained Keras model.
        x_data (np.ndarray): Input data for prediction.
        run_id (int): Identifier for the current experimental run.
        save_path_prefix (str, optional): Prefix for saving predictions. Defaults to "".

    Returns:
        np.ndarray: 1D array of model predictions.
    """
    y_pred_nn = model.predict(x_data, verbose=0)
    y_pred = y_pred_nn.flatten()

    if save_path_prefix:
        np.save(f"{save_path_prefix}_y_pred_run{run_id}.npy", y_pred)
        print(f"Saved y_pred for run {run_id}")
    return y_pred


def plot_emulator_fit(
    x_samples, y_true, y_pred, mu, sigma, n_samples, run_id, save_path=""
):
    """Plots the true function, sample data, and emulator predictions.

    Args:
        x_samples (np.ndarray): 1D array of input samples.
        y_true (np.ndarray): 1D array of true function outputs.
        y_pred (np.ndarray): 1D array of emulator predictions.
        mu (float): Mean of the true Gaussian PDF.
        sigma (float): Standard deviation of the true Gaussian PDF.
        n_samples (int): Number of samples.
        run_id (int): Identifier for the current experimental run.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    plt.figure(figsize=(10, 6))
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
    y_dense_true = ss.norm.pdf(x_dense, loc=mu, scale=sigma)  # [4, 5]

    plt.plot(x_dense, y_dense_true, "b-", label=f"True Gaussian PDF f(x)", linewidth=2)
    plt.plot(
        x_sorted, y_pred_sorted, "r--", label=f"Emulator $\hat{{f}}(x)$", linewidth=2
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


def plot_exceedance_probabilities(y_true, y_pred, run_id, save_path=""):
    """Calculates and plots empirical exceedance probabilities for true and predicted values.

    Args:
        y_true (np.ndarray): 1D array of true function outputs.
        y_pred (np.ndarray): 1D array of emulator predictions.
        run_id (int): Identifier for the current experimental run.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    plt.figure(figsize=(10, 6))

    # Plot ECCDF for true values (y)
    # matplotlib.pyplot.ecdf is available from version 3.8
    # For broader compatibility, we can implement it manually or use statsmodels
    # Using statsmodels for robustness if available, else manual.
    # Manual ECDF calculation
    def manual_ecdf(data_array):
        x = np.sort(data_array)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x_true_sorted, cdf_true = manual_ecdf(y_true)
    x_pred_sorted, cdf_pred = manual_ecdf(y_pred)
    plt.plot(
        x_true_sorted,
        1 - cdf_true,
        label="True Function $P(Y > t)$ (Manual)",
        marker=".",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        x_pred_sorted,
        1 - cdf_pred,
        label="Emulator $\hat{P}(\hat{Y} > t)$ (Manual)",
        marker="x",
        linestyle="--",
        color="red",
    )
    # Note: For a strict ECCDF (P(X>t)), the plotting points might need slight adjustment for steps.
    # Matplotlib's ecdf(complementary=True) handles this well. [11, 12]

    plt.xlabel("Threshold t (Function Output Value)")
    plt.ylabel("Exceedance Probability P(Value > t)")
    plt.title(f"Run {run_id}: Comparison of Empirical Exceedance Probabilities")
    plt.yscale("log")
    # plt.xscale('log') # Optional, depending on the range of y values
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved exceedance probability plot to {save_path}")
    plt.close()


def plot_all_emulator_exceedance_curves(y_true_ref, all_y_pred, n_runs, save_path=""):
    """Plots the true exceedance curve against all emulator exceedance curves.

    Args:
        y_true_ref (np.ndarray): Reference true output values for the true curve.
        all_y_pred (list of np.ndarray): List where each element is y_pred from a run.
        n_runs (int): Total number of runs.
        save_path (str, optional): Path to save the plot. If empty, plot is not saved.
    """
    plt.figure(figsize=(12, 7))

    # Plot true exceedance curve (reference)

    def manual_ecdf(data_array):
        x = np.sort(data_array)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x_true_sorted, cdf_true = manual_ecdf(y_true_ref)
    plt.plot(
        x_true_sorted,
        1 - cdf_true,
        "k-",
        linewidth=2,
        label="True Function $P(Y > t)$ (Reference, Manual)",
    )

    # Plot all emulator curves
    for i, y_pred_run in enumerate(all_y_pred):
        x_pred_sorted, cdf_pred = manual_ecdf(y_pred_run)
        plt.plot(
            x_pred_sorted,
            1 - cdf_pred,
            alpha=0.5,
            linestyle="--",
            label=f"Emulator Run {i+1} (Manual)" if i < 5 else None,
        )

    plt.xlabel("Threshold t (Function Output Value)")
    plt.ylabel("Exceedance Probability P(Value > t)")
    plt.title(f"Variability of Emulator Exceedance Probabilities ({n_runs} Runs)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved aggregate exceedance plot to {save_path}")
    plt.close()


# --- Main Experiment Script ---
if __name__ == "__main__":
    # python -m  surgenet.toy_example
    all_emulator_predictions = []
    y_true_reference = None  # To store y_true from the first run as a reference

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
        if (
            run == 0
        ):  # Store y_true from the first run as reference for the aggregate plot
            y_true_reference = y_t.copy()

        # Plot initial data for this run
        plot_initial_data(
            x_s,
            y_t,
            MU,
            SIGMA,
            N_SAMPLES,
            current_run_id,
            save_path=os.path.join(
                DATA_PATH, f"initial_data_plot_run{current_run_id}.pdf"
            ),
        )

        # 2. Build Model
        # Keras input shape expects (num_features,)
        # Our x_s_nn is (N_SAMPLES, 1), so input_shape is (1,)
        model = build_relu_nn_model(
            input_shape=(x_s_nn.shape[1],),
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

        # 4. Generate Predictions
        y_predicted = predict_with_model(
            trained_model,
            x_s_nn,
            current_run_id,
            save_path_prefix=os.path.join(DATA_PATH, f"predictions"),
        )
        all_emulator_predictions.append(y_predicted)

        # 5. Plot Emulator Fit for this run
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

        # 6. Plot Exceedance Probabilities for this run
        plot_exceedance_probabilities(
            y_t,
            y_predicted,
            current_run_id,
            save_path=os.path.join(
                FIGURE_PATH, f"exceedance_plot_run{current_run_id}.pdf"
            ),
        )

        print(f"--- Completed Run {run + 1} ---")

    # 7. Plot all emulator exceedance curves together
    if N_RUNS > 0 and y_true_reference is not None:
        plot_all_emulator_exceedance_curves(
            y_true_reference,
            all_emulator_predictions,
            N_RUNS,
            save_path=os.path.join(FIGURE_PATH, "all_emulator_exceedance_curves.pdf"),
        )

    print(
        "\nExperiment finished. Outputs are saved in the 'experiment_outputs' directory."
    )
