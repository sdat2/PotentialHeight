"""GP experiment module: check the gains from different kernels, and\or different prior mean functions."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from sithom.time import timeit
from sithom.plot import plot_defaults, label_subplots, get_dim
from worst.utils import retry_wrapper
from .constants import DATA_PATH, EXP_PATH

# from tf.keras.metrics import R2Score, RootMeanSquaredError


SAVE_DATA_PATH = os.path.join(DATA_PATH, "gpr_data.csv")
OUTPUT_COLUMNS = [
    "norm_x",
    "norm_y",
    "kernel",
    "mean_function",
    "log_likelihood",
    "log_likelihood_sem",
    "rmse",
    "rmse_sem",
    "r2",
    "r2_sem",
]
INPUT_COLUMNS = ["angle", "angle", "displacement", "trans_speed", "res"]


def gather_data(columns=INPUT_COLUMNS) -> None:
    """Gather data from existing experiments.

    Get all the LHS samples from the 3D experiments.

    Save it all as a large csv file
    x(displacement, bearing, angle), z (maximum storm height).
    """
    print(f"Gathering data from {EXP_PATH} to {SAVE_DATA_PATH}.")
    df = pd.DataFrame(columns=columns)  # create empty dataframe
    for i, b in [(1, 49), (3, 47), (25, 25), (50, 0)]:
        for t in range(10):
            if t == 0:
                name = f"i{i}b{b}"
            else:
                name = f"i{i}b{b}t{t}"
            exp_file_path = os.path.join(EXP_PATH, name, "experiments.json")
            if os.path.exists(exp_file_path):
                tmp_df = pd.read_json(exp_file_path, orient="index").iloc[
                    :i
                ]  # rows come from "0","1",…
                # drop the stray column whose name is the empty string
                tmp_df = tmp_df.drop(columns=[""])
                # put the columns in the order you prefer
                tmp_df = tmp_df[columns]

                df = pd.concat([df, tmp_df], ignore_index=True)

    # remove duplicates
    df = df.drop_duplicates()
    # remove rows with NaN values
    df = df.dropna()
    df.to_csv(SAVE_DATA_PATH, index=False)  # create empty csv file


def load_data(data_path: str = SAVE_DATA_PATH) -> pd.DataFrame:
    """Load data from a csv file.
    Args:
        data_path (str): Path to the csv file.
    Returns:
        pd.DataFrame: Dataframe containing the data.
    """
    if not os.path.exists(data_path):
        gather_data()
    return pd.read_csv(data_path)


def log_likelihood(
    model: gpflow.models.GPR, x_test: np.ndarray, y_test: np.ndarray, noisy: bool = True
) -> float:
    """Return the mean of predictive log-likelihood on a test set.

    Args:
        model (gpflow.models.GPR): The GP model.
        x_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test output data.
        noisy (bool, optional): Whether to include noise in the prediction. Defaults to True.

    Returns:
        float: The average log-likelihood.
    """
    if noisy:
        mu, var = model.predict_y(x_test)  # includes likelihood noise
    else:
        mu, var = model.predict_f(x_test)  # latent/noise-free

    dist = tfp.distributions.Normal(loc=mu, scale=tf.sqrt(var))
    logdens = dist.log_prob(y_test)  # shape (N, 1)
    return tf.reduce_mean(logdens)  # or tf.reduce_mean(logdens) .reduce_sum(logdens)


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str,
    mean_function: str,
) -> gpflow.models.GPR:
    """Fit a GP model to the data.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        kernel (str): Kernel to use.
        mean_function (str): Mean function to use.

    Returns:
        gpflow.models.GPR: Fitted GP model.
    """
    if kernel == "Matern52":
        k = gpflow.kernels.Matern52()
    elif kernel == "Matern12":
        k = gpflow.kernels.Matern12()
    elif kernel == "SE" or kernel == "SquaredExponential" or kernel == "RBF":
        k = gpflow.kernels.SquaredExponential()
    elif kernel == "RationalQuadratic":
        k = gpflow.kernels.RationalQuadratic()
    elif kernel == "Exponential":
        k = gpflow.kernels.Exponential()
    elif kernel == "Linear":
        k = gpflow.kernels.Linear()
    elif kernel == "Periodic":
        k = gpflow.kernels.Periodic()
    elif kernel == "Polynomial":
        k = gpflow.kernels.Polynomial()
    elif kernel == "Constant":
        k = gpflow.kernels.Constant()
    elif kernel == "White":
        k = gpflow.kernels.White()
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    if mean_function == "Constant":
        mean_function = gpflow.mean_functions.Constant()
    elif mean_function == "Linear":
        mean_function = gpflow.mean_functions.Linear()
    elif mean_function == "Polynomial":
        mean_function = gpflow.mean_functions.Polynomial()
    elif mean_function == "Zero":
        mean_function = gpflow.mean_functions.Zero()
    elif mean_function == "Exponential":
        mean_function = gpflow.mean_functions.Exponential()
    elif mean_function == "Periodic":
        mean_function = gpflow.mean_functions.Periodic()
    elif mean_function == "RationalQuadratic":
        mean_function = gpflow.mean_functions.RationalQuadratic()
    elif mean_function == "LinearPeriodic":
        mean_function = gpflow.mean_functions.LinearPeriodic()
    elif mean_function == "LinearExponential":
        mean_function = gpflow.mean_functions.LinearExponential()
    elif mean_function == "LinearRationalQuadratic":
        mean_function = gpflow.mean_functions.LinearRationalQuadratic()
    elif mean_function == "LinearWhite":
        mean_function = gpflow.mean_functions.LinearWhite()
    elif mean_function == "LinearConstant":
        mean_function = gpflow.mean_functions.LinearConstant()
    elif mean_function == "LinearPolynomial":
        mean_function = gpflow.mean_functions.LinearPolynomial()
    elif mean_function == "LinearGaussian":
        mean_function = gpflow.mean_functions.LinearGaussian()
    elif mean_function == "LinearGaussianPeriodic":
        mean_function = gpflow.mean_functions.LinearGaussianPeriodic()
    elif mean_function == "LinearGaussianRationalQuadratic":
        mean_function = gpflow.mean_functions.LinearGaussianRationalQuadratic()
    elif mean_function == "LinearGaussianWhite":
        mean_function = gpflow.mean_functions.LinearGaussianWhite()
    elif mean_function == "LinearGaussianConstant":
        mean_function = gpflow.mean_functions.LinearGaussianConstant()
    elif mean_function == "LinearGaussianPolynomial":
        mean_function = gpflow.mean_functions.LinearGaussianPolynomial()
    elif mean_function == "LinearGaussianExponential":
        mean_function = gpflow.mean_functions.LinearGaussianExponential()
    elif mean_function == "LinearGaussianPeriodicExponential":
        mean_function = gpflow.mean_functions.LinearGaussianPeriodicExponential()
    else:
        raise ValueError(f"Unknown mean function: {mean_function}")

    model = gpflow.models.GPR(data=(x, y), kernel=k, mean_function=mean_function)

    # model fit
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        method="L-BFGS-B",
        options={"maxiter": 100},
    )

    return model


@timeit
@retry_wrapper(max_retries=10)
def run_single_fit(
    norm_x: bool = True,
    norm_y: bool = True,
    seed: int = 42,
    n_train: int = 25,
    kernel: str = "Matern52",
    mean_function: str = "Constant",
) -> None:
    """Run a single fit of the GP model.

    Args:
        norm_x (bool, optional): Whether to normalize x. Defaults to True.
        norm_y (bool, optional): Whether to normalize y. Defaults to True.
        kernel (str, optional): Kernel to use. Defaults to "Matern52".
        mean_function (str, optional): Mean function to use. Defaults to "Constant".
    """

    df = load_data()
    # scale data
    x = df[INPUT_COLUMNS[:-1]].values
    y = df[INPUT_COLUMNS[-1]].values.reshape(-1, 1)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    if norm_x:
        x = (x - x_mean) / x_std
    if norm_y:
        y = (y - y_mean) / y_std
    # x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    # y = (y - np.mean(y)) / np.std(y)

    # split data into training and test sets
    # n_train = int(0.8 * len(x))
    print(f"Training on {n_train} samples, testing on {len(x) - n_train} samples.")
    # shuffle data
    idx = np.arange(len(x))
    np.random.seed(seed)
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    # split data
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # fit GP model
    model = fit_gp(x_train, y_train, kernel=kernel, mean_function=mean_function)

    # evaluate model
    ll = log_likelihood(model, x_test, y_test)
    y_pred, var = model.predict_y(x_test)  # includes likelihood noise
    rmse = np.sqrt(
        np.mean((y_pred - y_test) ** 2)
    )  # np.sqrt(mean_squared_error(y_test, model.predict(x_test)))
    if norm_y:
        rmse = rmse * y_std  # unscale rmse
        ll = ll - tf.math.log(y_std)  # unscale log likelihood
    r2 = 1 - (
        np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    )  # 1 - (mean_squared_error(y_test, model.predict(x_test)) / np.var(y_test))
    print(f"Log likelihood: {ll.numpy()}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    # save results
    return pd.DataFrame(
        {
            "norm_x": [norm_x],
            "norm_y": [norm_y],
            "kernel": [kernel],
            "mean_function": [mean_function],
            "log_likelihood": [ll.numpy()],
            "rmse": [rmse],
            "r2": [r2],
        }
    )


@timeit
def run_exp(
    kernels: tuple = ("Matern52", "Matern12", "SE", "RationalQuadratic"),
    mean_functions: tuple = (
        "Constant",
        "Linear",
        "Polynomial",
        "Zero",
        "Exponential",
        "Periodic",
    ),
    n_train: int = 25,
    repeats: int = 100,
) -> None:
    """Run the experiment.

    - Scale the data.
    - Split the data into training and test sets.
    - Fit the GP model to the training set.
    - Evaluate the model on the test set (log liklihood, r2, rmse).
    - Save the results in a csv file and a tex file.

    Args:
        kernels (tuple, optional): Kernels to use. Defaults to ("Matern52", "Matern12", "SE", "RationalQuadratic").
        mean_functions (tuple, optional): Mean functions to use. Defaults to ("Constant", "Linear", "Polynomial", "Zero", "Exponential", "Periodic").
        n_train (int, optional): Number of training samples. Defaults to 25.
        repeats (int, optional): Number of repeats for each experiment. Defaults to 100.

    """
    results = pd.DataFrame(
        columns=[
            "norm_x",
            "norm_y",
            "kernel",
            "mean_function",
            "log_likelihood",
            "log_likelihood_sem",
            "rmse",
            "rmse_sem",
            "r2",
            "r2_sem",
        ]
    )
    for norm_x in [True, False]:
        for norm_y in [True, False]:
            for kernel in kernels:
                for mean_function in mean_functions:
                    # empty tmp dataframe
                    tmp_results = pd.DataFrame(columns=OUTPUT_COLUMNS)
                    for seed in range(repeats):
                        print(
                            f"Running experiment with norm_x={norm_x}, norm_y={norm_y}, kernel={kernel}, mean_function={mean_function}, seed={seed}, n_train={n_train}"
                        )
                        tmp_results = pd.concat(
                            [
                                tmp_results,
                                run_single_fit(
                                    seed=seed,
                                    norm_x=norm_x,
                                    norm_y=norm_y,
                                    kernel=kernel,
                                    mean_function=mean_function,
                                    n_train=n_train,
                                ),
                            ],
                            ignore_index=True,
                        )
                    # Store mean and SEM for each metric
                    results = pd.concat(
                        [
                            results,
                            pd.DataFrame(
                                {
                                    "norm_x": [norm_x],
                                    "norm_y": [norm_y],
                                    "kernel": [kernel],
                                    "mean_function": [mean_function],
                                    "log_likelihood": [
                                        tmp_results["log_likelihood"].mean()
                                    ],
                                    "log_likelihood_sem": [
                                        tmp_results["log_likelihood"].std(ddof=1)
                                        / np.sqrt(len(tmp_results))
                                    ],
                                    "rmse": [tmp_results["rmse"].mean()],
                                    "rmse_sem": [
                                        tmp_results["rmse"].std(ddof=1)
                                        / np.sqrt(len(tmp_results))
                                    ],
                                    "r2": [tmp_results["r2"].mean()],
                                    "r2_sem": [
                                        tmp_results["r2"].std(ddof=1)
                                        / np.sqrt(len(tmp_results))
                                    ],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

    # save results
    results.to_csv(
        os.path.join(DATA_PATH, f"gp_results_n_train_{n_train}.csv"),
        index=False,
    )
    # save results in LaTeX table
    save_results_tex(n_train=n_train)


def save_results_tex(n_train: int = 25):
    """Save the results in a LaTeX table.

    Args:
        n_train (int, optional): Number of training samples. Defaults to 25.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, f"gp_results_n_train_{n_train}.csv"))

    # --- format <mean ± SEM> strings for LaTeX output, bolding the best ---

    # nice kernel names for LaTeX
    df["kernel"] = (
        df["kernel"]
        .str.replace("Matern52", r"Mat\'ern-\({5}/{2}\)")
        .str.replace("Matern32", r"Mat\'ern-\({3}/{2}\)")
        .str.replace("Matern12", r"Mat\'ern-\({1}/{2}\)")
        .str.replace("SE", "SE")
    )

    # find best values
    best_vals = {
        "log_likelihood": df["log_likelihood"].max(),  # highest is best
        "rmse": df["rmse"].min(),  # lowest is best
        "r2": df["r2"].max(),  # highest is best
    }

    # helper: format mean ± sem, bold if mean equals best (allow tiny tol)
    def fmt_val(metric: str, mean: float, sem: float) -> str:
        base = f"{mean:.3f} \\(\\pm\\) {sem:.3f}"
        if abs(mean - best_vals[metric]) < 1e-9:
            return f"\\textbf{{{mean:.3f}}} \\(\\pm\\)  \\textbf{{{sem:.3f}}}"
        return base

    for metric in ("log_likelihood", "rmse", "r2"):
        sem_col = metric + "_sem"
        df[metric] = df.apply(
            lambda row, m=metric: fmt_val(m, row[m], row[sem_col]), axis=1
        )
        df = df.drop(columns=[sem_col])

    # rename columns for LaTeX header
    df = df.rename(
        columns={
            "log_likelihood": r"\(\bar{\mathcal{L}}\) ",
            "rmse": "RMSE [m]",
            "r2": r"\(r^2\)",
            "kernel": "Kernel",
            "mean_function": "Mean F.",
            "norm_x": r"Norm. \(x\)",
            "norm_y": r"Norm. \(y\)",
        }
    )

    # write LaTeX table with df strings
    df.to_latex(
        os.path.join(DATA_PATH, f"gp_results_n_train_{n_train}.tex"),
        index=False,
        escape=False,
    )
    return  # end of run_exp


if __name__ == "__main__":
    # python -m adbo.gp_exp &> gp_exp.log
    # gather_data()
    # run_exp(kernels=("Matern52",), mean_functions=("Constant",), n_train=25)
    # save_results_tex(n_train=50)  # 100)

    run_exp(
        kernels=(
            "Matern52",
            "Matern32",
            "Matern12",
            "SE",
            # "RationalQuadratic",
            "Exponential",
            # "Linear",
            # "Periodic",
            "Polynomial",
            # "Constant",
            # "White",
        ),
        mean_functions=(
            "Constant",
            # "Linear",
            # "Polynomial",
            "Zero",
            # "Exponential",
            # "Periodic",
        ),
        n_train=100,
    )
