"""Run BayesOpt experiments"""

from typing import Callable
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
from trieste.acquisition import (
    # ExpectedImprovement,
    MinValueEntropySearch,
)
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.experimental.plotting.plotting import plot_bo_points, plot_function_2d
from sithom.time import timeit
from sithom.plot import plot_defaults
from sithom.io import write_json
import matplotlib.pyplot as plt
from adforce.wrap import run_wrapped
from src.constants import NEW_ORLEANS
from .rescale import rescale_inverse

plot_defaults()

ROOT: str = "/work/n01/n01/sithom/adcirc-swan/"


@timeit
def setup_tf(seed: int = 1793, log_name: str = "experiment1") -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print("trieste.__version__", trieste.__version__)
    print("tf.__version__", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.debugging.set_log_device_placement(True)
    TENSORBOARD_LOG_DIR = os.path.join("logs", "tensorboard", log_name)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    files = os.listdir(TENSORBOARD_LOG_DIR)
    for file in files:
        os.remove(os.path.join(TENSORBOARD_LOG_DIR, file))
    summary_writer = tf.summary.create_file_writer(TENSORBOARD_LOG_DIR)
    trieste.logging.set_tensorboard_writer(summary_writer)
    print("summary_writer", summary_writer)


@timeit
def observer_f(config: dict, exp_name: str = "bo_test") -> Callable:
    # set up folder for all experiments
    exp_dir = os.path.join(ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    call_number = -1
    output = {}

    def temp_dir() -> str:
        nonlocal exp_dir
        tmp_dir = os.path.join(exp_dir, f"exp_{call_number:04}")
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def add_query_to_output(real_query: np.ndarray, real_result: tf.Tensor) -> None:
        nonlocal output
        output[call_number] = {
            "dir": temp_dir(),
            "res": float(real_result),
            **{name: float(real_query[j]) for j, name in enumerate(config["order"])},
        }
        write_json(output, os.path.join(exp_dir, "experiments.json"))

    def obs(queries: tf.Tensor) -> tf.Tensor:
        nonlocal call_number
        # put in real space
        returned_results = []
        real_queries = rescale_inverse(queries, config)
        for i in range(real_queries.shape[0]):
            call_number += 1
            tmp_dir = temp_dir()
            inputs = {
                name: float(real_queries[i][j])
                for j, name in enumerate(config["order"])
            }
            inputs["impact_lon"] = NEW_ORLEANS.lon + inputs["displacement"]
            del inputs["displacement"]
            real_result = run_wrapped(out_path=tmp_dir, **inputs)
            add_query_to_output(real_queries[i], real_result)
            # flip sign to make it a minimisation problem
            returned_results.append(-real_result)

        return tf.constant(returned_results)
        # run the model
        # return the result

    return obs


@timeit
def run_bayesopt_exp(
    seed: int = 10, exp_name: str = "bo_test", init_steps: int = 10, daf_steps: int = 10
) -> None:
    setup_tf(seed=seed, log_name=exp_name)
    constraints_d = {
        "angle": {"min": -80, "max": 80, "units": "degrees"},
        "trans_speed": {"min": 0, "max": 15, "units": "m/s"},
        "displacement": {"min": -2, "max": 2, "units": "degrees"},
        "order": ("angle", "trans_speed", "displacement"),
    }
    # set up BayesOpt
    search_space = trieste.space.Box([0, 0, 0], [1, 1, 1])
    init_steps = 10
    initial_query_points = search_space.sample_sobol(init_steps)
    observer = observer_f(constraints_d, exp_name=exp_name)
    initial_data = observer(initial_query_points)
    gpr = trieste.models.gpflow.build_gpr(initial_data, search_space)
    model = trieste.models.gpflow.GaussianProcessRegression(gpr)
    acquisition_rule = EfficientGlobalOptimization(MinValueEntropySearch(search_space))
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    daf_steps = 10
    result = bo.optimize(
        daf_steps,
        initial_data,
        model,
        acquisition_rule,
        track_state=False,
    ).astuple()
    trieste.logging.set_summary_filter(lambda name: True)  # enable all summaries
    # print("result", result)
    # print("history", history)
    real_res = result[0].unwrap()
    dataset = real_res.dataset
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()
    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_bo_points(
        query_points,
        ax[0, 0],
        5,
        m_add="+",  # obs_values=observations
    )  # , arg_min_idx)
    plt.scatter(query_points[:, 0], query_points[:, 1], c=observations, cmap="viridis")
    ax[0, 0].set_xlabel(r"$x_1$")
    ax[0, 0].set_xlabel(r"$x_2$")
    plt.savefig(os.path.join("img", "bo_test5_mves.png"))


if __name__ == "__main__":
    # run_bayesopt_exp(seed=12, exp_name="bo_test5", init_steps=5, daf_steps=35)
    run_bayesopt_exp(seed=13, exp_name="bo_test6", init_steps=5, daf_steps=35)
    # python -m adbo.exp &> logs/bo_test6.log
    # python -m adbo.exp
    # python -m adbo.exp &> logs/exp.log
