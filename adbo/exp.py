"""Run BayesOpt experiments"""

from typing import Callable
import os
import math
import numpy as np
import time

start_tf_import = time.time()
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
from trieste.acquisition import (
    # ExpectedImprovement,
    MinValueEntropySearch,
)
from trieste.objectives import SingleObjectiveTestProblem
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.experimental.plotting.plotting import plot_bo_points
from trieste.objectives.single_objectives import check_objective_shapes

end_tf_import = time.time()
print("tf import time", end_tf_import - start_tf_import)  # takes about 270 seconds
from sithom.time import timeit
from sithom.plot import plot_defaults
from sithom.io import write_json
import matplotlib.pyplot as plt
from adforce.wrap import run_wrapped, select_point_f
from src.constants import NEW_ORLEANS
from .rescale import rescale_inverse

plot_defaults()

ROOT: str = "/work/n01/n01/sithom/adcirc-swan/"  # ARCHER2 path


@timeit
def setup_tf(seed: int = 1793, log_name: str = "experiment1") -> None:
    """
    Set up the tensorflow environment.

    Args:
        seed (int, optional): Random seed for numpy/tensorflow. Defaults to 1793.
        log_name (str, optional): Name for the log. Defaults to "experiment1".
    """
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
def objective_f(
    config: dict, exp_name: str = "bo_test", stationid: int = 3, wrap_test: bool = False
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Return a wrapper function for the ADCIRC model that is compatible with being used as an observer in trieste after processing.

    At each objective function call the model is run and the result is returned and saved.

    TODO: add wandb logging option.

    Args:
        config (dict): Dictionary with the constraints for the model.
        exp_name (str, optional): Name for folder. Defaults to "bo_test".
        stationid (int, optional): Which coast tidal gauge to sample near. Defaults to 3.
        wrap_test (bool, optional): If True, do not run the ADCIRC model. Defaults to False.

    Returns:
        Callable[[tf.Tensor], tf.Tensor]: trieste observer function.
    """
    # set up folder for all experiments
    exp_dir = os.path.join(ROOT, "exp", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    call_number = -1
    output = {}
    select_point = select_point_f(stationid)
    dimension_inputs = len(config["order"])
    print("dimension_inputs", dimension_inputs)

    def temp_dir() -> str:
        nonlocal exp_dir
        tmp_dir = os.path.join(exp_dir, f"exp_{call_number:04}")
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def add_query_to_output(real_query: tf.Tensor, real_result: tf.Tensor) -> None:
        nonlocal output
        output[call_number] = {
            "dir": temp_dir(),
            "res": float(real_result),
            **{name: float(real_query[j]) for j, name in enumerate(config["order"])},
        }
        write_json(output, os.path.join(exp_dir, "experiments.json"))

    @check_objective_shapes(d=dimension_inputs)
    def obj(x: tf.Tensor) -> tf.Tensor:
        """
        Run the ADCIRC model and return the result.

        Args:
            x (tf.Tensor): Possibly a batch of scaled queries.

        Returns:
            tf.Tensor: The negative of the result of the ADCIRC model at the selected point.
        """
        nonlocal call_number, select_point
        # put in real space
        returned_results = []  # new results, negative height [m]
        real_queries = rescale_inverse(x, config)  # convert to real space
        for i in range(real_queries.shape[0]):
            call_number += 1
            tmp_dir = temp_dir()
            inputs = {
                name: float(real_queries[i][j])
                for j, name in enumerate(config["order"])
            }
            if "displacement" in inputs:
                inputs["impact_lon"] = NEW_ORLEANS.lon + inputs["displacement"]
                del inputs["displacement"]
            if wrap_test:
                real_result = min(7 + np.random.normal(), 10)
            else:
                real_result = run_wrapped(
                    out_path=tmp_dir, select_point=select_point, **inputs
                )

            add_query_to_output(real_queries[i], real_result)
            # flip sign to make it a minimisation problem
            returned_results.append([-real_result])

        return tf.constant(returned_results, dtype=tf.float64)
        # Dataset(
        #    query_points=x, observations=tf.constant(returned_results, dtype=tf.float64)
        # )
        # run the model
        # return the result

    return obj


DEFAULT_CONSTRAINTS = {
    "angle": {"min": -80, "max": 80, "units": "degrees"},
    "trans_speed": {"min": 0, "max": 15, "units": "m/s"},
    "displacement": {"min": -2, "max": 2, "units": "degrees"},
    "order": ("angle", "trans_speed", "displacement"),  # order of input features
}


@timeit
def gp_model_out_callback(datasets, gp_models, state) -> bool:
    # use the early_stop_callback to save the GP at each step
    print("gp_model_out_callback", gp_models)
    for i, model in enumerate(gp_models):
        print(i, model, type(model))
        print(i, gp_models[model], type(gp_models[model]))
    #  model.save(os.path.join(f"gp_model_{i}.h5"))
    return False


@timeit
def run_bayesopt_exp(
    constraints: dict = DEFAULT_CONSTRAINTS,
    seed: int = 10,
    exp_name: str = "bo_test",
    init_steps: int = 10,
    daf_steps: int = 10,
    wrap_test: bool = False,
) -> None:
    """
    Run a Bayesian Optimisation experiment.

    Args:
        seed (int, optional): Seed to initialize. Defaults to 10.
        exp_name (str, optional): Experiment name. Defaults to "bo_test".
        init_steps (int, optional): How many sobol sambles. Defaults to 10.
        daf_steps (int, optional): How many acquisition points. Defaults to 10.
        wrap_test (bool, optional): Whether to prevent. Defaults to False.
    """
    setup_tf(seed=seed, log_name=exp_name)

    # set up BayesOpt
    dimensions_input = len(constraints["order"])
    search_space = trieste.space.Box([0] * dimensions_input, [1] * dimensions_input)
    initial_query_points = search_space.sample_sobol(init_steps)
    print("initial_query_points", initial_query_points, type(initial_query_points))
    init_objective = objective_f(constraints, exp_name=exp_name, wrap_test=wrap_test)
    put_through_sotp = False

    if put_through_sotp:
        obs_class = SingleObjectiveTestProblem(
            name="adcirc35k",
            search_space=search_space,
            objective=init_objective,
            minimizers=tf.constant(
                [[0.114614, 0.555649, 0.852547]], tf.float64
            ),  # what does the minimizer do?
            minimum=tf.constant([-10], tf.float64),
        )
        print("obs_class", obs_class, type(obs_class))
        observer = trieste.objectives.utils.mk_observer(obs_class.objective)
    else:
        # observer = trieste.objectives.utils.mk_observer(obs_class.objective)
        observer = trieste.objectives.utils.mk_observer(init_objective)
    print("observer", observer, type(observer))
    initial_data = observer(initial_query_points)
    print("initial_data", initial_data, type(initial_data))
    gpr = trieste.models.gpflow.build_gpr(initial_data, search_space)
    model = trieste.models.gpflow.GaussianProcessRegression(gpr)
    acquisition_rule = EfficientGlobalOptimization(MinValueEntropySearch(search_space))
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(
        daf_steps,
        initial_data,
        model,
        acquisition_rule,
        track_state=True,  # there was some issue with this on mac
        early_stop_callback=gp_model_out_callback,
    ).astuple()
    trieste.logging.set_summary_filter(lambda name: True)  # enable all summaries
    # print("result", result)
    # print("history", history)
    real_res = result[0].unwrap()
    dataset = real_res.dataset
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    # plot the results
    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_bo_points(
        query_points,
        ax[0, 0],
        5,
        m_init="o",
        m_add="+",  # obs_values=observations
    )  # , arg_min_idx)
    ax[0, 0].scatter(
        query_points[:, 0], query_points[:, 1], c=observations, cmap="viridis"
    )
    ax[0, 0].set_xlabel(r"$x_1$ [dimensionless]")
    ax[0, 0].set_ylabel(r"$x_2$ [dimensionless]")
    plt.savefig(os.path.join("img", exp_name + "_mves.png"))


if __name__ == "__main__":
    # run_bayesopt_exp(seed=12, exp_name="bo_test5", init_steps=5, daf_steps=35)
    # run_bayesopt_exp(seed=13, exp_name="bo_test8", init_steps=5, daf_steps=35)
    # python -m adbo.exp &> logs/bo_test10.log
    # python -m adbo.exp
    # python -m adbo.exp &> logs/exp.log
    # run_bayesopt_exp(seed=14, exp_name="bo_test10", init_steps=5, daf_steps=50)
    # python -m adbo.exp &> logs/test15.log
    # run_bayesopt_exp(seed=15, exp_name="bo_test11", init_steps=1, daf_steps=50)
    # run_bayesopt_exp(seed=15, exp_name="test12", init_steps=1, daf_steps=50)
    constraints_2d = {
        "angle": {"min": -80, "max": 80, "units": "degrees"},
        "displacement": {"min": -2, "max": 2, "units": "degrees"},
        "order": ("angle", "displacement"),  # order of input features
    }
    run_bayesopt_exp(
        seed=15,
        constraints=constraints_2d,
        exp_name="test21",
        init_steps=5,
        daf_steps=50,
        wrap_test=True,
    )
    # python -m adbo.exp &> logs/test21.log
    # run_bayesopt_exp(seed=16, exp_name="bo_test16", init_steps=5, daf_steps=50)
    #  python -m adbo.exp &> logs/bo_test17.log
    # run_bayesopt_exp(seed=18, exp_name="bo_test18", init_steps=5, daf_steps=100)
