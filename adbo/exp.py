"""Run BayesOpt experiments"""

from typing import Callable
import os
import math
import numpy as np
import xarray as xr
import time

start_tf_import = time.time()
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import trieste
from trieste.acquisition import (
    # ExpectedImprovement,
    MinValueEntropySearch,
)
from trieste.objectives import SingleObjectiveTestProblem
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.experimental.plotting.plotting import plot_bo_points, plot_regret
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
    Set up the tensorflow environment by seeding and setting up logging/tensorboard.

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
        """
        Return a temporary directory for the current experiment.

        Returns:
            str: Temporary directory based on the call number.
        """
        nonlocal exp_dir
        tmp_dir = os.path.join(exp_dir, f"exp_{call_number:04}")
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def add_query_to_output(real_query: tf.Tensor, real_result: tf.Tensor) -> None:
        nonlocal output
        output[call_number] = {
            "": temp_dir(),
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


DEFAULT_CONSTRAINTS: dict = {
    "angle": {"min": -80, "max": 80, "units": "degrees"},
    "trans_speed": {"min": 0, "max": 15, "units": "m/s"},
    "displacement": {"min": -2, "max": 2, "units": "degrees"},
    "order": ("angle", "trans_speed", "displacement"),  # order of input features
}


def gp_model_callback_maker(
    direc: str, dimension: int = 2
) -> Callable[[any, any, any], bool]:
    """
    Return a callback function that saves the GP model at each step.

    Args:
        direc (str): Directory to save the models.
        dimension (int, optional): Number of input dimensions. Defaults to 2.

    Returns:
        Callable[[any, any, any], bool]: Callback function for early_stop_callback.
    """
    # https://github.com/secondmind-labs/trieste/blob/develop/trieste/models/gpflow/models.py
    os.makedirs(direc, exist_ok=True)
    # saver = gpflow.saver.Saver()
    call: int = 0

    n = 100
    x1 = np.linspace(0, 1, num=n)
    x2 = np.linspace(0, 1, num=n)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.flatten(), X2.flatten()])
    ypred_list = []
    yvar_list = []

    @timeit
    def gp_model_callback(datasets, gp_models, state) -> bool:
        """
        Save the GP model at each step.

        Args:
            datasets (any): The datasets.
            gp_models (any): The GP models.
            state (any): The state.

        Returns:
            bool: Whether to stop the optimization.
        """
        # could either save the whole model or just the predictions at particular points.
        nonlocal call, direc, n, x1, x2, X1, X2, X, ypred_list, yvar_list
        call += 1  # increment the call number

        print("dimension", dimension)
        print("datasets", datasets)
        print("state", state)
        # use the early_stop_callback to save the GP at each step
        print("gp_model_out_callback", gp_models)
        for i, model in enumerate(gp_models):
            print(i, model, type(model))
            print(i, gp_models[model], type(gp_models[model]))
            print(
                "gp_models[model].model", gp_models[model].model
            )  # .save(f"gp_model_{i}.h5")
            ckpt = tf.train.Checkpoint(model=gp_models[model].model)
            manager = tf.train.CheckpointManager(ckpt, direc, max_to_keep=100)
            manager.save()
            # plt.show()
            plt.clf()
            plt.close()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            Y, Yvar = gp_models[model].predict_y(X)
            Y, Yvar = np.reshape(Y, (n, n)), np.reshape(Yvar, (n, n))
            ypred_list.append(Y)
            yvar_list.append(Yvar)
            print("np.array(ypred_list))", np.array(ypred_list).shape)
            print("np.array(yvar_list))", np.array(yvar_list).shape)
            print(
                "np.array([x + 1 for x in range(call)]))",
                [x + 1 for x in range(len(ypred_list))],
            )
            xr.Dataset(
                data_vars={
                    "ypred": (("call", "x1", "x2"), np.array(ypred_list)),
                    "yvar": (("call", "x1", "x2"), np.array(yvar_list)),
                },
                coords={
                    "x1": x1,
                    "x2": x2,
                    "call": [x + 1 for x in range(len(ypred_list))],
                },
            ).to_netcdf(os.path.join(direc, f"gp_model_{call}.nc"))

            im = axs[0].contourf(X1, X2, Y, levels=1000)
            # add colorbar to the plot with the right scale and the same size as the plot
            fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
            # axs[0].colorbar()
            axs[0].set_title("Mean")
            im = axs[1].contourf(X1, X2, np.sqrt(Yvar), levels=1000)
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
            # axs[1].colorbar()
            axs[1].set_title("Std. Dev., $\sigma$")
            axs[0].set_xlabel("x$_1$")
            axs[0].set_ylabel("x$_2$")
            axs[1].set_xlabel("x$_1$")
            # plt.show()
            plt.clf()
            plt.close()
            #  model.save(os.path.join(f"gp_model_{i}.h5"))
            # saver.save(os.path.join(direc, f"gp_model_{call}"), model)

        return False  # False means don't stop

    return gp_model_callback


@timeit
def run_bayesopt_exp(
    constraints: dict = DEFAULT_CONSTRAINTS,
    seed: int = 10,
    exp_name: str = "bo_test",
    direc: str = "exp",
    stationid: int = 3,
    init_steps: int = 10,
    daf_steps: int = 10,
    wrap_test: bool = False,
) -> None:
    """
    Run a Bayesian Optimisation experiment.

    Args:
        constraints (dict, optional): Dictionary with the constraints for the optimization. Defaults to DEFAULT_CONSTRAINTS.
        seed (int, optional): Seed to initialize. Defaults to 10.
        exp_name (str, optional): Experiment name. Defaults to "bo_test".
        init_steps (int, optional): How many sobol sambles. Defaults to 10.
        daf_steps (int, optional): How many acquisition points. Defaults to 10.
        wrap_test (bool, optional): Whether to prevent. Defaults to False.
    """
    os.makedirs(direc, exist_ok=True)
    setup_tf(seed=seed, log_name=exp_name)

    # set up BayesOpt
    dimensions_input = len(constraints["order"])
    search_space = trieste.space.Box([0] * dimensions_input, [1] * dimensions_input)
    initial_query_points = search_space.sample_sobol(init_steps)
    print("initial_query_points", initial_query_points, type(initial_query_points))
    init_objective = objective_f(
        constraints, stationid=stationid, exp_name=exp_name, wrap_test=wrap_test
    )
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
        early_stop_callback=gp_model_callback_maker(
            exp_name, dimension=dimensions_input
        ),
    ).astuple()
    trieste.logging.set_summary_filter(lambda name: True)  # enable all summaries
    # print("result", result)
    # print("history", history)
    real_res = result[0].unwrap()
    dataset = real_res.dataset
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    xr.Dataset(
        data_vars={
            "x1": (("call"), query_points[:, 0]),
            "x2": (("call"), query_points[:, 1]),
            "y": (("call"), observations.flatten()),
        },
        coords={"call": [x + 1 for x in range(len(observations))]},
    ).to_netcdf(os.path.join(direc, exp_name + "_mves.nc"))

    # plot the results
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_bo_points(
        query_points,
        ax,
        5,
        m_init="o",
        m_add="+",  # obs_values=observations
    )  # , arg_min_idx)
    ax.scatter(query_points[:, 0], query_points[:, 1], c=observations, cmap="viridis")
    ax.set_xlabel(r"$x_1$ [dimensionless]")
    ax.set_ylabel(r"$x_2$ [dimensionless]")
    plt.savefig(os.path.join("img", exp_name + "_mves.png"))
    # plt.show()
    plt.clf()
    plt.close()

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_regret(
        dataset.observations.numpy(),
        ax,
        num_init=5,
        show_obs=True,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    plt.savefig(os.path.join("img", exp_name + "_regret.png"))
    # plt.show()
    plt.clf()
    plt.close()


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
        seed=13,
        constraints=constraints_2d,
        exp_name="bo-test-2d-2",
        init_steps=5,
        daf_steps=50,
        wrap_test=False,
    )
    # python -m adbo.exp &> logs/test32.log
    # run_bayesopt_exp(seed=16, exp_name="bo_test16", init_steps=5, daf_steps=50)
    #  python -m adbo.exp &> logs/bo_test17.log
    # run_bayesopt_exp(seed=18, exp_name="bo_test18", init_steps=5, daf_steps=100)
