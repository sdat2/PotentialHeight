"""Run BayesOpt experiments.

Should contain all tensorflow imports so that other scripts
are easier to use/test without waiting for tensorflow to load.
"""

from typing import Callable, Optional, List, Tuple
import os
import copy
import json
import yaml
import numpy as np
import xarray as xr
import matplotlib
import time

start_tf_import = time.time()
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import trieste
from trieste.acquisition import (
    ExpectedImprovement,
    MinValueEntropySearch,
    NegativeLowerConfidenceBound,
)
from omegaconf import OmegaConf
from trieste.data import Dataset
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.experimental.plotting.plotting import plot_bo_points, plot_regret
from trieste.objectives.single_objectives import check_objective_shapes

end_tf_import = time.time()
print(
    "tf import time", end_tf_import - start_tf_import
)  # takes about 270 seconds on ARCHER2 (?why?)
from sithom.time import timeit, time_stamp
from sithom.misc import get_git_revision_hash
from sithom.plot import plot_defaults
from sithom.io import write_json
import matplotlib.pyplot as plt
from adforce.wrap import idealized_tc_observe
from adforce.constants import NEW_ORLEANS
from .wrap_utils import build_wrap_config
from .ani import plot_gps
from .rescale import rescale_inverse
from .constants import FIGURE_PATH, EXP_PATH, CONFIG_PATH, PROJECT_PATH

# Environment variable to override the root experiment directory
# (follows the WORSTSURGE_DATA_ROOT convention in adforce/constants.py).
EXP_DIR_ENV_VAR = "WORSTSURGE_EXP_DIR"
# On-disk names used by the BO loop (see objective_f/add_query_to_output).
BO_LEDGER_NAME = "experiments.json"
BO_CONFIG_NAME = "bo-config.json"

# archer 2:
matplotlib.use("Agg")
plot_defaults()


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
    print("gpflow.__version__", gpflow.__version__)
    print("np.__version__", np.__version__)
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
    cfg: dict,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Return a wrapper function for the ADCIRC model that is compatible with being used as an observer in trieste after processing.

    At each objective function call the model is run and the result is returned and saved.

    Returns:
        Callable[[tf.Tensor], tf.Tensor]: trieste observer function.
    """
    call_number = -1
    output = {}
    dimension_inputs = len(cfg["constraints"]["order"])
    exp_dir = cfg["exp_dir"]
    print("dimension_inputs", dimension_inputs)

    # Size--intensity tradeoff mode: when "vmax" is a BO dimension, each
    # sampled intensity gets its own CLE15 profile generated on the potential
    # size curve r(V) (see w22.tradeoff), instead of the fixed profile_name.
    tradeoff_curve = None
    if "vmax" in cfg["constraints"]["order"]:
        # lazy import: pulls in the numba CLE15 kernels, only needed here
        from w22.tradeoff import TradeoffCurve

        if not cfg.get("curve_path"):
            raise ValueError(
                "constraints include 'vmax' but no curve_path was supplied; "
                "generate one with w22.tradeoff.generate_curve and pass it "
                "to run_bayesopt_exp(curve_path=...)."
            )
        tradeoff_curve = TradeoffCurve.from_file(cfg["curve_path"])
        print(
            f"objective_f: size-intensity tradeoff mode, curve "
            f"{cfg['curve_path']} (vmax in [{tradeoff_curve.v_min:.2f}, "
            f"{tradeoff_curve.v_max:.2f}] m/s gradient wind)"
        )

    # When resuming an interrupted experiment, reload the ledger of completed
    # evaluations so that (a) call numbering continues after the last completed
    # run instead of overwriting exp_0000 onwards, and (b) previously saved
    # records are preserved when experiments.json is rewritten.
    if cfg.get("resume", True):
        ledger_path = os.path.join(exp_dir, BO_LEDGER_NAME)
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, "r", encoding="utf-8") as ledger_file:
                    output = {int(k): v for k, v in json.load(ledger_file).items()}
            except (ValueError, OSError) as e:
                print(
                    f"Warning: could not reload ledger {ledger_path} ({e}); "
                    "starting the ledger afresh."
                )
                output = {}
            if output:
                call_number = max(output)
                print(
                    f"objective_f: resuming after call_number={call_number} "
                    f"({len(output)} saved evaluations in {ledger_path})"
                )

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
        """
        Add the query and result to the output dictionary.

        Args:
            real_query (tf.Tensor): The resecaled query.
            real_result (tf.Tensor): The result with the correct sign.
        """
        nonlocal output
        output[call_number] = {
            "": temp_dir(),
            "res": float(real_result),
            **{
                name: float(real_query[j])
                for j, name in enumerate(cfg["constraints"]["order"])
            },
        }
        write_json(output, os.path.join(exp_dir, BO_LEDGER_NAME))

    @check_objective_shapes(d=dimension_inputs)
    def obj(x: tf.Tensor) -> tf.Tensor:
        """
        Run the ADCIRC model and return the result.

        Args:
            x (tf.Tensor): Possibly a batch of scaled queries.

        Returns:
            tf.Tensor: The negative of the result of the ADCIRC model at the selected point.
        """
        nonlocal call_number
        # put in real space
        returned_results = []  # new results, negative height [m]
        real_queries = rescale_inverse(x, cfg["constraints"])  # convert to real space
        for i in range(real_queries.shape[0]):
            call_number += 1
            inputs = {
                name: float(real_queries[i][j])
                for j, name in enumerate(cfg["constraints"]["order"])
            }
            # I want to generalize this so it's relative to the observation point not New Orleans
            tmp_dir = temp_dir()
            wrap_cfg = build_wrap_config(
                cfg, inputs, tmp_dir, tradeoff_curve=tradeoff_curve
            )
            print("read config", wrap_cfg)
            # if "displacement" in inputs:
            #    # assume impact lon relative to New Orleans
            #    inputs["impact_lon"] = NEW_ORLEANS.lon + inputs["displacement"]
            #    del inputs["displacement"]

            if cfg["wrap_test"]:
                real_result = min(7 + np.random.normal(), 10)
                yaml_str = OmegaConf.to_yaml(wrap_cfg)
                with open(os.path.join(tmp_dir, "wrap_cfg.yaml"), "w") as file:
                    file.write(yaml_str)
            else:
                # wrap_cfg.tc["displacement"].value = x
                try:
                    real_result = idealized_tc_observe(wrap_cfg)
                except ValueError as e:
                    # A NaN/fill max water level usually means the observation
                    # node stayed DRY for this query (e.g. slow oblique storms
                    # in the wider 4D space) — the honest objective there is
                    # zero surge, which is informative to the GP, unlike a
                    # crashed loop. Log loudly, score the point at 0.0, and
                    # keep optimizing. (Previously this re-raised, which
                    # killed multi-day 4D campaigns one dry query at a time.)
                    print(
                        f"Dry/invalid run scored as 0.0 m: "
                        f"call_number={call_number}, run_folder={tmp_dir}, "
                        f"query={inputs}, error={e}"
                    )
                    real_result = 0.0
                # out_path=tmp_dir,
                # profile_name=profile_name,
                # resolution=resolution,
                # **inputs,

            add_query_to_output(real_queries[i], real_result)
            # flip sign to make it a minimisation problem
            returned_results.append([-real_result])

        return tf.constant(returned_results, dtype=tf.float64)

    return obj


# maybe shift this to constants?
DEFAULT_CONSTRAINTS: dict = yaml.safe_load(
    open(os.path.join(CONFIG_PATH, "3d_constraints.yaml"))
)


def gp_model_callback_maker(
    direc: str,
    constraints: dict = DEFAULT_CONSTRAINTS,
    acq_rule: Optional[EfficientGlobalOptimization] = None,
    ckpt_manager: Optional[tf.train.CheckpointManager] = None,
) -> Callable[[any, any, any], bool]:
    """
    Return a callback function that saves the GP model at each step.

    TODO: problem from the animation: the x1 and x2 are not the right way around. Could be a problem in the scaled results.

    Args:
        direc (str): Directory to save the models.
        constraints (dict, optional): Dictionary with the constraints for the optimization. Defaults to DEFAULT_CONSTRAINTS.
        acq_rule (Optional[EfficientGlobalOptimization], optional): The acquisition rule. Defaults to None.
        ckpt_manager (Optional[tf.train.CheckpointManager], optional): Shared
            checkpoint manager (created and, when resuming, restored in
            ``run_bayesopt_exp``). If None, a manager is created once per model
            on first use rather than recreated at each callback. Defaults to None.

    Returns:
        Callable[[any, any, any], bool]: Callback function for early_stop_callback.
    """
    # https://github.com/secondmind-labs/trieste/blob/develop/trieste/models/gpflow/models.py
    os.makedirs(direc, exist_ok=True)
    # saver = gpflow.saver.Saver()
    call: int = 0
    dimensions = len(constraints["order"])
    # keep (Checkpoint, CheckpointManager) pairs alive between callback calls
    # so checkpoint bookkeeping (max_to_keep, numbering) is not reset each step.
    checkpoint_managers: dict = {}

    if dimensions == 1:  # if 1D save GP model output
        nx1 = 100
        x1 = np.linspace(0, 1, num=nx1)
        x1_r = rescale_inverse(np.column_stack([x1]), constraints=constraints)[:, 0]
        assert x1.shape == (nx1,)
        assert x1_r.shape == x1.shape
        x_input = np.column_stack([x1])
        assert x_input.shape == (nx1, 1)
        ypred_list: List[np.ndarray] = []
        ystd_list: List[np.ndarray] = []
        acq_list: List[np.ndarray] = []
        # x_r = rescale_inverse(np.column_stack([x1]), constraints=constraints)

    elif dimensions == 2:  # if 2D save GP model output

        nx1, nx2 = 100, 102  # resolution of the plot
        # added some asymmetry to try to see if axes flip.
        x1 = np.linspace(0, 1, num=nx1)
        x2 = np.linspace(0, 1, num=nx2)
        # x1, x2 = np.linspace(0, 1, num=nx1), np.linspace(0, 1, num=nx2)
        # To to get different lengths of x1 and
        # x2 to be rescaled, need to do them separately.
        x1_r = rescale_inverse(
            np.column_stack([x1, np.linspace(0, 1, num=nx1)]), constraints=constraints
        )[:, 0]
        x2_r = rescale_inverse(
            np.column_stack([np.linspace(0, 1, num=nx2), x2]), constraints=constraints
        )[:, 1]
        assert x2_r.shape == x2.shape
        assert x1.shape == (nx1,)
        assert x1_r.shape == x1.shape
        # x_r = rescale_inverse(np.column_stack([x1, x2]), constraints=constraints)
        # x1_r, x2_r = x_r[:, 0], x_r[:, 1]
        X1, X2 = np.meshgrid(x1, x2, indexing="ij")
        print("X1", X1.shape)
        assert X1.shape == (nx1, nx2)
        print("X2", X2.shape)
        assert X2.shape == (nx1, nx2)
        x_input = np.column_stack([X1.flatten(), X2.flatten()])
        assert x_input.shape == (nx1 * nx2, 2)
        assert np.allclose(x_input[:, 0].reshape((nx1, nx2)), X1)
        assert np.allclose(x_input[:, 1].reshape((nx1, nx2)), X2)
        ypred_list: List[np.ndarray] = []
        ystd_list: List[np.ndarray] = []
        acq_list: List[np.ndarray] = []

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
        nonlocal call, direc, nx1, nx2, x1, x2, X1, X2, x_input, ypred_list, ystd_list, constraints, dimensions
        call += 1  # increment the call number

        print("dimension", dimensions)
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
            # reuse a single CheckpointManager (passed in from run_bayesopt_exp,
            # which restores manager.latest_checkpoint when resuming) instead of
            # recreating a fresh one at every callback call.
            if ckpt_manager is not None:
                manager = ckpt_manager
            else:
                if model not in checkpoint_managers:
                    ckpt = tf.train.Checkpoint(model=gp_models[model].model)
                    checkpoint_managers[model] = (
                        ckpt,
                        tf.train.CheckpointManager(ckpt, direc, max_to_keep=100),
                    )
                manager = checkpoint_managers[model][1]
            manager.save()
            # plt.show()
            plt.clf()
            plt.close()
            if dimensions == 1:
                # fig, axs = plt.subplots(1, 1, figsize=(10, 5))
                y_mean, y_var = gp_models[model].predict_y(x_input)
                y_mean, y_var = y_mean.numpy(), y_var.numpy()
                y_mean, y_var = y_mean.reshape((nx1)), y_var.reshape((nx1))

                ypred_list.append(y_mean)
                ystd_list.append(np.sqrt(y_var))
                print("np.array(ypred_list))", np.array(ypred_list).shape)
                print("np.array(yvar_list))", np.array(ystd_list).shape)
                data_vars = {
                    "ypred": (
                        ("call", "x1"),
                        np.array(ypred_list),
                        {"units": "m", "long_name": "Mean prediction"},
                    ),
                    "ystd": (
                        ("call", "x1"),
                        np.array(ystd_list),
                        {"units": "m", "long_name": "Std. Dev. in prediction"},
                    ),
                }
                if acq_rule is not None:
                    if acq_rule.acquisition_function is not None:
                        acq = acq_rule.acquisition_function(
                            tf.expand_dims(x_input, axis=-2)
                        )
                        acq = np.reshape(acq, (nx1,))
                    else:
                        acq = np.zeros((nx1,))
                    acq_list.append(acq)
                    print("np.array(acq_list))", np.array(acq_list).shape)
                    data_vars["acq"] = (
                        ("call", "x1"),
                        np.array(acq_list),
                        {"units": "dimensionless", "long_name": "acquisition function"},
                    )
                    print("np.array(acq_list))", np.array(acq_list).shape)

                print(
                    "np.array([x + 1 for x in range(call)]))",
                    [x + 1 for x in range(len(ypred_list))],
                )
                xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "x1": (
                            ("x1"),
                            x1_r,
                            {"units": constraints[constraints["order"][0]]["units"]},
                        ),
                        "call": (
                            ("call"),
                            [x + 1 for x in range(len(ypred_list))],
                            {"units": "dimensionless", "long_name": "call number"},
                        ),
                    },
                ).to_netcdf(os.path.join(direc, f"gp_model_outputs.nc"))
            if dimensions == 2:
                # what's a good way to check that this puts the dimensions the right way around?
                # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                y_mean, y_var = gp_models[model].predict_y(x_input)
                y_mean, y_var = y_mean.numpy(), y_var.numpy()
                y_mean, y_var = y_mean.reshape((nx1, nx2)), y_var.reshape((nx1, nx2))

                ypred_list.append(y_mean)
                ystd_list.append(np.sqrt(y_var))
                print("np.array(ypred_list))", np.array(ypred_list).shape)
                print("np.array(yvar_list))", np.array(ystd_list).shape)
                data_vars = {
                    "ypred": (
                        ("call", "x1", "x2"),
                        np.array(ypred_list),
                        {"units": "m", "long_name": "Mean prediction"},
                    ),
                    "ystd": (
                        ("call", "x1", "x2"),
                        np.array(ystd_list),
                        {"units": "m", "long_name": "Std. Dev. in prediction"},
                    ),
                }
                if acq_rule is not None:
                    if acq_rule.acquisition_function is not None:
                        acq = acq_rule.acquisition_function(
                            tf.expand_dims(x_input, axis=-2)
                        )
                        acq = np.reshape(acq, (nx1, nx2))
                    else:
                        acq = np.zeros((nx1, nx2))
                    acq_list.append(acq)
                    print("np.array(acq_list))", np.array(acq_list).shape)
                    data_vars["acq"] = (
                        ("call", "x1", "x2"),
                        np.array(acq_list),
                        {"units": "dimensionless", "long_name": "acquisition function"},
                    )
                    print("np.array(acq_list))", np.array(acq_list).shape)

                print(
                    "np.array([x + 1 for x in range(call)]))",
                    [x + 1 for x in range(len(ypred_list))],
                )
                xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "x1": (
                            ("x1"),
                            x1_r,
                            {"units": constraints[constraints["order"][0]]["units"]},
                        ),
                        "x2": (
                            ("x2"),
                            x2_r,
                            {"units": constraints[constraints["order"][1]]["units"]},
                        ),
                        "call": (
                            ("call"),
                            [x + 1 for x in range(len(ypred_list))],
                            {"units": "dimensionless", "long_name": "call number"},
                        ),
                    },
                ).to_netcdf(os.path.join(direc, f"gp_model_outputs.nc"))

                # im = axs[0].contourf(X1, X2, y_mean, levels=1000)
                # # add colorbar to the plot with the right scale and the same size as the plot
                # fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
                # # axs[0].colorbar()
                # axs[0].set_title("Mean")
                # im = axs[1].contourf(X1, X2, np.sqrt(y_var), levels=1000)
                # fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
                # # axs[1].colorbar()
                # axs[1].set_title("Std. Dev., $\sigma$")
                # axs[0].set_xlabel("x$_1$")
                # axs[0].set_ylabel("x$_2$")
                # axs[1].set_xlabel("x$_1$")
                # # plt.show()
                # plt.clf()
                # plt.close()
            #  model.save(os.path.join(f"gp_model_{i}.h5"))
            # saver.save(os.path.join(direc, f"gp_model_{call}"), model)

        return False  # False means don't stop

    return gp_model_callback


def _load_bo_constraints(direc: str) -> Optional[dict]:
    """
    Load the constraints dictionary stored in an experiment's bo-config.json.

    Args:
        direc (str): Experiment directory.

    Returns:
        Optional[dict]: The constraints dictionary, or None if unavailable.
    """
    config_path = os.path.join(direc, BO_CONFIG_NAME)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file).get("constraints")
    except (ValueError, OSError) as e:
        print(f"Warning: could not read {config_path} ({e}).")
        return None


def _scan_completed_runs(direc: str, order: Optional[List[str]] = None) -> List[dict]:
    """
    Scan an experiment directory for completed BO evaluations (pure python/numpy).

    A run ``i`` counts as completed if the ``experiments.json`` ledger (written
    incrementally by ``objective_f`` after each successful ADCIRC evaluation)
    contains an entry for call number ``i`` with a finite result, and the
    corresponding ``exp_{i:04}`` run folder exists on disk. Only the contiguous
    prefix of completed runs (starting from call 0) is returned, so that call
    numbering stays consistent when resuming; anything after the first
    missing/corrupt record is skipped with a printed warning.

    Args:
        direc (str): Experiment directory containing exp_NNNN subfolders.
        order (Optional[List[str]], optional): Parameter names that each ledger
            record must contain (from constraints["order"]). Defaults to None
            (do not check parameters).

    Returns:
        List[dict]: Ledger records of completed runs, ordered by call number.
    """
    if not os.path.isdir(direc):
        return []
    run_folders = sorted(
        x
        for x in os.listdir(direc)
        if x.startswith("exp_") and os.path.isdir(os.path.join(direc, x))
    )
    ledger_path = os.path.join(direc, BO_LEDGER_NAME)
    if not os.path.exists(ledger_path):
        if run_folders:
            print(
                f"Warning: found {len(run_folders)} run folder(s) in {direc} but "
                f"no {BO_LEDGER_NAME}; treating experiment as having no completed runs."
            )
        return []
    try:
        with open(ledger_path, "r", encoding="utf-8") as ledger_file:
            ledger = json.load(ledger_file)
    except (ValueError, OSError) as e:
        print(
            f"Warning: could not read ledger {ledger_path} ({e}); "
            "treating experiment as having no completed runs."
        )
        return []

    def _is_finite_number(value: any) -> bool:
        return isinstance(value, (int, float)) and np.isfinite(value)

    records: List[dict] = []
    i = 0
    while str(i) in ledger:
        record = ledger[str(i)]
        run_folder = os.path.join(direc, f"exp_{i:04}")
        if not isinstance(record, dict) or not _is_finite_number(record.get("res")):
            print(
                f"Warning: ledger entry {i} in {ledger_path} has an invalid "
                f"result ('res'={record.get('res') if isinstance(record, dict) else record}); "
                f"stopping the resume scan at {i} completed run(s)."
            )
            break
        if not os.path.isdir(run_folder):
            print(
                f"Warning: ledger entry {i} has no run folder {run_folder}; "
                f"stopping the resume scan at {i} completed run(s)."
            )
            break
        if order is not None and not all(
            _is_finite_number(record.get(name)) for name in order
        ):
            print(
                f"Warning: ledger entry {i} in {ledger_path} is missing or has "
                f"invalid parameter value(s) (expected {list(order)}); "
                f"stopping the resume scan at {i} completed run(s)."
            )
            break
        records.append(record)
        i += 1

    # warn about run folders beyond the completed prefix (e.g. a run that was
    # interrupted mid-ADCIRC and never reached the ledger); these are skipped
    # and will be re-run (the folder is reused/overwritten) on resume.
    for folder in run_folders:
        try:
            idx = int(folder.split("_")[-1])
        except ValueError:
            continue
        if idx >= i:
            print(
                f"Warning: skipping incomplete/corrupt run folder "
                f"{os.path.join(direc, folder)} (no valid '{folder}' entry in "
                f"{BO_LEDGER_NAME})."
            )
    return records


def rebuild_initial_data(
    direc: str, constraints: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebuild the (queries, observations) arrays of a partially-completed BO
    experiment from the per-run records saved on disk (pure numpy; no
    tensorflow/trieste needed at call time).

    The queries are returned in the rescaled [0, 1] optimisation space (the
    space trieste's search box uses) and the observations carry the negated
    sign convention of the objective (negative max water level in metres), so
    they can be converted directly into a trieste ``Dataset``.

    Args:
        direc (str): Experiment directory containing exp_NNNN subfolders,
            experiments.json and (optionally) bo-config.json.
        constraints (Optional[dict], optional): Constraints dictionary with an
            "order" key and per-parameter "min"/"max". Defaults to None, in
            which case it is read from bo-config.json in ``direc``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (queries [N, D] in [0, 1], observations
            [N, 1], both float64), ordered by call number. Empty arrays with
            the right trailing dimensions if there are no completed runs.
    """
    if constraints is None:
        constraints = _load_bo_constraints(direc)
    order = (
        list(constraints["order"])
        if constraints is not None and "order" in constraints
        else None
    )
    ndim = len(order) if order is not None else 0
    empty = (
        np.zeros((0, ndim), dtype=np.float64),
        np.zeros((0, 1), dtype=np.float64),
    )
    records = _scan_completed_runs(direc, order=order)
    if not records:
        return empty
    if order is None:
        print(
            f"Warning: {len(records)} completed run(s) found in {direc} but no "
            f"constraints available (missing {BO_CONFIG_NAME}); cannot rebuild "
            "query points."
        )
        return empty
    real_queries = np.array(
        [[float(record[name]) for name in order] for record in records],
        dtype=np.float64,
    )
    mins = np.array([float(constraints[name]["min"]) for name in order])
    diffs = np.array(
        [
            float(constraints[name]["max"]) - float(constraints[name]["min"])
            for name in order
        ]
    )
    queries = (real_queries - mins) / diffs  # rescale to [0, 1] (see adbo.rescale)
    tol = 1e-6
    in_range = np.all((queries >= -tol) & (queries <= 1 + tol), axis=1)
    if not np.all(in_range):
        first_bad = int(np.argmin(in_range))
        print(
            f"Warning: completed run {first_bad} in {direc} lies outside the "
            "current constraint box (were the constraints changed?); only the "
            f"first {first_bad} run(s) will be reused."
        )
        queries, records = queries[:first_bad], records[:first_bad]
    queries = np.clip(queries, 0.0, 1.0)
    observations = -np.array(
        [[float(record["res"])] for record in records], dtype=np.float64
    )
    return queries, observations


def count_completed_runs(direc: str) -> int:
    """
    Count the completed BO evaluations in an experiment directory.

    Args:
        direc (str): Experiment directory.

    Returns:
        int: Number of completed runs (contiguous from call 0).
    """
    constraints = _load_bo_constraints(direc)
    order = (
        list(constraints["order"])
        if constraints is not None and "order" in constraints
        else None
    )
    return len(_scan_completed_runs(direc, order=order))


def run_exists(exp_name: str, num_runs: int, root_exp_direc: str = EXP_PATH) -> bool:
    """
    Check if the experiment has already been run to completion.

    A run counts as completed if it has both a valid entry in the
    experiments.json ledger and an exp_NNNN run folder (see
    ``_scan_completed_runs``). ``completed >= num_runs`` means done;
    ``0 < completed < num_runs`` means the experiment is resumable (this
    function returns False and ``run_bayesopt_exp`` resumes it).

    Args:
        exp_name (str): Name of the experiment.
        num_runs (int): Expected total number of runs (init_steps + daf_steps).
        root_exp_direc (str, optional): Root directory containing the
            experiment folder. Defaults to EXP_PATH.

    Returns:
        bool: Whether the experiment has already been fully run.
    """
    return count_completed_runs(os.path.join(root_exp_direc, exp_name)) >= num_runs


@timeit
def run_bayesopt_exp(
    constraints: dict = DEFAULT_CONSTRAINTS,
    seed: int = 10,
    profile_name: str = "2015_new_orleans_profile_r4i1p1f1",  # shipped: {2015,2100}_{new_orleans,galverston,miami}_profile_r4i1p1f1.json (the old 2025_*/2097_* names never existed)
    resolution: str = "mid",
    exp_name: str = "bo_test",
    root_exp_direc: Optional[str] = None,
    obs_lon: float = NEW_ORLEANS.lon,
    obs_lat: float = NEW_ORLEANS.lat,
    init_steps: int = 10,
    daf_steps: int = 10,
    daf: str = "mes",  # mes, ei, ucb
    kernel: str = "Matern52",  # Matern32, Matern52, SE
    wrap_test: bool = False,
    resume: bool = True,
    curve_path: Optional[str] = None,
) -> None:
    """
    Run a Bayesian Optimisation experiment.

    Args:
        constraints (dict, optional): Dictionary with the constraints for the optimization. Defaults to DEFAULT_CONSTRAINTS.
        seed (int, optional): Seed to initialize. Defaults to 10.
        profile_name (str, optional): Name of the profile. Defaults to "2015_new_orleans_profile_r4i1p1f1".
        exp_name (str, optional): Experiment name. Defaults to "bo_test".
        root_exp_direc (Optional[str], optional): Root directory for the
            experiments. Defaults to None, in which case the WORSTSURGE_EXP_DIR
            environment variable is used if set, falling back to EXP_PATH.
        init_steps (int, optional): How many sobol sambles. Defaults to 10.
        daf_steps (int, optional): How many acquisition points. Defaults to 10.
        obs_lon (float, optional): Longitude of the observation point. Defaults to NEW_ORLEANS.lon.
        obs_lat (float, optional): Latitude of the observation point. Defaults to NEW_ORLEANS.lat.
        resolution (str, optional): Resolution of the model. Defaults to "mid".
        daf (str, optional): Type of acquisition function. Defaults to "mes".
        kernel (str, optional): Kernel to use. Defaults to "Matern52".
        wrap_test (bool, optional): Whether to prevent. Defaults to False.
        resume (bool, optional): Whether to resume a partially-completed
            experiment by reusing the ADCIRC evaluations already saved on disk
            (see ``rebuild_initial_data``). Defaults to True.
        curve_path (Optional[str], optional): Path to a stored size-intensity
            tradeoff curve (``w22.tradeoff.generate_curve``). Required when
            the constraints include a "vmax" dimension: each BO sample then
            gets a CLE15 profile generated at its sampled intensity on the
            curve, and missing vmax bounds are filled from the curve domain
            ([Cat-1, potential intensity], gradient wind). Defaults to None.
    """
    if "vmax" in constraints["order"]:
        # avoid mutating a shared constraints dict (e.g. module default)
        constraints = copy.deepcopy(constraints)
        if curve_path is None:
            raise ValueError(
                "constraints include 'vmax' but curve_path is None; generate "
                "a tradeoff curve with w22.tradeoff.generate_curve first."
            )
        # lazy import: pulls in the numba CLE15 kernels
        from w22.tradeoff import TradeoffCurve

        _curve = TradeoffCurve.from_file(curve_path)
        vmax_c = constraints.setdefault("vmax", {})
        # fill missing bounds from the curve; clip explicit ones to its domain
        vmax_c["min"] = (
            _curve.v_min
            if vmax_c.get("min") is None
            else max(float(vmax_c["min"]), _curve.v_min)
        )
        vmax_c["max"] = (
            _curve.v_max
            if vmax_c.get("max") is None
            else min(float(vmax_c["max"]), _curve.v_max)
        )
        vmax_c.setdefault("units", "m/s (gradient wind)")
        if vmax_c["min"] >= vmax_c["max"]:
            raise ValueError(
                f"Empty vmax range [{vmax_c['min']}, {vmax_c['max']}] after "
                f"clipping to the curve domain [{_curve.v_min:.2f}, "
                f"{_curve.v_max:.2f}] m/s."
            )
        print(
            f"run_bayesopt_exp: vmax dimension in "
            f"[{vmax_c['min']:.2f}, {vmax_c['max']:.2f}] m/s (gradient wind), "
            f"profiles from {curve_path}"
        )

    if root_exp_direc is None:
        root_exp_direc = os.environ.get(EXP_DIR_ENV_VAR, EXP_PATH)
        print(
            f"run_bayesopt_exp: root experiment directory '{root_exp_direc}' "
            f"(set ${EXP_DIR_ENV_VAR} to override; default {EXP_PATH})"
        )
    direc = os.path.join(root_exp_direc, exp_name)
    num_total_runs = init_steps + daf_steps

    cfg = {
        "constraints": constraints,
        "seed": seed,
        "profile_name": profile_name,
        "exp_name": exp_name,
        "init_steps": init_steps,
        "daf_steps": daf_steps,
        "obs_lon": obs_lon,
        "obs_lat": obs_lat,
        "wrap_test": wrap_test,
        "resolution": resolution,
        "exp_dir": direc,
        "resume": resume,
        "curve_path": curve_path,
    }

    if run_exists(exp_name, num_runs=num_total_runs, root_exp_direc=root_exp_direc):
        print(
            f"Experiment {exp_name} already complete "
            f"(>= {num_total_runs} runs found in {direc})"
        )
        return

    os.makedirs(direc, exist_ok=True)

    # Rebuild any evaluations already completed by an interrupted session
    # (pure numpy scan of the exp_NNNN folders + experiments.json ledger).
    if resume:
        past_queries, past_observations = rebuild_initial_data(
            direc, constraints=constraints
        )
    else:
        past_queries = np.zeros((0, len(constraints["order"])), dtype=np.float64)
        past_observations = np.zeros((0, 1), dtype=np.float64)
    num_completed = int(past_queries.shape[0])
    if num_completed > 0:
        print(
            f"Resuming experiment {exp_name}: {num_completed}/{num_total_runs} "
            "runs already completed; reusing saved ADCIRC evaluations."
        )

    write_json(cfg, os.path.join(direc, BO_CONFIG_NAME))
    print("BO config", cfg)

    setup_tf(seed=seed, log_name=exp_name)

    # set up BayesOpt
    dimensions_input: int = len(constraints["order"])
    # assert dimensions_input == 2
    search_space = trieste.space.Box([0] * dimensions_input, [1] * dimensions_input)
    init_objective = objective_f(cfg)

    # observer = trieste.objectives.utils.mk_observer(obs_class.objective)
    observer = trieste.objectives.utils.mk_observer(init_objective)
    print("observer", observer, type(observer))

    past_data: Optional[Dataset] = None
    if num_completed > 0:
        past_data = Dataset(
            query_points=tf.constant(past_queries, dtype=tf.float64),
            observations=tf.constant(past_observations, dtype=tf.float64),
        )

    num_init_remaining = max(init_steps - num_completed, 0)
    if num_init_remaining > 0:
        # Draw the full seeded Sobol design (the same design as an
        # uninterrupted run, since setup_tf has just reseeded tf/numpy) and
        # only evaluate the initial points that are still missing.
        initial_query_points = search_space.sample_sobol(init_steps)[
            init_steps - num_init_remaining :
        ]
        print("initial_query_points", initial_query_points, type(initial_query_points))
        new_initial_data = observer(initial_query_points)
        initial_data = (
            past_data + new_initial_data if past_data is not None else new_initial_data
        )
    else:
        initial_data = past_data
    print("initial_data", initial_data, type(initial_data))

    # acquisition steps still to run (any beyond the initial design that were
    # already completed before an interruption count towards daf_steps).
    num_daf_remaining = daf_steps - max(num_completed - init_steps, 0)
    if num_daf_remaining <= 0:
        print(
            f"Experiment {exp_name}: no acquisition steps remaining "
            f"({num_completed} completed of {num_total_runs}); nothing to do."
        )
        return
    # set up bayesopt loop
    if kernel == "Matern52":
        gp_kernel = gpflow.kernels.Matern52()
    elif kernel == "Matern32":
        gp_kernel = gpflow.kernels.Matern32()
    elif kernel == "SE":
        gp_kernel = gpflow.kernels.SquaredExponential()
    else:
        raise ValueError(f"Unknown kernel {kernel}")

    gpr = trieste.models.gpflow.build_gpr(  # by default Matern 52
        initial_data, search_space, kernel=gp_kernel
    )  # should add kernel choice here.
    model = trieste.models.gpflow.GaussianProcessRegression(gpr)

    # Single checkpoint manager for the experiment: restore the latest GP
    # checkpoint when resuming (warm start of the hyperparameters), and pass
    # the manager to the callback so it is not recreated at every step.
    checkpoint = tf.train.Checkpoint(model=model.model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, direc, max_to_keep=100)
    if resume and num_completed > 0 and checkpoint_manager.latest_checkpoint:
        try:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            print(
                "Restored GP hyperparameters from checkpoint "
                f"{checkpoint_manager.latest_checkpoint}"
            )
        except Exception as e:
            print(
                f"Warning: could not restore GP checkpoint "
                f"{checkpoint_manager.latest_checkpoint} ({e}); "
                "continuing with a freshly initialised GP."
            )
    # choices mves,
    # TODO: make another choice for this.
    if daf == "mes":
        acquisition_func = MinValueEntropySearch(
            search_space
        )  # should add in acquisition function choice here.
    elif daf == "ei":
        acquisition_func = ExpectedImprovement(
            search_space  # , best_observation=initial_data.maximum()
        )
    elif daf == "ucb":
        acquisition_func = NegativeLowerConfidenceBound(search_space)
    else:
        raise ValueError(f"Unknown acquisition function {daf}")
    acquisition_rule = EfficientGlobalOptimization(acquisition_func)
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    # run bayesopt loop
    result = bo.optimize(
        num_daf_remaining,
        initial_data,
        model,
        acquisition_rule,
        track_state=True,  # there was some issue with this on mac
        early_stop_callback=gp_model_callback_maker(
            direc,
            constraints,
            acq_rule=acquisition_rule,
            ckpt_manager=checkpoint_manager,
        ),
    ).astuple()
    trieste.logging.set_summary_filter(lambda name: True)  # enable all summaries
    # print("result", result)
    # print("history", history)
    real_res = result[0].unwrap()
    dataset = real_res.dataset
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    rescaled_query_points = rescale_inverse(query_points, constraints)

    def save_results(
        rescaled_query_points: np.ndarray,  # shape [N, F]
        observations: np.ndarray,
        direc: str,
        exp_name: str,
    ) -> None:
        results_ds = xr.Dataset(
            data_vars={
                **{
                    "x"
                    + str(i): (
                        ("call"),
                        rescaled_query_points[:, i],
                        {"units": constraints[var]["units"], "long_name": var},
                    )
                    for i, var in enumerate(constraints["order"])
                },
                **{"y": (("call"), -observations.flatten(), {"units": "m"})},
            },
            coords={"call": [x + 1 for x in range(len(observations))]},
        )
        # provenance attributes (mirrors w22/ps.py style)
        try:
            git_hash = get_git_revision_hash(path=str(PROJECT_PATH))
        except Exception as e:
            git_hash = f"unknown ({e})"
        results_ds.attrs["bo_exp_calculated_at_git_hash"] = git_hash
        results_ds.attrs["bo_exp_calculated_at_time"] = time_stamp()
        results_ds.attrs["seed"] = seed
        results_ds.attrs["trieste_version"] = str(trieste.__version__)
        results_ds.attrs["tensorflow_version"] = str(tf.__version__)
        results_ds.attrs["gpflow_version"] = str(gpflow.__version__)
        results_ds.attrs["numpy_version"] = str(np.__version__)
        results_ds.to_netcdf(os.path.join(direc, exp_name + "_results.nc"))

    save_results(rescaled_query_points, observations, direc, exp_name)

    def plot_results() -> None:
        # plot the results for 2d
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_bo_points(
            query_points,
            ax,
            init_steps,
            m_init="o",
            m_add="+",  # obs_values=observations
        )  # , arg_min_idx)
        ax.scatter(
            query_points[:, 0], query_points[:, 1], c=observations, cmap="viridis"
        )
        ax.set_xlabel(r"$x_1$ [dimensionless]")
        ax.set_ylabel(r"$x_2$ [dimensionless]")
        # change name to allow choice.
        plt.savefig(os.path.join(FIGURE_PATH, "bo_exp", exp_name + "_results.pdf"))
        # plt.show()
        plt.clf()
        plt.close()

    def plt_regret() -> None:
        # plot the regret
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_regret(
            dataset.observations.numpy(),
            ax,
            num_init=init_steps,
            show_obs=True,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Regret [-m]")
        plt.savefig(os.path.join(FIGURE_PATH, "bo_exp", exp_name + "_regret.pdf"))
        # plt.show()
        plt.clf()
        plt.close()

    if len(constraints["order"]) == 2:
        plot_results()
    plt_regret()

    # plot the gp model changes for 2d case:
    if len(constraints["order"]) in (1, 2):
        plot_gps(path_in=direc, plot_acq=True)


if __name__ == "__main__":
    # python -m adbo.newexp
    run_bayesopt_exp(wrap_test=False, exp_name="Test3")
