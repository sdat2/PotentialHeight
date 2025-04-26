"""adbo.exp_2d.py"""

import os
import yaml
import argparse
from adforce.constants import NEW_ORLEANS
from .exp import run_bayesopt_exp
from .constants import CONFIG_PATH


def create_2d_ani_run() -> None:
    """
    Run a 2D experiment to make an animation of the GP model output being refined in BayesOpt.
    """
    constraints_2d: dict = yaml.safe_load(
        open(os.path.join(CONFIG_PATH, "2d_constraints.yaml"))
    )
    print("constraints_2d", constraints_2d)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--obs_lon", type=float, default=NEW_ORLEANS.lon)
    parser.add_argument("--obs_lat", type=float, default=NEW_ORLEANS.lat)
    parser.add_argument("--init_steps", type=int, default=25)
    parser.add_argument("--daf_steps", type=int, default=25)
    parser.add_argument("--resolution", type=str, default="mid")
    parser.add_argument("--exp_name", type=str, default="ani-2d-2097")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--kernel", type=str, default="Matern52")

    parser.add_argument(
        "--profile_name", type=str, default="2015_new_orleans_profile_r4i1p1f1"
    )

    args = parser.parse_args()

    run_bayesopt_exp(
        constraints=constraints_2d,
        seed=args.seed,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        profile_name=args.profile_name,
        exp_name=args.exp_name,
        resolution=args.resolution,
        init_steps=args.init_steps,
        daf_steps=args.daf_steps,
        wrap_test=args.test,
        kernel=args.kernel,
    )


if __name__ == "__main__":
    create_2d_ani_run()
    # TODO: check if the 3d experiments have finished.
    # run_3d_exp()
    # we could add an existence check to the run_bayesopt_exp function.
    # To exist, the directory with that name should exist, the correct number of subdirectories should be created, and the summary results should be stored.
    # Idea: animation with maximum storm heights for each new sample with track plotted on top.
    # Idea: create a 3D plot of the GP model output.
    # Idea: create a 3D plot of the GP model output with the acquisition function.
    # run_bayesopt_exp(seed=12, exp_name="bo_test5", init_steps=5, daf_steps=35)
    # run_bayesopt_exp(seed=13, exp_name="bo_test8", init_steps=5, daf_steps=35)
    # python -m adbo.exp_2d &> logs/bo_test10.log
    # python -m
    # python -m adbo.exp_2d &> logs/2d_ani.log
    # python -m adbo.exp_2d &> logs/exp.log
    # run_bayesopt_exp(seed=14, exp_name="bo_test10", init_steps=5, daf_steps=50)
    # python -m adbo.exp_2d &> logs/test15.log
    # run_bayesopt_exp(seed=15, exp_name="bo_test11", init_steps=1, daf_steps=50)
    # run_bayesopt_exp(seed=15, exp_name="test12", init_steps=1, daf_steps=50)
    # python -m adbo.exp_2d &> logs/bo-test-0-2015.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-midres2.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-midres3-2097.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-midres5.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-hres.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-hres2.log
    # python -m adbo.exp_2d &> logs/bo-test-2d-4.log
    # python -m adbo.exp_2d &> logs/test-2d.log
