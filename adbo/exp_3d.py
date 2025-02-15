"""adbo.exp_3d.py"""

import argparse
from adforce.constants import NEW_ORLEANS
from .constants import DEFAULT_CONSTRAINTS
from .newexp import run_bayesopt_exp


# @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bo_setup")
def run_3d_exp() -> None:
    """
    Run an experiment varying the angle, displacement and speed of the storm for a given tropical cyclone profile.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument(
        "--profile_name", type=str, default="2025_new_orleans_profile_r4i1p1f1"
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--obs_lon", type=float, default=NEW_ORLEANS.lon)
    parser.add_argument("--obs_lat", type=float, default=NEW_ORLEANS.lat)
    parser.add_argument("--init_steps", type=int, default=25)
    parser.add_argument("--daf_steps", type=int, default=25)
    parser.add_argument("--resolution", type=str, default="mid")
    parser.add_argument("--exp_name", type=str, default="bo")

    args = parser.parse_args()

    print(args)
    run_bayesopt_exp(
        seed=args.seed,
        profile_name=args.profile_name,
        constraints=DEFAULT_CONSTRAINTS,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        exp_name=args.exp_name,  # f"{args.setup}-{args.year}-i-{args.init_steps}",
        resolution="mid",
        init_steps=args.init_steps,
        daf_steps=args.daf_steps,
        wrap_test=args.test,
    )
    print(args)


if __name__ == "__main__":
    run_3d_exp()
    # TODO: check if the 3d experiments have finished.
    # run_3d_exp()
    # we could add an existence check to the run_bayesopt_exp function.
    # To exist, the directory with that name should exist, the correct number of subdirectories should be created, and the summary results should be stored.
    # Idea: animation with maximum storm heights for each new sample with track plotted on top.
    # Idea: create a 3D plot of the GP model output.
    # Idea: create a 3D plot of the GP model output with the acquisition function.
    # run_bayesopt_exp(seed=12, exp_name="bo_test5", init_steps=5, daf_steps=35)
    # run_bayesopt_exp(seed=13, exp_name="bo_test8", init_steps=5, daf_steps=35)
    # run_bayesopt_exp(
    #     seed=13,
    #     constraints=constraints_2d,
    #     exp_name="bo-test-2d-4",
    #     init_steps=30,
    #     daf_steps=50,
    #     wrap_test=False,
    # )
    # python -m adbo.exp_3d &> logs/test32.log
    # run_bayesopt_exp(seed=16, exp_name="bo_test16", init_steps=5, daf_steps=50)
    # python -m adbo.exp_3d &> logs/bo_test-2d-3.log
    # run_bayesopt_exp(seed=18, exp_name="bo_test18", init_steps=5, daf_steps=100)
    # stationid = 0
