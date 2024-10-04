"""adbo.exp_2d.py"""

import argparse
from adbo.exp import run_bayesopt_exp


def create_2d_ani_run() -> None:
    """
    Run a 2D experiment to make an animation of the GP model output being refined in BayesOpt.
    """
    constraints_2d: dict = {
        "angle": {"min": -80, "max": 80, "units": "degrees"},
        "displacement": {"min": -2, "max": 2, "units": "degrees"},
        "order": ("angle", "displacement"),  # order of input features
    }
    print("constraints_2d", constraints_2d)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--stationid", type=int, default=3)
    parser.add_argument("--resolution", type=str, default="mid-notide")
    parser.add_argument("--exp_name", type=str, default="ani-2d-3")
    args = parser.parse_args()

    run_bayesopt_exp(
        constraints=constraints_2d,
        seed=10,
        stationid=args.stationid,
        profile_name=str(args.year) + ".json",
        exp_name=args.exp_name,
        resolution=args.resolution,
        init_steps=25,
        daf_steps=25,
        wrap_test=args.test,
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
