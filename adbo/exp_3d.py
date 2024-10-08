"""adbo.exp_3d.py"""

import argparse
from adbo.exp import run_bayesopt_exp, DEFAULT_CONSTRAINTS


def run_3d_exp() -> None:
    """
    Run an experiment varying the angle, displacement and speed of the storm for a given tropical cyclone profile.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--stationid", type=int, default=3)
    parser.add_argument("--year", type=int, default=2097)
    parser.add_argument("--init_steps", type=int, default=25)
    parser.add_argument("--daf_steps", type=int, default=25)
    parser.add_argument("--seed_offset", type=int, default=22)
    parser.add_argument("--setup", type=str, default="mid-notide")
    args = parser.parse_args()

    # stationid: int = 3
    # year: int = 2097  # python -m adbo.exp_3d &> logs/bo-3-2097.log
    # python -m adbo.exp_3d --stationid=3 --year=2025 &> logs/bo-test-3-2025-new.log
    run_bayesopt_exp(
        seed=args.seed_offset + args.stationid + args.year,
        profile_name=f"{args.year}.json",
        constraints=DEFAULT_CONSTRAINTS,
        stationid=args.stationid,
        exp_name=f"{args.setup}-{args.stationid:01}-{args.year}-i-{args.init_steps}",
        resolution=args.setup,
        init_steps=args.init_steps,
        daf_steps=args.daf_steps,
        wrap_test=False,
    )


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
