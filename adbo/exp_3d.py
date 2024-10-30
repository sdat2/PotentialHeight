"""adbo.exp_3d.py"""

import hydra
from omegaconf import DictConfig
from adbo.constants import CONFIG_PATH
from adbo.exp import run_bayesopt_exp, DEFAULT_CONSTRAINTS


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bo_setup")
def run_3d_exp(args: DictConfig) -> None:
    """
    Run an experiment varying the angle, displacement and speed of the storm for a given tropical cyclone profile.
    """
    print(args)
    run_bayesopt_exp(
        seed=args.seed_offset + args.year,
        profile_name=f"{args.year}.json",
        constraints=DEFAULT_CONSTRAINTS,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        exp_name=args.exp_name,  # f"{args.setup}-{args.year}-i-{args.init_steps}",
        resolution=args.resolution,
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
