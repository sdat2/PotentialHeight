"""
Conduct a 1D experiment for ease of plotting.
"""

import os
from yaml import safe_load
import argparse
from adforce.constants import NEW_ORLEANS
from .exp import run_bayesopt_exp
from .constants import CONFIG_PATH


constraints = safe_load(open(os.path.join(CONFIG_PATH, "1d_constraints.yaml")))

print(constraints)


def create_1d_run() -> None:
    """
    Run a 1D experiment to make an animation of the GP model output being refined in BayesOpt.
    """
    args = argparse.ArgumentParser()
    args.add_argument("--test", type=bool, default=False)
    args.add_argument("--year", type=int, default=2025)
    args.add_argument("--obs_lon", type=float, default=NEW_ORLEANS.lon)
    args.add_argument("--obs_lat", type=float, default=NEW_ORLEANS.lat)
    args.add_argument("--init_steps", type=int, default=25)
    args.add_argument("--daf_steps", type=int, default=25)
    args.add_argument("--resolution", type=str, default="mid")
    args.add_argument("--exp_name", type=str, default="ani-1d-2015")
    args.add_argument("--seed", type=int, default=10)
    args.add_argument("--kernel", type=str, default="Matern52")
    args.add_argument(
        "--profile_name", type=str, default="2015_new_orleans_profile_r4i1p1f1"
    )
    args.add_argument("--wrap_test", type=bool, default=False)
    # adding ability to override constraints in the command line

    for key in constraints.keys():
        if key == "order":
            continue
        args.add_argument(f"--{key}_min", type=float, default=constraints[key]["min"])
        args.add_argument(f"--{key}_max", type=float, default=constraints[key]["max"])

    args = args.parse_args()

    for key in constraints.keys():
        if key == "order":
            continue
        constraints[key]["min"] = getattr(args, f"{key}_min")
        constraints[key]["max"] = getattr(args, f"{key}_max")

    print(args)
    run_bayesopt_exp(
        constraints=constraints,
        seed=args.seed,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        profile_name=args.profile_name,
        exp_name=args.exp_name,
        resolution=args.resolution,
        init_steps=args.init_steps,
        daf_steps=args.daf_steps,
        wrap_test=args.wrap_test,
        kernel=args.kernel,
    )


if __name__ == "__main__":
    create_1d_run()
