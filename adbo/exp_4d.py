"""adbo.exp_4d.py

Run a 4D Bayesian-optimization experiment varying the angle, displacement,
translation speed AND intensity of the storm, with each sampled intensity
placed on the size--intensity tradeoff curve r(V) of the potential size
model (see ``w22.tradeoff``). Requires a precomputed curve netCDF:

    from w22.tradeoff import env_from_point_ds, generate_curve
    env = env_from_point_ds("w22/data/new_orleans_august_ssp585_CESM2_r4i1p1f1_isothermal_pi4new.nc", 2025)
    generate_curve(env, "2025_new_orleans_r4i1p1f1")

Example::

    python -m adbo.exp_4d --curve_nc w22/data/curves/2025_new_orleans_r4i1p1f1.nc \
        --exp_name no-4d-2025 --init_steps 35 --daf_steps 35
"""

import argparse
import os

import yaml

from adforce.constants import NEW_ORLEANS
from .constants import CONFIG_PATH

CONSTRAINTS_4D_PATH = os.path.join(CONFIG_PATH, "4d_constraints.yaml")


def run_4d_exp() -> None:
    """Run the 4D (track + on-curve intensity) BO experiment from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve_nc", type=str, required=True)
    parser.add_argument("--test", action="store_true", help="wrap_test mode")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--obs_lon", type=float, default=NEW_ORLEANS.lon)
    parser.add_argument("--obs_lat", type=float, default=NEW_ORLEANS.lat)
    parser.add_argument("--init_steps", type=int, default=35)
    parser.add_argument("--daf_steps", type=int, default=35)
    parser.add_argument("--resolution", type=str, default="mid")
    parser.add_argument("--exp_name", type=str, default="bo-4d")
    parser.add_argument("--kernel", type=str, default="Matern52")
    parser.add_argument("--daf", type=str, default="mes")

    # constraint overrides (vmax bounds default to the curve domain)
    constraints = yaml.safe_load(open(CONSTRAINTS_4D_PATH))
    for dim in ("angle", "trans_speed", "displacement", "vmax"):
        for bound in ("min", "max"):
            parser.add_argument(
                f"--{dim}_{bound}",
                type=float,
                default=constraints[dim][bound],
            )
    args = parser.parse_args()
    print(args)

    for dim in ("angle", "trans_speed", "displacement", "vmax"):
        constraints[dim]["min"] = getattr(args, f"{dim}_min")
        constraints[dim]["max"] = getattr(args, f"{dim}_max")

    # heavy import (tensorflow/trieste) deferred until actually running
    from .exp import run_bayesopt_exp

    run_bayesopt_exp(
        constraints=constraints,
        seed=args.seed,
        exp_name=args.exp_name,
        resolution=args.resolution,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        init_steps=args.init_steps,
        daf_steps=args.daf_steps,
        daf=args.daf,
        kernel=args.kernel,
        wrap_test=args.test,
        curve_path=args.curve_nc,
    )


if __name__ == "__main__":
    run_4d_exp()
