"""1D intensity sweep along the size--intensity tradeoff curve.

Holds the storm track fixed (ideally at the 3D BO optimum found previously)
and sweeps the gradient-level intensity ``vmax`` along the potential-size
tradeoff curve r(V) (``w22.tradeoff``), running one ADCIRC evaluation per
intensity. This directly answers whether the worst-case surge storm sits at
the potential intensity or at a bigger-but-weaker point on the curve, at a
fraction of the cost of the full 4D optimization.

No tensorflow/trieste needed. Output format matches the BO experiments
(``exp_NNNN`` folders + ``experiments.json`` ledger), so the resume logic and
plotting utilities work unchanged.

Example (ARCHER2)::

    python -m adbo.sweep_vmax --curve_nc w22/data/curves/2025_new_orleans.nc \
        --exp_name no-sweep-2025 --num 15 \
        --angle 0.0 --trans_speed 7.71 --displacement 0.0

"""

import argparse
import os
from typing import Optional

import numpy as np
from sithom.io import read_json, write_json
from sithom.time import time_stamp, timeit

from adforce.constants import NEW_ORLEANS


@timeit
def sweep_vmax(
    curve_nc: str,
    exp_name: str = "vmax_sweep",
    num: int = 15,
    angle: float = 0.0,
    trans_speed: float = 7.71,
    displacement: float = 0.0,
    obs_lon: float = NEW_ORLEANS.lon,
    obs_lat: float = NEW_ORLEANS.lat,
    resolution: str = "mid",
    root_exp_direc: Optional[str] = None,
    wrap_test: bool = False,
    resume: bool = True,
) -> dict:
    """Run the intensity sweep.

    Args:
        curve_nc (str): Stored tradeoff curve (``w22.tradeoff.generate_curve``).
        exp_name (str): Experiment (output folder) name.
        num (int, optional): Number of intensities, evenly spaced over the
            curve domain [Cat-1, potential intensity]. Defaults to 15.
        angle (float, optional): Fixed track bearing [deg]. Defaults to 0.
        trans_speed (float, optional): Fixed translation speed [m/s].
            Defaults to 7.71.
        displacement (float, optional): Fixed impact-longitude offset [deg]
            from the observation point. Defaults to 0.
        obs_lon (float, optional): Observation longitude. Defaults to New
            Orleans.
        obs_lat (float, optional): Observation latitude. Defaults to New
            Orleans.
        resolution (str, optional): ADCIRC mesh resolution. Defaults to "mid".
        root_exp_direc (str, optional): Root experiment directory; defaults
            to $WORSTSURGE_EXP_DIR or the adbo EXP_PATH (as in adbo.exp).
        wrap_test (bool, optional): Skip ADCIRC, write configs + fake results
            (plumbing test). Defaults to False.
        resume (bool, optional): Skip intensities already in the ledger.
            Defaults to True.

    Returns:
        dict: The completed ledger (call index -> record).
    """
    # local imports keep `import adbo.sweep_vmax` light (numba/adforce heavy)
    from w22.tradeoff import TradeoffCurve
    from adforce.wrap import idealized_tc_observe
    from .constants import EXP_PATH
    from .wrap_utils import build_wrap_config

    if root_exp_direc is None:
        root_exp_direc = os.environ.get("WORSTSURGE_EXP_DIR", EXP_PATH)
    direc = os.path.join(root_exp_direc, exp_name)
    os.makedirs(direc, exist_ok=True)

    curve = TradeoffCurve.from_file(curve_nc)
    v_grid = np.linspace(curve.v_min, curve.v_max, num)

    cfg = {
        "obs_lon": obs_lon,
        "obs_lat": obs_lat,
        "resolution": resolution,
        "profile_name": None,  # always from the curve
        "curve_path": curve_nc,
        "angle": angle,
        "trans_speed": trans_speed,
        "displacement": displacement,
        "num": num,
        "wrap_test": wrap_test,
        "sweep_started_at": time_stamp(),
    }
    write_json(cfg, os.path.join(direc, "sweep-config.json"))

    ledger_path = os.path.join(direc, "experiments.json")
    ledger = {}
    if resume and os.path.exists(ledger_path):
        ledger = read_json(ledger_path)
        print(f"sweep_vmax: resuming, {len(ledger)} evaluations already done.")

    for i, v in enumerate(v_grid):
        key = str(i)
        if key in ledger:
            print(f"sweep_vmax: [{i + 1}/{num}] V={v:.2f} m/s already done, skipping.")
            continue
        tmp_dir = os.path.join(direc, f"exp_{i:04}")
        os.makedirs(tmp_dir, exist_ok=True)
        inputs = {
            "angle": angle,
            "trans_speed": trans_speed,
            "displacement": displacement,
            "vmax": float(v),
        }
        wrap_cfg = build_wrap_config(cfg, inputs, tmp_dir, tradeoff_curve=curve)
        print(f"sweep_vmax: [{i + 1}/{num}] V={v:.2f} m/s -> {tmp_dir}")
        if wrap_test:
            from omegaconf import OmegaConf

            with open(os.path.join(tmp_dir, "wrap_cfg.yaml"), "w") as file:
                file.write(OmegaConf.to_yaml(wrap_cfg))
            res = float(min(7 + np.random.normal(), 10))
        else:
            try:
                res = float(idealized_tc_observe(wrap_cfg))
            except ValueError as e:
                # sweep points are independent: record the failure and move on
                print(
                    f"sweep_vmax: FAILED at V={v:.2f} m/s ({tmp_dir}): {e}; "
                    "recording NaN and continuing."
                )
                res = float("nan")
        ledger[key] = {"": tmp_dir, "res": res, **inputs}
        write_json(ledger, ledger_path)

    n_bad = sum(1 for r in ledger.values() if not np.isfinite(r["res"]))
    print(f"sweep_vmax: complete, {len(ledger)}/{num} runs, {n_bad} failed.")
    return ledger


if __name__ == "__main__":
    # python -m adbo.sweep_vmax --curve_nc <curve.nc> --exp_name no-sweep-2025
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve_nc", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="vmax_sweep")
    parser.add_argument("--num", type=int, default=15)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--trans_speed", type=float, default=7.71)
    parser.add_argument("--displacement", type=float, default=0.0)
    parser.add_argument("--obs_lon", type=float, default=NEW_ORLEANS.lon)
    parser.add_argument("--obs_lat", type=float, default=NEW_ORLEANS.lat)
    parser.add_argument("--resolution", type=str, default="mid")
    parser.add_argument("--test", action="store_true", help="wrap_test mode")
    args = parser.parse_args()
    sweep_vmax(
        curve_nc=args.curve_nc,
        exp_name=args.exp_name,
        num=args.num,
        angle=args.angle,
        trans_speed=args.trans_speed,
        displacement=args.displacement,
        obs_lon=args.obs_lon,
        obs_lat=args.obs_lat,
        resolution=args.resolution,
        wrap_test=args.test,
    )
