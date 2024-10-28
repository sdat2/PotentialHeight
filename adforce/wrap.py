"""Wrap the adcirc call."""

import os, shutil
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from .constants import CONFIG_PATH, DATA_PATH, SETUP_PATH
from .fort22 import create_fort22
from .slurm import setoff_slurm_job_and_wait
from .mesh import xr_loader


def observe_max_point(cfg: DictConfig) -> float:
    """Observe the ADCIRC model."""
    mele_ds = xr_loader(os.path.join(cfg.files.run_folder, "maxele.63.nc"))
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    at_obs_loc = cfg.adcirc.attempted_observation_location.value

    distsq = (xs - at_obs_loc[0]) ** 2 + (ys - at_obs_loc[1]) ** 2
    min_p = np.argmin(distsq)
    maxele = mele_ds.zeta.isel(node=min_p).values
    point = (xs[min_p], ys[min_p])
    cfg.adcirc["actual_observation_location"]["value"] = point

    return maxele


def save_config(cfg: DictConfig) -> None:
    """Save the configuration file."""
    with open(os.path.join(cfg.files.run_folder, "config.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)


# version_base=None,
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="wrap_config")
def idealized_tc_observe(cfg: DictConfig) -> float:
    """Wrap the adcirc call.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        float: max water level at observation point.
    """
    cfg.files["run_folder"] = os.path.join(cfg.files.exp_path, cfg.name)
    os.makedirs(cfg.files.run_folder, exist_ok=True)
    # transfer relevant ADCIRC setup files
    shutil.copy(
        os.path.join(SETUP_PATH, "fort.15.mid.notide"),
        os.path.join(cfg.files.run_folder, "fort.15"),
    )
    shutil.copy(
        os.path.join(SETUP_PATH, "fort.13.mid"),
        os.path.join(cfg.files.run_folder, "fort.13"),
    )
    shutil.copy(
        os.path.join(SETUP_PATH, "fort.14.mid"),
        os.path.join(cfg.files.run_folder, "fort.14"),
    )

    print(cfg)
    # save config file
    save_config(cfg)
    # create forcing files
    create_fort22(cfg.files.run_folder, cfg.grid, cfg.tc)
    # run ADCIRC
    setoff_slurm_job_and_wait(cfg.files.run_folder, cfg)
    # observe ADCIRC
    maxele = observe_max_point(cfg)
    # save config file
    save_config(cfg)
    return maxele  # not yet implemented


if __name__ == "__main__":
    # python -m adforce.wrap
    idealized_tc_observe()
