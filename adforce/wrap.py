"""Wrap the adcirc call."""

import os, shutil
import numpy as np
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sithom.time import timeit
from .constants import CONFIG_PATH, SETUP_PATH
from .fort22 import create_fort22
from .slurm import setoff_slurm_job_and_wait
from .mesh import xr_loader


def observe_max_point(cfg: DictConfig) -> float:
    """Observe the ADCIRC model."""
    mele_ds = xr_loader(os.path.join(cfg.files.run_folder, "maxele.63.nc"))
    # print("mele_ds", mele_ds)
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    at_obs_loc = cfg.adcirc.attempted_observation_location.value

    distsq = (xs - at_obs_loc[0]) ** 2 + (ys - at_obs_loc[1]) ** 2
    min_p = np.argmin(distsq)
    maxele = mele_ds.zeta_max.isel(node=min_p).values
    point = (xs[min_p], ys[min_p])
    cfg.adcirc["actual_observation_location"]["value"][0] = float(point[0])
    cfg.adcirc["actual_observation_location"]["value"][1] = float(point[1])

    print(
        "point info:",
        mele_ds.isel(node=min_p)["depth"],
        "\n depth at point: ",
        mele_ds.isel(node=min_p)["depth"].values,
        " m",
    )

    return maxele


def save_config(cfg: DictConfig) -> None:
    """Save the configuration file.

    Args:
        cfg (DictConfig): configuration.
    """
    with open(os.path.join(cfg.files.run_folder, "config.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)


@timeit
def idealized_tc_observe(cfg: DictConfig) -> float:
    """Wrap the adcirc call.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        float: max water level at observation point.
    """
    os.makedirs(cfg.files.run_folder, exist_ok=True)
    # transfer relevant ADCIRC setup files
    assert cfg.adcirc.tide.value == False
    assert cfg.adcirc.resolution.value == "mid"
    # other options not yet implemented
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
    print("max height at obs point: ", maxele, "m\n\n")

    if cfg.ani:
        from adforce.ani import plot_heights_and_winds

        plot_heights_and_winds(os.path.join(cfg.files.run_folder), step_size=10)

    return maxele


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="wrap_config")
def main(cfg: DictConfig) -> float:
    """
    Command line call

    Args:
        cfg (DictConfig): Full config file.

    Returns:
        float: float.
    """
    cfg["name"] == str(cfg["name"])
    cfg.files["run_folder"] = os.path.join(str(cfg.files.exp_path), str(cfg.name))
    return idealized_tc_observe(cfg)


def get_default_config() -> OmegaConf:
    with initialize(
        config_path=os.path.relpath(CONFIG_PATH, os.path.dirname(__file__))
    ):
        # Compose the configuration as if you were running from the command line.
        cfg = compose(config_name="wrap_config")
    return cfg


if __name__ == "__main__":
    # python -m adforce.wrap name=changed_calendar_wrap
    # python -m adforce.wrap name="2097" tc.profile_name.value="2097_new_orleans_profile_r4i1p1f1"
    main()
