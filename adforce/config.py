"""A file to store the configuration formatters for the adforce package.

We use hydra to manage the configuration files. This file contains the dataclass that will be used to store the configuration parameters.

# TODO: Implement the configuration dataclass for the adforce package.

Attributes:
    ChavasCollision: A dataclass to store the impact location of the Chavas Lin and Emanuel (2015) profile collision on a straight track.
"""

import os
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class ChavasCollision:
    impact_lon: float
    impact_lat: float


def save_config(cfg: DictConfig) -> None:
    """Save the configuration file for a wrapped ADCIRC run.

    Args:
        cfg (DictConfig): configuration.
    """
    with open(os.path.join(cfg.files.run_folder, "config.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)


def load_config(path: str) -> DictConfig:
    """Load the configuration file for a wrapped ADCIRC run.

    Args:
        path (str): path to the config file.

    Returns:
        DictConfig: loaded configuration.
    """
    with open(path, "r") as fp:
        cfg = OmegaConf.load(fp.name)
    return cfg
