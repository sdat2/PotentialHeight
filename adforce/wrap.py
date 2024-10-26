"""Wrap the adcirc call."""

import os
import hydra
from omegaconf import DictConfig
from .constants import CONFIG_PATH, DATA_PATH


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="tc_param_wrap")
def idealized_tc_observe(cfg: DictConfig) -> None:
    """Wrap the adcirc call.

    Args:
        cfg (DictConfig): configuration.
    """

    print(cfg)


if __name__ == "__main__":
    # python -m adforce.wrap
    idealized_tc_observe()
