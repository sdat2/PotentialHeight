"""A file to store the configuration formatters for the adforce package.

We use hydra to manage the configuration files. This file contains the dataclass that will be used to store the configuration parameters.

Attributes:
    ChavasCollision: A dataclass to store the impact location of the Chavas Lin and Emanuel (2015) profile collision on a straight track.
"""

from dataclasses import dataclass
import hydra
from omegaconf import DictConfig


@dataclass
class ChavasCollision:
    impact_lon: float
    impact_lat: float
