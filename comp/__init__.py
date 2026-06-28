"""comp: compare historical ADCIRC surge simulations against NOAA tide gauges.

Validates the SurgeNet historical-storm ADCIRC dataset (Hugging Face
``sdat2/surgenet-train``) against de-tided NOAA CO-OPS observations.

Entry point::

    python -m comp.validate
"""

from .validate import run, validate_storm, metrics

__all__ = ["run", "validate_storm", "metrics"]
