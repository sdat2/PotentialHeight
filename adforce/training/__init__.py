"""SurgeNet ADCIRC training-data generation package.

This package is the split-up successor of the monolithic
``adforce/generate_training_data.py`` (which remains as a back-compat shim
and as the CLI entry point ``python -m adforce.generate_training_data``):

- ``storms``: ``Storm`` container and ``calculate_simulation_window`` (stdlib only).
- ``atcf``: IBTrACS -> ASWIP/ATCF text conversion (numpy/pandas/xarray only).
- ``cfl``: ``calculate_cfl_timestep`` CFL condition helper (duck-typed mesh).
- ``inputs``: adcircpy input-deck generation (``CustomAdcircRun``,
  ``generate_adcirc_inputs``) — heavy adcircpy import.
- ``driver``: ``drive_all_adcirc`` orchestration and the ``main()`` CLI —
  heavy imports (hydra config, tcpips IBTrACS selection, subprocess runner).

Names are re-exported lazily so that ``import adforce.training`` stays cheap:
the heavy adcircpy/hydra imports only happen when a name from ``inputs`` or
``driver`` is first touched.
"""

import importlib

__all__ = [
    "Storm",
    "calculate_simulation_window",
    "clean_radii",
    "convert_ibtracs_storm_to_aswip_input",
    "convert_ibtracs_storm_to_atcf",
    "calculate_cfl_timestep",
    "CustomAdcircRun",
    "generate_adcirc_inputs",
    "drive_all_adcirc",
    "main",
]

# name -> submodule holding it (resolved lazily on first attribute access)
_EXPORTS = {
    "Storm": "storms",
    "calculate_simulation_window": "storms",
    "_decode_char_array": "atcf",
    "clean_radii": "atcf",
    "convert_ibtracs_storm_to_aswip_input": "atcf",
    "convert_ibtracs_storm_to_atcf": "atcf",
    "calculate_cfl_timestep": "cfl",
    "CustomAdcircRun": "inputs",
    "generate_adcirc_inputs": "inputs",
    "FORT14_PATH": "inputs",
    "FORT13_PATH": "inputs",
    "drive_all_adcirc": "driver",
    "main": "driver",
    "RUNS_PARENT_DIR": "driver",
}


def __getattr__(name: str):
    """Lazily resolve re-exported names from their submodules (PEP 562)."""
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + list(_EXPORTS)))
