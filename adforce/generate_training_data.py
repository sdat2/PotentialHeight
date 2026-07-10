"""Back-compatibility shim: this module moved to the ``adforce.training`` package.

The monolithic training-data generator that used to live here was split into:

- ``adforce.training.storms``  — ``Storm``, ``calculate_simulation_window``
- ``adforce.training.atcf``    — ``_decode_char_array``, ``clean_radii``,
  ``convert_ibtracs_storm_to_aswip_input``, ``convert_ibtracs_storm_to_atcf``
- ``adforce.training.cfl``     — ``calculate_cfl_timestep``
- ``adforce.training.inputs``  — ``CustomAdcircRun``, ``generate_adcirc_inputs``
  (adcircpy-heavy; also ``FORT14_PATH``/``FORT13_PATH``)
- ``adforce.training.driver``  — ``drive_all_adcirc``, ``main`` CLI
  (also ``RUNS_PARENT_DIR``)

All of those names remain importable from this module (resolved lazily, so
importing this shim stays cheap), and the CLI is unchanged:

    python -m adforce.generate_training_data [--test-single] [--test-nosubprocess] [--recommended-dt 5.0] [--runs-parent-name test_runs]

The block of commented-out historical experiment variants (VortexTrack
probes, drive_katrina, etc.) that used to sit below the ``__main__`` block
was deleted in the split; see the git history of this file if you need them.
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

# name -> new home (resolved lazily on first attribute access, PEP 562)
_EXPORTS = {
    "Storm": "adforce.training.storms",
    "calculate_simulation_window": "adforce.training.storms",
    "_decode_char_array": "adforce.training.atcf",
    "clean_radii": "adforce.training.atcf",
    "convert_ibtracs_storm_to_aswip_input": "adforce.training.atcf",
    "convert_ibtracs_storm_to_atcf": "adforce.training.atcf",
    "calculate_cfl_timestep": "adforce.training.cfl",
    "CustomAdcircRun": "adforce.training.inputs",
    "generate_adcirc_inputs": "adforce.training.inputs",
    "FORT14_PATH": "adforce.training.inputs",
    "FORT13_PATH": "adforce.training.inputs",
    "drive_all_adcirc": "adforce.training.driver",
    "main": "adforce.training.driver",
    "RUNS_PARENT_DIR": "adforce.training.driver",
}


def __getattr__(name: str):
    """Lazily import moved names from their new adforce.training homes."""
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + list(_EXPORTS)))


if __name__ == "__main__":
    # python -m adforce.generate_training_data
    from adforce.training.driver import main

    main()
