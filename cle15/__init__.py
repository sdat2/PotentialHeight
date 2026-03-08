"""
cle15 — Chavas, Lin & Emanuel (2015) tropical cyclone wind profile.

Three implementations are provided:

* :mod:`cle15.cle15`  — pure-Python reference
* :mod:`cle15.cle15n` — numba-accelerated drop-in replacement
* :mod:`cle15.cle15m` — thin wrapper around the original MATLAB/Octave scripts

All three expose the same public API:
``run_cle15``, ``profile_from_stats``, ``process_inputs``.
"""
