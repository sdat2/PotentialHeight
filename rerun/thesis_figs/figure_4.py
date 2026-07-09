"""Regenerate thesis figure_4a / figure_4b (+ the 4a/4b-*-ours.csv) — fixed PS solver.

Purpose
-------
Recreate the Wang (2022) figure-4 reproduction that appears in the thesis /
potential-height paper:

- ``figure_4a`` — central-pressure ``p_m`` vs outer radius ``r_a``: the digitized
  W22 and CLE15 curves from the paper, OUR CLE15 curve (Octave
  ``cle15.cle15m.run_cle15``), OUR W22 Carnot curve (inline
  ``bisection(wang_diff(...))``), and the single fixed-solver point solution
  ``w22.ps.point_solution_ps`` (the green ``x``).
- ``figure_4b`` — the energy-budget terms (``W_PBL``, ``Q_S``, ``W_P``,
  ``-Q_Gibbs``, ``W_out``) vs outer radius, digitized vs ours.

These use the FIXED potential-size solver (2026-07 humidity/units bug fix:
``w22.w22_carnot.carnot_pm_from_y`` via ``w22.ps``). Canonical point solution:
r0 = 2128.81 km, rmax = 60.67 km, pm = 942.04 mbar, pc = 887.93 mbar at
CkCd = 1, rh = 0.9 (Wang 2022 targets: r0 = 2193 km, rmax = 64 km, pm = 944 mbar).

This is a faithful, durable copy of the recovered scratchpad ``plot_fig4.py``.
It is a thin driver only: no numerical logic lives here. It imports the repo
module ``w22.test_figures`` and calls ``test_figure_4()`` DIRECTLY. That repo
function is decorated with ``pytest.mark.skipif`` (Octave / digitized data /
``W22_SLOW_TESTS``) guards, but ``skipif`` only affects pytest *collection* —
calling the function directly bypasses the skips and runs the full ~95 Octave
subprocess calls (minutes). The candidate-name loop and the exception-swallowing
``try/except`` are preserved from the transcript exactly.

Durable inputs (read by ``test_figure_4``, all under ``w22/data/w22``)
----------------------------------------------------------------------
Digitized Wang (2022) figure data:
  ``4a-cle15.csv``, ``4a-w22.csv``, ``4b-Wpbl.csv``, ``4b-Qs.csv``,
  ``4b-Qgibbs.csv``, ``4b-Wout.csv``, ``4b-Wpbl-surplus.csv``.
Also requires GNU Octave on PATH (``cle15.cle15m.run_cle15`` shells out to it).

Outputs (written by ``test_figure_4``; repo-internal, durable)
--------------------------------------------------------------
- ``w22/data/w22/4a-cle15-ours.csv``
- ``w22/data/w22/4a-w22-ours.csv``
- ``w22/data/w22/4b-w22-ours.csv``
- ``w22/img/w22/figure_4a.pdf``
- ``w22/img/w22/figure_4b.pdf``
(``DATA_PATH`` / ``FIGURE_PATH`` are resolved by ``w22.constants`` relative to
the ``w22`` package; there is no scratchpad path to redirect. The ``data/ibtracs``
symlink is not relevant to this figure.)

How to run (macOS/exFAT-safe, under caffeinate + the RSS/wall watchdog)
----------------------------------------------------------------------
    caffeinate -is python rerun/mem_guard.py rerun/thesis_figs/figure_4.py 6000 3600

or directly (env guards are set at the top of this file):

    caffeinate -is python rerun/thesis_figs/figure_4.py

The env guards below (single-thread BLAS, capped LOKY workers, disabled HDF5
file locking) are required on macOS/exFAT to avoid BrokenProcessPool, file-lock
errors and multi-core memory blow-ups.
"""

import os
import warnings

# --- macOS/exFAT env-safety guards (must be set before heavy imports) ---
os.environ["LOKY_MAX_CPU_COUNT"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import w22.test_figures as tf

# figure_4a (single-solution crossing) — find the producing fn and run it.
for fn in ("test_figure_4a", "figure_4a", "test_figure_4"):
    if hasattr(tf, fn):
        print("running", fn, flush=True)
        try:
            getattr(tf, fn)()
            print("  OK", fn)
        except Exception as e:
            print("  FAIL", fn, type(e).__name__, str(e)[:200])
        break
print("DONE")
