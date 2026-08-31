"""Run-directory status for eval orchestration and discovery.

Generalizes the receipt logic of ``adforce.check_training_runs.status_list``
(the ``'Job completed successfully.'`` line that the wrap/slurm path prints
into ``<run>/slurm.out``) with provenance checking against the cell a
directory is supposed to hold.

Deliberately light on imports: the laptop scoring path must work without the
launch stack, so this module never imports ``adforce.wrap`` or slurmpy.
Slurm-queue polling (RUNNING) and the maxele-based fallback for subprocess
runs without a ``slurm.out`` belong to the launch backend (Commit E), not
here.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from .cells import ConfigCell, provenance_match

#: Written by the wrap slurm/subprocess paths on success (see
#: adforce/check_training_runs.py, which scans for the same line).
SUCCESS_MARKER = "Job completed successfully."


class RunStatus(str, Enum):
    MISSING = "missing"  # no directory or no config.yaml provenance
    FOREIGN = "foreign"  # config.yaml disagrees with the expected cell
    EXTRACTED = "extracted"  # success + gauge_ts.parquet already reduced
    SUCCESS = "success"  # success receipt in slurm.out
    FAILED = "failed"  # provenance present but no success receipt


def _has_success_receipt(run_dir: str) -> bool:
    out = os.path.join(run_dir, "slurm.out")
    if not os.path.exists(out):
        return False
    try:
        with open(out, errors="ignore") as f:
            return any(SUCCESS_MARKER in line for line in f)
    except OSError:
        return False


def run_status(run_dir: str, cell: Optional[ConfigCell] = None) -> RunStatus:
    """Classify one run directory.

    Args:
        run_dir (str): The per-storm run directory.
        cell (Optional[ConfigCell]): Expected cell; when given, the dumped
            ``config.yaml`` must match its physical axes or the directory is
            FOREIGN (never scored, never overwritten without ``overwrite``).

    Returns:
        RunStatus: See the enum docstrings.
    """
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isdir(run_dir) or not os.path.exists(cfg_path):
        return RunStatus.MISSING
    if cell is not None:
        from adforce.config import load_config  # omegaconf only, no wrap import

        try:
            run_cfg = load_config(cfg_path)
        except Exception:
            return RunStatus.MISSING
        if not provenance_match(run_cfg, cell):
            return RunStatus.FOREIGN
    if not _has_success_receipt(run_dir):
        return RunStatus.FAILED
    if os.path.exists(os.path.join(run_dir, "gauge_ts.parquet")):
        return RunStatus.EXTRACTED
    return RunStatus.SUCCESS
