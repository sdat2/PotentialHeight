"""Harvest the minimal artifact set of a remote sweep to the laptop.

Runs execute on GCP (spot VMs, docker); scoring runs locally. Per run only the small
artifacts travel (~MBs instead of the 5-8 GB ``fort.63.nc``):
``config.yaml`` (provenance), ``gauge_ts.parquet`` (the science),
``fort.61.nc`` (station cross-check), ``maxele.63.nc`` (run health),
``slurm.out`` (receipts), and the study's ``eval_manifest.json``.

Run (hydra; config root adforce/eval/config/harvest_config.yaml)::

    python -m adforce.eval.harvest remote=gcp-vm:/work/exp/eval study=kat-ida
    python -m adforce.eval.harvest remote=... study=... dry_run=false
"""

from __future__ import annotations

import os
import subprocess
from typing import List

import hydra
from omegaconf import DictConfig

from . import constants as C

#: files worth moving; everything else is excluded
INCLUDE = [
    "config.yaml",
    "gauge_ts.parquet",
    "fort.61.nc",
    "maxele.63.nc",
    "slurm.out",
    "eval_manifest.json",
]


def rsync_command(remote: str, study: str, dest: str) -> List[str]:
    """Build the rsync invocation (pure; unit-testable)."""
    cmd = ["rsync", "-av", "--prune-empty-dirs", "--include=*/"]
    cmd += [f"--include={name}" for name in INCLUDE]
    cmd += ["--exclude=*", f"{remote.rstrip('/')}/{study}/", f"{dest.rstrip('/')}/"]
    return cmd


def harvest(remote: str, study: str, dest: str = None, dry_run: bool = True) -> List[str]:
    """Rsync one study's minimal artifacts into the local runs mirror.

    Args:
        remote (str): ``host:/path/to/exp/eval`` (parent of study dirs).
        study (str): Study name (the ``<study>/`` subtree to pull).
        dest (str): Local mirror root (default ``data/comp/runs/<study>``).
        dry_run (bool): Print the command instead of running it.

    Returns:
        List[str]: The rsync argv (executed unless ``dry_run``).
    """
    dest = dest or os.path.join(C.COMP_DATA_PATH, "runs", study)
    cmd = rsync_command(remote, study, dest)
    print(" ".join(cmd))
    if not dry_run:
        os.makedirs(dest, exist_ok=True)
        subprocess.run(cmd, check=True)
    return cmd


@hydra.main(version_base=None, config_path="config", config_name="harvest_config")
def main(cfg: DictConfig) -> None:
    harvest(str(cfg.remote), str(cfg.study), dest=cfg.dest, dry_run=cfg.dry_run)


if __name__ == "__main__":
    main()
