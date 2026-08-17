"""Sweep orchestrator: launch the comparison matrix of adforce runs.

Replaces ``rerun/adcirc/hist_sweep.sh``'s env-var matrix with the eval matrix
YAML (``config/matrix/``). Every historical cell routes through
``adforce.training.driver.drive_storm`` -- the per-storm input generation
that builds each storm's fort.15 with the correct cold-start/tidal window.
``wrap.stage_input_files``'s static decks (Katrina-pinned tidal window) are
never used here, which is what makes tide-on cells for the other 18 storms
legal at all.

``dry_run: true`` is the default: ``python -m adforce.eval.launch`` prints
the plan table and exits without firing anything.

Deliberate limits (fail loudly, not silently):
* ``swan`` cells are BLOCKED -- the training driver's input generation does
  not stage SWAN decks (idealized-TC swan runs go through ``adforce.wrap``).
* the slurm backend is not implemented yet -- run the subprocess backend
  inside one allocation (as ``hist_sweep.sh`` did on GCP), or sbatch a shell
  loop; a throttled slurmpy backend is a known follow-up.

All runner imports (wrap, training.driver, tcpips) stay function-local so the
laptop scoring path never needs the launch stack.

Run (hydra; config root adforce/eval/config/launch_config.yaml)::

    python -m adforce.eval.launch study=kat-ida matrix=res_x_tide \
        'storms=["Katrina 2005","Ida 2021"]'            # plan only (dry_run)
    python -m adforce.eval.launch study=kat-ida matrix=res_x_tide \
        'storms=["Katrina 2005","Ida 2021"]' dry_run=false
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from . import constants as C
from .cells import AXIS_PATHS, ConfigCell, cell_id, expand_matrix, storm_to_slug
from .status import RunStatus, run_status


def _cell_from_overrides(overrides: dict, forcing: str = None) -> ConfigCell:
    """Matrix overrides -> ConfigCell.

    Wrap-tree paths (``adcirc.*.value``) map onto the physical axes; the
    pseudo-axes ``forcing`` (storm|tide|both) and ``mannings_n`` (fort.13
    default override) are eval-side and never touch the wrap config.
    """
    get = lambda axis, default: overrides.get(AXIS_PATHS[axis], default)
    tide = bool(get("tide", False))
    mannings = overrides.get("mannings_n")
    return ConfigCell(
        resolution=str(get("resolution", "mid")),
        tide=tide,
        swan=bool(get("swan", False)),
        forcing=str(forcing or overrides.get("forcing") or ("both" if tide else "storm")),
        mannings_n=float(mannings) if mannings is not None else None,
    )


def _deck_exists(resolution: str) -> bool:
    from adforce.constants import SETUP_PATH  # light: paths only

    return os.path.exists(os.path.join(SETUP_PATH, f"fort.14.{resolution}"))


def _spinup(cfg, cell: ConfigCell) -> float:
    if str(cfg.spinup_days) != "auto":
        return float(cfg.spinup_days)
    return 6.0 if (cell.tide or cell.forcing in ("tide", "both")) else 0.0


def _runs_root(cfg) -> str:
    root = str(cfg.runs_root)
    return root if os.path.isabs(root) else os.path.join(C.PROJ_PATH, root)


def _expand_cells(cfg: DictConfig) -> list:
    """Matrix cells (+ auto tide-only controls) as ``[(name, ConfigCell)]``.

    Shared by :func:`plan` and :func:`launch` so both see identical cells --
    including the control cells, which exist only here, not in the matrix.
    """
    cells = [
        (name, _cell_from_overrides(dict(ov)))
        for name, ov in expand_matrix(cfg.matrix)
    ]
    if cfg.controls:
        # tide-only control run per unique (resolution, swan) with a tide-on
        # cell -- the third leg of the tide-surge-interaction triple.
        seen = {(c.resolution, c.swan) for _, c in cells if c.tide}
        for res, swan in sorted(seen):
            ctrl = ConfigCell(resolution=res, tide=True, swan=swan, forcing="tide")
            cells.append((cell_id(ctrl), ctrl))
    return cells


def plan(cfg: DictConfig) -> pd.DataFrame:
    """Expand the matrix into per-(cell, storm) rows with status and action.

    Pure (no launching, no heavy imports): usable from the laptop to preview
    exactly what a remote launch would do.

    Returns:
        pd.DataFrame: columns ``cell, storm, slug, run_dir, status, action,
        reason`` -- action in {run, skip, blocked}.
    """
    cells = _expand_cells(cfg)
    storms = list(cfg.storms) if cfg.storms else list(C.STORMS)
    rows = []
    for name, cell in cells:
        for storm in storms:
            slug = storm_to_slug(storm, C.STORMS[storm])
            run_dir = os.path.join(_runs_root(cfg), str(cfg.study), name, slug)
            status = run_status(run_dir, cell)
            action, reason = "run", ""
            if cell.swan:
                action, reason = "blocked", "driver input generation has no SWAN staging"
            elif not _deck_exists(cell.resolution):
                action, reason = (
                    "blocked",
                    f"no fort.14.{cell.resolution} deck in adforce/setup",
                )
            elif status is RunStatus.FOREIGN and not cfg.overwrite:
                action, reason = "blocked", "run dir holds a different config (FOREIGN)"
            elif status in (RunStatus.SUCCESS, RunStatus.EXTRACTED) and cfg.skip_completed:
                action, reason = "skip", "already successful"
            rows.append(
                dict(
                    cell=name,
                    storm=storm,
                    slug=slug,
                    run_dir=run_dir,
                    status=status.value,
                    action=action,
                    reason=reason,
                )
            )
    return pd.DataFrame(rows)


def config_hash(resolved_cfg) -> str:
    """sha256 of the full resolved run config: the provenance fingerprint
    stored in the manifest so model-vs-model can detect drift in
    un-overridden defaults between launches (see plan gap G5)."""
    return hashlib.sha256(
        OmegaConf.to_yaml(resolved_cfg, resolve=False).encode()
    ).hexdigest()


def _write_manifest(runs_root: str, study: str, entries: dict) -> str:
    path = os.path.join(runs_root, study, "eval_manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.exists(path):
        try:
            existing = json.load(open(path))
        except Exception:
            pass
    existing.update(entries)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    return path


def launch(cfg: DictConfig) -> pd.DataFrame:
    """Execute the plan's ``run`` rows sequentially via the driver seam.

    Sequential on purpose (the hist_sweep disk-pressure lesson): each run is
    reduced to ``gauge_ts.parquet`` and stripped before the next starts when
    ``extract_after_run`` is set.
    """
    import glob as _glob

    from adforce.wrap import get_default_config  # runner imports stay local

    table = plan(cfg)
    todo = table[table.action == "run"]
    print(table.to_string(index=False))
    if todo.empty:
        print("nothing to launch")
        return table

    from tcpips.ibtracs import na_landing_tcs

    from adforce.training.driver import drive_storm
    from adforce.training.storms import Storm

    target = na_landing_tcs()
    base_cfg = get_default_config()
    cells = dict(_expand_cells(cfg))
    gauges = None

    for row in todo.itertuples():
        cell = cells[row.cell]
        i = int(row.slug.split("_")[0])  # archive index == IBTrACS enumeration
        raw_times = pd.to_datetime(target.time[i].values, errors="coerce")
        storm = Storm(
            sid=target.sid[i].item(),
            name=target.name[i].item(),
            time=raw_times.dropna().to_pydatetime().tolist(),
        )
        wrap_cfg = base_cfg.copy()
        for axis, path in AXIS_PATHS.items():
            OmegaConf.update(wrap_cfg, path, getattr(cell, axis), merge=False)
        os.makedirs(row.run_dir, exist_ok=True)
        OmegaConf.update(wrap_cfg.files, "run_folder", row.run_dir)
        # eval-side axes recorded alongside the wrap config: provenance for
        # friction cells and tide-only controls (see cells.provenance_match)
        OmegaConf.update(
            wrap_cfg,
            "eval_axes",
            {"forcing": cell.forcing, "mannings_n": cell.mannings_n},
            force_add=True,
        )
        OmegaConf.save(wrap_cfg, os.path.join(row.run_dir, "config.yaml"))
        entry = {
            f"{row.cell}/{row.slug}": dict(
                status="launched", config_hash=config_hash(wrap_cfg)
            )
        }
        _write_manifest(_runs_root(cfg), str(cfg.study), entry)

        try:
            drive_storm(
                storm,
                target.isel(storm=i),
                row.run_dir,
                wrap_cfg,
                resolution=cell.resolution,
                mode=cell.forcing,
                spinup_days=_spinup(cfg, cell),
                recommended_dt=cfg.recommended_dt,
                mannings_n=cell.mannings_n,
            )
            status = run_status(row.run_dir, cell)
            if cfg.extract_after_run and status is RunStatus.SUCCESS:
                from .extract import extract_run, gauge_frame

                gauges = gauge_frame() if gauges is None else gauges
                df = extract_run(row.run_dir, gauges)
                if not df.empty:
                    df.to_parquet(
                        os.path.join(row.run_dir, "gauge_ts.parquet"), index=False
                    )
                for pat in cfg.strip_after_extract:
                    for f in _glob.glob(os.path.join(row.run_dir, str(pat))):
                        os.remove(f)
                status = run_status(row.run_dir, cell)
        except Exception as e:  # per-storm failures never stop the sweep
            print(f"!!! FAILED {row.cell}/{row.slug}: {e}")
            status = RunStatus.FAILED
        entry[f"{row.cell}/{row.slug}"]["status"] = (
            status.value if isinstance(status, RunStatus) else str(status)
        )
        _write_manifest(_runs_root(cfg), str(cfg.study), entry)
    return plan(cfg)


@hydra.main(version_base=None, config_path="config", config_name="launch_config")
def main(cfg: DictConfig) -> None:
    if cfg.dry_run:
        table = plan(cfg)
        print(table.to_string(index=False))
        n = (table.action == "run").sum()
        print(f"\ndry_run=true: {n} runs would launch; pass dry_run=false to execute")
        return
    launch(cfg)


if __name__ == "__main__":
    main()
