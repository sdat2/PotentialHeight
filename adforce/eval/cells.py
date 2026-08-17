"""Configuration-cell identity for adforce run comparison.

A *cell* is one point in the comparison matrix (resolution x tide x swan x
forcing x physics tag). Its canonical string (:func:`cell_id`) names run
directories, cache files, and model-vs-model join keys; provenance is always
established from the ``<run>/config.yaml`` that ``adforce.config.save_config``
writes (never from a directory name alone).

Storm identity is kept separate from the cell: the same cell is run once per
storm, in ``<runs_root>/<study>/<cell_id>/<storm_slug>/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Wrap-config paths for the physical axes (the one place the wrap tree's
# {value, ...} nesting appears on the eval side).
AXIS_PATHS = {
    "resolution": "adcirc.resolution.value",
    "tide": "adcirc.tide.value",
    "swan": "adcirc.swan.value",
}


@dataclass(frozen=True, order=True)
class ConfigCell:
    """One comparison-matrix cell (storm-independent)."""

    resolution: str = "mid"  # low | mid | high
    tide: bool = False  # tidal forcing in fort.15
    swan: bool = False  # SWAN wave coupling (padcswan)
    forcing: str = "storm"  # storm | tide | both (tide = wind-off control run)
    physics_tag: str = "default"  # opaque label for future physics variants
    mannings_n: Optional[float] = None  # fort.13 default override (None = shipped 0.022)


def _onoff(b: bool) -> str:
    return "on" if b else "off"


def cell_id(c: ConfigCell) -> str:
    """Deterministic cell name, e.g. ``res-mid_tide-off_swan-off_f-storm``.

    The ``physics_tag`` is appended only when non-default, so today's standard
    cells stay short and stable.
    """
    parts = [
        f"res-{c.resolution}",
        f"tide-{_onoff(c.tide)}",
        f"swan-{_onoff(c.swan)}",
        f"f-{c.forcing}",
    ]
    if c.mannings_n is not None:
        parts.append(f"n{c.mannings_n:g}")
    if c.physics_tag != "default":
        parts.append(f"p-{c.physics_tag}")
    return "_".join(parts)


def dir_to_storm(d: str) -> Optional[str]:
    """``'22_MICHAEL_2018'`` -> ``'Michael 2018'`` (val_summary storm naming).

    Ported from ``rerun/adcirc/score_resolution.py`` so legacy GCP/ARCHER2
    sweep directories (``<i>_<NAME>_<YEAR>``) can be ingested read-only.
    Returns None when the basename does not match the convention.
    """
    m = re.match(r"\d+_(.+)_(\d{4})$", d)
    if m is None:
        return None
    name = m.group(1).replace("_", " ").title()
    return f"{name} {m.group(2)}"


def storm_to_slug(storm: str, fname: str) -> str:
    """``('Katrina 2005', '152_KATRINA_2005.nc')`` -> ``'152_KATRINA_2005'``.

    The slug matches both the HF archive filename stem and the run-dir
    basename produced by the training driver, so one name keys both sources.
    """
    return fname[:-3] if fname.endswith(".nc") else fname


def expand_matrix(matrix_cfg) -> list:
    """Expand the matrix-YAML grammar into ``[(cell_name, overrides), ...]``.

    Grammar (see ``adforce/eval/config/matrix/``): ``axes`` maps WRAP config
    paths (``adcirc.resolution.value`` etc.) to value lists whose cartesian
    product, minus partial-match ``exclude`` dicts, forms the grid;
    ``cells``/``include`` append fully explicit ``{name, overrides}`` entries.
    Cell names come from ``name_keys`` aliases in axes order (bools render
    ``on``/``off``), e.g. ``res-low_tide-off``.

    Raises:
        ValueError: On duplicate cell names or a ``baseline`` naming no cell.
    """
    from itertools import product

    axes = dict(matrix_cfg.get("axes") or {})
    name_keys = dict(matrix_cfg.get("name_keys") or {})
    excludes = [dict(e) for e in (matrix_cfg.get("exclude") or [])]

    def render(path, val) -> str:
        alias = name_keys.get(path, path.split(".")[-1])
        v = ("on" if val else "off") if isinstance(val, bool) else str(val)
        return f"{alias}-{v}"

    cells = []
    if axes:
        paths = list(axes)
        for combo in product(*(list(axes[p]) for p in paths)):
            ov = dict(zip(paths, combo))
            if any(all(ov.get(k) == v for k, v in ex.items()) for ex in excludes):
                continue
            cells.append(("_".join(render(p, ov[p]) for p in paths), ov))
    for group in ("cells", "include"):
        for item in matrix_cfg.get(group) or []:
            cells.append((str(item["name"]), dict(item["overrides"])))

    names = [n for n, _ in cells]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate cell names in matrix: {dups}")
    baseline = matrix_cfg.get("baseline")
    if baseline and baseline not in names:
        raise ValueError(f"baseline {baseline!r} is not one of the cells {names}")
    return cells


def provenance_match(run_cfg, cell: ConfigCell) -> bool:
    """Does a run's dumped ``config.yaml`` match this cell's physical axes?

    Args:
        run_cfg: OmegaConf ``DictConfig`` loaded from ``<run>/config.yaml``
            (via :func:`adforce.config.load_config`).
        cell (ConfigCell): The cell the directory is expected to hold.

    Returns:
        bool: True when resolution/tide/swan all agree, and -- for cells that
        carry non-default eval-side axes (mannings_n set, or a tide-only
        control) -- when the run's ``eval_axes`` block matches too. Legacy
        run dirs predate ``eval_axes``, so those checks only fire for
        non-default cells. Missing wrap keys count as a mismatch (FOREIGN)
        -- a run without provenance is never trusted.
    """
    try:
        adcirc = run_cfg.adcirc
        if not (
            str(adcirc.resolution.value) == cell.resolution
            and bool(adcirc.tide.value) == cell.tide
            and bool(adcirc.swan.value) == cell.swan
        ):
            return False
        axes = run_cfg.get("eval_axes") or {}
        if cell.mannings_n is not None:
            got = axes.get("mannings_n")
            if got is None or abs(float(got) - cell.mannings_n) > 1e-9:
                return False
        if cell.forcing == "tide" and axes.get("forcing") != "tide":
            return False
        return True
    except Exception:
        return False
