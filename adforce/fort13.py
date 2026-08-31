"""Minimal fort.13 (ADCIRC nodal-attribute) editing.

Purpose-built for the Manning's-n tidal-friction sweep
(``adforce/eval/tidal_diagnosis.md``): rewrite the DEFAULT value of the
``mannings_n_at_sea_floor`` attribute while leaving the per-node override
blocks untouched, so a friction cell differs from the control by exactly one
number.

fort.13 layout (relevant part): line 1 description, line 2 node count,
line 3 attribute count NATTR, then NATTR four-line header blocks
(name / units / values-per-node / default value(s)), then the per-attribute
data sections. Only the header block's default line is edited here.
"""

from __future__ import annotations

import re

MANNINGS = "mannings_n_at_sea_floor"


def _default_line_index(lines: list, attribute: str = MANNINGS) -> int:
    """Index of the attribute's default-value line in its HEADER block."""
    n_attr = int(lines[2].split()[0])
    i = 3
    for _ in range(n_attr):
        name = lines[i].strip()
        if name == attribute:
            return i + 3  # name / units / values-per-node / DEFAULT
        i += 4
    raise ValueError(f"attribute {attribute!r} not in fort.13 header ({n_attr} attrs)")


def read_mannings_default(path: str) -> float:
    """The uniform default Manning's n recorded in a fort.13 header."""
    with open(path) as f:
        lines = [ln for _, ln in zip(range(64), f)]  # headers live at the top
    return float(lines[_default_line_index(lines)].split()[0])


def write_mannings_default(src: str, dst: str, n: float) -> None:
    """Copy ``src`` -> ``dst`` with the default Manning's n replaced by ``n``.

    Per-node override sections are preserved byte-for-byte; only the single
    header default line changes (formatted to six decimals, matching the
    shipped decks).
    """
    with open(src) as f:
        lines = f.readlines()
    i = _default_line_index(lines)
    old = lines[i]
    indent = re.match(r"\s*", old).group(0)
    lines[i] = f"{indent}{n:.6f}\n"
    with open(dst, "w") as f:
        f.writelines(lines)
