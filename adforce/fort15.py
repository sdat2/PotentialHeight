"""Minimal fort.15 (ADCIRC control file) editing.

Purpose-built for the bottom-friction sensitivity sweep: with ``NWP = 0``
(both the driver-generated and the static decks -- fort.13 is never read),
the live friction knob is the ``NOLIBF = 2`` hybrid-friction line
``CF HBREAK FTHETA FGAMMA`` (shipped: ``0.0025 1 10 0.333333`` -- a uniform
quadratic coefficient, with Manning-like depth scaling only below
HBREAK = 1 m). Rewrite CF, keep the other three parameters.
"""

from __future__ import annotations

import re

#: matches the hybrid-friction line's comment in both deck styles:
#: "! CF HBREAK FTHETA FGAMMA" (adcircpy) and "! FFACTOR,HBREAK,FTHETA,FGAMMA"
_FRICTION_COMMENT = re.compile(r"!.*HBREAK", re.IGNORECASE)


def _friction_line_index(lines: list) -> int:
    hits = [i for i, ln in enumerate(lines) if _FRICTION_COMMENT.search(ln)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one CF/HBREAK line, found {len(hits)}")
    return hits[0]


def read_friction_cf(path: str) -> float:
    """The hybrid-friction coefficient CF recorded in a fort.15."""
    with open(path) as f:
        lines = f.readlines()
    return float(lines[_friction_line_index(lines)].split()[0])


def write_friction_cf(src: str, dst: str, cf: float) -> None:
    """Copy ``src`` -> ``dst`` with CF replaced (HBREAK/FTHETA/FGAMMA kept).

    Only the leading coefficient of the single friction line changes; the
    rest of the deck is preserved byte-for-byte.
    """
    with open(src) as f:
        lines = f.readlines()
    i = _friction_line_index(lines)
    old = lines[i]
    indent = re.match(r"\s*", old).group(0)
    rest = old.split(None, 1)[1]  # HBREAK FTHETA FGAMMA + comment
    lines[i] = f"{indent}{cf:g} {rest}"
    with open(dst, "w") as f:
        f.writelines(lines)
