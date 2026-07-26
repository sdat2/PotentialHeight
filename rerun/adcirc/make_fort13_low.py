"""Generate adforce/setup/fort.13.low from fort.13.mid by nearest-neighbour.

The low-resolution mesh (fort.14.low, 8,303 nodes) has no nodal-attribute
file, but a mesh-resolution sensitivity sweep must hold the *physics* fixed:
fort.13.mid carries mannings_n_at_sea_floor (spatially varying friction),
sea_surface_height_above_geoid (datum/steric offset) and
primitive_weighting_in_continuity_equation (Tau0). Dropping them (NWP=0)
would confound friction changes with resolution changes. This script samples
each mid-mesh attribute at the nearest mid node to every low node and writes
a structurally identical fort.13.low (same defaults; per-node exceptions
wherever the sampled value differs from the default).

Run once:  python rerun/adcirc/make_fort13_low.py
"""

import os

import numpy as np
from scipy.spatial import cKDTree

SETUP = os.path.join(os.path.dirname(__file__), "..", "..", "adforce", "setup")


def read_fort14_nodes(path: str) -> np.ndarray:
    with open(path) as f:
        f.readline()
        ne, nn = map(int, f.readline().split()[:2])
        xy = np.empty((nn, 2))
        for i in range(nn):
            p = f.readline().split()
            xy[i] = (float(p[1]), float(p[2]))
    return xy


def read_fort13(path: str):
    """Parse an ADCIRC fort.13: header, NP, attributes with defaults, then
    per-attribute exception blocks. Returns (header, np_nodes, attrs) where
    attrs is a list of dicts with name/units/nvals/default/values (full
    per-node array, single-valued attributes only)."""
    with open(path) as f:
        header = f.readline().rstrip("\n")
        np_nodes = int(f.readline().split()[0])
        nattr = int(f.readline().split()[0])
        attrs = []
        for _ in range(nattr):
            name = f.readline().strip()
            units = f.readline().rstrip("\n")
            nvals = int(f.readline().split()[0])
            assert nvals == 1, f"attribute {name}: only single-valued supported"
            default = float(f.readline().split()[0])
            attrs.append(dict(name=name, units=units, nvals=1, default=default))
        by_name = {a["name"]: a for a in attrs}
        for _ in range(nattr):
            name = f.readline().strip()
            nexc = int(f.readline().split()[0])
            a = by_name[name]
            a["values"] = np.full(np_nodes, a["default"])
            for _ in range(nexc):
                p = f.readline().split()
                a["values"][int(p[0]) - 1] = float(p[1])
    return header, np_nodes, attrs


def main() -> None:
    mid_xy = read_fort14_nodes(os.path.join(SETUP, "fort.14.mid"))
    low_xy = read_fort14_nodes(os.path.join(SETUP, "fort.14.low"))
    header, np_mid, attrs = read_fort13(os.path.join(SETUP, "fort.13.mid"))
    assert np_mid == len(mid_xy)
    tree = cKDTree(mid_xy)
    _, nearest = tree.query(low_xy)

    out = os.path.join(SETUP, "fort.13.low")
    with open(out, "w") as f:
        f.write(f"{header} (nearest-neighbour sample of fort.13.mid)\n")
        f.write(f"{len(low_xy)}\n{len(attrs)}\n")
        for a in attrs:
            f.write(f"{a['name']}\n{a['units']}\n1\n {a['default']:.6f}\n")
        for a in attrs:
            vals = a["values"][nearest]
            exc = np.where(vals != a["default"])[0]
            f.write(f"{a['name']}\n{len(exc)}\n")
            for i in exc:
                f.write(f"{i + 1} {vals[i]:.6f}\n")
            print(
                f"{a['name']}: default {a['default']}, "
                f"{len(exc)}/{len(low_xy)} exception nodes "
                f"(mid had {np.sum(a['values'] != a['default'])}/{np_mid})"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
