"""Score the low-resolution historical sweep against the mid (EC95d) archive.

SUPERSEDED by ``adforce.eval.pairs.resolution_bias_table`` (which reproduces
``rerun/results/resolution_bias.csv`` exactly -- pinned by
``tests/test_eval.py``); kept as the frozen record of the published run.

Joins the low-res gauge series (rerun/adcirc/extract_gauge_series.py output)
with comp.validate's val_summary.csv (mid-resolution sim peaks + de-tided
observed peaks + clean flags), keyed by (storm "Name YYYY", gauge sid). The
low sim peak is the max of the extracted series (run window == storm window
for storm-only mode, matching comp.validate's peak definition).

Prints the pooled and per-gauge low-vs-mid-vs-obs bias table over clean pairs
and writes rerun/results/resolution_bias.csv.

Run:  python rerun/adcirc/score_resolution.py \
          [--series data/comp/lowres/low_storm_gauge_series.parquet]
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def dir_to_storm(d: str) -> str:
    """'22_MICHAEL_2018' -> 'Michael 2018' (val_summary storm naming)."""
    m = re.match(r"\d+_(.+)_(\d{4})$", d)
    name = m.group(1).replace("_", " ").title()
    return f"{name} {m.group(2)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series",
        default=os.path.join(REPO, "data/comp/lowres/low_storm_gauge_series.parquet"),
    )
    ap.add_argument("--out", default=os.path.join(REPO, "rerun/results/resolution_bias.csv"))
    a = ap.parse_args()

    ser = pd.read_parquet(a.series)
    ser["storm_name"] = ser.storm.map(dir_to_storm)
    low = (
        ser.groupby(["storm_name", "sid"], as_index=False)
        .zeta.max()
        .rename(columns={"zeta": "low_peak", "storm_name": "storm"})
    )
    low["sid"] = low.sid.astype(str)

    val = pd.read_csv(os.path.join(REPO, "data/comp/out/val_summary.csv"))
    val["sid"] = val.sid.astype(str)
    df = val.merge(low, on=["storm", "sid"], how="left")
    clean = df[df.clean.astype(bool) & df.low_peak.notna() & (df.obs_peak >= 1.0)].copy()
    clean["mid_ratio"] = clean.sim_peak / clean.obs_peak
    clean["low_ratio"] = clean.low_peak / clean.obs_peak

    print(f"clean pairs with obs>=1 m and low coverage: {len(clean)}")
    for name, r in (("mid/obs", clean.mid_ratio), ("low/obs", clean.low_ratio)):
        print(f"  {name}: mean {r.mean():.2f}  median {np.median(r):.2f}")
    print(f"  low/mid peak ratio: mean {(clean.low_peak/clean.sim_peak).mean():.2f}")

    print("\nper-gauge (n>=2 clean pairs):")
    g = (
        clean.groupby("name")
        .agg(n=("storm", "size"), mid=("mid_ratio", "mean"), low=("low_ratio", "mean"))
        .query("n >= 2")
        .sort_values("mid", ascending=False)
    )
    print(g.to_string(float_format=lambda v: f"{v:.2f}"))

    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
