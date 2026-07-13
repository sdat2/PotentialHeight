"""Plots for the size--intensity tradeoff experiments.

- :func:`plot_vmax_sweep`: surge height against intensity from a 1D sweep
  (``adbo.sweep_vmax``), with the curve's r0/rmax context panels.
- :func:`plot_4d_samples`: surge against sampled intensity from a 4D BO
  experiment (``adbo.exp_4d``), split into initial-design vs acquisition
  samples.

Both read the ``experiments.json`` ledger, so they work on partial runs.

Example::

    python -m adbo.plot_tradeoff --exp_dir <exp>/no-sweep-2025 --mode sweep
"""

import argparse
import json
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_ledger(exp_dir: str) -> dict:
    with open(os.path.join(exp_dir, "experiments.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _fig_path(exp_dir: str, name: str) -> str:
    img_dir = os.path.join(exp_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return os.path.join(img_dir, name)


def _curve_context(curve_nc: Optional[str]):
    if curve_nc is None or not os.path.exists(curve_nc):
        return None
    from w22.tradeoff import TradeoffCurve

    return TradeoffCurve.from_file(curve_nc)


def plot_vmax_sweep(exp_dir: str, curve_nc: Optional[str] = None) -> str:
    """Plot surge height vs intensity for a 1D sweep experiment.

    Args:
        exp_dir (str): Sweep experiment directory (contains experiments.json).
        curve_nc (str, optional): Tradeoff curve for context panels; defaults
            to the path recorded in sweep-config.json.

    Returns:
        str: Path of the written figure.
    """
    ledger = _load_ledger(exp_dir)
    if curve_nc is None:
        cfg_path = os.path.join(exp_dir, "sweep-config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                curve_nc = json.load(f).get("curve_path")
    curve = _curve_context(curve_nc)

    v = np.array([rec["vmax"] for rec in ledger.values()])
    z = np.array([rec["res"] for rec in ledger.values()])
    order = np.argsort(v)
    v, z = v[order], z[order]
    ok = np.isfinite(z)

    nrows = 2 if curve is not None else 1
    fig, axs = plt.subplots(
        nrows, 1, sharex=True, figsize=(6, 3 * nrows), squeeze=False
    )
    ax = axs[0][0]
    ax.plot(v[ok], z[ok], "o-", color="tab:blue")
    if (~ok).any():
        for vf in v[~ok]:
            ax.axvline(vf, color="red", alpha=0.3, linestyle=":")
    if len(z[ok]):
        i_best = int(np.nanargmax(z[ok]))
        ax.plot(v[ok][i_best], z[ok][i_best], "*", color="tab:orange", markersize=15)
        ax.annotate(
            f"max {z[ok][i_best]:.2f} m @ {v[ok][i_best]:.1f} m/s",
            (v[ok][i_best], z[ok][i_best]),
            textcoords="offset points",
            xytext=(5, -12),
        )
    ax.set_ylabel("Max SSH at observation point [m]")
    ax.set_title("Surge along the size-intensity tradeoff curve")

    if curve is not None:
        ax.axvline(curve.v_max, color="green", linestyle="--", alpha=0.7)
        ax.annotate("$V_p$", (curve.v_max, ax.get_ylim()[0]), color="green", ha="right")
        ax2 = axs[1][0]
        vv = np.linspace(curve.v_min, curve.v_max, 200)
        ax2.plot(vv, [curve.rmax(x) / 1000 for x in vv], color="tab:purple")
        ax2.set_ylabel("$r_{\\mathrm{max}}(V)$ [km]", color="tab:purple")
        ax2b = ax2.twinx()
        ax2b.plot(vv, [curve.r0(x) / 1000 for x in vv], color="tab:gray")
        ax2b.set_ylabel("$r_0(V)$ [km]", color="tab:gray")
        ax2.set_xlabel("Maximum gradient wind speed $V$ [m s$^{-1}$]")
    else:
        ax.set_xlabel("Maximum gradient wind speed $V$ [m s$^{-1}$]")

    out = _fig_path(exp_dir, "vmax_sweep.pdf")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"plot_vmax_sweep: wrote {out}")
    return out


def plot_4d_samples(
    exp_dir: str, init_steps: Optional[int] = None, curve_nc: Optional[str] = None
) -> str:
    """Plot surge vs sampled intensity for a 4D BO experiment.

    Args:
        exp_dir (str): BO experiment directory.
        init_steps (int, optional): Number of initial-design samples; defaults
            to the value in bo-config.json.
        curve_nc (str, optional): Curve for the V_p marker; defaults to the
            path in bo-config.json.

    Returns:
        str: Path of the written figure.
    """
    ledger = _load_ledger(exp_dir)
    bo_cfg_path = os.path.join(exp_dir, "bo-config.json")
    if os.path.exists(bo_cfg_path):
        with open(bo_cfg_path, "r", encoding="utf-8") as f:
            bo_cfg = json.load(f)
        init_steps = init_steps if init_steps is not None else bo_cfg.get("init_steps")
        curve_nc = curve_nc if curve_nc is not None else bo_cfg.get("curve_path")
    curve = _curve_context(curve_nc)

    calls = np.array(sorted(int(k) for k in ledger))
    v = np.array([ledger[str(c)]["vmax"] for c in calls])
    z = np.array([ledger[str(c)]["res"] for c in calls])
    is_init = (
        calls < init_steps if init_steps is not None else np.ones_like(calls, bool)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        v[is_init], z[is_init], marker="x", color="tab:blue", label="Initial design"
    )
    if (~is_init).any():
        sc = ax.scatter(
            v[~is_init],
            z[~is_init],
            c=calls[~is_init],
            marker="o",
            cmap="viridis",
            label="Acquisition",
        )
        plt.colorbar(sc, ax=ax, label="Evaluation index")
    if len(z):
        i_best = int(np.nanargmax(z))
        ax.plot(v[i_best], z[i_best], "*", color="tab:orange", markersize=15)
    if curve is not None:
        ax.axvline(curve.v_max, color="green", linestyle="--", alpha=0.7)
        ax.annotate("$V_p$", (curve.v_max, ax.get_ylim()[0]), color="green", ha="right")
    ax.set_xlabel("Maximum gradient wind speed $V$ [m s$^{-1}$]")
    ax.set_ylabel("Max SSH at observation point [m]")
    ax.legend()

    out = _fig_path(exp_dir, "bo_4d_samples.pdf")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"plot_4d_samples: wrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["sweep", "4d"], default="sweep")
    parser.add_argument("--curve_nc", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "sweep":
        plot_vmax_sweep(args.exp_dir, curve_nc=args.curve_nc)
    else:
        plot_4d_samples(args.exp_dir, curve_nc=args.curve_nc)
