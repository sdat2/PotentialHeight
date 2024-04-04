"""Plot results from adcirc bayesian optimization experiments."""

import os
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.plot import plot_defaults, label_subplots
from tcpips.constants import FIGURE_PATH
from adbo.constants import EXP_PATH


def plot_diff() -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()
    exp1_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2025")
    exp2_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2097")
    exp1 = read_json(os.path.join(exp1_dir, "experiments.json"))
    exp2 = read_json(os.path.join(exp2_dir, "experiments.json"))
    print(exp1.keys())
    print(exp2.keys())

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    def plot_exp(exp: dict, label: str, color: str, marker_size: float = 1) -> None:
        calls = sorted(list(exp.keys()))
        res = [exp[call]["res"] for call in calls]
        displacement = [exp[call]["displacement"] for call in calls]
        angle = [exp[call]["angle"] for call in calls]
        trans_speed = [exp[call]["trans_speed"] for call in calls]

        axs[0].scatter(calls, res, label=label, color=color, s=marker_size)
        axs[1].scatter(calls, displacement, label=label, color=color, s=marker_size)
        axs[2].scatter(calls, angle, label=label, color=color, s=marker_size)
        axs[3].scatter(calls, trans_speed, label=label, color=color, s=marker_size)

    axs[0].set_ylabel("Result [m]")
    axs[1].set_ylabel("Displacement [\(^\circ\)]")
    axs[2].set_ylabel("Angle [\(^\circ\)]")
    axs[3].set_ylabel("Trans Speed [m/s]")
    axs[3].set_xlabel("Calls")
    axs[0].legend()
    label_subplots(axs)

    plot_exp(exp1, "2025", "blue")
    plot_exp(exp2, "2097", "red")
    figure_path = os.path.join(FIGURE_PATH, "2025-vs-2097-bayesopt.png")
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)


if __name__ == "__main__":
    # python -m adbo.plot
    plot_diff()
