"""Plot results from adcirc bayesian optimization experiments."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from sithom.plot import plot_defaults, label_subplots
from tcpips.constants import FIGURE_PATH
from adbo.constants import EXP_PATH


@timeit
def plot_diff() -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()
    # exp1_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2025")
    # exp2_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2097")
    exp1_dir = os.path.join(EXP_PATH, "bo-3-2025")
    exp2_dir = os.path.join(EXP_PATH, "bo-3-2097")
    exp1 = read_json(os.path.join(exp1_dir, "experiments.json"))
    exp2 = read_json(os.path.join(exp2_dir, "experiments.json"))
    # print(exp1.keys())
    # print(exp2.keys())

    _, axs = plt.subplots(4, 1, figsize=(8, 8))

    def plot_exp(exp: dict, label: str, color: str, marker_size: float = 1) -> None:
        """
        Plot experiment.

        Args:
            exp (dict): Experiment dictionary.
            label (str): Experiment label.
            color (str): Color of the markers and lines.
            marker_size (float, optional): Defaults to 1.
        """
        nonlocal axs
        calls = list(exp.keys())
        res = [float(exp[call]["res"]) for call in calls]
        displacement = [float(exp[call]["displacement"]) for call in calls]
        angle = [float(exp[call]["angle"]) for call in calls]
        trans_speed = [float(exp[call]["trans_speed"]) for call in calls]
        calls = [float(call) + 1 for call in calls]

        max_res = []
        maxr = -np.inf
        for r in res:
            if r > maxr:
                maxr = r
            max_res.append(maxr)

        axs[0].scatter(calls, res, label=label, color=color, s=marker_size)
        axs[0].plot(calls, max_res, color=color, linestyle="-", label=f"{label} max")
        axs[1].scatter(calls, displacement, label=label, color=color, s=marker_size)
        axs[2].scatter(calls, angle, label=label, color=color, s=marker_size)
        axs[3].scatter(calls, trans_speed, label=label, color=color, s=marker_size)

    def vline(sample: float) -> None:
        nonlocal axs
        axs[0].axvline(sample, color="black", linestyle="--")
        axs[1].axvline(sample, color="black", linestyle="--")
        axs[2].axvline(sample, color="black", linestyle="--")
        axs[3].axvline(sample, color="black", linestyle="--")

    axs[0].set_ylabel("Result [m]")
    axs[1].set_ylabel("Displacement [$^\circ$]")
    axs[2].set_ylabel("Angle [$^\circ$]")
    axs[3].set_ylabel("Trans Speed [m/s]")
    axs[3].set_xlabel("Samples")
    axs[0].legend()
    plt.legend()
    label_subplots(axs)

    plot_exp(exp1, "2025", "blue")
    plot_exp(exp2, "2097", "red")
    vline(25.5)
    figure_path = os.path.join(FIGURE_PATH, "2025-vs-2097-sid3.png")
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)


@timeit
def plot_many() -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()

    exps = {
        i: read_json(
            os.path.join(EXP_PATH, "bo-test-2d-midres" + str(i), "experiments.json")
        )
        for i in range(0, 5)
        if i != 3
    }
    print("exps", exps)

    # print(exp2.keys())

    _, axs = plt.subplots(3, 1, figsize=(8, 8))

    def plot_exp(exp: dict, label: str, color: str, marker_size: float = 1) -> None:
        """
        Plot experiment.

        Args:
            exp (dict): Experiment dictionary.
            label (str): Experiment label.
            color (str): Color of the markers and lines.
            marker_size (float, optional): Defaults to 1.
        """
        nonlocal axs
        calls = list(exp.keys())
        res = [float(exp[call]["res"]) for call in calls]
        displacement = [float(exp[call]["displacement"]) for call in calls]
        angle = [float(exp[call]["angle"]) for call in calls]
        # trans_speed = [float(exp[call]["trans_speed"]) for call in calls]
        calls = [float(call) + 1 for call in calls]

        max_res = []
        maxr = -np.inf
        for r in res:
            if r > maxr:
                maxr = r
            max_res.append(maxr)

        axs[0].scatter(calls, res, label=label, color=color, s=marker_size)
        axs[0].plot(calls, max_res, color=color, linestyle="-", label=f"{label} max")
        axs[1].scatter(calls, displacement, label=label, color=color, s=marker_size)
        axs[2].scatter(calls, angle, label=label, color=color, s=marker_size)
        # axs[3].scatter(calls, trans_speed, label=label, color=color, s=marker_size)

    def vline(sample: float) -> None:
        nonlocal axs
        axs[0].axvline(sample, color="black", linestyle="--")
        axs[1].axvline(sample, color="black", linestyle="--")
        axs[2].axvline(sample, color="black", linestyle="--")
        # axs[3].axvline(sample, color="black", linestyle="--")

    axs[0].set_ylabel("Result [m]")
    axs[1].set_ylabel("Displacement [$^\circ$]")
    axs[2].set_ylabel("Angle [$^\circ$]")
    axs[-1].set_xlabel("Samples")
    axs[0].legend()
    plt.legend()
    label_subplots(axs)

    colors = ["blue", "red", "green", "orange", "purple"]

    for exp_num, exp in exps.items():
        plot_exp(exp, f"sid{exp_num}", colors[exp_num])
    vline(5.5)
    figure_path = os.path.join(FIGURE_PATH, "2015.png")
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)


if __name__ == "__main__":
    # python -m adbo.plot
    # plot_diff()
    plot_many()
