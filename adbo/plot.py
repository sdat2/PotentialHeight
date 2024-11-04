"""Plot results from adcirc bayesian optimization experiments."""

from typing import Tuple, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from sithom.place import BoundingBox
from adforce.constants import NO_BBOX
from sithom.plot import plot_defaults, label_subplots
from tcpips.constants import FIGURE_PATH
from adbo.constants import EXP_PATH


# exp_names = [
#     "notide-" + str(stationid) + "-" + str(year) + "-midres"
#     for stationid in range(0, 6)
#     for year in [2025, 2097]
# ]
# print(exp_names)


stationid = [
    "8729840",
    "8735180",
    "8760922",
    "8761724",
    "8762075",
    "8762482",
    "8764044",
]

years = ["2025", "2097"]


@timeit
def plot_diff(
    exps: Tuple[str, str] = ("new-3d", "new-3d-2097"),
    figure_name="2025-vs-2097-new-orleans.pdf",
) -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()
    # exp1_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2025")
    # exp2_dir = os.path.join(EXP_PATH, "bo-test-2d-midres-agg-3-2097")
    exp1_dir = os.path.join(EXP_PATH, exps[0])
    exp2_dir = os.path.join(EXP_PATH, exps[1])
    paths = [os.path.join(direc, "experiments.json") for direc in [exp1_dir, exp2_dir]]
    if not all([os.path.exists(path) for path in paths]):
        print("One or more experiments do not exist.", paths)
        return

    exp1 = read_json(os.path.join(exp1_dir, "experiments.json"))
    exp2 = read_json(os.path.join(exp2_dir, "experiments.json"))
    # print(exp1.keys())
    # print(exp2.keys())

    _, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

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
        print(f"{label} max_res: {max_res[-1]} m")
        print(f"{label} max_res25: {max_res[-26]} m")

    def vline(sample: float) -> None:
        """vertical line.

        Args:
            sample (float): sample number.
        """
        nonlocal axs
        axs[0].axvline(sample, color="black", linestyle="--")
        axs[1].axvline(sample, color="black", linestyle="--")
        axs[2].axvline(sample, color="black", linestyle="--")
        axs[3].axvline(sample, color="black", linestyle="--")

    axs[0].set_ylabel("Max SSH at Point [m]")
    axs[1].set_ylabel(r"Track Displacement [$^\circ$]")
    axs[2].set_ylabel(r"Track Angle [$^\circ$]")
    axs[3].set_ylabel("Translation Speed [m s$^{-1}$]")
    axs[3].set_xlabel("Number of Samples")
    # axs[0].legend()
    label_subplots(axs)

    plot_exp(exp1, "2025", "blue")
    plot_exp(exp2, "2097", "red")
    axs[0].legend()
    vline(25.5)  # after 25 samples goes to Bayesian optimization
    plt.xlim(1, 50)

    # before that it is doing Latin Hypercube Sampling
    figure_path = os.path.join(FIGURE_PATH, figure_name)
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)
    plt.close()


@timeit
def plot_many(year="2025") -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()

    exps = {
        i: read_json(os.path.join(EXP_PATH, i + "-" + year, "experiments.json"))
        for i in stationid
    }
    print("exps", exps)

    # print(exp2.keys())

    _, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

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

    axs[0].set_ylabel("Max SSH at Point [m]")
    axs[1].set_ylabel(r"Track Displacement [$^\circ$]")
    axs[2].set_ylabel(r"Track Angle [$^\circ$]")
    axs[-1].set_xlabel("Number of Samples")
    label_subplots(axs)

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"][::-1]

    for exp_num, exp_key in enumerate(exps):
        plot_exp(exps[exp_key], f"{exp_key}", colors[exp_num])
    vline(25.5)

    # axs[0].legend()
    plt.legend()
    plt.xlim(1, 50)

    figure_path = os.path.join(FIGURE_PATH, "along-coast-" + year + ".pdf")
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)
    plt.close()


if False:
    # python -m adbo.plot
    for staionid in range(0, 6):
        plot_diff(
            exps=(
                f"notide-{staionid}-2025-midres",
                f"notide-{staionid}-2097-midres",
            ),
            figure_name=f"2025-vs-2097-sid{staionid}-midres.png",
        )
    # plot_diff()
    # plot_many()
    # pass


@timeit
def plot_places(
    bbox: Optional[BoundingBox] = NO_BBOX.pad(0.5),
) -> None:
    """
    Plot observation places.

    Args:
        bbox (optional, Optional[BoundingBox]): edge of bounding box.

    """
    lats: List[float] = [
        30.404389,
        30.25,
        28.932222,
        29.263,
        29.114167,
        29.788611,
        29.6675,
    ]  # Latitude in degrees North
    lons: List[float] = [
        -87.211194,
        -88.075,
        -89.4075,
        -89.957,
        -90.199167,
        -90.420278,
        -91.237611,
    ]  # Longitude in degrees East
    stationid: List[str] = [
        "8729840",
        "8735180",
        "8760922",
        "8761724",
        "8762075",
        "8762482",
        "8764044",
    ]
    plot_defaults()
    try:
        import cartopy
        import cartopy.crs as ccrs

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        # ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.STATES, linestyle=":")
        fd = dict(transform=ccrs.PlateCarree())
    except ImportError:
        print("Cartopy not installed. Using default plot.")
        fd = {}
        ax = plt.axes()

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"][::-1]

    for i, sid in enumerate(stationid):
        print(lons[i], lats[i], sid)
        ax.scatter(
            lons[i], lats[i], label=sid, color=colors[i], s=100, marker="x", **fd
        )  # color="blue"
        # print("fd", fd)
    ax.legend()
    if fd != {}:
        ax.set_yticks(
            [
                x
                for x in range(
                    int((bbox.lat[0] // 1) + 1),
                    int((bbox.lat[1] // 1) + 1),
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xticks(
            [
                x
                for x in range(
                    int((bbox.lon[0] // 1) + 1),
                    int((bbox.lon[1] // 1) + 1),
                )
            ],
            crs=ccrs.PlateCarree(),
        )
    bbox.ax_lim(ax)
    plt.xlabel("Longitude $^\circ$E")
    plt.ylabel("Latitude $^\circ$N")
    figure_name = os.path.join(FIGURE_PATH, "stationid_map.pdf")
    plt.savefig(figure_name)
    plt.close()
    print(f"Saved figure to {figure_name}")
    # CESM, GFDL, GISS, MIROC, UKESM


if __name__ == "__main__":
    # python -m adbo.plot
    # plot_diff()
    # plot_many("2025")
    # plot_many("2097")
    plot_places()
