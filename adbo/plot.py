"""Plot results from adcirc bayesian optimization experiments."""

from typing import Tuple, List, Optional, Dict
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from sithom.place import BoundingBox
from adforce.constants import NO_BBOX
from sithom.plot import plot_defaults, label_subplots
from tcpips.constants import FIGURE_PATH, DATA_PATH
from adbo.constants import EXP_PATH
from adforce.mesh import xr_loader


stationid: List[str] = [
    "8729840",
    "8735180",
    "8760922",
    "8761724",
    "8762075",
    "8762482",
    "8764044",
]

stationid_to_names: Dict[str, str] = {
    "8729840": "Pensacola (-87.211, 30.404)",
    "8735180": "Dauphin Island (-88.075, 30.250)",
    "8760922": "Pilots Station East, S.W. Pass (-89.407, 28.932)",
    "8761724": "Grand Isle (-89.957, 29.263)",
    "8762075": "Port Fourchon, Belle Pass (-90.199, 29.114)",
    "8762482": "West Bank 1, Bayou Gauche (-90.420, 29.789)",
    "8764044": "Berwick, Atchafalaya River (-91.238, 29.668)",
}
stationid_to_names = {}
ds = xr.open_dataset(os.path.join(DATA_PATH, "katrina_tides.nc"))
name_d = {}
for sid in ds.stationid.values:
    dss = ds.sel(stationid=sid)
    name_d[sid] = f"{dss.name.values}"  # ({dss.lon.values:.3f}, {dss.lat.values:.3f})"
stationid_to_names = name_d

years: List[str] = ["2025", "2097"]


@timeit
def plot_diff(
    exps: Tuple[str, str] = ("miami-2025", "miami-2097"),
    figure_name="2025-vs-2097-miami.pdf",
) -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()
    exp1_dir = os.path.join(EXP_PATH, exps[0])
    exp2_dir = os.path.join(EXP_PATH, exps[1])
    paths = [os.path.join(direc, "experiments.json") for direc in [exp1_dir, exp2_dir]]
    if not all([os.path.exists(path) for path in paths]):
        print("One or more experiments do not exist.", paths)
        return

    exp1 = read_json(os.path.join(exp1_dir, "experiments.json"))
    exp2 = read_json(os.path.join(exp2_dir, "experiments.json"))
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
        for ax in axs:
            ax.axvline(sample, color="black", linestyle="--")

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


def find_max(exp: dict) -> float:
    """
    Find max value in experiment.

    Args:
        exp (dict): Experiment dictionary.

    Returns:
        float: Maximum value.
    """
    res = [float(exp[call]["res"]) for call in exp.keys()]
    if len(res) > 0:
        return max(res)
    else:
        return float("nan")


def find_difference(stationid: str) -> Tuple[float, float, float]:
    """
    Find difference in max value between two years.

    Args:
        stationid (int): Station ID.

    Returns:
        float: Difference in max value.
    """
    fp1 = os.path.join(EXP_PATH, f"{stationid}-2025", "experiments.json")
    if os.path.exists(fp1):
        exp1 = read_json(fp1)
        max1 = find_max(exp1)
    else:
        max1 = float("nan")
    fp2 = os.path.join(EXP_PATH, f"{stationid}-2097", "experiments.json")
    if os.path.exists(fp2):
        exp2 = read_json(fp2)
        max2 = find_max(exp2)
    else:
        max2 = float("nan")
    return max2 - max1, max1, max2


def find_differences() -> None:
    """Find differences in max values."""
    diff_list = []
    diff_percent_list = []
    for sid in stationid:
        diff, max1, max2 = find_difference(sid)
        print(
            f"{stationid_to_names[sid]}, max1: {max1:.1f} m, max2: {max2:.1f} m, diff: {diff:.1f} m, {diff/max1*100:.0f} %"
        )
        diff_list.append(diff)
        diff_percent_list.append(diff / max1)
    print(f"Average difference: {np.mean(diff_list):.1f} m")
    print(f"Average percentage difference: {np.mean(diff_percent_list)*100:.0f} %")


@timeit
def plot_many(year="2025") -> None:
    """
    Plot difference between two years.
    """
    plot_defaults()

    def _safe_read(fp):
        if os.path.exists(fp):
            return read_json(fp)
        else:
            return None

    exps = {
        id: _safe_read(os.path.join(EXP_PATH, id + "-" + year, "experiments.json"))
        for id in stationid
    }
    # print("exps", exps)

    _, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    def read_exp(
        exp: dict, variables=("res", "displacement", "angle", "trans_speed")
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Read experiment.

        Args:
            exp (dict): Experiment dictionary.

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]: Calls, res, displacement, angle.
        """
        calls = list(exp.keys())
        output = {}
        for var in variables:
            output[var] = [float(exp[call][var]) for call in calls]
        calls = [float(call) + 1 for call in calls]
        return calls, *tuple([output[var] for var in variables])

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

        calls, res, displacement, angle, speed = read_exp(exp)

        max_res = []
        maxr = -np.inf
        for r in res:
            if r > maxr:
                maxr = r
            max_res.append(maxr)

        axs[0].scatter(
            calls, res, label=stationid_to_names[label], color=color, s=marker_size
        )
        axs[0].plot(calls, max_res, color=color, linestyle="-", label=f"{label} max")
        axs[1].scatter(
            calls,
            displacement,
            label=stationid_to_names[label],
            color=color,
            s=marker_size,
        )
        axs[2].scatter(
            calls, angle, label=stationid_to_names[label], color=color, s=marker_size
        )
        axs[3].scatter(
            calls, speed, label=stationid_to_names[label], color=color, s=marker_size
        )
        print(f"{label} max_res: {max_res[-1]} m")
        print(f"{label} max_res25: {max_res[-26]} m")
        # axs[3].scatter(calls, trans_speed, label=label, color=color, s=marker_size)

    def vline(sample: float) -> None:
        nonlocal axs
        for ax in axs:
            ax.axvline(sample, color="black", linestyle="--")

    axs[0].set_ylabel("Max SSH at Point [m]")
    axs[1].set_ylabel(r"Track Displacement [$^\circ$]")
    axs[2].set_ylabel(r"Track Angle [$^\circ$]")
    axs[3].set_ylabel("Translation Speed [m s$^{-1}$]")
    axs[-1].set_xlabel("Number of Samples")
    label_subplots(axs)

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"][::-1]

    for exp_num, exp_key in enumerate(exps):
        if exps[exp_key] is not None:
            plot_exp(exps[exp_key], f"{exp_key}", colors[exp_num])
    vline(25.5)  # LHS to DAF transition in current set up.

    # axs[0].legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.75), ncol=3)

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
    path_to_maxele: str = os.path.join(
        EXP_PATH, "8729840-2025", "exp_0001", "maxele.63.nc"
    ),
) -> None:
    """
    Plot observation places.

    Args:
        bbox (optional, Optional[BoundingBox]): edge of bounding box.
        path_to_maxele (str, optional): path to maxele file. Defaults to os.path.join(EXP_PATH, "8729840-2025", "exp_0001", "maxele.63.nc").

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

    mele_ds = xr_loader(path_to_maxele)
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    for i, sid in enumerate(stationid):
        print(lons[i], lats[i], sid)
        distsq = (xs - lons[i]) ** 2 + (ys - lats[i]) ** 2
        min_p = np.argmin(distsq)
        lons[i] = xs[min_p]
        lats[i] = ys[min_p]

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
            lons[i],
            lats[i],
            label=stationid_to_names[sid],
            color=colors[i],
            s=100,
            marker="x",
            **fd,
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
    plt.xlabel("Longitude [$^\circ$E]")
    plt.ylabel("Latitude [$^\circ$N]")
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
    # plot_places()
    find_differences()

"""
Pensacola (-87.211, 30.404), max1: 4.191 m, max2: 4.772 m, diff: 0.581 m, 0.139 %
Dauphin Island (-88.075, 30.250), max1: 3.567 m, max2: 3.946 m, diff: 0.379 m, 0.106 %
Pilots Station East, S.W. Pass (-89.407, 28.932), max1: 1.879 m, max2: 2.420 m, diff: 0.541 m, 0.288 %
Grand Isle (-89.957, 29.263), max1: 5.886 m, max2: 7.141 m, diff: 1.255 m, 0.213 %
Port Fourchon, Belle Pass (-90.199, 29.114), max1: 4.903 m, max2: 5.274 m, diff: 0.371 m, 0.076 %
West Bank 1, Bayou Gauche (-90.420, 29.789), max1: 11.694 m, max2: 12.714 m, diff: 1.020 m, 0.087 %
Berwick, Atchafalaya River (-91.238, 29.668), max1: 9.911 m, max2: 11.102 m, diff: 1.192 m, 0.120 %
Average difference: 0.763 m
Average percentage difference: 14.708 %
"""
