"""Plot results from adcirc bayesian optimization experiments."""

from typing import Tuple, List, Optional, Dict
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from sithom.place import BoundingBox
from adforce.constants import NO_BBOX
from sithom.plot import plot_defaults, label_subplots
from tcpips.constants import FIGURE_PATH, DATA_PATH
from adbo.constants import EXP_PATH, DEFAULT_CONSTRAINTS
from adforce.mesh import xr_loader

# from pandas.plotting import parallel_coordinates

COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink"][::-1]
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
    "8729840": "Pensacola",  # (-87.211, 30.404)
    "8735180": "Dauphin Island",  #  (-88.075, 30.250)",
    "8760922": "Pilots Station East, S.W. Pass",  # (-89.407, 28.932)",
    "8761724": "Grand Isle",  # (-89.957, 29.263)",
    "8762075": "Port Fourchon, Belle Pass ",  # (-90.199, 29.114)",
    "8762482": "West Bank 1, Bayou Gauche",  # (-90.420, 29.789)",
    "8764044": "Berwick, Atchafalaya River",  # (-91.238, 29.668)",
}
stationid_to_names = {}
ds = xr.open_dataset(os.path.join(DATA_PATH, "katrina_tides.nc"))
name_d = {}
for sid in ds.stationid.values:
    dss = ds.sel(stationid=sid)
    name_d[sid] = f"{dss.name.values}"  # ({dss.lon.values:.3f}, {dss.lat.values:.3f})"
stationid_to_names = name_d

years: List[str] = ["2025", "2097"]

LABELS = {
    "res": "Max SSH at Point, $z$ [m]",
    "displacement": r"Track Displacement, $c$ [$^\circ$E]",
    "angle": r"Track Angle, $\chi$ [$^\circ$]",
    "trans_speed": r"Translation Speed, $V_t$ [m s$^{-1}$]",
}


def listify(exp: dict, key: str) -> List[float]:
    return [float(exp[call][key]) for call in exp.keys()]


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

        res = listify(exp, "res")
        displacement = listify(exp, "displacement")
        angle = listify(exp, "angle")
        trans_speed = listify(exp, "trans_speed")
        calls = [float(call) + 1 for call in calls]

        # get current max as list for each step to plot regret line.
        max_res: list = []
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
        if len(max_res) > 25:
            print(f"{label} max_res25: {max_res[-26]} m")

    def vline(sample: float) -> None:
        """vertical line.

        Args:
            sample (float): sample number.
        """
        nonlocal axs
        for ax in axs:
            ax.axvline(sample, color="black", linestyle="--")

    axs[0].set_ylabel("Max SSH at Point, $z$ [m]")
    axs[1].set_ylabel(LABELS["displacement"])
    axs[2].set_ylabel(LABELS["angle"])
    axs[3].set_ylabel(LABELS["trans_speed"])
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

    res = listify(exp, "res")
    if len(res) > 0:
        return max(res)
    else:
        return float("nan")


def find_argmax(exp: dict) -> Optional[int]:
    """
    Find argmax value in experiment.

    Args:
        exp (dict): Experiment dictionary.

    Returns:
        int: Index of maximum value.
    """
    res = listify(exp, "res")
    if len(res) > 0:
        return int(np.argmax(res))
    else:
        return None


def find_difference(stationid: str) -> Tuple[float, float, float]:
    """
    Find difference in max value between two years.

    Args:
        stationid (int): Station ID.

    Returns:
        float: Difference in max value.
    """

    def get_max(fp: str) -> float:
        if os.path.exists(fp):
            exp = read_json(fp)
            mx = find_max(exp)
        else:
            mx = float("nan")
        return mx

    fp1 = os.path.join(EXP_PATH, f"{stationid}-2025", "experiments.json")
    max1 = get_max(fp1)
    fp2 = os.path.join(EXP_PATH, f"{stationid}-2097", "experiments.json")
    max2 = get_max(fp2)
    return max2 - max1, max1, max2


def find_argdifference(stationid: str) -> Tuple[float, float, float]:
    """
    Find difference in max value between two years.

    Args:
        stationid (int): Station ID.

    Returns:
        float: Difference in max value.
    """

    def get_argmax(fp):
        if os.path.exists(fp):
            exp = read_json(fp)
            mix = find_argmax(exp)
            mx = {
                key: listify(exp, key)[mix]
                for key in ("displacement", "angle", "trans_speed", "res")
            }
        else:
            mx = {
                key: float("nan")
                for key in ("displacement", "angle", "trans_speed", "res")
            }
        return mx

    fp1 = os.path.join(EXP_PATH, f"{stationid}-2025", "experiments.json")
    argmax1 = get_argmax(fp1)
    fp2 = os.path.join(EXP_PATH, f"{stationid}-2097", "experiments.json")
    argmax2 = get_argmax(fp2)
    print("2025", argmax1, "2097", argmax2)

    def take_diff(am1: dict, am2: dict) -> dict:
        out = {}
        for key in am1:
            out[key] = am2[key] - am1[key]
        return out

    return argmax1, argmax2, take_diff(argmax1, argmax2)


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
    axs[1].set_ylabel(LABELS["displacement"])
    axs[2].set_ylabel(LABELS["angle"])
    axs[3].set_ylabel(LABELS["trans_speed"])
    axs[-1].set_xlabel("Number of Samples")
    label_subplots(axs)

    for exp_num, exp_key in enumerate(exps):
        if exps[exp_key] is not None:
            plot_exp(exps[exp_key], f"{exp_key}", COLORS[exp_num])
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

    for i, sid in enumerate(stationid):
        print(lons[i], lats[i], sid)
        ax.scatter(
            lons[i],
            lats[i],
            label=stationid_to_names[sid],
            color=COLORS[i],
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


def custom_parallel_coordinates(
    df: pd.DataFrame,
    class_column: str = None,
    cols: list = None,
    colors: dict = None,
    constraints: dict = None,
):
    """
    Plots a parallel coordinates chart where each column has its own vertical scale.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the features (and optionally a class column).
    cols : list
        List of columns in df to include as parallel axes (y-axes).
    class_column : str, optional
        Column in df to use for grouping/coloring lines (e.g., 'group' or 'class').
        If None, lines are all the same color.
    colors : dict, optional
        A dictionary mapping each class value (from class_column) to a desired color.
        Example: {'A': 'blue', 'B': 'green', 'C': 'red'}
    title : str, optional
        Title of the plot.
    """
    # If class_column is given, separate the class labels for coloring
    if class_column and class_column in df.columns:
        unique_classes = df[class_column].unique()
    else:
        unique_classes = [None]

    # Figure and axis setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each column, determine its data min and max (for labeling the axis ticks)
    mins = df[cols].min()
    maxs = df[cols].max()
    if constraints:
        for col in cols:
            if col in constraints:
                mins[col] = constraints[col]["min"]
                maxs[col] = constraints[col]["max"]
            if col == "res":
                mins[col] = 0.0

    y_ticks = np.linspace(0, 1, num=7).tolist()
    y_tick_lalels = [""] * 7
    ax.set_yticks(y_ticks, labels=y_tick_lalels)
    # We'll create an x-position for each column
    x_positions = np.arange(len(cols))

    # Plot each row in the DataFrame as one line
    for idx, row in df.iterrows():
        # If we have a class column, pick a color based on row's class value; else default
        if class_column and class_column in row and colors:
            c = colors.get(row[class_column], "gray")
        elif class_column and class_column in row:
            # If no custom colors dict is provided, pick from a colormap or just gray
            label_value = row[class_column]
            color_idx = np.where(unique_classes == label_value)[0][0]
            c = plt.cm.tab10(color_idx % 10)
        else:
            c = "gray"

        # For each column, normalize its value to 0..1 based on that column's min and max
        y_vals = []
        for col in cols:
            y_norm = (
                (row[col] - mins[col]) / (maxs[col] - mins[col])
                if maxs[col] != mins[col]
                else 0.0
            )
            y_vals.append(y_norm)

        # Plot the line across the x_positions
        ax.plot(x_positions, y_vals, color=c, alpha=0.7, label=row[class_column])

    # Now we label each "vertical axis" with its original scale
    # We'll add a vertical line at each x-position, and we add min/max tick labels
    for i, col in enumerate(cols):
        # Draw a vertical line
        ax.axvline(x=i, color="black", linestyle="--", linewidth=0.5)

        # Text label for the column name
        ax.text(
            i,
            1.05,
            LABELS[col],
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=9,
            # fontweight="bold",
        )

        # Min/Max numerical labels. They appear just below/above the data range (0.0 and 1.0 after normalization)
        ax.text(i - 0.05, 0, f"{mins[col]:.3g}", ha="center", va="top", fontsize=12)
        ax.text(i - 0.05, 1, f"{maxs[col]:.3g}", ha="center", va="bottom", fontsize=12)

    # Remove default x-axis tick marks and labels since we handle them manually
    ax.set_xticks([])
    # Set y-limits to [0,1] because we normalized each column into that range
    ax.set_ylim([-0.05, 1.05])
    # ax.set_ylabel("Normalized Values (each axis scaled independently)")

    plt.tight_layout()
    # plt.show()
    # plot legend below plot if class_column is given
    if class_column:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
        )


def plot_multi_argmax():
    def _listify(ld: List[dict[str, float]], key) -> List[float]:
        return [ld[i][key] for i in range(len(ld))]

    def to_pd(am_l: List[dict]) -> pd.DataFrame:
        out = {}
        for key in am_l[0]:
            out[key] = _listify(am_l, key)
        return pd.DataFrame(out)

    def mean_std(am_l: List[dict]) -> dict:
        out = {}
        for key in am_l[0]:
            l = _listify(am_l, key)
            if len(l) == 0:
                out[key] = (float("nan"), float("nan"))
            elif len(l) == 1:
                out[key] = (np.mean(l), float("nan"))
            else:
                out[key] = (np.mean(l), np.std(l))
        return out

    am1_l = []
    am2_l = []
    amd_l = []
    for sid in stationid:
        am1, am2, amd = find_argdifference(sid)
        am1_l += [am1]
        am2_l += [am2]
        amd_l += [amd]
    am1_res = mean_std(am1_l)
    am2_res = mean_std(am2_l)
    amd_res = mean_std(amd_l)
    print("\n\n2025\n", am1_res, "\n\n2097\n", am2_res, "\n\nDiff\n", amd_res)
    for i, (aml, name) in enumerate(
        [(am1_l, "2025"), (am2_l, "2097"), (amd_l, "diff")]
    ):
        df = to_pd(aml)
        df["stationid"] = [stationid_to_names[sid] for sid in stationid]
        if name == "diff":
            constraints = None
        else:
            constraints = DEFAULT_CONSTRAINTS
        custom_parallel_coordinates(
            df,
            cols=["res", "displacement", "angle", "trans_speed"],
            class_column="stationid",
            colors={
                stationid_to_names[sid]: COLORS[i] for i, sid in enumerate(stationid)
            },
            constraints=constraints,
        )
        # plt.title("Parallel Coordinates Plot of Max and Argmax")
        plt.xlabel(
            "Maximum SSH $z^{*}$ at point and corresponding arguments $c$, $\chi$, $V_t$"
        )
        # plt.ylabel("Values")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_PATH, f"parallel_coordinates_{name}.pdf"))
        # splt.show()
        plt.clf()


if __name__ == "__main__":
    # python -m adbo.plot
    for point in ["miami", "new-orleans", "galverston"]:
        plot_diff(
            exps=(f"{point}-2025", f"{point}-2097"),
            figure_name=f"2025-vs-2097-{point}.pdf",
        )
    plt.clf()
    plot_many("2025")
    plot_many("2097")
    # plot_places()
    find_differences()
    plot_multi_argmax()


"""
Old (isopycnal approx, profile from centre of GOM).
Pensacola (-87.211, 30.404), max1: 4.191 m, max2: 4.772 m, diff: 0.581 m, 13.9 %
Dauphin Island (-88.075, 30.250), max1: 3.567 m, max2: 3.946 m, diff: 0.379 m, 10.6 %
Pilots Station East, S.W. Pass (-89.407, 28.932), max1: 1.879 m, max2: 2.420 m, diff: 0.541 m, 28.8 %
Grand Isle (-89.957, 29.263), max1: 5.886 m, max2: 7.141 m, diff: 1.255 m, 21.3 %
Port Fourchon, Belle Pass (-90.199, 29.114), max1: 4.903 m, max2: 5.274 m, diff: 0.371 m, 7.6 %
West Bank 1, Bayou Gauche (-90.420, 29.789), max1: 11.694 m, max2: 12.714 m, diff: 1.020 m, 8.7 %
Berwick, Atchafalaya River (-91.238, 29.668), max1: 9.911 m, max2: 11.102 m, diff: 1.192 m, 12.0 %
Average difference: 0.763 m
Average percentage difference: 14.708 %

New (isothermal approx) profile from near New Orleans:
Pensacola, max1: 5.6 m, max2: 6.0 m, diff: 0.4 m, 7 %
Dauphin Island, max1: 3.9 m, max2: 4.5 m, diff: 0.6 m, 16 %
Pilots Station East, S.W. Pass, max1: 2.4 m, max2: 2.7 m, diff: 0.3 m, 13 %
Grand Isle, max1: 8.2 m, max2: 9.3 m, diff: 1.1 m, 13 %
Port Fourchon, Belle Pass, max1: 6.3 m, max2: 6.9 m, diff: 0.6 m, 10 %
West Bank 1, Bayou Gauche, max1: 15.0 m, max2: 16.6 m, diff: 1.5 m, 10 %
Berwick, Atchafalaya River, max1: 12.6 m, max2: 13.7 m, diff: 1.2 m, 9 %
Average difference: 0.8 m
Average percentage difference: 11 %
"""
