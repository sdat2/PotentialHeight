"""Plot results from adcirc Bayesian optimization experiments."""

from typing import Tuple, List, Optional, Dict
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sithom.io import read_json
from sithom.time import timeit
from sithom.place import BoundingBox
from adforce.constants import NO_BBOX, NEW_ORLEANS, GALVERSTON, MIAMI
from sithom.plot import plot_defaults, label_subplots, get_dim
from tcpips.constants import FIGURE_PATH, DATA_PATH
from adforce.mesh import xr_loader
from adforce.ani import single_wind_and_height_step
from .constants import EXP_PATH, DEFAULT_CONSTRAINTS
from .constants import DATA_PATH as DATA_PATH_ADBO


# from pandas.plotting import parallel_coordinates
# This might not be a great way of choosing colors
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "black"][::-1]
stationid: List[str] = [
    "new-orleans",
    "8729840",
    "8735180",
    "8760922",
    "8761724",
    "8762075",
    "8762482",
    "8764044",
]

stationid_to_names: Dict[str, str] = {
    "new-orleans": "New Orleans",  # (-90.070, 29.950)
    "miami": "Miami",  # (-80.191, 25.761)
    "galverston": "Galveston",  # (-94.797, 29.301)
    "8729840": "Pensacola",  # (-87.211, 30.404)
    "8735180": "Dauphin Island",  #  (-88.075, 30.250)",
    "8760922": "Pilots Station East, S.W. Pass",  # (-89.407, 28.932)",
    "8761724": "Grand Isle",  # (-89.957, 29.263)",
    "8762075": "Port Fourchon, Belle Pass ",  # (-90.199, 29.114)",
    "8762482": "West Bank 1, Bayou Gauche",  # (-90.420, 29.789)",
    "8764044": "Berwick, Atchafalaya River",  # (-91.238, 29.668)",
}
stationid_to_bbox: Dict[str, BoundingBox] = {
    "new-orleans": NO_BBOX.pad(1),
    "miami": MIAMI.bbox(3),
    "galverston": GALVERSTON.bbox(3),
}
# stationid_to_names = {}
# ds = xr.open_dataset(os.path.join(DATA_PATH, "katrina_tides.nc"))
# name_d = {}
# for sid in ds.stationid.values:
#    dss = ds.sel(stationid=sid)
#    name_d[sid] = f"{dss.name.values}"  # ({dss.lon.values:.3f}, {dss.lat.values:.3f})"
# stationid_to_names = name_d

years: List[str] = ["2015", "2100"]

subfolder = f"{years[0]}vs{years[1]}"
if not os.path.exists(os.path.join(FIGURE_PATH, subfolder)):
    os.makedirs(os.path.join(FIGURE_PATH, subfolder))

LABELS = {
    "res": "Max SSH, $z$ [m]",
    "displacement": r"Track Displacement, $c$ [$^\circ$E]",
    "angle": r"Track Angle, $\chi$ [$^\circ$]",
    "trans_speed": r"Translation Speed, $V_t$ [m s$^{-1}$]",
}


def listify(exp: dict, key: str) -> List[float]:
    """
    Listify the values of a key in a dictionary.

    Args:
        exp (dict): Experiment dictionary.
        key (str): Key to listify.

    Returns:
        List[float]: List of values.
    """
    return [float(exp[call][key]) for call in exp.keys()]


@timeit
def plot_diff(
    exps: Tuple[str, str] = ("miami-2015", "miami-2100"),
    figure_name="2015-vs-2100-miami.pdf",
) -> None:
    """
    Plot difference between two years.

    Args:
        exps (Tuple[str, str], optional): Experiment names. Defaults to ("miami-2015", "miami-2100").
        figure_name (str, optional): Figure name. Defaults to "2015-vs-2100-miami.pdf".
    """
    plot_defaults()
    # miami-2015 and miami-2100 are the original experiments.
    # there are now some new experiments called miami-2015-1, miami-2015-2
    # and miami-2100-1, miami-2100-2 etc.
    # so we need to check if they exist and use them if they do.
    # they are repeats of the original experiments with different random seeds.
    exp1_dirs = [os.path.join(EXP_PATH, f"{exps[0]}")]
    exp1_dirs += [
        os.path.join(EXP_PATH, f"{exps[0]}-{i}")
        for i in range(1, 4)
        if os.path.exists(os.path.join(EXP_PATH, f"{exps[0]}-{i}"))
    ]
    exp2_dirs = [os.path.join(EXP_PATH, f"{exps[1]}")]
    exp2_dirs += [
        os.path.join(EXP_PATH, f"{exps[1]}-{i}")
        for i in range(1, 4)
        if os.path.exists(os.path.join(EXP_PATH, f"{exps[1]}-{i}"))
    ]
    if len(exp1_dirs) == 0 or len(exp2_dirs) == 0:
        print("One or more experiments do not exist.")
        return

    exp1_dicts = [
        read_json(os.path.join(exp1_dir, "experiments.json")) for exp1_dir in exp1_dirs
    ]
    exp2_dicts = [
        read_json(os.path.join(exp2_dir, "experiments.json")) for exp2_dir in exp2_dirs
    ]
    _, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    def plot_exp(
        exp: dict, label: str, color: str, marker_size: float = 1, use_label=False
    ) -> None:
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
        if use_label:
            label_d_max = {"label": f"{label} max"}
            label_d_scatter = {"label": label}
        else:
            label_d_max = {}
            label_d_scatter = {}
        axs[0].scatter(calls, res, **label_d_scatter, color=color, s=marker_size)
        axs[0].plot(calls, max_res, color=color, linestyle="-", **label_d_max)
        axs[1].scatter(
            calls, displacement, **label_d_scatter, color=color, s=marker_size
        )
        axs[2].scatter(calls, angle, **label_d_scatter, color=color, s=marker_size)
        axs[3].scatter(
            calls, trans_speed, **label_d_scatter, color=color, s=marker_size
        )
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
    axs[3].set_xlabel("Number of Samples, $s$")

    label_subplots(axs)
    [
        plot_exp(exp1, "2015", "blue", use_label=(i == 0))
        for i, exp1 in enumerate(exp1_dicts)
    ]
    [
        plot_exp(exp2, "2100", "red", use_label=(i == 0))
        for i, exp2 in enumerate(exp2_dicts)
    ]
    axs[0].legend()
    vline(25.5)  # after 25 samples goes to Bayesian optimization
    plt.xlim(1, 50)

    # before that it is doing Latin Hypercube Sampling
    figure_path = os.path.join(FIGURE_PATH, subfolder, figure_name)
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
        return np.nanmax(res)
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
        return int(np.nanargmax(res))
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

    fp1 = os.path.join(EXP_PATH, f"{stationid}-2015", "experiments.json")
    max1 = get_max(fp1)
    fp2 = os.path.join(EXP_PATH, f"{stationid}-2100", "experiments.json")
    max2 = get_max(fp2)
    return max2 - max1, max1, max2


def find_argdifference(stationid: str) -> Tuple[float, float, float]:
    """
    Find difference in max value between two years.

    Args:
        stationid (int): Station ID.

    Returns:
        Tuple[float, float, float]: argmax1, argmax2, take_diff(argmax1, argmax2)
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

    fp1 = os.path.join(EXP_PATH, f"{stationid}-2015", "experiments.json")
    argmax1 = get_argmax(fp1)
    fp2 = os.path.join(EXP_PATH, f"{stationid}-2100", "experiments.json")
    argmax2 = get_argmax(fp2)
    print("2015", argmax1, "2100", argmax2)

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
    print(f"Average difference: {np.nanmean(diff_list):.1f} m")
    print(f"Average percentage difference: {np.nanmean(diff_percent_list)*100:.0f} %")


@timeit
def plot_many(year: str = "2015") -> None:
    """
    Plot difference between two years.

    Args:
        year (str, optional): Year to plot. Defaults to "2015".
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

    figure_path = os.path.join(FIGURE_PATH, subfolder, "along-coast-" + year + ".pdf")
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)
    plt.close()


if False:
    # python -m adbo.plot
    for staionid in range(0, 6):
        plot_diff(
            exps=(
                f"notide-{staionid}-2015-midres",
                f"notide-{staionid}-2100-midres",
            ),
            figure_name=f"2015-vs-2100-sid{staionid}-midres.png",
        )
    # plot_diff()
    # plot_many()
    # pass


@timeit
def plot_places(
    bbox: Optional[BoundingBox] = NO_BBOX.pad(0.5),
    path_to_maxele: str = os.path.join(
        EXP_PATH, "8729840-2015", "exp_0001", "maxele.63.nc"
    ),
) -> None:
    """
    Plot observation places.

    Args:
        bbox (optional, Optional[BoundingBox]): edge of bounding box.
        path_to_maxele (str, optional): path to maxele file. Defaults to os.path.join(EXP_PATH, "8729840-2015", "exp_0001", "maxele.63.nc").

    """
    lats: List[float] = [
        NEW_ORLEANS.lat,
        30.404389,
        30.25,
        28.932222,
        29.263,
        29.114167,
        29.788611,
        29.6675,
    ]  # Latitude in degrees North
    lons: List[float] = [
        NEW_ORLEANS.lon,
        -87.211194,
        -88.075,
        -89.4075,
        -89.957,
        -90.199167,
        -90.420278,
        -91.237611,
    ]  # Longitude in degrees East
    stationid: List[str] = [
        "new-orleans",
        "8729840",
        "8735180",
        "8760922",
        "8761724",
        "8762075",
        "8762482",
        "8764044",
    ]

    # put original in pandas dataframe
    df = pd.DataFrame(
        {
            "Name": [stationid_to_names[x] for x in stationid],
            "StationID": stationid,
            "Original Latitiude": lats,
            "Original Longitude": lons,
        }
    )

    mele_ds = xr_loader(path_to_maxele)
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    for i, sid in enumerate(stationid):
        print(lons[i], lats[i], sid)
        distsq = (xs - lons[i]) ** 2 + (ys - lats[i]) ** 2
        min_p = np.argmin(distsq)
        lons[i] = xs[min_p]
        lats[i] = ys[min_p]

    df["Node Latitude"] = lats
    df["Node Longitude"] = lons

    # lets get node depth for those points too
    # df["Node Depth"] = mele_ds
    print("mele_ds", mele_ds.variables)

    data_path = os.path.join(DATA_PATH, "stations")
    os.makedirs(data_path, exist_ok=True)

    df.to_csv(os.path.join(data_path, "stationid_lat_lon.csv"), index=False)
    df.to_latex(
        os.path.join(data_path, "stationid_lat_lon.tex"), index=False  # decimal=3,
    )

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
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
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
    figure_name = os.path.join(FIGURE_PATH, subfolder, "stationid_map.pdf")
    plt.tight_layout()
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {figure_name}")
    # CESM, GFDL, GISS, MIROC, UKESM


def custom_parallel_coordinates(
    df: pd.DataFrame,
    class_column: str = None,
    cols: list = None,
    colors: dict = None,
    constraints: dict = None,
    balance: bool = False,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots a parallel coordinates chart where each column has its own vertical scale.

    Args:
        df (pd.DataFrame): DataFrame with data to plot.
        class_column (str, optional): Column to use for coloring. Defaults to None.
        cols (list, optional): Columns to plot. Defaults to None.
        colors (dict, optional): Colors to use for each class. Defaults to None.
        constraints (dict, optional): Constraints for each column. Defaults to None.
        balance (bool, optional): Balance the data. Defaults to False.
        legend (bool, optional): Show legend. Defaults to True.
        ax (Optional[plt.Axes], optional): Axis to plot on. Defaults to None.

    Returns:
        plt.Axes: Axis with plot.
    """
    NEW_LABELS = {
        "res": "Max SSH, $z^*$ [m]",
        "displacement": r"Track Displacement, $c^*$ [$^\circ$E]",
        "angle": r"Track Angle, $\chi^*$ [$^\circ$]",
        "trans_speed": r"Translation Speed, $V_t^*$ [m s$^{-1}$]",
    }

    # If class_column is given, separate the class labels for coloring
    if class_column and class_column in df.columns:
        unique_classes = df[class_column].unique()
    else:
        unique_classes = [None]

    # Figure and axis setup
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

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
    elif balance:
        for col in cols:
            if col != "res":
                mins[col] = min(mins[col], -maxs[col])
                maxs[col] = max(maxs[col], -mins[col])
    # an odd number so there is a middle line
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
            1.1,
            NEW_LABELS[col],
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=9,
            # fontweight="bold",
        )

        # Min/Max numerical labels. They appear just below/above the data range (0.0 and 1.0 after normalization)
        ax.text(i - 0.05, 0, f"{mins[col]:.3g}", ha="center", va="top", fontsize=12)
        ax.text(
            i - 0.05,
            0.5,
            f"{(mins[col]+maxs[col])/2:.3g}",
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.text(i - 0.05, 1, f"{maxs[col]:.3g}", ha="center", va="bottom", fontsize=12)

    # Remove default x-axis tick marks and labels since we handle them manually
    ax.set_xticks([])
    # Set y-limits to [0,1] because we normalized each column into that range
    ax.set_ylim([-0.1, 1.1])
    # ax.set_ylabel("Normalized Values (each axis scaled independently)")

    plt.tight_layout()
    # plt.show()
    # plot legend below plot if class_column is given
    if class_column and legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
        )
    return ax


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
                out[key] = (np.nanmean(l), float("nan"))
            else:
                out[key] = (np.nanmean(l), np.nanstd(l))
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
    print("\n\n2015\n", am1_res, "\n\n2100\n", am2_res, "\n\nDiff\n", amd_res)
    _, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i, (aml, name) in enumerate(
        [(am1_l, "2015"), (am2_l, "2100"), (amd_l, "diff")]
    ):
        ax = axs[i]
        df = to_pd(aml)
        df["stationid"] = [stationid_to_names[sid] for sid in stationid]
        if name == "diff":
            constraints = None
        else:
            constraints = DEFAULT_CONSTRAINTS
        if name == "diff":
            legend = True
            balance = True
        else:
            legend = False
            balance = False
        custom_parallel_coordinates(
            df,
            cols=["res", "displacement", "angle", "trans_speed"],
            class_column="stationid",
            colors={
                stationid_to_names[sid]: COLORS[i] for i, sid in enumerate(stationid)
            },
            constraints=constraints,
            ax=ax,
            legend=legend,
            balance=balance,
        )
        # plt.title("Parallel Coordinates Plot of Max and Argmax")
        if name != "diff":
            ax.set_xlabel(
                "Maximum SSH $z^{*}$ at point and corresponding arguments $c^{*}$, $\chi^*$, $V_t^{*}$ for "
                + name
            )
        else:
            ax.set_xlabel(
                "Difference in SSH $z^{*}$ at point and corresponding arguments $c^{*}$, $\chi^*$, $V_t^{*}$ between "
                + f" {years[0]} and {years[1]}"
            )  # plt.ylabel("Values")
        # plt.grid(True)
        # ax.set_grid(True)
        # add grid to axis object
        ax.grid(True)

    plt.tight_layout()
    label_subplots(axs, x_pos=-0.03)
    plt.savefig(os.path.join(FIGURE_PATH, subfolder, f"parallel_coordinates_all.pdf"))
    # splt.show()
    plt.clf()


def get_max_from_ib_list(
    iblist: List[Tuple[int, int]],
) -> Tuple[List[float], List[int]]:
    """
    Get max from ib list.

    Args:
        iblist (List[Tuple[int, int]]): List of tuples.
    """
    res_list = []
    b_list = []
    for i, b in iblist:
        print("LHS points", i, "DAF points", b)
        exp_name = f"i{i}b{b}"
        if not os.path.exists(os.path.join(EXP_PATH, exp_name)):
            print(f"Experiment {exp_name} does not exist.")
            # continue
        else:
            exp = read_json(os.path.join(EXP_PATH, exp_name, "experiments.json"))
            # calls = list(exp.keys())
            res = listify(exp, "res")
            res_list.append(max(res))
            b_list.append(b)
            print(f"Max res for {exp_name} is {max(res)}")
    return res_list, b_list


def plot_bo_exp() -> None:
    plot_defaults()
    _, _ = plt.subplots(1, 1, figsize=get_dim())
    res_list, b_list = get_max_from_ib_list([(1, 9), (5, 5), (9, 1), (10, 0)])
    plt.plot(b_list, res_list, color="orange", label="Total Calls 10")
    res_list, b_list = get_max_from_ib_list([(20, 0), (19, 1), (10, 10), (1, 19)])
    plt.plot(b_list, res_list, color="purple", label="Total Calls 20")
    res_list, b_list = get_max_from_ib_list(
        [(50, 0), (49, 1), (35, 15), (25, 25), (15, 35), (1, 49)]
    )
    plt.plot(b_list, res_list, color="blue", label="Total Calls 50")
    plt.xlabel("BO points")
    plt.ylabel("Max SSH over experiment, $z^{*}$ [m]")
    plt.title("Max SSH over experiment $z^{*}$ vs. BO points")
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 50)
    plt.savefig(
        os.path.join(FIGURE_PATH, subfolder, "bo_vary_init_vary_daf_exp.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def plot_bo_comp() -> None:
    # try redoing 25i 25b for 2015.
    # had mulitple trials (11) for each experiment
    # naming now i{i}b{b}t{trial} apart from 0th where its i{i}b{b}
    plt.clf()
    plt.close()
    plot_defaults()
    res_lol = []
    for i, b in [(50, 0), (25, 25), (3, 47)]:  #  (1, 49)
        res_lol += [[]]
        for t in [i for i in range(11)]:  # if i not in [4, 5, 9, 10]]:
            if t == 0 and i in {50, 25}:
                exp_name = f"i{i}b{b}"
            else:
                exp_name = f"i{i}b{b}t{t}"
            if not os.path.exists(os.path.join(EXP_PATH, exp_name)):
                print(f"Experiment {exp_name} does not exist.")
                res_lol[-1].append([float("nan")] * 50)
            else:
                exp = read_json(os.path.join(EXP_PATH, exp_name, "experiments.json"))
                res = listify(exp, "res")
                if len(res) < 50:
                    print(f"Experiment {exp_name} does not have 50 samples.")
                    res += [float("nan")] * (50 - len(res))
                print(f"Max res for {exp_name} is {max(res)}")
                res_lol[-1].append(res)

    res_array = np.array(res_lol)
    # TODO: replace with nan safe operations that discount nans in averages and maximums etc.
    # use nanmax, nanmean, nanstd etc.
    global_max = np.nanmax(res_array)
    # take cumulative maximum over each trial
    cum_max_array = np.maximum.accumulate(res_array, axis=2)
    # take cumulative maximum over each trial
    trial_label = [
        "50 LHS points",
        "25 LHS, 25 BO points",
        "3 LHS, 47 BO Points",
    ]  # , "10 LHS, 40 BO points"]
    colors = ["blue", "red", "green"]  # , "orange"]

    def plot_ensemble(
        array: np.ndarray, i: int, letter: str, trial_label: str, color: str
    ) -> None:
        for j in range(array.shape[1]):
            if j == 0:
                labels = {
                    "label": f"({letter.upper()}) {trial_label} trials ({array.shape[1]})"
                }
            else:
                labels = {}
            plt.plot(
                np.arange(50) + 1,
                array[i][j],
                color=color,
                linestyle="--",
                linewidth=0.5,
                **labels,
            )
        plt.plot(
            np.arange(50) + 1,
            np.mean(array[i], axis=0),
            label=f"({letter.upper()}) Mean",
            color=color,
            linewidth=1,
        )
        plt.fill_between(
            np.arange(50) + 1,
            np.nanpercentile(array[i], 5, axis=0),
            np.nanpercentile(array[i], 95, axis=0),
            label=f"({letter.upper()}) 5% to 95% envelope",
            color=color,
            alpha=0.4,
        )

    plot_ensemble(res_array, 0, "a", trial_label[0], colors[0])
    plot_ensemble(res_array, 1, "b", trial_label[1], colors[1])
    plot_ensemble(res_array, 2, "c", trial_label[2], colors[2])
    plt.axvline(25, color="black", linestyle="--")
    plt.xlabel("Samples, $s$ (LHS + BO points) [dimensionless]")
    plt.ylabel("Max SSH for sample, $z$ [m]")
    plt.xlim(1, 50)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_PATH, subfolder, "bo_exp_2525vs50_trials.pdf"),
        bbox_inches="tight",
    )

    plt.clf()
    plt.close()
    plot_ensemble(cum_max_array, 0, "a", trial_label[0], colors[0])
    plot_ensemble(cum_max_array, 1, "b", trial_label[1], colors[1])
    plot_ensemble(cum_max_array, 2, "c", trial_label[2], colors[2])

    plt.axvline(25, color="black", linestyle="--")
    plt.xlabel("Samples, $s$ (LHS + BO points) [dimensionless]")
    plt.ylabel("Max SSH over experiment, $z^{*}$ [m]")
    plt.xlim(1, 50)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_PATH, subfolder, "bo_comp_2525vs50.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()
    regret_array = global_max - cum_max_array
    plot_ensemble(regret_array, 0, "a", trial_label[0], colors[0])
    plot_ensemble(regret_array, 1, "b", trial_label[1], colors[1])
    plot_ensemble(regret_array, 2, "c", trial_label[2], colors[2])
    plt.axvline(25, color="black", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Samples, $s$ (LHS + BO points) [dimensionless]")
    plt.ylabel("Approximate Simple Regret [m]")
    plt.xlim(1, 50)
    # 2 columns below figure
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_PATH, subfolder, "bo_regret_2525vs50.pdf"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # set y-axis to be semi-log
    # r"Empirical Regret for dataset $i$ at step $s$, $\max\left(\max\left(\vec{z}^1\right), \cdots \max\left(\vec{z}^n\right)\right) - \max\left(\vec{z}^i_{1,\cdots,s}\right)$ [m]"


@timeit
def make_argmax_table(daf: str = "mes") -> None:
    """We want to make a table of the argmax values for each experiment.

    Args:
        daf (str, optional): DAF name. Defaults to "mes". Also ran an "ei" experiment.

    Columns:
        - Name (e.g New Orleans, Galverston, Miami)
        - Year (e.g 2015, 2100)
        - Trial (e.g 0, 1, 2)
        - Argmax index (1 to 50)
        - Displacement
        - Angle
        - Translation Speed
        - Max SSH
    """
    df = pd.DataFrame(
        columns=[
            "Name",
            "Year",
            "Trial",
            r"\(i\)",
            r"\(c\) [\(^{\circ}\)]",
            r"\(\chi\) [\(^{\circ}\)]",
            r"\(V_t\) [m s\(^{-1}\)]",
            r"\(z^{*}\) [m]",
        ]
    )
    for point in ["new-orleans", "miami", "galverston"]:
        for year in ["2015", "2100"]:
            for trial in range(12):
                if trial == 0 and daf == "mes":
                    exp_name = f"{point}-{year}"
                elif daf == "mes":
                    exp_name = f"{point}-{year}-{trial}"
                else:
                    exp_name = f"{point}-{year}-{trial}-{daf}"
                if not os.path.exists(os.path.join(EXP_PATH, exp_name)):
                    print(f"Experiment {exp_name} does not exist.")
                    continue
                else:
                    exp = read_json(
                        os.path.join(EXP_PATH, exp_name, "experiments.json")
                    )
                    # calls = list(exp.keys())
                    res = listify(exp, "res")
                    max_res = max(res)
                    argmax_index = res.index(max_res)
                    displacement = listify(exp, "displacement")[argmax_index]
                    angle = listify(exp, "angle")[argmax_index]
                    trans_speed = listify(exp, "trans_speed")[argmax_index]
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Name": [stationid_to_names[point]],
                                    "Year": [year],
                                    "Trial": [trial],
                                    r"\(i\)": [argmax_index],
                                    r"\(c\) [\(^{\circ}\)]": [displacement],
                                    r"\(\chi\) [\(^{\circ}\)]": [angle],
                                    r"\(V_t\) [m s\(^{-1}\)]": [trans_speed],
                                    r"\(z^{*}\) [m]": [max_res],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
    # save to csv
    # add 1 to argmax index
    df[r"\(i\)"] = df[r"\(i\)"] + 1
    if daf == "mes":
        tex_out_path = os.path.join(DATA_PATH_ADBO, "argmax_table.tex")
    else:
        tex_out_path = os.path.join(DATA_PATH_ADBO, f"argmax_table_{daf}.tex")
    df.to_latex(tex_out_path, index=False, escape=False)  # decimal=3,
    paths_to_plot = []
    N2ID = {v: k for k, v in stationid_to_names.items()}
    for _, row in df.iterrows():
        # exp_name = f"{row['Name']}-{row['Year']}-{int(row['Trial'])}"
        if int(row["Trial"]) == 0:
            exp_name = f"{N2ID[row['Name']]}-{row['Year']}"
        else:
            exp_name = f"{N2ID[row['Name']]}-{row['Year']}-{int(row['Trial'])}"
        # print(exp_name)
        j = int(row[r"\(i\)"]) - 1
        paths_to_plot.append(os.path.join(EXP_PATH, exp_name, f"exp_{j:04}"))

    if daf == "mes":
        csv_out_path = os.path.join(DATA_PATH_ADBO, "argmax_table.csv")
    else:
        csv_out_path = os.path.join(DATA_PATH_ADBO, f"argmax_table_{daf}.csv")

    # add to dataframe
    df["Path"] = paths_to_plot
    df.to_csv(csv_out_path, index=False)

    figure_path = os.path.join(FIGURE_PATH, "argmax_snapshots")
    os.makedirs(figure_path, exist_ok=True)

    for i, row in df.iterrows():
        if row["Name"] == "New Orleans":
            qk_loc = {"x_pos": 0.5, "y_pos": 1.05}
        else:
            qk_loc = {"x_pos": 1, "y_pos": -0.05}
        if daf == "mes":
            file_name = (
                f"{N2ID[row['Name']]}_{row['Year']}_{int(row['Trial'])}_snapshot.pdf"
            )
        else:
            file_name = f"{N2ID[row['Name']]}_{row['Year']}_{int(row['Trial'])}_{daf}_snapshot.pdf"
        single_wind_and_height_step(
            path_in=row["Path"],
            bbox=stationid_to_bbox[N2ID[row["Name"]]],
            time_i=None,
            coarsen=3,
            plot_loc=True,
            figure_name=os.path.join(
                figure_path,
                file_name,
            ),
            **qk_loc,
        )


if __name__ == "__main__":
    # python -m adbo.plot
    make_argmax_table("ei")
    make_argmax_table("mes")
    # for point in ["new-orleans", "miami", "galverston"]:
    #    plot_diff(
    #        exps=(f"{point}-2015", f"{point}-2100"),
    #        figure_name=f"2015-vs-2100-{point}.pdf",
    #    )
    # plt.clf()
    # plot_many("2015")
    # plot_many("2100")
    # plot_places()
    # find_differences()
    # plot_multi_argmax()
    # plot_bo_exp()
    # plot_bo_comp()
