"""afdorce.ani.py"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import imageio
from tqdm import tqdm
from sithom.io import read_json
from sithom.plot import plot_defaults, feature_grid, label_subplots
from sithom.time import timeit

matplotlib.use("Agg")


@timeit
def plot_gps(
    path_in: str = "mult1",
    gp_file: str = "gp_model_outputs.nc",
    plot_acq: bool = False,
    add_name: str = "",
    verbose: bool = False,
    save_pdf: set = set(),
) -> None:
    """
    Plot GPs.

    Function currently assumes that the GP file contains the following variables:
    - ypred: GP prediction
    - yvar: GP variance
    - acq: acquisition function

    And has the following dimensions:
    - call: number of calls
    - x1: angle (bearing) [degrees from North]
    - x2: displacement [degrees east of New Orleans]

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        gp_file (str, optional): name of GP file. Defaults to "gp_model_outputs.nc".
        plot_acq (bool, optional): Plot acquisition function. Defaults to False.
        add_name (str, optional): Additional name for the gif. Defaults to "".
    """
    plot_defaults()

    exp = read_json(os.path.join(path_in, "experiments.json"))
    calls = list(exp.keys())
    res = [float(exp[call]["res"]) for call in calls]
    displacement = [float(exp[call]["displacement"]) for call in calls]
    angle = [float(exp[call]["angle"]) for call in calls]
    calls = [float(call) + 1 for call in calls]

    img_folder = os.path.join(path_in, "img")
    gif_folder = os.path.join(path_in, "gif")

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)

    figure_names = []

    ds = xr.open_dataset(os.path.join(path_in, gp_file))

    print(ds)

    ds["ypred"] = -ds.ypred

    vminm, vmaxm = ds.ypred.min().values, ds.ypred.max().values
    if "yvar" in ds:
        ds["ystd"] = ds.yvar**0.5  # take square root of variance to get std
    _, vmaxstd = 0, ds.ystd.max().values
    if plot_acq and "acq" in ds:
        vminacq, vmaxacq = ds.acq.min().values, ds.acq.max().values

    # x1_units = ds.x1.attrs["units"]
    # x2_units = ds.x2.attrs["units"]
    len_active_points = len(ds.call.values)
    num_init_data_points = len(calls) - len_active_points
    print("len_active_points", len_active_points)
    print("num_init_data_points", num_init_data_points)
    side_length = 3

    for call_i in tqdm(range(len_active_points), desc="Plotting GPs"):
        if plot_acq and "acq" in ds:  # plot three panels
            fig, axs = feature_grid(
                ds.isel(call=call_i),
                [["ypred", "ystd", "acq"]],
                [["m", "m", None]],
                [
                    [
                        r"GP mean, $\mu_{\hat{f}}$",
                        r"GP std. dev., $\sigma_{\hat{f}}$",
                        r"Acquisition function, $\alpha_t$",
                    ]
                ],
                [
                    [
                        [vminm, vmaxm, "cmo.amp"],
                        [0, vmaxstd, "cmo.amp"],
                        [vminacq, vmaxacq, "cmo.amp"],
                    ],
                ],
                ["", "", ""],
                figsize=(side_length * 3, side_length),
                xy=(
                    ("x1", r"Track Bearing, $\chi$", r"$^{\circ}$"),
                    ("x2", r"Track Displacement, $c$", r"$^{\circ}$E"),
                ),
            )
        else:  # plot 2 panels
            fig, axs = feature_grid(
                ds.isel(call=call_i),
                [["ypred", "ystd"]],
                [["m", "m"]],
                [
                    [
                        r"GP mean, $\mu_{\hat{f}}$",
                        r"GP std. dev., $\sigma_{\hat{f}}$",
                    ]
                ],
                [[[vminm, vmaxm, "cmo.amp"], [0, vmaxstd, "cmo.amp"]]],
                ["", ""],
                figsize=(side_length * 2, side_length),
                xy=(
                    ("x1", r"Track Bearing, $\chi$", r"$^{\circ}$"),
                    (
                        "x2",
                        r"Track Displacement, $c$",
                        r"$^{\circ}$E",
                    ),
                ),
            )
        # this seems not to work properly
        if verbose:
            # initial data points
            print(
                "intial_points",
                angle[:num_init_data_points],
                displacement[:num_init_data_points],
            )
            # BO/active data points
            print(
                "active_points",
                angle[num_init_data_points : -len_active_points + call_i],
            )
        for i in range(axs.ravel().shape[0]):
            axs.ravel()[i].scatter(
                angle[:num_init_data_points],
                displacement[:num_init_data_points],
                marker="x",
                s=55,
                color="blue",
            )
            #
            axs.ravel()[i].scatter(
                angle[num_init_data_points : num_init_data_points + call_i + 1],
                displacement[num_init_data_points : num_init_data_points + call_i + 1],
                s=100,
                marker="+",
                color="green",
            )
        label_subplots(axs, override="outside")
        if call_i in save_pdf:
            figure_name = os.path.join(img_folder, f"gp_{call_i}.pdf")
            plt.savefig(figure_name)
        plt.suptitle(
            "Additional sample " + str(call_i + 1),
        )
        figure_name = os.path.join(img_folder, f"gp_{call_i}.png")
        figure_names.append(figure_name)
        plt.savefig(figure_name)
        plt.close(fig)
        plt.clf()

    gif_name = os.path.join(gif_folder, add_name + "gp.gif")
    print("gif_name", gif_name)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for filename in figure_names:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Saved gif to {gif_name}")
    print("Active data points: ", len_active_points)
    print("Total data points: ", call_i + len_active_points + 1)


if __name__ == "__main__":
    # python -m adbo.ani
    plot_gps(
        path_in="/work/n02/n02/sdat2/adcirc-swan/exp/old/2d-ani-new-redo-4",
        add_name="",
        plot_acq=True,
        save_pdf={
            0,
            1,
            2,
            3,
            24,
        },
    )
