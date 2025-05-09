"""afdorce.ani.py"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import imageio
from tqdm import tqdm
from sithom.io import read_json
from sithom.plot import plot_defaults, feature_grid, label_subplots
from sithom.time import timeit

matplotlib.use("Agg")

VAR_TO_LABEL = {
    "angle": r"Track Bearing, $\chi$",
    "displacement": r"Track Displacement, $c$",
    "ypred": r"GP mean, $\mu_{\hat{f}}$",
    "ystd": r"GP std. dev., $\sigma_{\hat{f}}$",
    "acq": r"Acquisition function, $\alpha_t$",
}
VAR_TO_UNITS = {
    "angle": r"$^{\circ}$",
    "displacement": r"$^{\circ}$E",
    "ypred": "m",
    "ystd": "m",
    "acq": "",
}


@timeit
def plot_gps(
    path_in: str = "mult1",
    gp_file: str = "gp_model_outputs.nc",
    plot_acq: bool = False,
    add_name: str = "",
    verbose: bool = False,
    save_pdf: set = set(),
    side_length=3,
) -> None:
    """
    Plot GPs. Now supports 1D and 2D plots.

    Function currently assumes that the GP file contains the following variables:
    - ypred: GP prediction
    - yvar: GP variance
    - acq: acquisition function

    And has the following dimensions:
    - call: number of calls
    Either x1 or x1 and x2.

    Reads bo-config.json to determine the order of the variables.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        gp_file (str, optional): name of GP file. Defaults to "gp_model_outputs.nc".
        plot_acq (bool, optional): Plot acquisition function. Defaults to False.
        add_name (str, optional): Additional name for the gif. Defaults to "".
    """
    plot_defaults()
    # read config yaml
    config = read_json(os.path.join(path_in, "bo-config.json"))["constraints"]
    F = len(config["order"])  # number of features

    if F > 2:
        raise ValueError("Only 2D and 1D plots are supported")

    exp = read_json(os.path.join(path_in, "experiments.json"))
    if len(exp) == 0:
        raise ValueError("No experiments found in experiments.json")
    calls = list(exp.keys())
    res = [float(exp[call]["res"]) for call in calls]
    x_vals = []
    for key in config["order"]:
        x_vals.append([float(exp[call][key]) for call in calls])

    X = np.array(x_vals)
    assert X.shape == (F, len(calls)), "X shape mismatch"

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

    len_active_points = len(ds.call.values)
    num_init_data_points = len(calls) - len_active_points
    print("len_active_points", len_active_points)
    print("num_init_data_points", num_init_data_points)

    for call_i in tqdm(range(len_active_points), desc="Plotting GPs"):
        if F == 1 and "acq" in ds:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(ds.x1.values, ds.ypred.isel(call=call_i).values)
            # axs[0].set_ylabel(r"GP mean, $\mu_{\hat{f}}$")
            axs[0].fill_between(
                ds.x1.values,
                ds.ypred.isel(call=call_i).values - ds.ystd.isel(call=call_i).values,
                ds.ypred.isel(call=call_i).values + ds.ystd.isel(call=call_i).values,
                alpha=0.2,
            )
            axs[0].set_ylabel(r"Storm surge height [m]")
            # axs[1].plot(ds.x1.values, ds.ystd.isel(call=call_i).values)
            # axs[1].set_ylabel(r"GP std. dev., $\sigma_{\hat{f}}$")
            axs[1].plot(ds.x1.values, ds.acq.isel(call=call_i).values)
            axs[1].set_ylabel(r"Acquisition function, $\alpha_t$")
            axs[1].set_xlabel(
                VAR_TO_LABEL[config["order"][0]]
                + " ["
                + VAR_TO_UNITS[config["order"][0]]
                + "]"
            )
            # axs[0].set_title("Additional sample " + str(call_i + 1))
            axs[0].set_ylim(vminm - vmaxstd, vmaxm + vmaxstd)
            # axs[1].set_ylim(0, vmaxstd)
            axs[1].set_ylim(vminacq, vmaxacq)
            axs[0].set_xlim(
                ds.x1.min().values,
                ds.x1.max().values,
            )

        elif plot_acq and "acq" in ds:  # plot three panels
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
                xy=tuple(
                    [
                        (
                            f"x{i+1}",
                            VAR_TO_LABEL[config["order"][i]],
                            VAR_TO_UNITS[config["order"][i]],
                        )
                        for i in range(F)
                    ]
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
                    (
                        "x1",
                        VAR_TO_LABEL[config["order"][0]],
                        VAR_TO_UNITS[config["order"][0]],
                    ),
                    (
                        "x2",
                        VAR_TO_LABEL[config["order"][1]],
                        VAR_TO_UNITS[config["order"][1]],
                    ),
                ),
            )

        if verbose and F == 2:
            # initial data points
            print(
                "intial_points",
                X[0, :num_init_data_points],
                X[1, :num_init_data_points],
            )
            # BO/active data points
            print(
                "active_points",
                X[0, num_init_data_points : -len_active_points + call_i],
                X[1, num_init_data_points : -len_active_points + call_i],
            )
        if F == 1:
            for i in range(axs.ravel().shape[0]):
                axs[i].scatter(
                    X[0, :num_init_data_points],
                    res[:num_init_data_points],
                    marker="x",
                    s=5,
                    color="blue",
                )
                #
                axs[i].scatter(
                    X[0, num_init_data_points : num_init_data_points + call_i + 1],
                    res[num_init_data_points : num_init_data_points + call_i + 1],
                    s=5,
                    marker="+",
                    color="green",
                )
        elif F == 2:
            for i in range(axs.ravel().shape[0]):
                axs.ravel()[i].scatter(
                    X[0, :num_init_data_points],
                    X[1, :num_init_data_points],
                    marker="x",
                    s=55,
                    color="blue",
                )
                #
                axs.ravel()[i].scatter(
                    X[0, num_init_data_points : num_init_data_points + call_i + 1],
                    X[1, num_init_data_points : num_init_data_points + call_i + 1],
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
        path_in="/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/adcirc-swan/worstsurge/exp/2d-ani-ei/exp_0049",
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
