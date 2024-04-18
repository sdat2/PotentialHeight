import os
import xarray as xr
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from sithom.io import read_json
from sithom.plot import plot_defaults, feature_grid
from sithom.time import timeit

import matplotlib

matplotlib.use("Agg")


@timeit
def plot_gps(
    path_in: str = "mult1",
    gp_file: str = "gp_model_outputs.nc",
    plot_acq: bool = False,
    add_name: str = "",
) -> None:
    """
    Plot GPs.

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
        ds["ystd"] = ds.yvar**0.5
    vminstd, vmaxstd = ds.ystd.min().values, ds.ystd.max().values
    if plot_acq and "acq" in ds:
        vminacq, vmaxacq = ds.acq.min().values, ds.acq.max().values

    # x1_units = ds.x1.attrs["units"]
    # x2_units = ds.x2.attrs["units"]
    len_active_points = len(ds.call.values)
    num_init_data_points = len(calls) - len_active_points
    print("len_active_points", len_active_points)
    print("num_init_data_points", num_init_data_points)

    for call_i in tqdm(range(len_active_points), desc="Plotting GPs"):
        if plot_acq and "acq" in ds:
            fig, axs = feature_grid(
                ds.isel(call=call_i),
                [["ypred", "ystd", "acq"]],
                [["m", "m", None]],
                [
                    [
                        "GP prediction, $\hat{y}$",
                        "GP standard deviation, $\sigma_{\hat{y}}$",
                        r"Acquisition function, $\alpha$",
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
                figsize=(4 * 3, 4),
                xy=(
                    ("x1", "Angle", "$^{\circ}$"),
                    ("x2", "Displacement", "$^{\circ}$"),
                ),
            )
        else:

            fig, axs = feature_grid(
                ds.isel(call=call_i),
                [["ypred", "ystd"]],
                [["m", "m"]],
                [
                    [
                        "GP prediction, $\hat{y}$",
                        "GP standard deviation, $\sigma_{\hat{y}}$",
                    ]
                ],
                [[[vminm, vmaxm, "cmo.amp"], [0, vmaxstd, "cmo.amp"]]],
                ["", ""],
                figsize=(4 * 2, 4),
                xy=(
                    ("x1", "Track Bearing", "$^{\circ}$"),
                    ("x2", "Displacement East of New Orleans", "$^{\circ}$"),
                ),
            )
        # this seems not to work properly
        # initial data points
        print(
            "intial_points",
            angle[:num_init_data_points],
            displacement[:num_init_data_points],
        )
        axs.ravel()[0].scatter(
            angle[:num_init_data_points],
            displacement[:num_init_data_points],
            marker="x",
            s=2,
            color="blue",
        )
        # BO/active data points
        print(
            "active_points", angle[num_init_data_points : -len_active_points + call_i]
        )
        axs.ravel()[0].scatter(
            angle[num_init_data_points : num_init_data_points + call_i + 1],
            displacement[num_init_data_points : num_init_data_points + call_i + 1],
            s=2,
            marker="+",
            color="green",
        )
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
        path_in="/work/n01/n01/sithom/adcirc-swan/exp/bo-test-2d-4",
        add_name="",
        plot_acq=False,
    )
