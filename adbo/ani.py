import os
import xarray as xr
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

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
    """
    plot_defaults()
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

    for call_i in tqdm(range(len(ds.call.values)), desc="Plotting GPs"):
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
                    ("x1", "Angle", "$^{\circ}$"),
                    ("x2", "Displacement", "$^{\circ}$"),
                ),
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


if __name__ == "__main__":
    # python -m adbo.ani
    plot_gps(
        path_in="/work/n01/n01/sithom/adcirc-swan/exp/test-2d-3",
        add_name="",
        plot_acq=True,
    )
