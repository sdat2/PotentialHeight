from typing import Optional, List, Tuple
import xarray as xr


def plot_features(
    plot_ds: xr.Dataset,
    features: List[List[str]],
    units: Optional[List[List[str]]] = None,
    names: Optional[List[List[str]]] = None,
    vlim: Optional[List[List[Tuple[str, float, float]]]] = None,
    super_titles: Optional[List[str]] = None,
) -> None:
    """
    A wrapper around the feature_grid function to plot the features of a dataset for the potential intensity inputs/outputs.

    Args:
        plot_ds (xr.Dataset): The dataset to plot data from.
        features (List[List[str]]): List of feature names to plot.
        units (Optional[List[List[str]]], optional): Units to plot. Defaults to None.
        names (Optional[List[List[str]]], optional): Names to plot. Defaults to None.
        vlim (Optional[List[List[Tuple[str, float, float]]]], optional): Colormap/vlim to plot. Defaults to None.
        super_titles (Optional[List[str]], optional): Supertitles to plot. Defaults to None.
    """

    if names is None:
        names = [
            [
                plot_ds[features[x][y]].attrs["long_name"]
                for y in range(len(features[x]))
            ]
            for x in range(len(features))
        ]
    if units is None:
        units = [
            [plot_ds[features[x][y]].attrs["units"] for y in range(len(features[x]))]
            for x in range(len(features))
        ]
    if vlim is None:
        vlim = [[None for y in range(len(features[x]))] for x in range(len(features))]
    if super_titles is None:
        super_titles = ["" for x in range(len(features))]

    fig, axs = feature_grid(
        plot_ds,
        features,
        units,
        names,
        vlim,
        super_titles,
        figsize=(12, 6),
    )
    label_subplots(axs)
