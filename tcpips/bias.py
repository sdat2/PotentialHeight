"""Compare the CMIP6 historical period against ERA5.

We might want to load the raw variables, or the derived variables like potential intensity and potential size.

For each ensemble member then might want to calculate the mean bias for each grid point for each variable, but also if this bias has a trend, and perhaps how significant the bias is.

For the ensemble as a whole we might want to compare how the distribution of mean biases and their trends compare, and whether there are significant difference between different families of models.

TODO: do hemisphere filtering to get August/March.

TODO: Find linear trend in bias, CMIP6, and ERA5. Work out significance of each.

TODO: Compare the distribution of biases, trends, and significance between different models and ensemble members.

"""
import os
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from sithom.time import timeit
from .constants import CDO_PATH, PI4_PATH, PS_PATH, BIAS_PATH
from .era5 import get_all_regridded_data, select_seasonal_hemispheric_data


def load_cmip6_data(exp = "historical", model = "CESM2", member = "r4i1p1f1") -> Optional[xr.Dataset]:
    """Load CMIP6 data for a specific experiment, model, and member.

    This function is a placeholder for loading CMIP6 data.
    The actual implementation would depend on the data source and format.

    Args:
        exp (str): Experiment name (default: "historical").
        model (str): Model name (default: "CESM2").
        member (str): Ensemble member (default: "r1i1p1f1").

    Returns:
        xarray.Dataset: Loaded CMIP6 dataset.
    """
    # Placeholder for actual implementation
    print(f"Loading CMIP6 data for {exp}, {model}, {member}...")
    paths = [os.path.join(CDO_PATH, exp, "ocean", model, member) + ".nc", os.path.join(CDO_PATH, exp, "atmos", model, member) + ".nc", os.path.join(PI4_PATH, exp, model, member) + ".zarr", os.path.join(PS_PATH, exp, model, member) + ".zarr"]
    ds_list = []
    for path in paths:
        if os.path.exists(path):
            if path.endswith('.zarr'):
                ds = xr.open_zarr(path, consolidated=True)
            else:
                ds = xr.open_dataset(path, chunks={"time": 1})  # Adjust chunk size as needed
            ds_list.append(ds)
            print(f"Loaded dataset from {path}")
        else:
            print(f"Dataset not found at {path}")
    one_day_tolerance = pd.to_timedelta("1D")  # , unit="D")
    for i, ds in enumerate(ds_list[1:]):
        ds_list[i+1] = ds.reindex(
        time=ds_list[0].time,
        method="nearest",  # tolerance="1D"
        tolerance=one_day_tolerance,
    )
        ds_list[i+1] = ds_list[i+1].drop_vars("time_bounds", errors="ignore")
    if ds_list:
        combined_ds = xr.merge(ds_list)
        print(f"Combined dataset contains variables: {list(combined_ds.data_vars)}")
        # remove time_bounds, nbnd,
        combined_ds = combined_ds.drop_vars("time_bounds", errors="ignore")
        # combined_ds = combined_ds.drop_vars("nbnd", errors="ignore")
        return combined_ds
    else:
        print("No datasets were loaded.")
        return None


@timeit
def calc_bias(start_year: int = 1980,
              end_year: int = 2014,
              model: str = "CESM2",
              member: str ="r4i1p1f1",
              exp: str="historical") -> None:
    """Calc mean biases over specified period.

    Args:
        start_year (int): Start year for the analysis (default: 1980).
        end_year (int): End year for the analysis (default: 2014).
        model (str): Model name (default: "CESM2").
        member (str): Ensemble member (default: "r4i1p1f1").
        exp (str): Experiment name (default: "historical").

    """
    era5_ds = get_all_regridded_data(start_year=1980, end_year=2024).sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    print("era5", era5_ds)
    print(f"Loaded ERA5 data from {start_year} to {end_year}.")
    print(f"Data variables: {list(era5_ds.data_vars)}")
    cmip6_ds = load_cmip6_data().sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    if cmip6_ds is not None:
        print(f"Loaded CMIP6 data with variables: {list(cmip6_ds.data_vars)}")

    vars_to_compare = set(era5_ds.data_vars).intersection(set(era5_ds.data_vars)).intersection(["sst", "vmax", "pmin", "otl"])
    print(f"Variables to compare: {vars_to_compare}")
    # ok, let's put the time coordinates on the same axis
    cmip6_ds = cmip6_ds.convert_calendar('standard', use_cftime=False)
    # print("Old CMIP6 time axis:", cmip6_ds.time)
    #cmip6_ds[vars_to_compare].mean(dim="time", keep_attrs=True).to_netcdf("cmip6_mean_old_time.nc", mode="w")
    cmip6_ds = cmip6_ds.reindex(
        time=era5_ds.time,
        method="nearest",
        # tolerance = 1 month
        tolerance=pd.to_timedelta("30D"),
        # tolerance=pd.to_timedelta("1"),
    )

    print("CMIP6", cmip6_ds)
    cmip6_ds = cmip6_ds.rename({"plev": "pressure_level"})
    cmip6_ds = cmip6_ds.drop_vars("time_bounds", errors="ignore")
    print("Aligned CMIP6 data to ERA5 time axis.")
    bias_ds = cmip6_ds[vars_to_compare] - era5_ds[vars_to_compare]
    #cmip6_ds[vars_to_compare].mean(dim="time", keep_attrs=True).to_netcdf("cmip6_mean.nc", mode="w")
    #era5_ds[vars_to_compare].mean(dim="time", keep_attrs=True).to_netcdf("era5_mean.nc", mode="w")

    print("bias", bias_ds)
    print(f"Bias dataset variables: {list(bias_ds.data_vars)}")
    #bias_ds.mean(dim="time", keep_attrs=True).to_netcdf("cmip6_bias.nc", mode="w")
    bias_ds.mean(dim="time", keep_attrs=True).to_netcdf(".nc", mode="w")
    historical_mean = cmip6_ds[vars_to_compare].mean(dim="time", keep_attrs=True)
    folder = os.path.join(BIAS_PATH, exp, model)
    os.makedirs(folder, exist_ok=True)
    historical_mean.to_netcdf(os.path.join(folder, member+".nc"), mode="w")
    print(f"Saved historical mean to {os.path.join(folder, member+ '.nc')}")


# TODO: write an ensemble reading function
@timeit
def example_new_orleans_plot() -> None:
    # plot example for a point near New Orleans.
    import matplotlib.pyplot as plt
    from .constants import FIGURE_PATH
    from sithom.plot import plot_defaults, label_subplots
    # let's just go for vmax to start with.
    # panel 1: ERA5 + CMIP6 ensemble members + CMIP6 multi-model mean
    # panel 2: bias
    plot_defaults()
    fig, axs = plt.subplots(2, 1, sharex=True)
    era5_ds = get_all_regridded_data().sel(
        time=slice("1980-01-01", "2014-12-31")
    )["vmax"].sel(lat=29.95-1, lon=-90.07, method="nearest")
    cmip6_ds = load_cmip6_data().sel(
        time=slice("1980-01-01", "2014-12-31")
    )["vmax"].sel(lat=29.95-1, lon=-90.07, method="nearest")
    cmip6_ds = cmip6_ds.convert_calendar('standard', use_cftime=False)
    cmip6_ds = select_seasonal_hemispheric_data(cmip6_ds.reindex(
        time=era5_ds.time,
        method="nearest",
        # tolerance = 1 month
        tolerance=pd.to_timedelta("30D"),
        # tolerance=pd.to_timedelta("1"),
    ),
    lon="lon", lat="lat")
    era5_ds = select_seasonal_hemispheric_data(era5_ds, lon="lon", lat="lat")
    bias_ds = cmip6_ds - era5_ds
    print(f"ERA5 data: {era5_ds}")
    print(f"ERA5 values {era5_ds.values}" )
    print(f"CMIP6 data: {cmip6_ds}")
    print(f"CMIP6 values {cmip6_ds.values}" )
    print(f"Bias data: {bias_ds}")
    era5_ds.plot(ax=axs[0], label="ERA5", color="blue")
    axs[0].axhline(era5_ds.mean(dim="year").values, color="blue", linestyle="--", label="ERA5 Mean")
    cmip6_ds.plot(ax=axs[0], label="CMIP6", color="orange")
    axs[0].axhline(cmip6_ds.mean(dim="year").values, color="orange", linestyle="--", label="CMIP6 Mean")
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    axs[0].set_title("")
    bias_ds.plot(ax=axs[1], label="Bias", color="red")
    axs[1].axhline(bias_ds.mean(dim="year").values, color="red", linestyle="--", label="Bias mean")
    axs[1].set_title("")
    axs[0].set_ylabel("$V_p$ [m s$^{-1}$]")
    axs[0].set_xlabel("")
    axs[0].set_xlim(1980, 2014)
    axs[1].set_ylabel("Bias, $\Delta V_p$ [m s$^{-1}$]")
    axs[1].set_xlabel("")
    label_subplots(axs)
    plt.savefig(os.path.join(FIGURE_PATH, "new_orleans_vmax_bias.pdf"), dpi=300)
    plt.close()
    print(f"Saved figure to {os.path.join(FIGURE_PATH, 'new_orleans_vmax_bias.pdf')}")


if __name__ == "__main__":
    # python -m tcpips.bias
    # Load the CMIP6 data
    # cmip6_data = load_cmip6_data()
    # for model in ["CESM2"]:
    #     for member in ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"]:
    #         print(f"Calculating bias for {model}, {member}...")
    #         calc_bias(model=model, member=member)
    example_new_orleans_plot()

