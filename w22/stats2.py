"""Generate tables of results.
"""
import os
from typing import Tuple, List, Dict
import math
import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf
from uncertainties import ufloat
from sithom.curve import fit
from .plot import get_timeseries
from .constants import DATA_PATH


# CONFIGURATION: Models and variables to process
MODELS_TO_PROCESS = {
    "HadGEM3-GC31-MM": ["r1i1p1f3", "r2i1p1f3", "r3i1p1f3"],
    "MIROC6": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
    "CESM2": ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"],
}

VARIABLES_TO_PROCESS = [# "sst", ""
"vmax_3", "r0_3", "rmax_3", # "r0_1",
"rmax_1"]


def safe_grad(xt: np.ndarray, yt: np.ndarray) -> ufloat:
    """
    Calculate the gradient of the data with error handling.

    Args:
        xt (np.ndarray): The x data.
        yt (np.ndarray): The y data.

    Returns:
        ufloat: The gradient.
    """
    # get rid of nan values
    valid_mask = ~np.isnan(xt) & ~np.isnan(yt)
    xt, yt = xt[valid_mask], yt[valid_mask]
    if len(xt) < 2:
        return ufloat(np.nan, np.nan)
    # normalize the data between 0 and 10
    xrange = np.max(xt) - np.min(xt)
    yrange = np.max(yt) - np.min(yt)
    if xrange == 0 or yrange == 0:
        return ufloat(0, 0)
    xt_norm = (xt - np.min(xt)) / xrange * 10
    yt_norm = (yt - np.min(yt)) / yrange * 10
    # fit the data with linear fit using OLS
    param, _ = fit(xt_norm, yt_norm)  # defaults to y=mx+c fit
    return param[0] * yrange / xrange


def safe_corr(xt: np.ndarray, yt: np.ndarray) -> float:
    """
    Calculate the correlation of the data with error handling.

    Args:
        xt (np.ndarray): The x data.
        yt (np.ndarray): The y data.

    Returns:
        float: The correlation.
    """
    valid_mask = ~np.isnan(xt) & ~np.isnan(yt)
    if np.sum(valid_mask) < 2:
        return np.nan
    corr = ma.corrcoef(ma.masked_invalid(xt), ma.masked_invalid(yt))
    return corr[0, 1]


def calculate_detrended_cv(timeseries_data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calculates the detrended standard deviation and coefficient of variation.

    This function performs a linear regression on the input time series to
    remove the trend. It then calculates the standard deviation of the
    residuals (the detrended data) and the coefficient of variation (CV)
    relative to the mean of the original, non-detrended data.

    Args:
        timeseries_data: A 1-dimensional numpy array of numerical data
                         representing the time series.

    Returns:
        A tuple containing:
        - residuals (np.ndarray): The detrended time series.
        - std_residuals (float): The standard deviation of the residuals.
        - detrended_cv (float): The coefficient of variation of the detrended
                                data, normalized by the original mean.

    Doctests:
    >>> # Test with a perfect linear trend and no noise
    >>> perfect_series = np.arange(10) * 2 + 50
    >>> res, std_res, cv = calculate_detrended_cv(perfect_series)
    >>> np.allclose(std_res, 0.0)
    True
    >>> np.allclose(cv, 0.0)
    True
    >>> # Test with some noise - CORRECTED VALUE
    >>> noisy_series = np.array([51, 51, 54, 55, 56, 59, 60, 62, 63, 65])
    >>> res, std_res, cv = calculate_detrended_cv(noisy_series)
    >>> round(std_res, 4)
    0.5865
    >>> round(cv, 4)
    0.0102
    """
    timeseries_data = np.asarray(timeseries_data)
    n = len(timeseries_data)
    if timeseries_data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array.")
    if n < 3:
        # Cannot compute with N-2 degrees of freedom
        return np.array([]), np.nan, np.nan

    time_axis = np.arange(n)

    slope, intercept, _, _, _ = ss.linregress(time_axis, timeseries_data)
    trend_line = slope * time_axis + intercept
    residuals = timeseries_data - trend_line

    sum_of_squared_residuals = np.sum(residuals**2)
    std_residuals = np.sqrt(sum_of_squared_residuals / (n - 2))

    original_mean = np.mean(timeseries_data)
    if original_mean == 0:
        detrended_cv = np.nan
    else:
        detrended_cv = std_residuals / np.abs(original_mean)

    return residuals, std_residuals, detrended_cv


def analyze_bias(ground_truth: np.ndarray, model: np.ndarray) -> dict:
    """
    Analyzes the bias between a model and ground truth data.

    Calculates the mean of the bias and the trend in the bias, and determines
    if they are statistically significant from zero.

    Args:
        ground_truth: A 1D numpy array of the true values.
        model: A 1D numpy array of the modeled values.

    Returns:
        A dictionary containing the mean bias, the p-value for the mean bias,
        the trend of the bias (slope), and the p-value for the trend.

    Doctests:
    >>> truth = np.linspace(0, 10, 10)
    >>> # Model has a constant bias of +2 and a slight upward trend in bias
    >>> model_data = truth + 2 + np.linspace(0, 0.5, 10)
    >>> results = analyze_bias(truth, model_data)
    >>> round(results['mean_bias'], 4)
    2.25
    >>> results['mean_bias_p_value'] < 0.001
    True
    >>> round(results['bias_trend'], 4)
    0.0556
    >>> results['bias_trend_p_value'] < 0.001
    True
    """
    if ground_truth.shape != model.shape:
        raise ValueError("Input arrays must have the same shape.")

    bias = model - ground_truth
    t_stat_mean, p_value_mean = ss.ttest_1samp(bias, 0.0, nan_policy='omit')
    time_axis = np.arange(len(bias))
    lin_reg_result = ss.linregress(time_axis, bias)

    return {
        'mean_bias': np.nanmean(bias),
        'mean_bias_p_value': p_value_mean,
        'bias_trend': lin_reg_result.slope,
        'bias_trend_p_value': lin_reg_result.pvalue,
    }


# TODO 3: Refactored function to be more extensible
def timeseries_relationships(
    timeseries_ds: xr.Dataset,
    place: str,
    member: str,
    year_min: int,
    year_max: int,
    variables: List[str],
) -> pd.DataFrame:
    """
    Calculates trend and correlation statistics for a set of timeseries variables.

    Args:
        timeseries_ds (xr.Dataset): Dataset containing the time series.
        place (str): Name of the location.
        member (str): Identifier for the model member (e.g., 'r1i1p1f1').
        year_min (int): Start year of the analysis period.
        year_max (int): End year of the analysis period.
        variables (List[str]): A list of variable names to analyze.

    Returns:
        pd.DataFrame: A single-row DataFrame with calculated statistics.
    """
    results = {
        "place": place, "member": member,
        "year_min": year_min, "year_max": year_max,
    }
    ds_period = timeseries_ds.sel(time=slice(str(year_min), str(year_max)))
    ts = {var: ds_period[var].values for var in variables if var in ds_period}
    if not ts: return pd.DataFrame()

    years = ds_period["time"]# .dt.year.values

    # Correlations and gradients with time (years)
    for var in ts:
        results[f"rho_{var}"] = safe_corr(ts[var], years)
        if var != 'sst': # SST gradient with time not in original analysis
            scale = 1000 if 'r0' in var or 'rmax' in var else 1
            fit_val = safe_grad(years, ts[var] / scale)
            results[f"fit_{var}"] = fit_val.n
            results[f"fit_{var}_err"] = fit_val.s

    # Cross-correlations and gradients with SST
    if "sst" in ts:
        for var in ts:
            if var != "sst":
                results[f"rho_{var}_sst"] = safe_corr(ts[var], ts["sst"])
                scale = 1000 if 'r0' in var or 'rmax' in var else 1
                fit_val_sst = safe_grad(ts["sst"], ts[var] / scale)
                results[f"fit_{var}_sst"] = fit_val_sst.n
                results[f"fit_{var}_sst_err"] = fit_val_sst.s

    return pd.DataFrame([results])


def temporal_relationship_data(place: str = "new_orleans", pi_version: int = 4) -> None:
    """
    Generates and saves temporal relationship statistics for multiple models.

    Args:
        place (str): The place to get the data for.
        pi_version (int): The potential intensity version to use.
    """
    print(f"\n--- Generating Temporal Relationship Tables for {place} ---")
    df_l = []

    # Process CMIP models
    for model, members in MODELS_TO_PROCESS.items():
        for member_str in members:
            try:
                timeseries_ds = get_timeseries(
                    model=model, place=place, member=member_str, pi_version=pi_version
                )
                # Future period
                df_l.append(timeseries_relationships(
                    timeseries_ds, place, member_str, 2014, 2100, VARIABLES_TO_PROCESS))
                # Historical period
                df_l.append(timeseries_relationships(
                    timeseries_ds, place, member_str, 1980, 2014, VARIABLES_TO_PROCESS))
            except Exception as e:
                print(f"Could not process {model}/{member_str}. Skipping. Error: {e}")

    # Process ERA5
    try:
        timeseries_ds_era5 = get_timeseries(place=place, model="ERA5", pi_version=pi_version)
        for start_year in [1980, 1940]:
            df_l.append(timeseries_relationships(
                timeseries_ds_era5, place, "ERA5", start_year, 2024, VARIABLES_TO_PROCESS))
    except Exception as e:
        print(f"Could not process ERA5. Skipping. Error: {e}")

    if not df_l:
        print("No data processed. Aborting table generation.")
        return

    df = pd.concat(df_l, ignore_index=True)
    df.to_csv(os.path.join(DATA_PATH, f"{place}_temporal_relationships_pi{pi_version}.csv"), index=False)
    print("Saved temporal relationships data to CSV.")

    df.drop(columns=["place"], inplace=True)
    corr_cols = ['member', 'year_min', 'year_max'] + [c for c in df.columns if c.startswith('rho_')]
    fit_cols = ['member', 'year_min', 'year_max'] + [c for c in df.columns if c.startswith('fit_')]

    # Save correlation table
    df_str_corr = dataframe_to_latex_table(df[corr_cols])
    with open(os.path.join(DATA_PATH, f"{place}_temporal_correlation_pi{pi_version}.tex"), "w") as f:
        f.write(df_str_corr)
    print("Saved temporal correlation data to LaTeX table.")

    # Save fit/gradient table
    df_str_fit = dataframe_to_latex_table(df[fit_cols])
    with open(os.path.join(DATA_PATH, f"{place}_temporal_fit_pi{pi_version}.tex"), "w") as f:
        f.write(df_str_fit)
    print("Saved temporal gradient data to LaTeX table.")


# TODO 2: New function to generate bias and variability tables
def generate_assessment_tables(
    place: str = "new_orleans",
    pi_version: int = 4,
    comparison_period: Tuple[int, int] = (1980, 2014)
) -> None:
    """
    Generates tables to assess biases and variability for models compared to ERA5.

    Args:
        place (str): The place for which to generate the assessment.
        pi_version (int): The potential intensity version to use.
        comparison_period (Tuple[int, int]): The (start, end) year for the comparison.
    """
    print(f"\n--- Generating Bias and Variability Assessment for {place} ---")
    results = []
    start_yr, end_yr = str(comparison_period[0]), str(comparison_period[1])

    try:
        era5_ds = get_timeseries(model="ERA5", place=place, pi_version=pi_version)
        era5_ds_period = era5_ds.sel(time=slice(start_yr, end_yr))
    except Exception as e:
        print(f"Could not load ERA5 data. Skipping assessment. Error: {e}")
        return

    for model, members in MODELS_TO_PROCESS.items():
        for member_str in members:
            try:
                model_ds = get_timeseries(model=model, place=place, member=member_str, pi_version=pi_version)
                model_ds_period = model_ds.sel(time=slice(start_yr, end_yr))

                common_time = np.intersect1d(era5_ds_period.time, model_ds_period.time)
                if len(common_time) < 5:
                    print(f"Not enough overlapping data for {model}/{member_str}. Skipping.")
                    continue

                era5_aligned = era5_ds_period.sel(time=common_time)
                model_aligned = model_ds_period.sel(time=common_time)

                for var in VARIABLES_TO_PROCESS:
                    if var not in model_aligned or var not in era5_aligned: continue

                    era5_data = era5_aligned[var].values
                    model_data = model_aligned[var].values

                    valid_mask = ~np.isnan(era5_data) & ~np.isnan(model_data)
                    if np.sum(valid_mask) < 5: continue

                    bias_res = analyze_bias(era5_data[valid_mask], model_data[valid_mask])
                    _, _, model_cv = calculate_detrended_cv(model_data[valid_mask])
                    _, _, era5_cv = calculate_detrended_cv(era5_data[valid_mask])

                    results.append({
                        "model": model, "member": member_str, "variable": var,
                        "mean_bias": bias_res['mean_bias'],
                        "mean_bias_p": bias_res['mean_bias_p_value'],
                        "bias_trend": bias_res['bias_trend'],
                        "bias_trend_p": bias_res['bias_trend_p_value'],
                        "model_cv": model_cv, "era5_cv": era5_cv,
                    })
            except Exception as e:
                print(f"Could not assess {model}/{member_str}. Skipping. Error: {e}")

    if not results:
        print("No assessment results generated.")
        return

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_PATH, f"{place}_assessment_pi{pi_version}.csv"), index=False)
    print("Saved assessment data to CSV.")

    # Format for LaTeX table
    def p_to_star(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    df['mean_bias'] = df.apply(lambda r: f"{r.mean_bias:.2f}{p_to_star(r.mean_bias_p)}", axis=1)
    df['bias_trend'] = df.apply(lambda r: f"{r.bias_trend:.3f}{p_to_star(r.bias_trend_p)}", axis=1)

    era5_cv_df = df[['variable', 'era5_cv']].drop_duplicates().reset_index(drop=True)
    print("\nERA5 Detrended Coefficient of Variation (1980-2014):")
    print(era5_cv_df)

    df_tex = df[['model', 'member', 'variable', 'mean_bias', 'bias_trend', 'model_cv']]
    df_str_tex = dataframe_to_latex_table(df_tex)
    with open(os.path.join(DATA_PATH, f"{place}_assessment_pi{pi_version}.tex"), "w") as f:
        f.write(df_str_tex)
    print("Saved assessment data to LaTeX table.")


def format_single_latex_sci(value: float, sig_figs: int = 2) -> str:
    # ... (function unchanged)
    assert isinstance(sig_figs, int) and sig_figs >= 1, "sig_figs must be a positive integer"
    assert isinstance(value, (int, float)), "value must be a number"
    if value == 0:
        return f"\\({0.0:.{sig_figs-1}f}\\)"
    if not math.isfinite(value):
        return "\\(\\infty\\)" if value > 0 else "\\(-\\infty\\)"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    mantissa = round(mantissa, sig_figs - 1)
    if abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    mantissa_str = f"{{:.{sig_figs-1}f}}".format(mantissa)
    if exponent == 0:
        return f"\\({mantissa_str}\\)"
    return f"\\({mantissa_str} \\times 10^{{{exponent}}}\\)"

def format_error_latex_sci(nominal: float, error: float) -> str:
    # ... (function unchanged)
    assert isinstance(nominal, (int, float)), "nominal must be a number"
    assert isinstance(error, (int, float)), "error must be a number"
    if error == 0: return format_single_latex_sci(nominal)
    if not math.isfinite(error) or not math.isfinite(nominal): return "--"
    exponent = math.floor(math.log10(abs(nominal))) if nominal != 0 else math.floor(math.log10(abs(error)))
    nominal_rescaled = nominal / (10**exponent)
    error_rescaled = error / (10**exponent)
    rounding_decimals = -math.floor(math.log10(abs(error_rescaled))) if error_rescaled != 0 else 1
    if rounding_decimals < 0: rounding_decimals = 1
    nominal_rounded = round(nominal_rescaled, rounding_decimals)
    error_rounded = round(error_rescaled, rounding_decimals)
    fmt_str = f"{{:.{rounding_decimals}f}}"
    nominal_str, error_str = fmt_str.format(nominal_rounded), fmt_str.format(error_rounded)
    if exponent == 0:
        return f"\\({nominal_str} \\pm {error_str}\\)"
    return f"\\(\\left({nominal_str} \\pm {error_str}\\right)\\times 10^{{{exponent}}}\\)"


def dataframe_to_latex_table(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a publication-quality LaTeX table string.
    """
    df_proc = df.copy()

    def _generate_header_map(columns: list[str]) -> Dict[str, str]:
        symbol_map = {
            "vmax_3": "V_p", "rmax_3": "r_3", "rmax_1": "r_1", "r0_1": "r_{a1}",
            "r0_3": "r_{a3}", "sst": "T_s", "years": "t",
        }
        unit_map = {
            "vmax_3": r"\text{m s}^{-1}", "r0_3": r"\text{km}", "r0_1": r"\text{km}",
            "rmax_3": r"\text{km}", "rmax_1": r"\text{km}", "sst": r"^{\circ}\text{C}",
            "years": r"\text{yr}",
        }
        header_map = {}
        for col in columns:
            if col.startswith("rho_"):
                parts = col.split('_')[1:]
                dep, indep = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], 'years')
                header_map[col] = f"\\(\\rho({symbol_map.get(dep, dep)}, {symbol_map.get(indep, indep)})\\)"
            elif col.startswith("fit_"):
                parts = col.split('_')[1:]
                dep, indep = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], 'years')
                dep_sym, ind_sym = symbol_map.get(dep, dep), symbol_map.get(indep, indep)
                dep_unit, ind_unit = unit_map.get(dep), unit_map.get(indep)
                unit_str = f" [\\({dep_unit}\\;{ind_unit}^{{-1}}\\)]" if dep_unit and ind_unit else ""
                header_map[col] = f"\\(m({ind_sym}, {dep_sym})\\){unit_str}"

        # Static headers
        header_map.update({
            "member": "Member", "place": "Place", "year_min": "Start", "year_max": "End",
            "model": "Model", "variable": "Variable", "mean_bias": "Mean Bias",
            "bias_trend": "Bias Trend", "model_cv": "Model CV",
        })
        return header_map

    # TODO: we need to get the calculate the bias over the three model members for each model,
    # we need to have the columns as the variables and the rows as the models with proper latex names e.g. \(V_p\)
    # and we need to seperate different properties into different tables

    header_map = _generate_header_map(list(df_proc.columns))
    err_cols_to_drop = []

    for col in df_proc.columns:
        err_col = f"{col}_err"
        if err_col in df_proc.columns:
            df_proc[col] = df_proc.apply(
                lambda r: format_error_latex_sci(r[col], r[err_col]) if pd.notnull(r[col]) and pd.notnull(r[err_col]) else "", axis=1)
            err_cols_to_drop.append(err_col)
        elif pd.api.types.is_numeric_dtype(df_proc[col]):
             df_proc[col] = df_proc[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "")

    df_proc.drop(columns=err_cols_to_drop, inplace=True)
    final_order = [col for col in df.columns if not col.endswith("_err")]
    df_proc.rename(columns=header_map, inplace=True)
    df_proc = df_proc[[header_map.get(c, c) for c in final_order]]

    col_format = "l" * len(df_proc.columns)
    return df_proc.to_latex(index=False, escape=False, header=True, column_format=col_format, caption=" ")


def wide_dataframe_to_latex(
    df: pd.DataFrame, caption: str, label: str, filename: str
) -> None:
    """
    Converts a wide-format pandas DataFrame to a publication-quality LaTeX table.

    This function handles DataFrames with multi-level columns (metric, variable)
    and formats them into a clean LaTeX table with grouped headers.

    Args:
        df (pd.DataFrame): The wide-format DataFrame to convert.
        caption (str): The LaTeX table caption.
        label (str): The LaTeX label for cross-referencing.
        filename (str): The full path to save the .tex file.
    """
    # Symbol and name mapping for variables and metrics
    symbol_map = {
        "vmax_3": "$V_p$", "rmax_3": "$r_3$", "rmax_1": "$r_1$",
        "r0_1": "$r_{a1}$", "r0_3": "$r_{a3}$", "sst": "$T_s$",
    }
    metric_map = {
        "mean_bias": "Mean Bias", "rmse": "RMSE",
        "variability_ratio": "Variability Ratio",
        "corr_variability": "Variability Corr. ($\\rho$)",
    }

    # Create a copy to avoid modifying the original DataFrame
    df_latex = df.copy()

    # Create new multi-level column headers from the existing ones
    new_cols = []
    current_cols = df_latex.columns.to_flat_index()
    for metric, var in current_cols:
        metric_name = metric_map.get(metric, metric)
        var_symbol = symbol_map.get(var, var)
        new_cols.append((var_symbol, metric_name)) # Group by variable symbol first
    df_latex.columns = pd.MultiIndex.from_tuples(new_cols)

    # Convert to LaTeX string
    latex_string = df_latex.to_latex(
        escape=False,
        column_format="l" + "c" * len(df_latex.columns),
        multicolumn_format="c",
        caption=caption,
        label=label,
        position="!htbp"
    )

    # Save to file
    with open(filename, "w") as f:
        f.write(latex_string)
    print(f"Saved LaTeX table to {filename}")


def generate_wide_assessment_tables(
    place: str = "new_orleans",
    pi_version: int = 4,
    comparison_period: Tuple[int, int] = (1980, 2024)
) -> None:
    """
    Generates wide-format assessment tables with aggregated model statistics.

    This function calculates bias, RMSE, and variability metrics for CMIP models
    against ERA5. It then aggregates results across ensemble members for each
    model and pivots the data to create publication-ready 'wide' format tables
    for mean state assessment and variability assessment.

    Args:
        place (str): The place for which to generate the assessment.
        pi_version (int): The potential intensity version to use.
        comparison_period (Tuple[int, int]): The (start, end) year for comparison.
    """
    print(f"\n--- Generating Wide Assessment Tables for {place} ---")
    results = []
    start_yr, end_yr = str(comparison_period[0]), str(comparison_period[1])

    try:
        era5_ds = get_timeseries(model="ERA5", place=place, pi_version=pi_version)
        era5_ds_period = era5_ds.sel(time=slice(start_yr, end_yr))
    except Exception as e:
        print(f"Could not load ERA5 data. Skipping assessment. Error: {e}")
        return

    for model, members in MODELS_TO_PROCESS.items():
        for member_str in members:
            try:
                model_ds = get_timeseries(model=model, place=place, member=member_str, pi_version=pi_version)
                model_ds_period = model_ds.sel(time=slice(start_yr, end_yr))

                # Align timestamps
                common_time = np.intersect1d(era5_ds_period.time, model_ds_period.time)
                if len(common_time) < 5:
                    continue
                era5_aligned = era5_ds_period.sel(time=common_time)
                model_aligned = model_ds_period.sel(time=common_time)

                for var in VARIABLES_TO_PROCESS:
                    if var not in model_aligned or var not in era5_aligned: continue

                    era5_data = era5_aligned[var].values
                    model_data = model_aligned[var].values
                    if "r" in var:
                        era5_data = era5_data / 1000.0  # Convert from m to km
                        model_data = model_data / 1000.0
                    valid_mask = ~np.isnan(era5_data) & ~np.isnan(model_data)
                    if np.sum(valid_mask) < 5: continue

                    era5_valid, model_valid = era5_data[valid_mask], model_data[valid_mask]
                    bias_res = analyze_bias(era5_valid, model_valid)
                    model_residuals, _, model_cv = calculate_detrended_cv(model_valid)
                    era5_residuals, _, era5_cv = calculate_detrended_cv(era5_valid)
                    rmse = np.sqrt(np.mean((model_valid - era5_valid)**2))
                    corr_variability = safe_corr(model_residuals, era5_residuals)

                    results.append({
                        "model": model, "member": member_str, "variable": var,
                        "mean_bias": bias_res['mean_bias'], "rmse": rmse,
                        "variability_ratio": model_cv / era5_cv if era5_cv != 0 else np.nan,
                        "model_cv": model_cv,
                        "corr_variability": corr_variability,
                    })
            except Exception as e:
                print(f"Could not assess {model}/{member_str}. Skipping. Error: {e}")

    if not results:
        print("No assessment results generated.")
        return

    df_long = pd.DataFrame(results)
    agg_funcs = {
        'mean_bias': ['mean', 'std'], 'rmse': ['mean', 'std'],
        'variability_ratio': ['mean', 'std'], 'corr_variability': ['mean', 'std'],
        'model_cv': ['mean', 'std'],
    }
    df_agg = df_long.groupby(['model', 'variable']).agg(agg_funcs).reset_index()
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

    for metric in agg_funcs.keys():
        mean_col, std_col = f'{metric}_mean', f'{metric}_std'
        df_agg[metric] = df_agg.apply(
            lambda r: f"${r[mean_col]:.2f} \\pm {r[std_col]:.2f}$"
            if pd.notnull(r[mean_col]) else "NaN", axis=1
        )
        df_agg.drop(columns=[mean_col, std_col], inplace=True)

    # --- Create and Save Bias/RMSE Table ---
    df_bias_rmse = df_agg.pivot(index='model', columns='variable', values=['mean_bias']) # 'rmse'
    df_bias_rmse.index.name = "Model"
    csv_path_br = os.path.join(DATA_PATH, f"{place}_assessment_bias_rmse_pi{pi_version}.csv")
    tex_path_br = os.path.join(DATA_PATH, f"{place}_assessment_bias_rmse_pi{pi_version}.tex")
    df_bias_rmse.to_csv(csv_path_br)
    print(f"\nSaved Bias and RMSE data to {csv_path_br}")
    wide_dataframe_to_latex(
        df=df_bias_rmse,
        caption=f"Model Mean State Assessment for {place.replace('_', ' ').title()} (1980-2014)",
        label=f"tab:{place}_bias_rmse",
        filename=tex_path_br
    )

    # --- Create and Save Variability Table ---
    df_variability = df_agg.pivot(index='model', columns='variable', values=['model_cv']) # 'variability_ratio', 'corr_variability'
    df_variability.index.name = "Model"
    csv_path_var = os.path.join(DATA_PATH, f"{place}_assessment_variability_pi{pi_version}.csv")
    tex_path_var = os.path.join(DATA_PATH, f"{place}_assessment_variability_pi{pi_version}.tex")
    df_variability.to_csv(csv_path_var)
    print(f"\nSaved Variability data to {csv_path_var}")
    wide_dataframe_to_latex(
        df=df_variability,
        caption=f"Model Variability Assessment for {place.replace('_', ' ').title()} (1980-2014)",
        label=f"tab:{place}_variability",
        filename=tex_path_var
    )


import statsmodels.api as sm
from tcpips.era5 import trend_with_neweywest_full

def analyze_bias_newey_west_ensemble(
    model_aligned_ds: xr.Dataset, era5_aligned_ds: xr.Dataset, var: str
) -> dict:
    """
    Analyzes bias using OLS with Newey-West standard errors on the ensemble mean.

    This function first calculates the bias for a given variable, then computes the
    mean across the ensemble members, and finally calculates the trend of this
    ensemble-mean bias time series.

    Args:
        model_aligned_ds (xr.Dataset): Dataset with multiple members, aligned to ERA5.
        era5_aligned_ds (xr.Dataset): ERA5 dataset, aligned to the model.
        var (str): The variable to analyze.

    Returns:
        A dictionary containing the mean bias, bias trend, and the trend's p-value.
    """
    # Calculate bias against ERA5 (retains the 'member' dimension)
    bias_ds = model_aligned_ds[var] - era5_aligned_ds[var]

    # Calculate the mean across the ensemble members to get a single time series
    ensemble_mean_bias = bias_ds.mean(dim="member").values

    # Analyze this single time series for its trend
    bias_trend, bias_intercept, bias_trend_p = trend_with_neweywest_full(ensemble_mean_bias)

    return {
        'mean_bias': np.nanmean(ensemble_mean_bias),
        'bias_trend': bias_trend,
        'bias_trend_p_value': bias_trend_p,
    }


def generate_newey_west_assessment_tables(
    place: str = "new_orleans",
    pi_version: int = 4,
    comparison_period: Tuple[int, int] = (1980, 2024)
) -> None:
    """
    Generates assessment tables using an ensemble-mean Newey-West approach.
    """
    print(f"\n--- Generating Newey-West Ensemble Assessment for {place} ---")
    results = []
    start_yr, end_yr = str(comparison_period[0]), str(comparison_period[1])

    try:
        era5_ds = get_timeseries(model="ERA5", place=place, pi_version=pi_version)
        era5_ds_period = era5_ds.sel(time=slice(start_yr, end_yr))
    except Exception as e:
        print(f"Could not load ERA5 data. Skipping assessment. Error: {e}")
        return

    for model, members in MODELS_TO_PROCESS.items():
        model_data_all_members = []
        for member_str in members:
            try:
                model_ds = get_timeseries(model=model, place=place, member=member_str, pi_version=pi_version)
                model_data_all_members.append(model_ds)
            except Exception as e:
                print(f"Could not load data for {model}/{member_str}. Skipping member.")

        if not model_data_all_members: continue

        full_model_ds = xr.concat(model_data_all_members, dim=pd.Index(members, name="member"))
        full_model_ds_period = full_model_ds.sel(time=slice(start_yr, end_yr))

        common_time = np.intersect1d(era5_ds_period.time, full_model_ds_period.time)
        if len(common_time) < 5: continue
        era5_aligned = era5_ds_period.sel(time=common_time)
        model_aligned = full_model_ds_period.sel(time=common_time)

        for var in VARIABLES_TO_PROCESS:
            if var not in model_aligned or var not in era5_aligned: continue
            if "r" in var:
                era5_aligned[var] = era5_aligned[var] / 1000.0  # Convert from m to km
                model_aligned[var] = model_aligned[var] / 1000.0

            # --- Analyze with the Newey-West ensemble function ---
            analysis_res = analyze_bias_newey_west_ensemble(model_aligned, era5_aligned, var)

            # (CV and RMSE would still be calculated per-member then averaged/ranged)

            # Helper for significance stars
            def p_to_star(p):
                return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

            results.append({
                "model": model, "variable": var,
                "mean_bias": f"${analysis_res['mean_bias']:.2f}$",
                "bias_trend": f"${analysis_res['bias_trend']:.3f}${p_to_star(analysis_res['bias_trend_p_value'])}",
            })

    if not results:
        print("No assessment results generated.")
        return

    df = pd.DataFrame(results)
    df_pivot = df.pivot(index='model', columns='variable', values=['mean_bias', 'bias_trend'])
    df_pivot.index.name = "Model"

    tex_path = os.path.join(DATA_PATH, f"{place}_assessment_neweywest_pi{pi_version}.tex")

    wide_dataframe_to_latex(
        df=df_pivot,
        caption=f"Model Bias Assessment for {place.replace('_', ' ').title()} (1980-2024). Trends computed on the ensemble mean with Newey-West standard errors.",
        label=f"tab:{place}_bias_neweywest",
        filename=tex_path
    )


def analyze_bias_mixed_model(model_df: pd.DataFrame) -> dict:
    """
    Analyzes bias using a linear mixed-effects model.

    This treats 'member' as a random effect to account for non-independence,
    providing a robust estimate of the overall mean bias (fixed intercept) and
    the trend in bias (fixed slope for time).

    Args:
        model_df: DataFrame with columns ['time', 'bias', 'member'].

    Returns:
        A dictionary with the mean bias and trend, their standard errors,
        p-values, and the variance of the inter-member variability.
    """
    # Ensure time is a numerical predictor (e.g., years from start)
    model_df['time_idx'] = model_df['time']  - model_df['time'].min()

    # Fit the mixed-effects model.
    # Fixed effects: Intercept (mean bias) + time_idx (bias trend)
    # Random effect: A random intercept for each member
    md = smf.mixedlm("bias ~ 1 + time_idx", model_df, groups=model_df["member"])
    mdf = md.fit(reml=False) # Use ML for better p-values

    # Extract results
    fixed_effects = mdf.params
    p_values = mdf.pvalues
    std_errs = mdf.bse
    random_effects_var = mdf.cov_re.iloc[0, 0] # Variance of the random intercept

    return {
        'mean_bias': fixed_effects['Intercept'],
        'mean_bias_se': std_errs['Intercept'],
        'mean_bias_p': p_values['Intercept'],
        'bias_trend': fixed_effects['time_idx'],
        'bias_trend_se': std_errs['time_idx'],
        'bias_trend_p': p_values['time_idx'],
        'inter_member_variance': random_effects_var,
    }


def generate_mixed_model_assessment_tables(
    place: str = "new_orleans",
    pi_version: int = 4,
    comparison_period: Tuple[int, int] = (1980, 2024)
) -> None:
    """
    Generates assessment tables using a robust mixed-effects model approach.
    """
    print(f"\n--- Generating Mixed-Effects Model Assessment for {place} ---")
    results = []
    start_yr, end_yr = str(comparison_period[0]), str(comparison_period[1])

    try:
        era5_ds = get_timeseries(model="ERA5", place=place, pi_version=pi_version)
        era5_ds_period = era5_ds.sel(time=slice(start_yr, end_yr))
    except Exception as e:
        print(f"Could not load ERA5 data. Skipping assessment. Error: {e}")
        return

    for model, members in MODELS_TO_PROCESS.items():
        # --- Collect data from all members of a model ---
        model_data_all_members = []
        for member_str in members:
            try:
                model_ds = get_timeseries(model=model, place=place, member=member_str, pi_version=pi_version)
                model_data_all_members.append(model_ds)
            except Exception as e:
                print(f"Could not load data for {model}/{member_str}. Skipping member.")

        if not model_data_all_members: continue

        # Concatenate datasets from all members
        full_model_ds = xr.concat(model_data_all_members, dim=pd.Index(members, name="member"))
        full_model_ds_period = full_model_ds.sel(time=slice(start_yr, end_yr))

        # Align with ERA5
        common_time = np.intersect1d(era5_ds_period.time, full_model_ds_period.time)
        if len(common_time) < 5: continue
        era5_aligned = era5_ds_period.sel(time=common_time)
        model_aligned = full_model_ds_period.sel(time=common_time)

        for var in VARIABLES_TO_PROCESS:
            if var not in model_aligned or var not in era5_aligned: continue


            # Create a DataFrame suitable for the mixed model
            bias_df = (model_aligned[var] - era5_aligned[var]).to_dataframe(name='bias').reset_index()
            if "r" in var:
                bias_df['bias'] = bias_df['bias'] / 1000.0  # Convert from m to km
            bias_df.dropna(inplace=True)
            if bias_df.shape[0] < 10: continue

            # --- Analyze with the new function ---
            bias_res = analyze_bias_mixed_model(bias_df)

            # (Note: CV calculation would still be done per-member then averaged, as it's a property of each timeseries)

            results.append({
                "model": model, "variable": var,
                "mean_bias": f"${bias_res['mean_bias']:.2f} \\pm {bias_res['mean_bias_se']:.2f}$",
                "bias_trend": f"${bias_res['bias_trend']:.3f} \\pm {bias_res['bias_trend_se']:.3f}$",
                # You could also add p-value stars if desired
            })

    if not results:
        print("No assessment results generated.")
        return

    df = pd.DataFrame(results)

    # --- Pivot and Save Table ---
    df_pivot = df.pivot(index='model', columns='variable', values=['mean_bias', 'bias_trend'])
    df_pivot.index.name = "Model"

    tex_path = os.path.join(DATA_PATH, f"{place}_assessment_mixed_model_pi{pi_version}.tex")

    # A simplified wide_dataframe_to_latex call
    wide_dataframe_to_latex(
        df=df_pivot,
        caption=f"Model Bias Assessment for {place.replace('_', ' ').title()} (1980-2024). Values are fixed effects $\\pm$ standard error from a mixed-effects model.",
        label=f"tab:{place}_bias_mixed_model",
        filename=tex_path
    )


if __name__ == "__main__":
    # python -m w22.stats2
    # Example usage for one location
    #LOCATION = "new_orleans"
    LOCATION = "hong_kong"
    PI_VERSION = 4

    # Generate tables for temporal trends and correlations
    #temporal_relationship_data(place=LOCATION, pi_version=PI_VERSION)

    # Generate tables for bias and variability assessment
    # generate_assessment_tables(place=LOCATION, pi_version=PI_VERSION)
    #generate_wide_assessment_tables(place=LOCATION, pi_version=PI_VERSION)
    # generate_wide_assessment_tables(place="new_orleans", pi_version=PI_VERSION)

    generate_mixed_model_assessment_tables(place="hong_kong", pi_version=PI_VERSION)
    generate_mixed_model_assessment_tables(place="new_orleans", pi_version=PI_VERSION)
    generate_newey_west_assessment_tables(place="hong_kong", pi_version=PI_VERSION)
    generate_newey_west_assessment_tables(place="new_orleans", pi_version=PI_VERSION)
