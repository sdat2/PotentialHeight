"""Generate tables of results."""
import os
import numpy as np
import numpy.ma as ma
from uncertainties import ufloat
import math
import xarray as xr
import pandas as pd
from sithom.curve import fit
from .plot import get_timeseries
from .constants import DATA_PATH


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
    xt, yt = xt[~np.isnan(xt)], yt[~np.isnan(xt)]
    xt, yt = xt[~np.isnan(yt)], yt[~np.isnan(yt)]
    # normalize the data between 0 and 10
    xrange = np.max(xt) - np.min(xt)
    yrange = np.max(yt) - np.min(yt)
    xt = (xt - np.min(xt)) / xrange * 10
    yt = (yt - np.min(yt)) / yrange * 10
    # fit the data with linear fit using OLS
    param, _ = fit(xt, yt)  # defaults to y=mx+c fit
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
    corr = ma.corrcoef(ma.masked_invalid(xt), ma.masked_invalid(yt))
    return corr[0, 1]



def timeseries_relationships(
    timeseries_ds: xr.Dataset,
    place: str,
    member: int,
    year_min: int = 2014,
    year_max: int = 2100,
) -> pd.DataFrame:
    """
    Timeseries relationships function.

    Args:
        timeseries_ds (xr.Dataset): _description_
        place (str): place for description.
        member (int): _description_
        year_min (int, optional): _description_. Defaults to 2014.
        year_max (int, optional): _description_. Defaults to 2100.

    Returns:
        pd.DataFrame: _description_
    """
    vars = ("sst", "vmax_3", "r0_3", "rmax_3")

    var_timeseries = {var: timeseries_ds[var].sel(time=slice(year_min, year_max)).values for var in vars}
    print(var_timeseries)

    ssts = timeseries_ds["sst"].sel(time=slice(year_min, year_max)).values
    vmaxs = timeseries_ds["vmax_3"].sel(time=slice(year_min, year_max)).values
    r0s_3 = timeseries_ds["r0_3"].sel(time=slice(year_min, year_max)).values
    years = np.array(
        [
            time
            for time in timeseries_ds["time"].sel(time=slice(year_min, year_max)).values
        ]
    )
    rmaxs = timeseries_ds["rmax_3"].sel(time=slice(year_min, year_max)).values

    rho_vmax = safe_corr(vmaxs, years)
    rho_r0 = safe_corr(r0s_3, years)
    rho_rmax = safe_corr(rmaxs, years)
    rho_sst = safe_corr(ssts, years)
    rho_sst_vmax = safe_corr(ssts, vmaxs)
    rho_sst_r0 = safe_corr(ssts, r0s_3)
    rho_sst_rmax = safe_corr(ssts, rmaxs)


    # work out gradient with error bars for same period
    fit_vmax = safe_grad(years, vmaxs)
    fit_r0 = safe_grad(years, r0s_3 / 1000)
    fit_rmax = safe_grad(years, rmaxs / 1000)
    fit_r0_sst = safe_grad(ssts, r0s_3 / 1000)
    fit_rmax_sst = safe_grad(ssts, rmaxs / 1000)

    # print("fit_r0_sst timeseries", fit_r0_sst, "km $^{\circ}$C$^{-1}$")
    fit_vmax_sst = safe_grad(ssts, vmaxs)
    # print("fit_vmax_sst timeseries", fit_vmax_sst, "m s$^{-1}$C$ ^{-1}$")
    # print("fit_vmax_years timeseries", fit_vmax, "m s$^{-1}$ yr$^{-1}$")
    # print("fit_r0_years timeseries", fit_r0, "km yr$^{-1}$")
    df = pd.DataFrame(
        {
            "place": [place],
            "member": [member],
            "year_min": [year_min],
            "year_max": [year_max],
            "rho_vmax": [rho_vmax],
            "rho_r0": [rho_r0],
            "rho_rmax": [rho_rmax],
            "rho_sst_rmax": [rho_sst_rmax],
            "rho_sst": [rho_sst],
            "rho_sst_vmax": [rho_sst_vmax],
            "rho_sst_r0": [rho_sst_r0],
            "fit_vmax": [fit_vmax.n],
            "fit_vmax_err": [fit_vmax.s],
            "fit_r0": [fit_r0.n],
            "fit_r0_err": [fit_r0.s],
            "fit_rmax" : [fit_rmax.n],
            "fit_rmax_err" : [fit_rmax.s],
            "fit_r0_sst": [fit_r0_sst.n],
            "fit_r0_sst_err": [fit_r0_sst.s],
            "fit_rmax_sst": [fit_rmax_sst.n],
            "fit_rmax_sst_err": [fit_rmax_sst.s],
            "fit_vmax_sst": [fit_vmax_sst.n],
            "fit_vmax_sst_err": [fit_vmax_sst.s],
        }
    )
    return df


def temporal_relationship_data(place: str = "new_orleans", pi_version: int = 4) -> None:
    """Get the temporal relationships data for the given place and potential intensity version.

    Args:
        place (str): The place to get the data for (default is "new_orleans").
        pi_version (int): The potential intensity version to use (default is 4).
    """
    df_l = []
    for member in [4, 10, 11]:
        timeseries_ds = get_timeseries(
            model="CESM2",
            place=place, member=member, pi_version=pi_version
        )
        df_l.append(
            timeseries_relationships(
                timeseries_ds,
                place=place,
                member="r" + str(member) + "i1p1f1",
                year_min=2014,
                year_max=2100,
            )
        )
        df_l.append(
            timeseries_relationships(
                timeseries_ds,
                place=place,
                member="r" + str(member) + "i1p1f1",
                year_min=1980,
                year_max=2024,
            )
        )
    timeseries_ds = get_timeseries(place=place, model="ERA5", pi_version=pi_version)
    df_l.append(
        timeseries_relationships(
            timeseries_ds,
            place=place,
            member="era5",
            year_min=1980,
            year_max=2024,
        )
    )
    df_l.append(
        timeseries_relationships(
            timeseries_ds,
            place=place,
            member="era5",
            year_min=1940,
            year_max=2024,
        )
    )
    df = pd.concat(df_l, ignore_index=True)
    file_name = os.path.join(
            DATA_PATH, f"{place}_temporal_relationships_pi{pi_version}new.csv"
        )

    df.to_csv(
        file_name,
        index=False,
    )
    print("Saved temporal relationships data to CSV.")
    df.drop(columns=["place"], inplace=True)

    # data frame was getting too large for latex table, so splitting into correlations and fits
    df_str = dataframe_to_latex_table(
        df[[col for col in df.columns if not col.startswith("fit_")]]
    )
    print(df_str)
    file_name = os.path.join(DATA_PATH, f"{place}_temporal_correlation_CESM2_pi{pi_version}.tex")
    with open(
        file_name,
        "w",
    ) as f:
        f.write(df_str)
    print("Saved temporal relationships data to LaTeX table.")

    df_str = dataframe_to_latex_table(
        df[[col for col in df.columns if not col.startswith("rho_")]]
    )
    print(df_str)
    file_name = os.path.join(DATA_PATH, f"{place}_temporal_fit_CESM2_pi{pi_version}.tex")
    with open(
        file_name, "w"
    ) as f:
        f.write(df_str)
    print("Saved temporal relationships data to LaTeX table.")
    return None


def format_single_latex_sci(value: float, sig_figs: int = 2) -> str:
    """Formats a single number as m x 10^e.

    Args:
        value (float): The value to format.
        sig_figs (int): The number of significant figures to use (default is 2
            which gives 1 decimal place for the mantissa).

    Returns:
        str: The formatted string in LaTeX scientific notation.

    Doctest:
        >>> print(format_single_latex_sci(12345))
        \(1.2 \\times 10^{4}\)
        >>> print(format_single_latex_sci(0.0012345))
        \(1.2 \\times 10^{-3}\)
        >>> print(format_single_latex_sci(0))
        \(0.0\)
        >>> print(format_single_latex_sci(float('inf')))
        \(\infty\)
    """
    assert isinstance(sig_figs, int) and sig_figs >= 1, "sig_figs must be a positive integer"
    assert isinstance(value, (int, float)), "value must be a number"
    if value == 0:
        return f"\\({0.0:.{sig_figs-1}f}\\)"
    if not math.isfinite(value):
        return "\\(\\infty\\)" if value > 0 else "\\(-\\infty\\)"

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    # Round mantissa to specified significant figures
    mantissa = round(mantissa, sig_figs - 1)

    # Correct for rounding rollovers (e.g., 9.99 -> 10.0)
    if abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1

    mantissa_str = f"{{:.{sig_figs-1}f}}".format(mantissa)

    if exponent == 0:
        return f"\\({mantissa_str}\\)"
    return f"\\({mantissa_str} \\times 10^{{{exponent}}}\\)"


def format_error_latex_sci(nominal: float, error: float) -> str:
    """Formats a nominal ± error pair with a common exponent.

    Args:
        nominal (float): The nominal value.
        error (float): The error value.

    Returns:
        str: The formatted string in LaTeX scientific notation.

    Doctest:
        >>> print(format_error_latex_sci(12345, 67))
        \(\left(1.234 \pm 0.007\\right)\\times 10^{4}\)
        >>> print(format_error_latex_sci(0.0012345, float('inf')))
        \(1.234 \pm \infty\)
    """

    assert isinstance(nominal, (int, float)), "nominal must be a number"
    assert isinstance(error, (int, float)), "error must be a number"

    if error == 0 or not math.isfinite(error) or not math.isfinite(nominal):
        return format_single_latex_sci(nominal) + " \\pm 0"

    # Determine the common exponent from the nominal value
    exponent = (
            math.floor(math.log10(abs(nominal)))
            if nominal != 0
            else math.floor(math.log10(abs(error)))
        )

    # Rescale numbers to the common exponent
    nominal_rescaled = nominal / (10**exponent)
    error_rescaled = error / (10**exponent)

    # Determine decimal places for rounding from the error's first significant digit
    if error_rescaled == 0:
        rounding_decimals = 1
    else:
        rounding_decimals = -math.floor(math.log10(abs(error_rescaled)))

    if rounding_decimals < 0:
        rounding_decimals = 1

    # Round the rescaled numbers to the determined decimal place
    nominal_rounded = round(nominal_rescaled, rounding_decimals)
    error_rounded = round(error_rescaled, rounding_decimals)

    fmt_str = f"{{:.{rounding_decimals}f}}"
    # error here?
    nominal_str = fmt_str.format(nominal_rounded)
    error_str = fmt_str.format(error_rounded)
    if exponent == 0:
        return f"\\({nominal_str} \\pm {error_str}\\)"

    return f"\\(\\left({nominal_str} \\pm {error_str}\\right)\\times 10^{{{exponent}}}\\)"


def dataframe_to_latex_table(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a publication-quality LaTeX table string.

    This function automatically generates formal LaTeX headers and formats
    numerical values into a human-readable scientific notation suitable for
    academic papers.

    Key features:
    - **Dynamic Header Generation**: Parses column names like 'rho_vmax' or
      'fit_r0_sst' into LaTeX expressions, e.g., '\\(\\rho(V_p)\\)'.
    - **Advanced Number Formatting**:
        - Single numbers are formatted as \\(m \\times 10^{e}\\).
        - Value ± error pairs are formatted as \\((\\textit{m}_n \\pm \\textit{m}_e) \\times 10^{E}\\),
          where the exponent is factored out and values are rounded
          systematically based on the error's magnitude.
    - **Custom Table Style**: Uses '\\topline', '\\midline', '\\bottomline' for
      table rules.

    Args:
        df (pd.DataFrame): The input DataFrame. Column names are expected to
            follow conventions like 'rho_{...}' and 'fit_{...}'.

    Returns:
        str: A string containing the fully formatted LaTeX table.

    Doctest:
        >>> # Example DataFrame mimicking a typical analysis output
        >>> data = {
        ...     'place': ['Atlantic'],
        ...     'member': ['member_01'],
        ...     'rho_vmax': [0.891],
        ...     'fit_vmax': [43.2],
        ...     'fit_vmax_err': [5.73],
        ...     'fit_r0_sst': [-1.54],
        ...     'fit_r0_sst_err': [0.11],
        ... }
        >>> df_test = pd.DataFrame(data)

    """

    # --- Main Function Logic ---
    df_proc = df.copy()

    def _generate_header_map(columns: list[str]) -> dict[str, str]:
        symbol_map = {
            "vmax": "V_p",
            "r0": "r_a",
            "sst": "T_s",
            "t0": "T_0",
            "rmax": r"r_{\mathrm{max}}",
            "rh": r"\mathcal{H}_e",
            "p": "p_a",
            "msl": "p_a",
            "years": "t",
        }
        unit_map = {
            "vmax": r"\text{m s}^{-1}",
            "r0": "\\text{km}",
            "rmax": "\\text{km}",
            "sst": r"^{\circ}\text{C}",
            "years": "\\text{yr}",
        }
        header_map = {}
        for col in columns:
            if col.startswith("rho_"):
                parts = col.split("_")[1:]
                symbols = [symbol_map.get(p, p) for p in parts]
                if len(symbols) == 1:
                    symbols.append("t")
                header_map[col] = f"\\(\\rho({', '.join(symbols)})\\)"
            elif col.startswith("fit_"):
                parts = col.split("_")[1:]
                dep_var, ind_var = parts[0], parts[1] if len(parts) > 1 else "years"
                dep_sym, ind_sym = symbol_map.get(dep_var, dep_var), symbol_map.get(
                    ind_var, ind_var
                )
                dep_unit, ind_unit = unit_map.get(dep_var), unit_map.get(ind_var)
                unit_str = (
                    f" [\\({dep_unit} \;{ind_unit}^{{-1}}\\)]"
                    if dep_unit and ind_unit
                    else ""
                )
                header_map[col] = f"\\(m({ind_sym}, {dep_sym})\\){unit_str}"
        header_map["member"] = "Member"
        header_map["place"] = "Place"
        header_map["year_min"] = "Start"
        header_map["year_max"] = "End"
        return header_map

    header_map = _generate_header_map(list(df_proc.columns))
    err_cols_to_drop = []

    for col in df_proc.columns:
        err_col = f"{col}_err"
        if err_col in df_proc.columns:
            df_proc[col] = df_proc.apply(
                lambda row: (
                    format_error_latex_sci(row[col], row[err_col])
                    if pd.notnull(row[col]) and pd.notnull(row[err_col])
                    else ""
                ),
                axis=1,
            )
            err_cols_to_drop.append(err_col)
        elif col in header_map and not col in [
            "place",
            "member",
            "year_min",
            "year_max",
        ]:
            df_proc[col] = df_proc[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else ""
            )

    df_proc.drop(columns=err_cols_to_drop, inplace=True)

    final_order = [col for col in df.columns if not col.endswith("_err")]
    df_proc.rename(columns=header_map, inplace=True)
    final_renamed_order = [header_map.get(col, col) for col in final_order]
    df_proc = df_proc[final_renamed_order]

    col_format = "l" * len(df_proc.columns)
    latex_str = df_proc.to_latex(
        index=False, escape=False, header=True, column_format=col_format, caption=" "
    )

    return latex_str

if __name__ == "__main__":
    # python -m w22.stats
    temporal_relationship_data(place = "new_orleans")
