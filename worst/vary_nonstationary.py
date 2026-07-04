"""Synthetic non-stationary EVT experiment with a time-varying upper bound.

This module extends the existing bounded-vs-unbounded experiments by adding
non-stationarity in the upper bound over time. The aim is to show a general
statistical point: if the physical cap evolves, then a model that can ingest a
time-varying cap should outperform stationary fits for long return periods.
"""

from typing import List, Tuple
import os
import hydra
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from scipy.stats import genextreme
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
from sithom.plot import get_dim, label_subplots, plot_defaults
from .constants import CONFIG_PATH, DATA_PATH, FIGURE_PATH
from .utils import alpha_from_z_star_beta_gamma, legend_below
from .tens import fit_gev_upper_bound_not_known, fit_gev_upper_bound_known

FIT_NAMES = [
    "stationary_unbounded",
    "stationary_bounded",
    "nonstationary_unbounded",
    "nonstationary_bounded",
]

FIT_LABELS = {
    "stationary_unbounded": "II-s: Stationary unbounded",
    "stationary_bounded": "I-s: Stationary bounded",
    "nonstationary_unbounded": "II-ns: Non-stationary unbounded",
    "nonstationary_bounded": "I-ns: Non-stationary bounded",
}


def _fit_value(fit: DictConfig, key: str, default: float) -> float:
    """Read a fit setting with a default if missing."""
    if key in fit and fit[key] is not None:
        return float(fit[key])
    return float(default)


def _fit_steps(fit: DictConfig) -> int:
    """Read optimization step count from fit settings."""
    if "steps" in fit and fit.steps is not None:
        return int(fit.steps)
    if "maxiter" in fit and fit.maxiter is not None:
        return int(fit.maxiter)
    return 1000


def _adam_optimizer(learning_rate: float):
    """Return a Keras-compatible Adam optimizer."""
    legacy = getattr(tf.keras.optimizers, "legacy", None)
    if legacy is not None and hasattr(legacy, "Adam"):
        try:
            return legacy.Adam(learning_rate=learning_rate)
        except Exception:
            pass
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def fit_stationary_unbounded(
    data: np.ndarray,
    fit: DictConfig,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """Fit stationary unbounded GEV using the same TensorFlow routine as other scripts."""
    return fit_gev_upper_bound_not_known(
        data,
        opt_steps=_fit_steps(fit),
        lr=_fit_value(fit, "lr", 0.01),
        alpha_guess=_fit_value(fit, "alpha_guess", float(np.mean(data))),
        beta_guess=_fit_value(fit, "beta_guess", 1.0),
        gamma_guess=_fit_value(fit, "gamma_guess", -0.1),
        force_weibull=bool(fit.force_weibull) if "force_weibull" in fit else False,
        verbose=verbose,
    )


def fit_stationary_bounded(
    data: np.ndarray,
    z_star_assumed: float,
    fit: DictConfig,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """Fit stationary bounded GEV using the same TensorFlow routine as other scripts."""
    return fit_gev_upper_bound_known(
        data,
        z_star_assumed,
        opt_steps=_fit_steps(fit),
        lr=_fit_value(fit, "lr", 0.01),
        beta_guess=_fit_value(fit, "beta_guess", 1.0),
        gamma_guess=_fit_value(fit, "gamma_guess", -0.1),
        verbose=verbose,
    )


def _fit_method(fit: DictConfig) -> str:
    """Optimiser for the non-stationary fits: 'lbfgs' (default) or 'adam'."""
    if "method" in fit and fit.method is not None:
        return str(fit.method)
    return "lbfgs"


def _ns_lbfgs_unbounded(data, time_axis, fit, force_weibull):
    """L-BFGS MLE of (alpha0, alpha1, beta, gamma); loc(t)=alpha0+alpha1 t."""
    def nll(p):
        a0, a1, beta = p[0], p[1], np.exp(p[2])
        gamma = -np.exp(p[3]) if force_weibull else p[3]
        lp = genextreme.logpdf(data, c=-gamma, loc=a0 + a1 * time_axis, scale=beta)
        return 1e12 if not np.all(np.isfinite(lp)) else -float(np.sum(lp))
    a0 = _fit_value(fit, "alpha_guess", float(np.mean(data)))
    a1 = float(np.polyfit(time_axis, data, 1)[0]) if len(data) > 1 else 0.0
    bg = _fit_value(fit, "beta_guess", 1.0)
    gg = _fit_value(fit, "gamma_guess", -0.1)
    x0 = [a0, a1, float(np.log(max(bg, 1e-3))),
          float(np.log(max(-gg, 1e-3)) if force_weibull else gg)]
    r = minimize(nll, x0, method="L-BFGS-B")
    gamma = float(-np.exp(r.x[3]) if force_weibull else r.x[3])
    return float(r.x[0]), float(r.x[1]), float(np.exp(r.x[2])), gamma


def _ns_lbfgs_bounded(data, z_star_assumed_t, fit):
    """L-BFGS MLE of (beta, gamma) with known time-varying bound; loc(t)=z*(t)+beta/gamma."""
    def nll(p):
        beta, gamma = np.exp(p[0]), -np.exp(p[1])
        lp = genextreme.logpdf(data, c=-gamma, loc=z_star_assumed_t + beta / gamma, scale=beta)
        return 1e12 if not np.all(np.isfinite(lp)) else -float(np.sum(lp))
    bg = _fit_value(fit, "beta_guess", 1.0)
    gg = _fit_value(fit, "gamma_guess", -0.1)
    r = minimize(nll, [float(np.log(max(bg, 1e-3))), float(np.log(max(-gg, 1e-3)))],
                 method="L-BFGS-B")
    return float(np.exp(r.x[0])), float(-np.exp(r.x[1]))


def fit_nonstationary_unbounded(
    data: np.ndarray,
    time_axis: np.ndarray,
    fit: DictConfig,
    verbose: bool = False,
) -> Tuple[float, float, float, float]:
    """Fit non-stationary unbounded GEV with linear alpha(t); L-BFGS by default, Adam optional."""
    force_weibull = bool(fit.force_weibull) if "force_weibull" in fit else False
    if _fit_method(fit) == "lbfgs":
        return _ns_lbfgs_unbounded(data, time_axis, fit, force_weibull)
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(time_axis, dtype=tf.float32)

    alpha0_guess = _fit_value(fit, "alpha_guess", float(np.mean(data)))
    beta_guess = _fit_value(fit, "beta_guess", 1.0)
    gamma_guess = _fit_value(fit, "gamma_guess", -0.1)
    if len(data) > 1:
        slope_guess = float(np.polyfit(time_axis, data, 1)[0])
    else:
        slope_guess = 0.0

    alpha0 = tf.Variable(alpha0_guess, dtype=tf.float32)
    alpha1 = tf.Variable(slope_guess, dtype=tf.float32)
    beta = tf.Variable(beta_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg())

    force_weibull = bool(fit.force_weibull) if "force_weibull" in fit else False
    if force_weibull:
        neg_gamma = tf.Variable(-gamma_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg())
    else:
        neg_gamma = tf.Variable(-gamma_guess, dtype=tf.float32)

    optimizer = _adam_optimizer(learning_rate=_fit_value(fit, "lr", 0.01))

    @tf.function
    def train_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            dist = tfd.GeneralizedExtremeValue(
                loc=alpha0 + alpha1 * t_tf,
                scale=beta,
                concentration=-neg_gamma,
            )
            log_likelihoods = dist.log_prob(data_tf)
            loss = -tf.reduce_sum(log_likelihoods)
        grads = tape.gradient(loss, [alpha0, alpha1, beta, neg_gamma])
        optimizer.apply_gradients(zip(grads, [alpha0, alpha1, beta, neg_gamma]))
        return loss

    for step in range(_fit_steps(fit)):
        loss = train_step()
        if step % 100 == 0 and verbose:
            print(
                "NS unbounded step",
                step,
                "loss",
                float(loss.numpy()),
                "alpha0",
                float(alpha0.numpy()),
                "alpha1",
                float(alpha1.numpy()),
                "beta",
                float(beta.numpy()),
                "gamma",
                float((-neg_gamma).numpy()),
            )

    return (
        float(alpha0.numpy()),
        float(alpha1.numpy()),
        float(beta.numpy()),
        float((-neg_gamma).numpy()),
    )


def fit_nonstationary_bounded(
    data: np.ndarray,
    z_star_assumed_t: np.ndarray,
    fit: DictConfig,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Fit non-stationary bounded GEV with known z_star(t); L-BFGS by default, Adam optional."""
    if _fit_method(fit) == "lbfgs":
        return _ns_lbfgs_bounded(data, z_star_assumed_t, fit)
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    z_star_tf = tf.convert_to_tensor(z_star_assumed_t, dtype=tf.float32)

    beta_guess = _fit_value(fit, "beta_guess", 1.0)
    gamma_guess = _fit_value(fit, "gamma_guess", -0.1)
    beta = tf.Variable(beta_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg())
    neg_gamma = tf.Variable(-gamma_guess, dtype=tf.float32, constraint=tf.keras.constraints.NonNeg())

    optimizer = _adam_optimizer(learning_rate=_fit_value(fit, "lr", 0.01))

    @tf.function
    def train_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            gamma = -tf.maximum(neg_gamma, 1e-6)
            alpha_t = z_star_tf + beta / gamma
            dist = tfd.GeneralizedExtremeValue(
                loc=alpha_t,
                scale=beta,
                concentration=gamma,
            )
            log_likelihoods = dist.log_prob(data_tf)
            loss = -tf.reduce_sum(log_likelihoods)
        grads = tape.gradient(loss, [beta, neg_gamma])
        optimizer.apply_gradients(zip(grads, [beta, neg_gamma]))
        return loss

    for step in range(_fit_steps(fit)):
        loss = train_step()
        if step % 100 == 0 and verbose:
            print(
                "NS bounded step",
                step,
                "loss",
                float(loss.numpy()),
                "beta",
                float(beta.numpy()),
                "gamma",
                float((-neg_gamma).numpy()),
            )

    return float(beta.numpy()), float((-neg_gamma).numpy())


def _name_base(config: DictConfig) -> str:
    """Build a compact identifier from config values for outputs."""
    return (
        f"z0_{config.z_star_start:.2f}_"
        + f"trend_{config.z_star_trend.min:.4f}_{config.z_star_trend.max:.4f}_{config.z_star_trend.steps}_"
        + f"beta_{config.beta:.2f}_gamma_{config.gamma:.2f}_"
        + f"ny_{config.n_years}_Nr_{config.seed_steps_Nr}_"
        + f"sigma_{config.z_star_assumed_sigma:.2f}"
    )


def _data_name(config: DictConfig) -> str:
    """Return the netCDF cache file path for the current configuration."""
    return os.path.join(DATA_PATH, f"vary_nonstationary_{_name_base(config)}.nc")


def _use_cache(config: DictConfig) -> bool:
    """Return whether cached data should be used when available.

    The deprecated `reload` option is honored if explicitly set.
    """
    if "reload" in config and config.reload is not None:
        return bool(config.reload)
    if "use_cache" in config:
        return bool(config.use_cache)
    return True


def _plot_only(config: DictConfig) -> bool:
    """Return whether to skip simulation and only plot existing cached data."""
    if "plot_only" in config:
        return bool(config.plot_only)
    return False


def run_single_experiment(
    trend: float,
    seed: int,
    config: DictConfig,
    quantiles: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one seed and one trend setting, returning fitted and true RVs."""
    n_years = int(config.n_years)
    years = np.arange(n_years, dtype=float)
    time_axis = years / max(1.0, years[-1])

    rng = np.random.default_rng(seed)

    z_star_true = float(config.z_star_start) + trend * years
    alpha_true_t = alpha_from_z_star_beta_gamma(z_star_true, config.beta, config.gamma)

    data = genextreme.rvs(
        c=-config.gamma,
        loc=alpha_true_t,
        scale=config.beta,
        size=n_years,
        random_state=rng,
    )

    # Uncertainty in the assumed physical bound. The default "systematic" model
    # draws one offset per experiment and shifts the whole bound trend by it: the
    # bound has the right shape but an uncertain level, representing structural /
    # parameter uncertainty (model choice, supergradient factor, surge-model bias),
    # which -- unlike independent year-to-year noise -- cannot be averaged away.
    sigma = float(config.z_star_assumed_sigma)
    mode = str(config.bound_uncertainty_mode) if "bound_uncertainty_mode" in config else "systematic"
    if mode == "per_year":
        bound_noise = rng.normal(0.0, sigma, size=n_years)
    else:  # "systematic"
        bound_noise = np.full(n_years, rng.normal(0.0, sigma))
    z_star_assumed_t = z_star_true + bound_noise
    # A bounded GEV gives zero likelihood to data above the cap; if the assumed
    # bound falls below an observed maximum, replace it with the empirical max
    # (the same adjustment used in the stationary experiments).
    z_star_assumed_t = np.maximum(
        z_star_assumed_t,
        data + float(config.upper_bound_padding),
    )

    fit_cfg = config.fit

    alpha_su, beta_su, gamma_su = fit_stationary_unbounded(
        data,
        fit=fit_cfg,
        verbose=bool(config.verbose),
    )
    # The elementwise clip above keeps z*(t) >= data(t), but the *mean* bound
    # handed to the stationary fit can still fall below the sample maximum,
    # which leaves no feasible parameters. Clip it to the empirical max plus
    # padding, as in vary_noise.py.
    z_star_assumed_stationary = max(
        float(np.mean(z_star_assumed_t)),
        float(data.max()) + float(config.upper_bound_padding),
    )
    alpha_sb, beta_sb, gamma_sb = fit_stationary_bounded(
        data,
        z_star_assumed_stationary,
        fit=fit_cfg,
        verbose=bool(config.verbose),
    )
    a0_nu, a1_nu, beta_nu, gamma_nu = fit_nonstationary_unbounded(
        data,
        time_axis,
        fit=fit_cfg,
        verbose=bool(config.verbose),
    )
    beta_nb, gamma_nb = fit_nonstationary_bounded(
        data,
        z_star_assumed_t,
        fit=fit_cfg,
        verbose=bool(config.verbose),
    )

    true_eval_loc = float(alpha_true_t[-1])
    rv_true = genextreme.isf(
        quantiles,
        c=-config.gamma,
        loc=true_eval_loc,
        scale=config.beta,
    )

    loc_su = alpha_su
    loc_sb = alpha_sb
    loc_nu = a0_nu + a1_nu * time_axis[-1]
    loc_nb = float(alpha_from_z_star_beta_gamma(z_star_assumed_t[-1], beta_nb, gamma_nb))

    rv_est = np.array(
        [
            genextreme.isf(quantiles, c=-gamma_su, loc=loc_su, scale=beta_su),
            genextreme.isf(quantiles, c=-gamma_sb, loc=loc_sb, scale=beta_sb),
            genextreme.isf(quantiles, c=-gamma_nu, loc=loc_nu, scale=beta_nu),
            genextreme.isf(quantiles, c=-gamma_nb, loc=loc_nb, scale=beta_nb),
        ]
    )

    return rv_est, rv_true


def get_fit_ds(config: DictConfig) -> xr.Dataset:
    """Compute or reload simulation results for all trend/seed combinations."""
    data_name = _data_name(config)
    if os.path.exists(data_name) and _use_cache(config):
        print("Reloading data file", data_name)
        return xr.open_dataset(data_name)

    print("Remaking data file", data_name)
    quantiles = list(config.quantiles)
    trends = np.linspace(
        float(config.z_star_trend.min),
        float(config.z_star_trend.max),
        int(config.z_star_trend.steps),
    )
    seeds = np.arange(int(config.seed_steps_Nr), dtype=int) + int(config.seed_offset)

    rv_est = np.full((len(FIT_NAMES), len(quantiles), len(seeds), len(trends)), np.nan)
    rv_true = np.full((len(quantiles), len(seeds), len(trends)), np.nan)

    n_failed = 0
    for i_trend, trend in enumerate(tqdm(trends, desc="Trend")):
        for i_seed, seed in enumerate(seeds):
            try:
                est, tru = run_single_experiment(
                    trend=float(trend),
                    seed=int(seed),
                    config=config,
                    quantiles=quantiles,
                )
                rv_est[:, :, i_seed, i_trend] = est
                rv_true[:, i_seed, i_trend] = tru
            except Exception as exc:
                n_failed += 1
                if config.verbose:
                    print(
                        f"Fit failed for trend={trend:.4f}, seed={seed}: {exc}",
                    )

    n_experiments = len(seeds) * len(trends)
    n_degenerate = int(np.isnan(rv_est).any(axis=(0, 1)).sum()) - n_failed
    print(
        f"Seed-experiments raising errors: {n_failed}/{n_experiments}; "
        f"with degenerate (NaN) fits: {n_degenerate}/{n_experiments}"
    )

    ds = xr.Dataset(
        data_vars={
            "rv_est": (
                ("fit", "rp", "seed", "trend"),
                rv_est,
                {"units": "m"},
            ),
            "rv_true": (
                ("rp", "seed", "trend"),
                rv_true,
                {"units": "m"},
            ),
        },
        coords={
            "fit": FIT_NAMES,
            "rp": (
                ("rp"),
                [int(1 / q) for q in quantiles],
                {"units": "years", "long_name": "Return period"},
            ),
            "seed": seeds,
            "trend": (
                ("trend"),
                trends,
                {"units": "m/year", "long_name": "Upper-bound trend"},
            ),
        },
    )
    ds.to_netcdf(data_name)
    return ds


def _seed_summary(da: xr.DataArray, lp: float, up: float) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Summarize over seed axis using mean and 5-95 envelope statistics."""
    mn = da.mean(dim="seed", skipna=True)
    lower = da.quantile(lp, dim="seed", skipna=True).drop_vars("quantile")
    upper = da.quantile(up, dim="seed", skipna=True).drop_vars("quantile")
    return mn, lower, upper, upper - lower


def _safe_ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Return ratio while masking non-positive or non-finite denominators."""
    good = np.isfinite(denominator) & (denominator > 0)
    return xr.where(good, numerator / denominator, np.nan)


def _show_stationary_reference(config: DictConfig) -> bool:
    """Return whether stationary fits should be shown as optional references."""
    if "figure" in config and "show_stationary_reference" in config.figure:
        return bool(config.figure.show_stationary_reference)
    return False


def plot_fit_ds(config: DictConfig, ds: xr.Dataset) -> None:
    """Plot RV estimates with envelopes, uncertainty metric, and separate bias metric."""
    plot_defaults()

    lp = float(config.figure.lp)
    up = float(config.figure.up)
    est_mn, est_low, est_up, est_range = _seed_summary(ds["rv_est"], lp, up)
    true_mn, true_low, true_up, _ = _seed_summary(ds["rv_true"], lp, up)
    bias = est_mn - true_mn
    show_stationary_reference = _show_stationary_reference(config)

    ns_fits = ["nonstationary_unbounded", "nonstationary_bounded"]
    stationary_fits = ["stationary_unbounded", "stationary_bounded"]

    fig, axs = plt.subplots(
        4,
        1,
        figsize=get_dim(ratio=0.6180339887498949 * 2.0),
        sharex=True,
        height_ratios=[1.0, 1.0, 0.85, 0.85],
    )

    for irp, rp in enumerate(ds.rp.values.tolist()):
        axs[irp].plot(
            ds.trend,
            true_mn.sel(rp=rp),
            color="black",
            linewidth=1.8,
            label="True end-year RV",
        )
        axs[irp].fill_between(
            ds.trend,
            true_low.sel(rp=rp),
            true_up.sel(rp=rp),
            color="black",
            alpha=0.12,
            linestyle="-",
        )

        for fit in ns_fits:
            color = config.color[fit]
            axs[irp].plot(
                ds.trend,
                est_mn.sel(fit=fit, rp=rp),
                color=color,
                linewidth=1.2,
                label=FIT_LABELS[fit],
            )
            axs[irp].fill_between(
                ds.trend,
                est_low.sel(fit=fit, rp=rp),
                est_up.sel(fit=fit, rp=rp),
                color=color,
                alpha=0.17,
            )

        if show_stationary_reference:
            for fit in stationary_fits:
                axs[irp].plot(
                    ds.trend,
                    est_mn.sel(fit=fit, rp=rp),
                    color=config.color[fit],
                    linewidth=0.9,
                    linestyle=":",
                    alpha=0.8,
                )

        axs[irp].set_ylabel(f"RV{int(rp)} [m]")

    ratio_ns = _safe_ratio(
        est_range.sel(fit="nonstationary_unbounded"),
        est_range.sel(fit="nonstationary_bounded"),
    )

    linestyles = ["-", "--"]
    for irp, rp in enumerate(ds.rp.values.tolist()):
        axs[2].plot(
            ds.trend,
            ratio_ns.sel(rp=rp),
            label=f"II-ns / I-ns envelope RV{int(rp)}",
            color=config.color.nonstationary_unbounded,
            linestyle=linestyles[irp],
        )

    for irp, rp in enumerate(ds.rp.values.tolist()):
        for fit in ns_fits:
            axs[3].plot(
                ds.trend,
                bias.sel(fit=fit, rp=rp),
                color=config.color[fit],
                linestyle=linestyles[irp],
                linewidth=1.2,
                label=f"{FIT_LABELS[fit]} RV{int(rp)}",
            )

        if show_stationary_reference:
            for fit in stationary_fits:
                axs[3].plot(
                    ds.trend,
                    bias.sel(fit=fit, rp=rp),
                    color=config.color[fit],
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.7,
                )

    axs[2].hlines(
        1.0,
        float(ds.trend.min()),
        float(ds.trend.max()),
        color="grey",
        linestyles="dashed",
    )
    axs[2].set_ylabel("5-95 envelope ratio")

    axs[3].hlines(
        0.0,
        float(ds.trend.min()),
        float(ds.trend.max()),
        color="grey",
        linestyles="dashed",
    )
    axs[3].set_ylabel("Bias [m]")
    axs[3].set_xlabel(r"Upper-bound trend, $dz^*/dt$ [m year$^{-1}$]")

    top_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="True end-year RV"),
        Line2D(
            [0],
            [0],
            color=config.color.nonstationary_unbounded,
            linewidth=1.2,
            label=FIT_LABELS["nonstationary_unbounded"],
        ),
        Line2D(
            [0],
            [0],
            color=config.color.nonstationary_bounded,
            linewidth=1.2,
            label=FIT_LABELS["nonstationary_bounded"],
        ),
    ]
    if show_stationary_reference:
        top_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=config.color.stationary_unbounded,
                    linewidth=0.9,
                    linestyle=":",
                    label="II-s: Stationary unbounded (ref)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=config.color.stationary_bounded,
                    linewidth=0.9,
                    linestyle=":",
                    label="I-s: Stationary bounded (ref)",
                ),
            ]
        )

    # Keep x-axis tight to the actual trend coordinates (no default padding).
    trend_min = float(np.nanmin(ds.trend.values))
    trend_max = float(np.nanmax(ds.trend.values))
    if np.isclose(trend_min, trend_max):
        trend_min -= 1e-9
        trend_max += 1e-9
    for ax in axs:
        ax.set_xlim(trend_min, trend_max)
        ax.margins(x=0)

    axs[0].legend(handles=top_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    axs[2].legend(loc="upper left")
    axs[3].legend(loc="upper right", ncol=2)
    label_subplots(axs, override="outside")

    fig_name = os.path.join(FIGURE_PATH, f"vary_nonstationary_{_name_base(config)}.pdf")
    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()


# Distinct colours (config default uses two greens, which clash in print).
_NS_COLORS = {
    "stationary_unbounded": "#d95f02",
    "stationary_bounded": "#7570b3",
    "nonstationary_unbounded": "#1f77b4",
    "nonstationary_bounded": "#2ca02c",
}
_NS_LINESTYLE = {
    "stationary_unbounded": ":",
    "stationary_bounded": ":",
    "nonstationary_unbounded": "-",
    "nonstationary_bounded": "-",
}


def plot_ns_skill(config: DictConfig, ds: xr.Dataset) -> None:
    """Summary figures using the same metrics as the stationary EVT experiments:
    the mean return value with its 5--95% envelope, the bias, and the 5--95% range.

    Two figures are written: (1) the return value with envelope vs trend (cf.
    Fig.~vary_z_star_sigma a,b), and (2) bias and 5--95% range vs trend for both
    return periods (cf. Fig.~evt2). The message: only the stationary fits are biased,
    and that bias grows with the trend; the non-stationary bounded fit (I-ns) matches
    the unbounded fit (II-ns) for bias while roughly halving the 5--95% range.
    """
    plot_defaults()

    est, tru = ds["rv_est"], ds["rv_true"]
    bias = (est.mean("seed", skipna=True) - tru.mean("seed", skipna=True))
    est_mn = est.mean("seed", skipna=True)
    true_mn = tru.mean("seed", skipna=True)
    elo = est.quantile(0.05, "seed", skipna=True).drop_vars("quantile")
    ehi = est.quantile(0.95, "seed", skipna=True).drop_vars("quantile")
    tlo = tru.quantile(0.05, "seed", skipna=True).drop_vars("quantile")
    thi = tru.quantile(0.95, "seed", skipna=True).drop_vars("quantile")
    width = ehi - elo
    trends = ds.trend.values
    rps = ds.rp.values.tolist()
    name = _name_base(config)

    # --- Figure 1: return value + 5-95% envelope vs trend --------------------
    fig, axs = plt.subplots(1, len(rps), figsize=get_dim(ratio=0.42), sharex=True,
                            squeeze=False)
    for j, rp in enumerate(rps):
        ax = axs[0, j]
        ax.plot(trends, true_mn.sel(rp=rp), color="black", lw=1.8, label="true")
        ax.fill_between(trends, tlo.sel(rp=rp), thi.sel(rp=rp), color="black", alpha=0.10)
        for fit in ("nonstationary_unbounded", "nonstationary_bounded"):
            ax.plot(trends, est_mn.sel(fit=fit, rp=rp), color=_NS_COLORS[fit], lw=1.4,
                    label=FIT_LABELS[fit])
            ax.fill_between(trends, elo.sel(fit=fit, rp=rp), ehi.sel(fit=fit, rp=rp),
                            color=_NS_COLORS[fit], alpha=0.18)
        for fit in ("stationary_unbounded", "stationary_bounded"):
            ax.plot(trends, est_mn.sel(fit=fit, rp=rp), color=_NS_COLORS[fit],
                    lw=1.0, ls=":", label=FIT_LABELS[fit])
        ax.set_title(f"RV{int(rp)}")
        ax.set_xlabel(r"upper-bound trend, $dz^*/dt$ [m year$^{-1}$]")
        ax.set_xlim(float(trends.min()), float(trends.max())); ax.margins(x=0)
    axs[0, 0].set_ylabel("return value [m]")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout()
    f1 = os.path.join(FIGURE_PATH, f"vary_nonstationary_rv_{name}.pdf")
    plt.savefig(f1); plt.close(); print("wrote", f1)

    # --- Figure 2: bias (top) and 5-95% range (bottom) vs trend --------------
    fig, axs = plt.subplots(2, len(rps), figsize=get_dim(ratio=0.85), sharex=True,
                            squeeze=False)
    for j, rp in enumerate(rps):
        for fit in FIT_NAMES:
            kw = dict(color=_NS_COLORS[fit], linestyle=_NS_LINESTYLE[fit], lw=1.6,
                      marker=("o" if "nonstationary" in fit else None), markersize=3)
            axs[0, j].plot(trends, bias.sel(fit=fit, rp=rp), label=FIT_LABELS[fit], **kw)
            axs[1, j].plot(trends, width.sel(fit=fit, rp=rp), **kw)
        axs[0, j].axhline(0.0, color="grey", ls="--", lw=0.8)
        axs[0, j].set_title(f"RV{int(rp)}")
        axs[1, j].set_xlabel(r"upper-bound trend, $dz^*/dt$ [m year$^{-1}$]")
        for ax in (axs[0, j], axs[1, j]):
            ax.set_xlim(float(trends.min()), float(trends.max())); ax.margins(x=0)
    axs[0, 0].set_ylabel("bias [m]")
    axs[1, 0].set_ylabel("5--95% range [m]")
    legend_below(fig, axs[0, 0], ncol=2)        # shared legend below -> no data overlap
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    f2 = os.path.join(FIGURE_PATH, f"vary_nonstationary_bias_range_{name}.pdf")
    plt.savefig(f2, bbox_inches="tight"); plt.close(); print("wrote", f2)

    # --- console summary (bias, 5-95% range, range ratio II-ns/I-ns) ---------
    it = int(np.argmin(np.abs(trends - 0.01)))
    print(f"\nSkill at trend = {trends[it]:.3f} m/year:")
    for rp in rps:
        print(f"  RV{int(rp)}:   bias    5-95%range")
        for fit in FIT_NAMES:
            print(f"    {FIT_LABELS[fit]:34s} "
                  f"{float(bias.sel(fit=fit, rp=rp)[it]):+5.2f}   "
                  f"{float(width.sel(fit=fit, rp=rp)[it]):5.2f}")
        ratio = float(width.sel(fit="nonstationary_unbounded", rp=rp)[it]
                      / width.sel(fit="nonstationary_bounded", rp=rp)[it])
        print(f"    -> 5-95% range ratio II-ns / I-ns = {ratio:.2f}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_nonstationary")
def run_vary_nonstationary(config: DictConfig) -> None:
    """Hydra entrypoint for non-stationary synthetic EVT experiment."""
    print(config)
    if _plot_only(config):
        data_name = _data_name(config)
        if not os.path.exists(data_name):
            raise FileNotFoundError(
                "plot_only=true but no cached dataset found at "
                + data_name
                + ". Run once with use_cache=false to generate it."
            )
        print("Plot-only mode, loading cached data file", data_name)
        ds = xr.open_dataset(data_name)
    else:
        ds = get_fit_ds(config)
    print(ds)
    plot_ns_skill(config, ds)


if __name__ == "__main__":
    # python -m worst.vary_nonstationary
    run_vary_nonstationary()
