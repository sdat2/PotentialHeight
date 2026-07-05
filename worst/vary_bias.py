"""Vary systematic bias of the assumed upper bound, and decompose why a wrong bound can still help.

This extends worst/vary_noise.py (pure random error in the calculated upper bound,
paper Fig. vary_z_star_sigma) to SYSTEMATIC error. For each Monte Carlo resample of
the true GEV (z_star, beta, gamma) we assume an upper bound

    z_hat_star = z_star + b + sigma * eps,    eps ~ N(0, 1),

where b is a fixed bias [m] swept over a grid and sigma is the random error level [m].
As in vary_noise.py the assumed bound is clipped to the empirical maximum plus a small
padding, z_star_assumed = max(z_hat_star, max(z) + pad), because a bounded GEV assigns
zero likelihood to data above its bound. We then fit:

    (I)  bounded GEV at the assumed bound (worst.tens.fit_gev_upper_bound_known), and
    (II) unbounded GEV (worst.tens.fit_gev_upper_bound_not_known, once per resample),

and record the 1 in 100 and 1 in 500 year return values over Nr resamples.

Mechanism decomposition: the paper conjectures that a noisy/biased bound still helps
because (i) bounds sampled too LOW are replaced by the empirical maximum (the clip
above), and (ii) bounds sampled too HIGH are compensated by the scale/shape estimates
co-adjusting (more negative gamma_hat pulls the return curve back down). To make this
visible we additionally record, per grid cell: the fraction of resamples where the
clip engaged, the assumed bound actually used, and the fitted (beta_hat, gamma_hat)
of method (I). A second experiment axis repeats the bias sweep for several generating
shapes gamma, since how far the sample maximum sits below z_star (and therefore how
early the clip can rescue a low bound) is controlled by gamma.

The headline number is b*: the bias at which method (I)'s RMSE of the 500-year return
value crosses method (II)'s, i.e. how much systematic bias in the potential height the
bounded fit tolerates before it underperforms not using the bound at all.

Run:
    python -m worst.vary_bias
    python -m worst.vary_bias regenerate=true
"""

from typing import Dict, List, Tuple
import os
import time
import numpy as np
import xarray as xr
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from tqdm import tqdm
from sithom.plot import plot_defaults, label_subplots, get_dim
from sithom.time import timeit, time_stamp
from sithom.misc import get_git_revision_hash
from .utils import alpha_from_z_star_beta_gamma, legend_below
from .tens import (
    seed_all,
    gen_data,
    fit_gev_upper_bound_not_known,
    fit_gev_upper_bound_known,
)
from .constants import CONFIG_PATH, DATA_PATH, FIGURE_PATH, PROJ_PATH


def _name_base(config: DictConfig) -> str:
    """Build a compact identifier from config values for cache/figure names.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        str: Unique name based on the configuration.
    """
    sig = "-".join(f"{float(s):.2f}" for s in config.sigmas)
    gam = "-".join(f"{float(g):.2f}" for g in config.gammas)
    return (
        f"z_star_{config.z_star:.2f}_ns_{config.ns}_Nr_{config.seed_steps_Nr}_"
        + f"beta_{config.beta:.2f}_"
        + f"bias_{config.bias.min:.2f}_{config.bias.max:.2f}_{config.bias.steps}_"
        + f"sigmas_{sig}_gammas_{gam}"
    )


def _data_dir(config: DictConfig) -> str:
    """Directory for the cached netCDF (config override for tests, else DATA_PATH).

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        str: Output directory.
    """
    if "data_dir" in config and config.data_dir:
        return str(config.data_dir)
    return DATA_PATH


def _data_name(config: DictConfig) -> str:
    """Return the netCDF cache file path for the current configuration.

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        str: Path of cached netCDF file.
    """
    return os.path.join(_data_dir(config), f"vary_bias_{_name_base(config)}.nc")


@timeit
def get_fit_ds(config: DictConfig) -> xr.Dataset:
    """Compute or reload the bias-sweep Monte Carlo experiment.

    Grid: generating shape gamma x bound noise sigma x bound bias b. For each
    (gamma, seed) one sample of ns block maxima and one standard normal draw eps
    are generated (exactly as in vary_noise.py: seed_all(seed), TF sampling, then
    np.random.normal), and shared across the whole (sigma, b) grid so that method
    (I) fits are paired across cells. Method (II) is fitted once per (gamma, seed).

    Args:
        config (DictConfig): Hydra config object.

    Returns:
        xr.Dataset: Per-seed return values, clip flags and fitted parameters.
    """
    data_name = _data_name(config)
    if os.path.exists(data_name) and not config.regenerate:
        print("Reloading data file", data_name)
        return xr.open_dataset(data_name)

    print("Remaking data file", data_name)
    quantiles = list(config.quantiles)
    gammas = [float(g) for g in config.gammas]
    sigmas = [float(s) for s in config.sigmas]
    biases = np.linspace(
        float(config.bias.min), float(config.bias.max), int(config.bias.steps)
    )
    seeds = np.arange(int(config.seed_steps_Nr), dtype=int) + int(config.seed_offset)
    pad = float(config.upper_bound_padding)
    z_star = float(config.z_star)
    beta = float(config.beta)
    ns = int(config.ns)
    fit = config.fit

    ng, nsig, nb, nr, nq = len(gammas), len(sigmas), len(biases), len(seeds), len(quantiles)

    rv_ubk = np.full((ng, nsig, nb, nr, nq), np.nan)
    rv_ubu = np.full((ng, nr, nq), np.nan)
    rv_true = np.full((ng, nq), np.nan)
    clip_engaged = np.zeros((ng, nsig, nb, nr), dtype=np.int8)
    z_star_assumed = np.full((ng, nsig, nb, nr), np.nan)
    beta_hat = np.full((ng, nsig, nb, nr), np.nan)
    gamma_hat = np.full((ng, nsig, nb, nr), np.nan)

    n_cells = ng * nsig * nb
    cell = 0
    t_start = time.time()

    for ig, gamma in enumerate(gammas):
        alpha = alpha_from_z_star_beta_gamma(z_star, beta, gamma)
        rv_true[ig] = genextreme.isf(quantiles, c=-gamma, loc=alpha, scale=beta)

        # per-seed data, eps draw and method (II) fit, shared across the (sigma, b) grid
        zs_list: List[np.ndarray] = []
        eps_arr = np.full(nr, np.nan)
        max_arr = np.full(nr, np.nan)
        for i_seed, seed in enumerate(
            tqdm(seeds, desc=f"gamma={gamma:+.2f} data + method II fits")
        ):
            seed_all(int(seed))
            # Cast to float64: gen_data returns float32, and the L-BFGS-B fits in
            # worst.tens use finite-difference gradients with eps ~1e-8, which are
            # meaningless on a float32-quantised likelihood (the optimizer then
            # stalls at its starting guess). float64 restores correct convergence
            # (verified against the Adam path and scipy's genextreme.fit).
            zs = gen_data(alpha, beta, gamma, ns).astype(np.float64)
            eps_arr[i_seed] = float(np.random.normal(0.0, 1.0))
            max_arr[i_seed] = float(np.max(zs))
            zs_list.append(zs)
            # alpha_guess=null in the config defaults to mean(data): with the fixed
            # guess 0.0 the unbounded L-BFGS fit converges to a spurious Frechet
            # optimum (gamma_hat ~ +0.9, exploding RVs) whenever the generating
            # location is far from zero (e.g. gamma=-0.4 gives alpha=4.5), which
            # would make method (II) a strawman baseline across the gamma sweep.
            alpha_guess = (
                float(fit.alpha_guess)
                if fit.alpha_guess is not None
                else float(np.mean(zs))
            )
            try:
                a2, b2, g2 = fit_gev_upper_bound_not_known(
                    zs,
                    opt_steps=fit.steps,
                    lr=fit.lr,
                    alpha_guess=alpha_guess,
                    beta_guess=fit.beta_guess,
                    gamma_guess=fit.gamma_guess,
                    verbose=config.verbose,
                )
                rv_ubu[ig, i_seed] = genextreme.isf(quantiles, c=-g2, loc=a2, scale=b2)
            except Exception as exc:
                print(f"Unbounded fit failed for gamma={gamma}, seed={seed}: {exc}")

        for i_sig, sigma in enumerate(sigmas):
            for ib, b in enumerate(biases):
                cell += 1
                t_cell = time.time()
                for i_seed in range(nr):
                    z_hat = z_star + b + sigma * eps_arr[i_seed]
                    floor = max_arr[i_seed] + pad
                    clipped = z_hat < floor
                    z_assumed = max(z_hat, floor)
                    clip_engaged[ig, i_sig, ib, i_seed] = int(clipped)
                    z_star_assumed[ig, i_sig, ib, i_seed] = z_assumed
                    try:
                        a1, b1, g1 = fit_gev_upper_bound_known(
                            zs_list[i_seed],
                            z_assumed,
                            opt_steps=fit.steps,
                            lr=fit.lr,
                            beta_guess=fit.beta_guess,
                            gamma_guess=fit.gamma_guess,
                            verbose=config.verbose,
                        )
                    except Exception as exc:
                        print(f"Bounded fit failed (b={b}, sigma={sigma}): {exc}")
                        a1 = b1 = g1 = float("nan")
                    beta_hat[ig, i_sig, ib, i_seed] = b1
                    gamma_hat[ig, i_sig, ib, i_seed] = g1
                    if np.isfinite(b1):
                        rv_ubk[ig, i_sig, ib, i_seed] = genextreme.isf(
                            quantiles, c=-g1, loc=a1, scale=b1
                        )
                dt = time.time() - t_cell
                eta = (time.time() - t_start) / cell * (n_cells - cell)
                cf = float(np.mean(clip_engaged[ig, i_sig, ib]))
                rv500 = float(np.nanmean(rv_ubk[ig, i_sig, ib, :, -1]))
                print(
                    f"cell {cell}/{n_cells} gamma={gamma:+.2f} sigma={sigma:.2f} "
                    f"b={b:+.2f} m: clip_frac={cf:.2f}, rv500_mn={rv500:.2f} m "
                    f"[{dt:.1f} s, eta {eta / 60:.1f} min]"
                )

    ds = xr.Dataset(
        data_vars={
            "rv_ubk": (
                ("gamma", "sigma", "bias", "seed", "rp"),
                rv_ubk,
                {"units": "m", "long_name": "Return value, method I (bounded at assumed bound)"},
            ),
            "rv_ubu": (
                ("gamma", "seed", "rp"),
                rv_ubu,
                {"units": "m", "long_name": "Return value, method II (unbounded)"},
            ),
            "rv_true": (
                ("gamma", "rp"),
                rv_true,
                {"units": "m", "long_name": "True return value of generating GEV"},
            ),
            "clip_engaged": (
                ("gamma", "sigma", "bias", "seed"),
                clip_engaged,
                {
                    "long_name": "Empirical-maximum safeguard engaged",
                    "description": "1 if z_hat_star < max(z) + pad so the assumed bound was replaced",
                },
            ),
            "z_star_assumed": (
                ("gamma", "sigma", "bias", "seed"),
                z_star_assumed,
                {"units": "m", "long_name": "Assumed upper bound after clipping"},
            ),
            "beta_hat": (
                ("gamma", "sigma", "bias", "seed"),
                beta_hat,
                {"units": "m", "long_name": "Fitted scale, method I"},
            ),
            "gamma_hat": (
                ("gamma", "sigma", "bias", "seed"),
                gamma_hat,
                {"long_name": "Fitted shape, method I"},
            ),
        },
        coords={
            "gamma": (("gamma"), gammas, {"long_name": "Generating GEV shape"}),
            "sigma": (
                ("sigma"),
                sigmas,
                {"units": "m", "long_name": "Random error s.d. of assumed upper bound"},
            ),
            "bias": (
                ("bias"),
                biases,
                {"units": "m", "long_name": "Systematic bias of assumed upper bound"},
            ),
            "seed": (("seed"), seeds, {"long_name": "Random seed"}),
            "rp": (
                ("rp"),
                [int(1 / q) for q in quantiles],
                {"units": "years", "long_name": "Return period"},
            ),
        },
        attrs={
            "description": (
                "Systematic bias + random noise in the assumed GEV upper bound: "
                "bounded (I) vs unbounded (II) return value estimates, clip "
                "fraction and fitted parameters. See worst/vary_bias.py."
            ),
            "z_star": z_star,
            "beta": beta,
            "ns": ns,
            "upper_bound_padding": pad,
            "vary_bias_calculated_at_git_hash": _git_hash(),
            "vary_bias_calculated_at_time": time_stamp(),
        },
    )
    ds.to_netcdf(data_name)
    print("Saved", data_name)
    return ds


def _git_hash() -> str:
    """Git revision hash for provenance attrs (never fatal).

    Returns:
        str: Git hash, or "unknown" if git is unavailable.
    """
    try:
        return get_git_revision_hash(path=str(PROJ_PATH))
    except Exception:
        return "unknown"


def _sigma_colors(n: int) -> np.ndarray:
    """Line colors for the sigma levels.

    Args:
        n (int): Number of sigma levels.

    Returns:
        np.ndarray: RGBA colors.
    """
    return plt.cm.viridis(np.linspace(0.0, 0.85, n))


def _crossings(x: np.ndarray, diff: np.ndarray) -> List[float]:
    """Zero crossings of diff(x) by linear interpolation.

    Args:
        x (np.ndarray): Grid.
        diff (np.ndarray): Function values on the grid.

    Returns:
        List[float]: Interpolated crossing locations.
    """
    out = []
    for i in range(len(x) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        if d0 == 0.0:
            out.append(float(x[i]))
        elif d0 * d1 < 0.0:
            out.append(float(x[i] - d0 * (x[i + 1] - x[i]) / (d1 - d0)))
    if len(diff) and np.isfinite(diff[-1]) and diff[-1] == 0.0:
        out.append(float(x[-1]))
    return out


def _rmse(err: xr.DataArray) -> xr.DataArray:
    """Root-mean-square error over the seed dimension (skipna).

    Args:
        err (xr.DataArray): Error (estimate minus truth) with a seed dim.

    Returns:
        xr.DataArray: RMSE.
    """
    return np.sqrt((err**2).mean("seed", skipna=True))


def plot_rv_bias(config: DictConfig, ds: xr.Dataset) -> str:
    """(a) Key figure: mean return-value error vs bound bias b, per sigma, per gamma.

    Method II's mean error (dashed line) and 5-95% error envelope (band) form the
    reference: while method I's curve stays inside/below that reference, the biased
    bound is still competitive with not using a bound at all.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.

    Returns:
        str: Figure path.
    """
    plot_defaults()
    gammas = ds.gamma.values.tolist()
    sigmas = ds.sigma.values.tolist()
    rps = ds.rp.values.tolist()
    biases = ds.bias.values
    colors = _sigma_colors(len(sigmas))
    lp, up = float(config.figure.lp), float(config.figure.up)

    err_i = ds["rv_ubk"] - ds["rv_true"]
    err_ii = ds["rv_ubu"] - ds["rv_true"]

    fig, axs = plt.subplots(
        len(rps),
        len(gammas),
        figsize=get_dim(ratio=0.85),
        sharex=True,
        squeeze=False,
    )
    for jg, gamma in enumerate(gammas):
        for irp, rp in enumerate(rps):
            ax = axs[irp, jg]
            e2 = err_ii.sel(gamma=gamma, rp=rp)
            ax.axhspan(
                float(e2.quantile(lp, dim="seed", skipna=True)),
                float(e2.quantile(up, dim="seed", skipna=True)),
                color=config.color.max_unknown,
                alpha=0.15,
                label="II: 5%-95% error envelope",
            )
            ax.axhline(
                float(e2.mean("seed", skipna=True)),
                color=config.color.max_unknown,
                linestyle="--",
                linewidth=1.2,
                label="II: Unbounded fit, mean error",
            )
            for k, sigma in enumerate(sigmas):
                ax.plot(
                    biases,
                    err_i.sel(gamma=gamma, sigma=sigma, rp=rp).mean("seed", skipna=True),
                    color=colors[k],
                    linewidth=1.2,
                    label=rf"I: $\sigma_{{\hat{{z}}^*}}={sigma:.1f}$ m, mean error",
                )
            ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlim(float(biases.min()), float(biases.max()))
            ax.margins(x=0)
            if irp == 0:
                ax.set_title(rf"$\gamma={gamma:+.1f}$")
            if jg == 0:
                ax.set_ylabel(f"RV{int(rp)} error [m]")
            if irp == len(rps) - 1:
                ax.set_xlabel(r"Upper-bound bias, $b$ [m]")
    legend_below(fig, axs[0, 0], ncol=2)
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGURE_PATH, f"vary_bias_rv_error_{_name_base(config)}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)
    return out


def plot_clip_fraction(config: DictConfig, ds: xr.Dataset) -> str:
    """(b) Mechanism figure: fraction of resamples where the empirical-max clip engaged.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.

    Returns:
        str: Figure path.
    """
    plot_defaults()
    gammas = ds.gamma.values.tolist()
    sigmas = ds.sigma.values.tolist()
    biases = ds.bias.values
    colors = _sigma_colors(len(sigmas))
    frac = ds["clip_engaged"].mean("seed")

    fig, axs = plt.subplots(
        1, len(gammas), figsize=get_dim(ratio=0.45), sharey=True, squeeze=False
    )
    for jg, gamma in enumerate(gammas):
        ax = axs[0, jg]
        for k, sigma in enumerate(sigmas):
            ax.plot(
                biases,
                frac.sel(gamma=gamma, sigma=sigma),
                color=colors[k],
                linewidth=1.2,
                label=rf"$\sigma_{{\hat{{z}}^*}}={sigma:.1f}$ m",
            )
        ax.set_title(rf"$\gamma={gamma:+.1f}$")
        ax.set_xlabel(r"Upper-bound bias, $b$ [m]")
        ax.set_xlim(float(biases.min()), float(biases.max()))
        ax.margins(x=0)
        ax.set_ylim(-0.02, 1.02)
    axs[0, 0].set_ylabel(
        r"Clip fraction, $P\left[\hat{z}^* < \max(\vec{z}) + \epsilon\right]$"
    )
    legend_below(fig, axs[0, 0], ncol=min(4, len(sigmas)), y=-0.06)
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(FIGURE_PATH, f"vary_bias_clip_fraction_{_name_base(config)}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)
    return out


def plot_param_compensation(config: DictConfig, ds: xr.Dataset) -> str:
    """(c) Mechanism figure: mean fitted gamma_hat (top) and beta_hat (bottom) vs b.

    Shows the shape/scale co-adjustment: a too-high bound is compensated by a more
    negative gamma_hat (and larger beta_hat), pulling the return curve back down.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.

    Returns:
        str: Figure path.
    """
    plot_defaults()
    gammas = ds.gamma.values.tolist()
    sigmas = ds.sigma.values.tolist()
    biases = ds.bias.values
    colors = _sigma_colors(len(sigmas))
    gam_mn = ds["gamma_hat"].mean("seed", skipna=True)
    bet_mn = ds["beta_hat"].mean("seed", skipna=True)

    fig, axs = plt.subplots(
        2, len(gammas), figsize=get_dim(ratio=0.7), sharex=True, squeeze=False
    )
    for jg, gamma in enumerate(gammas):
        for k, sigma in enumerate(sigmas):
            axs[0, jg].plot(
                biases,
                gam_mn.sel(gamma=gamma, sigma=sigma),
                color=colors[k],
                linewidth=1.2,
                label=rf"$\sigma_{{\hat{{z}}^*}}={sigma:.1f}$ m",
            )
            axs[1, jg].plot(
                biases,
                bet_mn.sel(gamma=gamma, sigma=sigma),
                color=colors[k],
                linewidth=1.2,
            )
        axs[0, jg].axhline(
            float(gamma), color=config.color.true_gev, linestyle="--", linewidth=0.9
        )
        axs[1, jg].axhline(
            float(config.beta), color=config.color.true_gev, linestyle="--", linewidth=0.9
        )
        axs[0, jg].set_title(rf"$\gamma={gamma:+.1f}$")
        axs[1, jg].set_xlabel(r"Upper-bound bias, $b$ [m]")
        for ax in (axs[0, jg], axs[1, jg]):
            ax.set_xlim(float(biases.min()), float(biases.max()))
            ax.margins(x=0)
    axs[0, 0].set_ylabel(r"Mean fitted shape, $\hat{\gamma}$")
    axs[1, 0].set_ylabel(r"Mean fitted scale, $\hat{\beta}$ [m]")
    legend_below(fig, axs[0, 0], ncol=min(4, len(sigmas)))
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGURE_PATH, f"vary_bias_params_{_name_base(config)}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)
    return out


def plot_rmse(config: DictConfig, ds: xr.Dataset) -> str:
    """(d) Supporting figure for the headline b*: RMSE of method I vs b, with II reference.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.

    Returns:
        str: Figure path.
    """
    plot_defaults()
    gammas = ds.gamma.values.tolist()
    sigmas = ds.sigma.values.tolist()
    rps = ds.rp.values.tolist()
    biases = ds.bias.values
    colors = _sigma_colors(len(sigmas))
    rmse_i = _rmse(ds["rv_ubk"] - ds["rv_true"])
    rmse_ii = _rmse(ds["rv_ubu"] - ds["rv_true"])

    fig, axs = plt.subplots(
        len(rps),
        len(gammas),
        figsize=get_dim(ratio=0.85),
        sharex=True,
        squeeze=False,
    )
    for jg, gamma in enumerate(gammas):
        for irp, rp in enumerate(rps):
            ax = axs[irp, jg]
            ax.axhline(
                float(rmse_ii.sel(gamma=gamma, rp=rp)),
                color=config.color.max_unknown,
                linestyle="--",
                linewidth=1.2,
                label="II: Unbounded fit",
            )
            for k, sigma in enumerate(sigmas):
                ax.plot(
                    biases,
                    rmse_i.sel(gamma=gamma, sigma=sigma, rp=rp),
                    color=colors[k],
                    linewidth=1.2,
                    label=rf"I: $\sigma_{{\hat{{z}}^*}}={sigma:.1f}$ m",
                )
            ax.set_xlim(float(biases.min()), float(biases.max()))
            ax.margins(x=0)
            if irp == 0:
                ax.set_title(rf"$\gamma={gamma:+.1f}$")
            if jg == 0:
                ax.set_ylabel(f"RV{int(rp)} RMSE [m]")
            if irp == len(rps) - 1:
                ax.set_xlabel(r"Upper-bound bias, $b$ [m]")
    legend_below(fig, axs[0, 0], ncol=2)
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGURE_PATH, f"vary_bias_rmse_{_name_base(config)}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)
    return out


def plot_fit_ds(config: DictConfig, ds: xr.Dataset) -> None:
    """Make all figures for the bias-sweep experiment.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.
    """
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plot_rv_bias(config, ds)
    plot_clip_fraction(config, ds)
    plot_param_compensation(config, ds)
    plot_rmse(config, ds)


def report_crossings(config: DictConfig, ds: xr.Dataset) -> Dict[Tuple[float, float, int], dict]:
    """Report b*: where method I's error crosses method II's, per (gamma, sigma, rp).

    Two error metrics are used. RMSE is the headline, but method II's RMSE is
    dominated by rare catastrophic unbounded fits (shape estimate near or above
    zero explodes the long-period return value), so the robust median absolute
    error (MedAE) is also reported to compare typical-case performance. Also
    prints the clip-engagement fractions at the ends and centre of the b-grid,
    the mechanism behind the asymmetric tolerance to negative bias.

    Args:
        config (DictConfig): Hydra config object.
        ds (xr.Dataset): Experiment dataset.

    Returns:
        Dict[Tuple[float, float, int], dict]: (gamma, sigma, rp) -> crossing summary
            with RMSE keys (rmse_ii, crossings, always_better, never_better) and
            MedAE keys (medae_ii, medae_crossings, medae_always_better,
            medae_never_better).
    """
    biases = ds.bias.values
    err_i = ds["rv_ubk"] - ds["rv_true"]
    err_ii = ds["rv_ubu"] - ds["rv_true"]
    metrics = {
        "RMSE": (_rmse(err_i), _rmse(err_ii)),
        "MedAE": (
            abs(err_i).median("seed", skipna=True),
            abs(err_ii).median("seed", skipna=True),
        ),
    }
    frac = ds["clip_engaged"].mean("seed")
    out: Dict[Tuple[float, float, int], dict] = {}
    for metric, (m_i, m_ii) in metrics.items():
        print(f"\n=== b* summary ({metric}): where {metric}(I) crosses {metric}(II) ===")
        for gamma in ds.gamma.values.tolist():
            for rp in ds.rp.values.tolist():
                r2 = float(m_ii.sel(gamma=gamma, rp=rp))
                print(f"\ngamma={gamma:+.2f}, RV{int(rp)}: {metric}(II) = {r2:.2f} m")
                for sigma in ds.sigma.values.tolist():
                    r1 = m_i.sel(gamma=gamma, sigma=sigma, rp=rp).values
                    diff = r1 - r2
                    cross = _crossings(biases, diff)
                    better = diff <= 0
                    if better.all():
                        msg = "I beats II over the whole b-grid"
                    elif not better.any():
                        msg = "I never beats II on this b-grid"
                    else:
                        msg = "b* = " + ", ".join(f"{c:+.2f} m" for c in cross)
                    cf = frac.sel(gamma=gamma, sigma=sigma).values
                    print(
                        f"  sigma={sigma:.1f} m: {msg} "
                        f"(clip frac at b={biases[0]:+.0f}/0/{biases[-1]:+.0f} m: "
                        f"{float(cf[0]):.2f}/{float(cf[len(cf) // 2]):.2f}/{float(cf[-1]):.2f})"
                    )
                    entry = out.setdefault((float(gamma), float(sigma), int(rp)), {})
                    prefix = "" if metric == "RMSE" else "medae_"
                    entry["rmse_ii" if metric == "RMSE" else "medae_ii"] = r2
                    entry[prefix + "crossings"] = cross
                    entry[prefix + "always_better"] = bool(better.all())
                    entry[prefix + "never_better"] = bool((~better).all())
    return out


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_bias")
def run_vary_bias(config: DictConfig) -> None:
    """Run the systematic-bias upper-bound experiment.

    Args:
        config (DictConfig): Hydra config object.
    """
    print("config", config)
    ds = get_fit_ds(config)
    print(ds)
    plot_fit_ds(config, ds)
    report_crossings(config, ds)


if __name__ == "__main__":
    # python -m worst.vary_bias
    run_vary_bias()
