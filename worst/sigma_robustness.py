"""Non-stationary EVT robustness to upper-bound uncertainty.

Companion to :mod:`worst.vary_nonstationary`. Fixes the bound trend at a representative
value and sweeps the systematic level uncertainty ``sigma_z*`` in the supplied physical
bound, reporting the bias and 5--95% range of the four estimators (stationary/non-stationary
x bounded/unbounded). Shows how far the non-stationary bounded advantage survives before the
bounded 5--95% range overtakes the unbounded one.

Run::

    python -m worst.sigma_robustness
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from sithom.plot import get_dim, label_subplots, plot_defaults

from .constants import CONFIG_PATH, DATA_PATH, FIGURE_PATH
from .utils import legend_below
from .vary_nonstationary import FIT_LABELS, FIT_NAMES, run_single_experiment

# representative bound trend (~1 cm/yr potential-height rise) and uncertainty grid
TREND = 0.01
SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
_COLORS = {
    "stationary_unbounded": "#d95f02", "stationary_bounded": "#7570b3",
    "nonstationary_unbounded": "#1f77b4", "nonstationary_bounded": "#2ca02c",
}
_LS = {
    "stationary_unbounded": ":", "stationary_bounded": ":",
    "nonstationary_unbounded": "-", "nonstationary_bounded": "-",
}


def get_sigma_ds(n_seed: int = 120, maxiter: int = 600) -> xr.Dataset:
    """Sweep ``sigma_z*`` at fixed ``TREND``; cache and return the result dataset."""
    cfg = OmegaConf.load(os.path.join(CONFIG_PATH, "vary_nonstationary.yaml"))
    cfg.fit.maxiter = maxiter
    cfg.verbose = False
    quantiles = list(cfg.quantiles)
    out = os.path.join(DATA_PATH, f"sigma_robustness_trend{TREND}_Nr{n_seed}.nc")
    if os.path.exists(out):
        return xr.open_dataset(out)

    est = np.full((len(SIGMAS), len(FIT_NAMES), len(quantiles), n_seed), np.nan)
    tru = np.full((len(SIGMAS), len(quantiles), n_seed), np.nan)
    for i_s, sig in enumerate(tqdm(SIGMAS, desc="sigma")):
        cfg.z_star_assumed_sigma = float(sig)
        for seed in range(n_seed):
            try:
                e, t = run_single_experiment(TREND, seed, cfg, quantiles)
                est[i_s, :, :, seed] = e
                tru[i_s, :, seed] = t
            except Exception as exc:  # pragma: no cover - occasional TF fit failure
                print(f"  fail sigma={sig} seed={seed}: {exc}")
    ds = xr.Dataset(
        {"rv_est": (("sigma", "fit", "rp", "seed"), est),
         "rv_true": (("sigma", "rp", "seed"), tru)},
        coords={"sigma": SIGMAS, "fit": FIT_NAMES,
                "rp": [int(1 / q) for q in quantiles], "seed": np.arange(n_seed)},
    )
    ds.to_netcdf(out)
    return ds


def plot_sigma_ds(ds: xr.Dataset) -> None:
    """Bias (top) and 5--95% range (bottom) vs sigma_z*, per return period.

    Matches the paper figure style (sithom serif via ``plot_defaults``, text-width
    sizing via ``get_dim``, (a)--(d) panel labels) and the sibling
    ``vary_nonstationary`` bias/range figure it accompanies."""
    import matplotlib.pyplot as plt
    plot_defaults()

    err = ds.rv_est - ds.rv_true
    bias = err.mean("seed", skipna=True)
    width = (ds.rv_est.quantile(0.95, "seed", skipna=True)
             - ds.rv_est.quantile(0.05, "seed", skipna=True))
    rps = ds.rp.values.tolist()
    fig, axs = plt.subplots(2, len(rps), figsize=get_dim(ratio=0.85), sharex=True,
                            squeeze=False)
    for j, rp in enumerate(rps):
        for f in FIT_NAMES:
            kw = dict(color=_COLORS[f], ls=_LS[f], lw=1.6,
                      marker=("o" if "nonstationary" in f else None), markersize=3)
            axs[0, j].plot(ds.sigma, bias.sel(fit=f, rp=rp), label=FIT_LABELS[f], **kw)
            axs[1, j].plot(ds.sigma, width.sel(fit=f, rp=rp), **kw)
        axs[0, j].axhline(0, color="grey", ls="--", lw=0.8)
        axs[0, j].set_title(f"RV{int(rp)}  (trend $= {TREND}$ m year$^{{-1}}$)")
        axs[1, j].set_xlabel(r"Bound level uncertainty $\sigma_{z^*}$ [m]")
        for ax in (axs[0, j], axs[1, j]):
            ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("Bias [m]")
    axs[1, 0].set_ylabel("5--95% range [m]")
    legend_below(fig, axs[0, 0], ncol=2)        # shared legend below -> no data overlap
    label_subplots(axs.ravel().tolist(), override="outside")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGURE_PATH, "sigma_robustness.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main() -> None:
    ds = get_sigma_ds()
    print(ds)
    plot_sigma_ds(ds)


if __name__ == "__main__":
    # python -m worst.sigma_robustness
    main()
