"""Paper figures for the non-stationary bounded-EVT experiment (paper appendix B).

Thin plotting layer over the L-BFGS sweep caches written by :mod:`worst.vary_nonstationary`
(trend sweep) and :mod:`worst.sigma_robustness` (bound-uncertainty sweep). Renders the two
appendix figures -- bias and 5--95% range for the four estimators
(stationary/non-stationary x bounded/unbounded) -- straight into the thesis ``img/`` tree and
prints the headline numbers (range ratios, sigma crossovers) used in the appendix/rebuttal.

Run (after the sweeps have been cached)::

    python -m worst.ns_evt_figs

Generates ``evt_ns_bias_range.pdf`` (vs bound trend) and ``evt_ns_sigma.pdf`` (vs bound
uncertainty).  The latest matching ``.nc`` cache is used; regenerate caches with
``python -m worst.vary_nonstationary`` / ``python -m worst.sigma_robustness``.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import DATA_PATH
from .evt_theory import _paper_img_path

FITS = ["stationary_unbounded", "stationary_bounded",
        "nonstationary_unbounded", "nonstationary_bounded"]
LAB = {"stationary_unbounded": "II-s  stationary unbounded",
       "stationary_bounded": "I-s  stationary bounded",
       "nonstationary_unbounded": "II-ns  non-stationary unbounded",
       "nonstationary_bounded": "I-ns  non-stationary bounded"}
COL = {"stationary_unbounded": "#d95f02", "stationary_bounded": "#7570b3",
       "nonstationary_unbounded": "#1f77b4", "nonstationary_bounded": "#2ca02c"}
LS = {"stationary_unbounded": ":", "stationary_bounded": ":",
      "nonstationary_unbounded": "-", "nonstationary_bounded": "-"}


def _latest(pattern: str) -> str:
    cands = sorted(glob.glob(os.path.join(DATA_PATH, pattern)), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(
            f"no sweep cache matching {pattern!r} in {DATA_PATH}; run the sweep module first.")
    return cands[-1]


def _bias_range(ds: xr.Dataset):
    err = ds.rv_est - ds.rv_true
    return (err.mean("seed", skipna=True),
            ds.rv_est.quantile(0.95, "seed", skipna=True)
            - ds.rv_est.quantile(0.05, "seed", skipna=True))


def _panel(ds: xr.Dataset, xc: str, xlabel: str, out: str, title: str):
    bias, width = _bias_range(ds)
    x = ds[xc].values
    rps = ds.rp.values.tolist()
    fig, axs = plt.subplots(2, len(rps), figsize=(9.2, 6.4), sharex=True, squeeze=False)
    for j, rp in enumerate(rps):
        for f in FITS:
            kw = dict(color=COL[f], ls=LS[f], lw=1.7,
                      marker=("o" if "nonstationary" in f else None), ms=3)
            axs[0, j].plot(x, bias.sel(fit=f, rp=rp), **kw)
            axs[1, j].plot(x, width.sel(fit=f, rp=rp), label=LAB[f], **kw)
        axs[0, j].axhline(0, color="grey", ls="--", lw=0.8)
        axs[0, j].set_title(f"RV{int(rp)}")
        axs[1, j].set_xlabel(xlabel)
        for ax in (axs[0, j], axs[1, j]):
            ax.grid(alpha=0.3); ax.margins(x=0)
    axs[0, 0].set_ylabel("bias [m]"); axs[1, 0].set_ylabel("5--95% range [m]")
    axs[1, 0].legend(fontsize=7, loc="upper left"); fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("  wrote", out)
    return bias, width


def make_trend_fig(out_dir: str | None = None) -> None:
    """``evt_ns_bias_range.pdf``: bias and range vs the upper-bound trend; prints the
    II-ns/I-ns range ratio at the representative 1 cm/yr trend."""
    out_dir = out_dir or _paper_img_path()
    nc = _latest("vary_nonstationary_*_ny_80_Nr_*_sigma_0.25.nc")
    print("  trend sweep:", os.path.basename(nc))
    dt = xr.open_dataset(nc)
    bias, width = _panel(dt, "trend", r"upper-bound trend, $dz^*/dt$ [m yr$^{-1}$]",
                         os.path.join(out_dir, "evt_ns_bias_range.pdf"),
                         "Non-stationary EVT: bias and 5--95% range vs trend")
    it = int(np.argmin(np.abs(dt.trend.values - 0.01)))
    for rp in dt.rp.values:
        r = float(width.sel(fit="nonstationary_unbounded", rp=rp)[it]
                  / width.sel(fit="nonstationary_bounded", rp=rp)[it])
        print(f"    trend=0.01  RV{int(rp)}: I-ns bias "
              f"{float(bias.sel(fit='nonstationary_bounded', rp=rp)[it]):+.2f}  "
              f"range-ratio II/I={r:.2f}")


def make_sigma_fig(out_dir: str | None = None) -> None:
    """``evt_ns_sigma.pdf``: robustness to bound-level uncertainty; prints the sigma
    crossover per return period."""
    out_dir = out_dir or _paper_img_path()
    nc = _latest("sigma_robustness_*.nc")
    print("  sigma sweep:", os.path.basename(nc))
    dss = xr.open_dataset(nc)
    _panel(dss, "sigma", r"bound level uncertainty $\sigma_{z^*}$ [m]",
           os.path.join(out_dir, "evt_ns_sigma.pdf"),
           "Non-stationary EVT: robustness to bound uncertainty")
    w = dss.rv_est.quantile(0.95, "seed") - dss.rv_est.quantile(0.05, "seed")
    s = dss.sigma.values
    for rp in dss.rp.values:
        rI = w.sel(fit="nonstationary_bounded", rp=rp).values
        rII = w.sel(fit="nonstationary_unbounded", rp=rp).values
        below = rI < rII
        if below.all():
            cx = s[-1]
        elif not below[0]:
            cx = s[0]
        else:
            i = int(np.argmax(~below))
            cx = float(np.interp(0, [rI[i - 1] - rII[i - 1], rI[i] - rII[i]], [s[i - 1], s[i]]))
        print(f"    sigma crossover RV{int(rp)}: ~{cx:.2f} m")


def main() -> None:
    print("evt_ns_bias_range.pdf:")
    make_trend_fig()
    print("evt_ns_sigma.pdf:")
    make_sigma_fig()


if __name__ == "__main__":
    main()
