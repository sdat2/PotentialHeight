"""Three-way tidal-amplitude check: model vs HAMTIDE vs NOAA harcon.

The friction sweep left a friction-INSENSITIVE diurnal residual (O1 ~ +34%,
K1 ~ +11%; ``tidal_diagnosis.md`` take-2). Two candidate sources remain:
the HAMTIDE basin solution itself (which forces the EC95d open boundary),
or our model's own basin response. Separate them by comparing, at every
panel gauge and for each constituent:

* ``amp_model``  -- utide fit of the tide-only run (``tideconst_*`` CSV),
* ``amp_noaa``   -- NOAA's published harmonic constituent (CO-OPS harcon),
* ``amp_ham``    -- HAMTIDE11a's assimilated field sampled at the gauge.

``model/ham ~ 1`` with ``ham/noaa ~ model/noaa`` => the model faithfully
reproduces an over-amplified HAMTIDE Gulf solution (database issue);
``ham/noaa ~ 1`` => the amplification is generated inside our model run.
Caveat: HAMTIDE is a 0.125-degree global solution -- coastal samples are
indicative, not exact; gauges with no wet HAMTIDE cell within ``max_deg``
are skipped.

Run (hydra; config root adforce/eval/config/tidedb_config.yaml)::

    python -m adforce.eval.tidedb model_csv=data/comp/out/tideconst_cf0.0025.csv
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from . import constants as C
from .extract import gauge_frame

#: HAMTIDE11a per-constituent files (adcircpy's tidal source for the runs)
HAMTIDE_URL = (
    "https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/hamtide/"
    "{c}.hamtide11a.nc"
)
HARCON_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
    "{sid}/harcon.json?units=metric"
)
HAMTIDE_CACHE = os.path.join(C.COMP_DATA_PATH, "hamtide")


def harcon(sid: str) -> Dict[str, float]:
    """NOAA published harmonic-constituent amplitudes [m] for a station
    (cached; empty dict when the station publishes none)."""
    import requests

    C.ensure_dirs()
    fp = os.path.join(C.COOPS_CACHE, f"{sid}_harcon.json")
    if os.path.exists(fp):
        data = json.load(open(fp))
    else:
        try:
            r = requests.get(HARCON_URL.format(sid=sid), timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = {}
        json.dump(data, open(fp, "w"))
    out = {}
    for hc in data.get("HarmonicConstituents", []):
        try:
            out[str(hc["name"]).strip().upper()] = float(hc["amplitude"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def hamtide_field(constituent: str, box=((260.0, 285.0), (22.0, 32.0))):
    """HAMTIDE amplitude field [m] for one constituent as (amp2d, lon, lat).

    Fetched as an OPeNDAP subset of the Gulf/Florida box (a few hundred KB,
    cached) rather than the full ~60 MB global file -- both to be polite and
    because laptop disk is tight.
    """
    import xarray as xr

    os.makedirs(HAMTIDE_CACHE, exist_ok=True)
    fp = os.path.join(HAMTIDE_CACHE, f"{constituent.lower()}_gulf.hamtide11a.nc")
    if not os.path.exists(fp):
        url = HAMTIDE_URL.replace("fileServer", "dodsC").format(c=constituent.lower())
        full = xr.open_dataset(url)
        lon_name = [v for v in ("LON", "lon") if v in full.coords or v in full][0]
        lat_name = [v for v in ("LAT", "lat") if v in full.coords or v in full][0]
        (lo0, lo1), (la0, la1) = box
        sub = full.sel({lon_name: slice(lo0, lo1), lat_name: slice(la0, la1)}).load()
        sub.to_netcdf(fp)
        full.close()
    ds = xr.open_dataset(fp)
    if "AMPL" in ds:
        amp = np.asarray(ds.AMPL.values, dtype=float)
    else:  # RE/IM decomposition
        amp = np.hypot(
            np.asarray(ds.RE.values, dtype=float), np.asarray(ds.IM.values, dtype=float)
        )
    lon = np.asarray(ds[[v for v in ("LON", "lon") if v in ds][0]].values)
    lat = np.asarray(ds[[v for v in ("LAT", "lat") if v in ds][0]].values)
    return amp / 100.0, lon, lat  # cm -> m


def sample_field(amp2d, lon, lat, glon: float, glat: float, max_deg: float = 0.5):
    """Nearest finite grid value within ``max_deg`` (HAMTIDE masks land)."""
    glon360 = glon % 360 if lon.max() > 180 else glon
    i = np.searchsorted(lat, glat)
    j = np.searchsorted(lon, glon360)
    n = int(np.ceil(max_deg / max(abs(np.diff(lat).mean()), 1e-6)))
    best, bestd = np.nan, np.inf
    for ii in range(max(0, i - n), min(len(lat), i + n + 1)):
        for jj in range(max(0, j - n), min(len(lon), j + n + 1)):
            v = amp2d[ii, jj]
            if np.isfinite(v) and v > 0:
                d = np.hypot(lat[ii] - glat, lon[jj] - glon360)
                if d < bestd and d <= max_deg:
                    best, bestd = float(v), d
    return best


def threeway_table(
    model_csv: str,
    constituents=("M2", "S2", "K1", "O1"),
    out: Optional[str] = None,
) -> pd.DataFrame:
    """Per (gauge, constituent): model, NOAA-harcon and HAMTIDE amplitudes.

    ``amp_model`` is the median over storms of the tideconst fit;
    gauges/constituents missing on any side are dropped row-wise.
    """
    fits = pd.read_csv(model_csv, dtype={"sid": str})
    model = (
        fits.groupby(["sid", "constituent"], as_index=False)
        .agg(gauge=("gauge", "first"), amp_model=("amp_model", "median"))
    )
    gauges = {str(r.sid): (r.lat, r.lon) for r in gauge_frame().itertuples()}
    fields = {c: hamtide_field(c) for c in constituents}
    rows = []
    for r in model.itertuples():
        if r.constituent not in constituents or r.sid not in gauges:
            continue
        glat, glon = gauges[r.sid]
        noaa = harcon(r.sid).get(r.constituent, np.nan)
        ham = sample_field(*fields[r.constituent], glon, glat)
        if not (np.isfinite(noaa) and noaa > 0.01 and np.isfinite(ham)):
            continue
        rows.append(
            dict(
                sid=r.sid,
                gauge=r.gauge,
                constituent=r.constituent,
                amp_model=r.amp_model,
                amp_noaa=round(noaa, 3),
                amp_ham=round(ham, 3),
                model_over_noaa=round(r.amp_model / noaa, 3),
                ham_over_noaa=round(ham / noaa, 3),
                model_over_ham=round(r.amp_model / ham, 3),
            )
        )
    df = pd.DataFrame(rows)
    if len(df):
        pooled = df.groupby("constituent").agg(
            n=("sid", "size"),
            model_over_noaa=("model_over_noaa", "median"),
            ham_over_noaa=("ham_over_noaa", "median"),
            model_over_ham=("model_over_ham", "median"),
        )
        print(pooled.reindex([c for c in constituents if c in pooled.index]).to_string(
            float_format=lambda v: f"{v:.2f}"
        ))
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


@hydra.main(version_base=None, config_path="config", config_name="tidedb_config")
def main(cfg: DictConfig) -> None:
    C.ensure_dirs()
    out = cfg.out or f"{C.OUT_PATH}/tidedb_threeway.csv"
    threeway_table(str(cfg.model_csv), constituents=tuple(cfg.constituents), out=out)


if __name__ == "__main__":
    main()
