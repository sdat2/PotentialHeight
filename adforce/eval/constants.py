"""Constants and configuration for the ``adforce.eval`` (observational comparison) module.

This module validates the historical ADCIRC storm-surge simulations (the SurgeNet
training set, published on Hugging Face) against de-tided NOAA CO-OPS tide-gauge
observations. See :mod:`adforce.eval.validate`.
"""

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
SRC_PATH = Path(__file__).parent
PROJ_PATH = SRC_PATH.parent.parent  # adforce/eval/ -> repo root
DATA_PATH = os.path.join(PROJ_PATH, "data")
COMP_DATA_PATH = os.path.join(DATA_PATH, "comp")  # caches (git-ignored)
HF_STORM_CACHE = os.path.join(COMP_DATA_PATH, "hf_storms")  # downloaded storm netCDFs
COOPS_CACHE = os.path.join(COMP_DATA_PATH, "coops_cache")  # raw CO-OPS responses
TS_CACHE = os.path.join(
    COMP_DATA_PATH, "ts_cache"
)  # per-storm de-tided time series (Parquet)
FIGURE_PATH = os.path.join(PROJ_PATH, "img", "comp")  # quick-look PNGs (tracked)
OUT_PATH = os.path.join(COMP_DATA_PATH, "out")  # summary tables / metrics


# Paper artifacts. The worstsurge code repo is symlinked into the thesis tree
# (thesis/worstsurge -> worstsurge), so the module's own parent is NOT the thesis
# root: the Environmental Data Science paper builds from a separate <thesis> that
# holds paper/appendix.tex and img/. We write the final PDFs and the generated
# LaTeX table straight there, so ``python -m adforce.eval.validate`` reproduces exactly
# what the paper \includegraphics/\inputs -- no manual copy/convert step that
# could silently drift. The root is located by looking for paper/appendix.tex
# (override with WORSTSURGE_PAPER_ROOT); absent it, we fall back to module dirs.
def _find_paper_root():
    env = os.environ.get("WORSTSURGE_PAPER_ROOT")
    candidates = ([Path(env)] if env else []) + [
        Path.home() / "thesis",
        PROJ_PATH.parent,
        PROJ_PATH,
    ]
    for c in candidates:
        if (c / "paper" / "appendix.tex").is_file():
            return c
    return None


PAPER_ROOT = _find_paper_root()
PAPER_IMG_PATH = str(PAPER_ROOT / "img") if PAPER_ROOT else FIGURE_PATH
PAPER_TEX_PATH = str(PAPER_ROOT / "paper") if PAPER_ROOT else OUT_PATH

for _p in (
    COMP_DATA_PATH,
    HF_STORM_CACHE,
    COOPS_CACHE,
    TS_CACHE,
    FIGURE_PATH,
    OUT_PATH,
    PAPER_IMG_PATH,
    PAPER_TEX_PATH,
):
    os.makedirs(_p, exist_ok=True)

# --- data sources ----------------------------------------------------------
# Historical ADCIRC simulations (228 IBTrACS N-Atlantic landfalling TCs, 1980-2024,
# EC95d mesh, NWS=20 GAHM forcing, tides excluded -> output is pure surge).
HF_REPO = "sdat2/surgenet-train"

# NOAA CO-OPS metadata + data-getter endpoints.
COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

# --- gauge panel -----------------------------------------------------------
# NW Gulf of Mexico box (Texas coast -> Florida panhandle): lon (min,max), lat (min,max).
# Covers the New Orleans and Galveston study regions of the three-city paper.
GAUGE_BOX = dict(lon=(-97.6, -84.0), lat=(27.3, 30.9))
# Florida-peninsula box (Key West -> Fernandina Beach, both coasts), covering
# the Miami study region, for the Atlantic/Florida storms added when the paper
# moved from the single New Orleans site to the three-city comparison
# (New Orleans / Galveston / Miami). A separate box (rather than widening
# GAUGE_BOX) keeps the Gulf storms' time-series cache valid: the box is part of
# the cache tag (see adforce.eval.validate._ts_cache_tag).
FLORIDA_BOX = dict(lon=(-82.3, -79.7), lat=(24.4, 30.8))

# Storms scored against FLORIDA_BOX instead of the (default Gulf) GAUGE_BOX.
FLORIDA_STORMS = {
    "Frances 2004",
    "Jeanne 2004",
    "Matthew 2016",
    "Irma 2017",
    "Nicole 2022",
}


def box_for(storm: str) -> dict:
    """Gauge-selection box for a storm (Florida storms use FLORIDA_BOX)."""
    return FLORIDA_BOX if storm in FLORIDA_STORMS else GAUGE_BOX


# Storm display-name -> Hugging Face filename within HF_REPO.
# Well-observed, surge-relevant Gulf landfalls 2005-2023, plus Florida
# peninsula/Atlantic events 2004-2022 for the Miami region (the 228-storm
# archive contains no Andrew 1992, Wilma 2005 or Ian 2022). Keep this dict in
# chronological order: the report table iterates it in insertion order.
STORMS = {
    "Frances 2004": "86_FRANCES_2004.nc",
    "Jeanne 2004": "99_JEANNE_2004.nc",
    "Katrina 2005": "152_KATRINA_2005.nc",
    "Rita 2005": "166_RITA_2005.nc",
    "Gustav 2008": "25_GUSTAV_2008.nc",
    "Ike 2008": "29_IKE_2008.nc",
    "Isaac 2012": "245_ISAAC_2012.nc",
    "Matthew 2016": "202_MATTHEW_2016.nc",
    "Harvey 2017": "239_HARVEY_2017.nc",
    "Irma 2017": "242_IRMA_2017.nc",
    "Nate 2017": "249_NATE_2017.nc",
    "Michael 2018": "22_MICHAEL_2018.nc",
    "Barry 2019": "53_BARRY_2019.nc",
    "Laura 2020": "140_LAURA_2020.nc",
    "Delta 2020": "161_DELTA_2020.nc",
    "Ida 2021": "220_IDA_2021.nc",
    "Nicholas 2021": "227_NICHOLAS_2021.nc",
    "Nicole 2022": "290_NICOLE_2022.nc",
    "Idalia 2023": "32_IDALIA_2023.nc",
}

# Documented instrument failures at landfall: the gauge was destroyed or capped
# mid-storm, so the observed peak is truncated. Excluded from skill scores -- the
# model captures the surge that broke the gauge (appears as spurious over-prediction).
# (storm display-name, CO-OPS station id)
KNOWN_FAILED = {
    ("Ida 2021", "8761724"),  # Grand Isle, LA
    ("Ida 2021", "8762075"),  # Port Fourchon, Belle Pass, LA
}

# Events that are poor surge tests (rainfall / inland-flood dominated, small surge).
POOR_SURGE_EVENTS = {"Harvey 2017"}

# Gauge time series used as the example-panel figure in the paper
# (img/comp_val_examples.pdf). Each entry is (storm, gauge-name substring); the
# substring is matched case-insensitively against the CO-OPS station name.
EXAMPLE_PANELS = [
    ("Katrina 2005", "Dauphin Island"),
    ("Katrina 2005", "Pilots Station East"),
    ("Ida 2021", "Shell Beach"),
    ("Ida 2021", "Grand Isle"),
    ("Ike 2008", "Galveston Pier 21"),  # Galveston study region
    ("Irma 2017", "Virginia Key"),  # Miami study region
    ("Ida 2021", "Amerada Pass"),  # LAWMA, left-of-track set-down
    ("Laura 2020", "Calcasieu Pass"),
]

# --- node selection --------------------------------------------------------
MAX_NODE_DEG = 0.12  # max distance (deg) from gauge to an acceptable mesh node
WET_MIN_M = 0.3  # node must stay wetter than this all storm (avoids drying spikes)
KNN = 60  # nearest-node candidates to search through

# --- "clean" gauge-storm pair filter --------------------------------------
MAX_TIMING_HR = 6.0  # |sim peak time - obs peak time| must be within this
MIN_OBS_PEAK_M = 0.4  # observed peak residual must exceed this (meaningful surge)

# --- time-series skill -----------------------------------------------------
# The peak-to-peak score collapses each gauge to one number; we also score the
# full storm hydrograph by interpolating the (2-hourly) simulated surge onto the
# (hourly) observed residual times over their overlap and taking the temporal
# correlation and RMSE. Requires at least this many overlapping hourly samples.
TS_MIN_OVERLAP = 24

# --- uncertainty -----------------------------------------------------------
N_BOOTSTRAP = 2000  # resamples for percentile CIs on pooled bias/RMSE/r
BOOTSTRAP_SEED = 0  # fixed so reported CIs are reproducible

# Minimum hourly samples in a calendar year for a stable utide harmonic fit.
UTIDE_MIN_SAMPLES = 2000

# --- annual-maximum pipeline (adforce.eval.annual_max) ------------------------------
ANNUAL_MAX_CACHE = os.path.join(
    COMP_DATA_PATH, "annual_max"
)  # residuals + annual maxima
os.makedirs(ANNUAL_MAX_CACHE, exist_ok=True)
AM_START_YEAR = 1980  # default year range for annual-maximum extraction
AM_END_YEAR = 2025
# A year only contributes an annual maximum if at least this fraction of its
# hours have data. Below this the annual maximum is likely biased low (the true
# peak may fall in a gap) and the harmonic fit degrades. 0.8 keeps most usable
# years at long-record Gulf gauges while excluding badly broken ones.
MIN_YEAR_COVERAGE = 0.8
COOPS_RETRIES = 3  # re-requests for a year whose cached response is a transport failure
COOPS_SLEEP_S = 0.5  # polite pause after each live (non-cached) CO-OPS request
