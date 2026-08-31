"""ADCIRC input-deck generation (adcircpy-heavy) for the training runs.

Split out of ``adforce/generate_training_data.py``. ``generate_adcirc_inputs``
writes, per storm run directory: ``pre_aswip_fort.22`` and ``atcf.txt`` (via
``adforce.training.atcf``), ``fort.14``/``fort.15``/``fort.22``/``driver.sh``
(via adcircpy's ``AdcircRun.write``) and a copy of the static ``fort.13``.

This module imports adcircpy at module level, so it is kept out of pytest's
``--doctest-modules`` collection (see pytest.ini) and out of any light import
paths.
"""

from typing import Dict
import os
import re
import math
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import xarray as xr
from adcircpy import AdcircMesh, AdcircRun
from adcircpy.forcing.winds import BestTrackForcing

from ..constants import SETUP_PATH
from .storms import Storm, calculate_simulation_window
from .atcf import convert_ibtracs_storm_to_aswip_input, convert_ibtracs_storm_to_atcf
from .cfl import calculate_cfl_timestep

FORT14_PATH = os.path.join(SETUP_PATH, "fort.14.mid")
FORT13_PATH = os.path.join(SETUP_PATH, "fort.13.mid")


class CustomAdcircRun(AdcircRun):
    """
    An AdcircRun subclass that overrides the default namelists
    to match the "notide-example" (File 2) configuration.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the parent AdcircRun and the custom NRAMP storage.

        Args:
            *args: Positional arguments forwarded to ``AdcircRun``.
            **kwargs: Keyword arguments forwarded to ``AdcircRun``.
        """
        # This initializes the parent AdcircRun class normally
        super().__init__(*args, **kwargs)
        # We add a private variable to store our custom NRAMP value
        self._custom_nramp = None

    @property
    def NRAMP(self) -> int:
        """
        Overrides the parent NRAMP property. If a custom value has been
        set, it returns that value. Otherwise, it falls back to the
        original logic of the parent class.
        """
        if self._custom_nramp is not None:
            return self._custom_nramp
        else:
            # This calls the original NRAMP logic from the Fort15 class
            return super().NRAMP

    @NRAMP.setter
    def NRAMP(self, value: int):
        """
        This is the new setter. It allows you to assign a value
        directly to the NRAMP property.
        """
        # You can add validation for the NRAMP value if you wish
        valid_nramp_values = [0, 1]
        if value not in valid_nramp_values:
            print(f"Warning: {value} is not a standard NRAMP value.")
        self._custom_nramp = int(value)

    @property
    def namelists(self) -> Dict[str, Dict[str, str]]:
        """
        Overrides the Fort15.namelists property.
        """
        # Get the default namelist dictionary from the parent class
        nlists = super().namelists
        # --- 1. Modify metControl ---
        # Change DragLawString from 'default' to 'Powell'
        # Note: The quotes are nested ('"Powell"') because the
        # namelist writer will strip one set.
        nlists["metControl"]["DragLawString"] = "'Powell'"

        # Change WindDragLimit (File 1: 0.0025, File 2: 0.0020)
        nlists["metControl"]["WindDragLimit"] = 0.0020

        # --- 2. Add owiWindNetcdf (if needed) ---
        # This namelist is specific to NWS=13 in your File 2.
        # You should check if your new run also uses NWS=13.
        if self.NWS == 13:
            # Get the forcing start date string in the right format
            start_str = self.forcing_start_date.strftime("%Y%m%d.%H%M%S")

            nlists["owiWindNetcdf"] = {
                "NWS13ColdStartString": f"'{start_str}'",
                "NWS13GroupForPowell": "2",
            }

        return nlists


def _pad_track_to(path: str, stamp: str) -> None:
    """Prepend copies of the first track record at 6-hourly stamps from
    ``stamp`` (YYYYMMDDHH) up to the record's own time, if the file starts
    later. A single back-dated record is NOT enough: GAHM's coverage check
    (nws20get) evidently assumes regularly spaced records, so a lone record
    6 days before the next one still fails with "not enough data". Each pad
    line string-replaces the unique 10-digit datetime token, so fixed-width
    (aswip) and comma (ATCF) formats both keep their layout."""
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return
    m = re.search(r"\b(\d{10})\b", lines[0])
    if m is None or m.group(1) <= stamp:
        return  # malformed or already covers the window start
    first_stamp = m.group(1)
    t = datetime.strptime(stamp, "%Y%m%d%H")
    t0 = datetime.strptime(first_stamp, "%Y%m%d%H")
    stamps = []
    while t < t0:
        stamps.append(t.strftime("%Y%m%d%H"))
        t += timedelta(hours=6)
    # ASWIP groups consecutive identical-position records into one "cycle"
    # (that is how multi-isotach cycles are encoded), so identical pads all
    # merge into cycle 1 and trip GAHM's 4-isotach limit. March the padded
    # storm eastward by 0.1 deg per record into its genesis point so every
    # pad is its own cycle; width-preserving replace keeps the fixed-format
    # aswip file aligned.
    mlon = re.search(r"\b(\d{2,4})([EW])\b", lines[0])
    pads = []
    n = len(stamps)
    for k, sstamp in enumerate(stamps):
        line = lines[0].replace(first_stamp, sstamp)
        if mlon:
            off = n - k  # earliest pad is furthest from the genesis point
            val = int(mlon.group(1)) + (-off if mlon.group(2) == "W" else off)
            token = str(val).rjust(len(mlon.group(1))) + mlon.group(2)
            line = line.replace(mlon.group(0), token)
        pads.append(line)
    lines[:0] = pads
    with open(path, "w") as f:
        f.writelines(lines)
    return len(pads)


def _renumber_aswip_tau(path: str, n_pads: int, pad_hours_step: int = 6) -> None:
    """Rewrite the TAU column (cols 31-33, aswip FORMAT ``2x,i3`` after
    castType) as hours since the (padded) file start. ASWIP keys BOTH its
    cycle grouping and its time axis on this column
    (``cycleTime = iFcstInc*3600``; wind/aswip.F): verbatim-copied pads with
    TAU=0 collapse into one >4-"isotach" cycle, and an unshifted TAU range
    under-covers the run window (the real cause of every nws20get error in
    this saga). Multi-isotach records (identical original TAU) keep sharing
    a TAU, preserving their grouping."""
    with open(path) as f:
        lines = f.readlines()
    out = []
    shift = n_pads * pad_hours_step
    for i, line in enumerate(lines):
        if i < n_pads:
            tau = i * pad_hours_step  # each pad is its own single-line cycle
        else:
            # original records: shift, preserving shared TAUs (multi-isotach)
            tau = int(line[30:33]) + shift
        out.append(line[:30] + f"{tau:3d}" + line[33:])
    with open(path, "w") as f:
        f.writelines(out)


def generate_adcirc_inputs(
    storm: Storm,
    storm_ds: xr.Dataset,
    output_dir: str,
    recalculate_timestep=False,
    recommended_dt=1.0,
    resolution: str = "mid",
    wind: bool = True,
    tides: bool = False,
    spinup_days: float = 0.0,
    mannings_n=None,
    friction_cf=None,
) -> None:
    """
    Generates a complete set of ADCIRC inputs for a single storm.
    This creates:
    - pre_aswip_fort.22 (from IBTrACS data)
    - atcf.txt (from IBTrACS data)
    - fort.15 (via adcircpy)
    - fort.13 (copied)

    Side effects: creates ``output_dir`` if needed and writes the input deck
    into it (``driver.write`` also produces fort.14, fort.22 and driver.sh);
    prints progress.

    Args:
        storm (Storm): Storm object containing storm metadata.
        storm_ds (xr.Dataset): xarray.Dataset for the storm from IBTrACS.
        output_dir (str): Directory to save the generated ADCIRC input files.
        recalculate_timestep (bool): Whether to recalculate the timestep
        recommended_dt (float): Timestep in seconds used when
            ``recalculate_timestep`` is False (or as the fallback on CFL
            failure). Defaults to 1.0.

    Returns:
        None
    """
    # 1. Load Mesh (per-resolution: fort.14.{mid,low}; fort.13.low is the
    # nearest-neighbour resample of fort.13.mid -- see
    # rerun/adcirc/make_fort13_low.py -- so friction/datum physics are held
    # fixed across the mesh-resolution sensitivity sweep)
    fort14_path = os.path.join(os.path.dirname(FORT14_PATH), f"fort.14.{resolution}")
    fort13_path = os.path.join(os.path.dirname(FORT13_PATH), f"fort.13.{resolution}")
    mesh = AdcircMesh.open(fort14_path, crs="epsg:4326")
    if recalculate_timestep:
        try:
            # You might adjust maxvel based on the storm's intensity if needed
            # For a major hurricane like Katrina, 10.0 m/s might be a safer estimate
            recommended_dt = calculate_cfl_timestep(mesh, cfl_target=0.7, maxvel=10.0)

            # Optional: Round down slightly for safety margin or to nicer number
            recommended_dt = (
                math.floor(recommended_dt * 10) / 10
            )  # e.g., round down to nearest 0.1s

            # Ensure dt is not excessively small (e.g., less than 0.1s might be problematic)
            if recommended_dt < 0.1:
                print(
                    f"Warning: Calculated dt ({recommended_dt:.4f}s) is very small. Using 0.1s."
                )
                recommended_dt = 0.1

        except (ValueError, AttributeError) as e:
            print(f"Error calculating CFL timestep: {e}. Defaulting to 2.5s.")
            recommended_dt = recommended_dt  # Fallback timestep    print(f"✅ Calculated recommended timestep (CFL=0.7): {recommended_dt:.2f} seconds")
    else:
        recommended_dt = recommended_dt  # Default timestep

    os.makedirs(output_dir, exist_ok=True)

    # 2a. Tidal forcing (for the tide-surge-interaction runs). HAMTIDE is the
    # only adcircpy tidal database that needs no local file (fetched over
    # OPeNDAP at write time); the eight major constituents match the
    # fort.15.*.tide templates used by the idealized decks. Nodal factors and
    # equilibrium arguments are computed by adcircpy for this run's window.
    if tides:
        from adcircpy.forcing.tides import Tides
        from adcircpy.forcing.tides.tides import TidalSource

        tidal_forcing = Tides(tidal_source=TidalSource.HAMTIDE)
        for c in ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"):
            tidal_forcing.use_constituent(c)
        mesh.add_forcing(tidal_forcing)

    # Simulation window first: the wind block needs sim_start to pad the
    # track files back over the tidal spinup (GAHM refuses a run window not
    # fully covered by fort.22 -- "nws20get: There aren't enough data").
    sim_start, sim_end = calculate_simulation_window(
        storm, extra_days=0, spinup_days=spinup_days
    )

    # 2b. Wind forcing (GAHM). Omitted for tide-only runs -> NWS=0 fort.15.
    if wind:
        convert_ibtracs_storm_to_aswip_input(
            ds=storm_ds,
            output_atcf_path=os.path.join(output_dir, "pre_aswip_fort.22"),
        )
        # This is the file adcircpy will read
        convert_ibtracs_storm_to_atcf(
            ds=storm_ds,
            output_atcf_path=os.path.join(output_dir, "atcf.txt"),
        )
        # adcircpy reads atcf.txt to get metadata for fort.15
        if spinup_days > 0:
            # prepend genesis-record copies over the spinup (6-hourly,
            # walking into the genesis point) so GAHM has met coverage of
            # the full run window, then renumber the aswip file's TAU column
            # -- aswip's cycle grouping AND time axis (see _renumber_aswip_tau)
            stamp = sim_start.strftime("%Y%m%d%H")
            aswip_path = os.path.join(output_dir, "pre_aswip_fort.22")
            n_pads = _pad_track_to(aswip_path, stamp)
            _pad_track_to(os.path.join(output_dir, "atcf.txt"), stamp)
            if n_pads:
                _renumber_aswip_tau(aswip_path, n_pads)
        wind_forcing = BestTrackForcing(
            Path(os.path.join(output_dir, "atcf.txt")), nws=20
        )  # NWS=20 for GAHM
        mesh.add_forcing(wind_forcing)

    # 3. (simulation window computed above, before the wind forcing)

    # 4. Configure AdcircRun Driver
    driver = CustomAdcircRun(
        mesh=mesh,
        start_date=sim_start,
        end_date=sim_end,  # spinup_duration=spinup
        # spinup_time=timedelta(days=1.0),
    )

    # --- Customize fort.15 parameters ---
    driver.timestep = recommended_dt
    driver.ICS = 20  # coordinate system (24 seems to have a big instability bug?)
    driver.ITITER = -1
    driver.CONVCR = 1.0e-7
    # longer ramp when tides are on: ramps the boundary/potential forcing
    # over the spinup window instead of shocking the basin
    driver.DRAMP = 2.0 if tides else 1.0
    driver.NRAMP = 1

    # set the output timestep in the netcdfs
    driver.set_elevation_surface_output(sampling_rate=timedelta(seconds=200))
    driver.set_velocity_surface_output(sampling_rate=timedelta(seconds=200))
    driver.set_meteorological_surface_output(sampling_rate=timedelta(seconds=200))

    # Set NWS=20 in fort.15
    # driver.fort15.NWS = 20 # GAHM model

    # 5. Write files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver.write(output_dir, overwrite=True)  # This creates fort.15

    # 6. Copy static files
    # fort.14 is copied by driver.write()
    if mannings_n is not None:
        # VERIFIED NO-OP (2026-08-17 GCP sweep: three cells, byte-identical
        # outputs): the generated fort.15 has NWP=0, so ADCIRC never reads
        # fort.13. Refuse rather than silently burn compute; the live knob is
        # friction_cf (fort.15 CF). See adforce/eval/tidal_diagnosis.md.
        raise ValueError(
            "mannings_n is a no-op: decks are generated with NWP=0 (fort.13 "
            "unread). Use friction_cf, or implement NWP=1 nodal attributes."
        )
    shutil.copy(fort13_path, os.path.join(output_dir, "fort.13"))
    if friction_cf is not None:
        # bottom-friction sensitivity (adforce/eval/tidal_diagnosis.md):
        # rewrite the NOLIBF=2 hybrid-friction CF in the generated fort.15
        from adforce.fort15 import write_friction_cf

        f15 = os.path.join(output_dir, "fort.15")
        write_friction_cf(f15, f15, float(friction_cf))

    print(
        f"Successfully generated inputs for {storm.name} {storm.year} in {output_dir}"
    )
