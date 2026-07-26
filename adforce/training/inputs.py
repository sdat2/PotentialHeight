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
import math
from pathlib import Path
import shutil
from datetime import timedelta
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
        wind_forcing = BestTrackForcing(
            Path(os.path.join(output_dir, "atcf.txt")), nws=20
        )  # NWS=20 for GAHM
        mesh.add_forcing(wind_forcing)

    # 3. Calculate Simulation Window (spinup extends the window BACKWARD so
    # tides equilibrate before the storm; the GAHM fort.22 only covers the
    # storm lifetime, so verify met behaviour with spinup on a single storm
    # before fleet use)
    sim_start, sim_end = calculate_simulation_window(
        storm, extra_days=0, spinup_days=spinup_days
    )

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
    shutil.copy(fort13_path, os.path.join(output_dir, "fort.13"))

    print(
        f"Successfully generated inputs for {storm.name} {storm.year} in {output_dir}"
    )
