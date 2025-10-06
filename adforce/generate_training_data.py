"""Generate ADCIRC input files for historical U.S. landfalling storms."""

# --- MASTER SCRIPT: 01_generate_inputs.py ---
from typing import Tuple
import os
import numpy as np
import pandas as pd
import xarray as xr
import shutil
from datetime import datetime, timedelta
from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.forcing.winds import BestTrackForcing
from tcpips.ibtracs import na_landing_tcs

# --- Configuration ---
# BASE_MODEL_DIR = './base_model/'
from .constants import SETUP_PATH, PROJ_PATH

RUNS_PARENT_DIR = os.path.join(PROJ_PATH, "runs")
os.makedirs(RUNS_PARENT_DIR, exist_ok=True)

FORT14_PATH = os.path.join(SETUP_PATH, "fort.14.mid")
FORT13_PATH = os.path.join(SETUP_PATH, "fort.13.mid")

# --- Function Definitions (from Section 2) ---


def calculate_simulation_window(
    storm, spinup_days: float = 10.0, extra_days: float = 2.0
) -> Tuple[datetime, datetime, timedelta]:
    """
    Calculate the simulation start and end dates based on storm data.

    Args:
    - storm: Storm object with 'time' attribute (list of datetime objects).
    - spinup_days: Number of days for model spin-up before storm start.
    - extra_days: Number of days to continue simulation after storm end.

    Returns:
    - simulation_start_date: Start date for the simulation (datetime).
    - simulation_end_date: End date for the simulation (datetime).
    - spinup_duration: Duration of the spin-up period (timedelta).
    """
    storm_start_time = storm.time
    storm_end_time = storm.time[-1]
    spinup_duration = timedelta(days=spinup_days)
    start_date = storm_start_time
    simulation_start_date = start_date - spinup_duration
    simulation_end_date = storm_end_time + timedelta(days=extra_days)
    return simulation_start_date, simulation_end_date, spinup_duration


def generate_adcirc_inputs(storm, output_dir):
    """
    Generates a complete set of ADCIRC inputs for a single storm.
    """
    # 1. Load Mesh
    mesh = AdcircMesh.open(FORT14_PATH, crs="epsg:4326")

    # 2. Add Forcings
    tidal_forcing = Tides()
    tidal_forcing.use_all()
    mesh.add_forcing(tidal_forcing)

    wind_forcing = BestTrackForcing(storm.id, nws=20)  # NWS=20 for GAHM
    mesh.add_forcing(wind_forcing)

    # 3. Calculate Simulation Window
    sim_start, sim_end, spinup = calculate_simulation_window(storm)

    # 4. Configure AdcircRun Driver
    driver = AdcircRun(
        mesh=mesh, start_date=sim_start, end_date=sim_end, spinup_duration=spinup
    )

    # --- Customize fort.15 parameters ---
    driver.timestep = 2.0
    driver.DRAMP = 10.0
    driver.set_elevation_surface_output(sampling_rate=timedelta(minutes=30))
    driver.set_velocity_surface_output(sampling_rate=timedelta(minutes=30))

    # 5. Write files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    driver.write(output_dir, overwrite=True)

    # 6. Copy static files
    shutil.copy(FORT14_PATH, output_dir)

    print(
        f"Successfully generated inputs for {storm.name} {storm.year} in {output_dir}"
    )


# Assuming your Storm class is defined as before:
class Storm:
    def __init__(self, sid, name, time):
        self.id = sid.decode()
        self.name = name.decode()
        self.time = time
        if self.time:
            self.year = self.time[0].year
        else:
            self.year = None


# --- DEBUGGING CODE ---
# We will just check the first storm to see what's happening.

if __name__ == "__main__":
    target_storms_ds = na_landing_tcs()
    target_storms = []
    for i in range(len(target_storms_ds.sid)):
        # Isolate the raw numpy array of times for the storm
        raw_times = target_storms_ds.time[i].values

        # Use pandas to reliably convert the array to datetime objects
        # 'coerce' will automatically handle 'NaT' values by turning them into NaT
        pd_timestamps = pd.to_datetime(raw_times, errors="coerce")

        # Drop any NaT values and convert the result to a list of
        # standard Python datetime objects.
        time_list = pd_timestamps.dropna().to_pydatetime().tolist()
        # --- END FIX ---

        storm = Storm(
            sid=target_storms_ds.sid[i].item(),
            name=target_storms_ds.name[i].item(),
            time=time_list,
        )
        target_storms.append(storm)

    print(f"Found {len(target_storms)} U.S. landfalling storms in IBTrACS.")

    # 2. Loop and generate inputs for each storm
    for storm in target_storms:
        storm_name_safe = storm.name.upper().replace(" ", "_")
        run_directory = os.path.join(RUNS_PARENT_DIR, f"{storm_name_safe}_{storm.year}")

        try:
            print(
                f"Generating inputs for {storm.name} {storm.year}, {storm.id} {type(storm.id)}..."
            )
            generate_adcirc_inputs(storm, run_directory)
        except Exception as e:
            print(f"!!! FAILED to generate inputs for {storm.name} {storm.year}: {e}")
            print(" traceback:")
            import traceback

            traceback.print_exc()
