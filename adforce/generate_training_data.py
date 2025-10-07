"""Generate ADCIRC input files for historical U.S. landfalling storms."""

# --- MASTER SCRIPT: 01_generate_inputs.py ---
from typing import Tuple
import os
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import shutil
from datetime import datetime, timedelta
from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.forcing.winds import BestTrackForcing
from stormevents.nhc import VortexTrack

from tcpips.ibtracs import na_landing_tcs


# --- Configuration ---
# BASE_MODEL_DIR = './base_model/'
from .constants import SETUP_PATH, PROJ_PATH

RUNS_PARENT_DIR = os.path.join(PROJ_PATH, "runs")
os.makedirs(RUNS_PARENT_DIR, exist_ok=True)

FORT14_PATH = os.path.join(SETUP_PATH, "fort.14.mid")
FORT13_PATH = os.path.join(SETUP_PATH, "fort.13.mid")

# --- Function Definitions (from Section 2) ---
# Assuming your Storm class is defined as before:
class Storm:
    # I hate classes, but this was to mirror tropycal
    def __init__(self, sid, name, time):
        self.id = sid.decode()
        self.name = name.decode()
        self.time = time
        if self.time:
            self.year = self.time[0].year
        else:
            self.year = None


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
    storm_start_time = storm.time[0]
    storm_end_time = storm.time[-1]
    spinup_duration = timedelta(days=spinup_days)
    start_date = storm_start_time
    simulation_start_date = start_date - spinup_duration
    simulation_end_date = storm_end_time + timedelta(days=extra_days)
    return simulation_start_date, simulation_end_date, spinup_duration



def convert_ibtracs_netcdf_to_enriched_atcf(
    ds: xr.Dataset,
    output_atcf_path: str,
    storm_name: str,
    storm_year: int
):
    """
    Loads IBTrACS NetCDF data and creates a complete ATCF file, including
    asymmetry (wind radii) and size (RMW, POCI, ROCI) information.

    Args:
        netcdf_path (Path): Path to the IBTrACS .nc file.
        output_atcf_path (Path): Path to save the new .txt ATCF file.
        storm_name (str): The name of the storm to extract (e.g., 'KATRINA').
        storm_year (int): The year/season of the storm (e.g., 2005).
    """

    # Define all the variables we want to keep for the enriched format
    keep_vars = [
        'name', 'season', 'number', 'basin', 'time',
        'usa_lat', 'usa_lon', 'usa_wind', 'usa_pres', 'usa_status',
        'usa_r34', 'usa_r50', 'usa_r64',  # Wind Radii for asymmetry
        'usa_rmw', 'usa_poci', 'usa_roci'   # Storm structure/size
    ]


    filtered_ds = ds[keep_vars]
    df_all = filtered_ds.to_dataframe()

    df_all.reset_index(inplace=True)

    # Decode byte strings and strip whitespace
    for col in ['name', 'basin', 'usa_status']:
        if col in df_all.columns:
            df_all[col] = df_all[col].str.decode('utf-8').str.strip().str.replace('\x00', '')

    storm_df = df_all

    print("Formatting data into enriched ATCF structure...")
    atcf_lines = []
    storm_number = storm_df['number'].iloc[0]

    for _, row in storm_df.iterrows():
        # --- Part 1: Basic Identifiers (same as before) ---
        atcf_basin = row['basin'].ljust(2)
        atcf_cyclone_num = str(storm_number).zfill(2)
        atcf_datetime = row['time'].strftime('%Y%m%d%H')
        atcf_technique = 'BEST'.rjust(4)

        # --- Part 2: Core Storm State (same as before) ---
        lat_val = int(abs(row['usa_lat']) * 10)
        lat_hem = 'N' if row['usa_lat'] >= 0 else 'S'
        atcf_lat = f"{lat_val}{lat_hem}".rjust(5)

        lon_val = int(abs(row['usa_lon']) * 10)
        lon_hem = 'W' if row['usa_lon'] < 0 else 'E'
        atcf_lon = f"{lon_val}{lon_hem}".rjust(6)

        atcf_wind = str(int(np.nan_to_num(row['usa_wind']))).rjust(4)
        atcf_pres = str(int(np.nan_to_num(row['usa_pres']))).rjust(5)
        atcf_status = row['usa_status'].ljust(3)

        # --- Part 3: Enriched Data for Asymmetry and Size ---
        # The 'quadrant' dimension becomes an index (0, 1, 2, 3) in the DataFrame
        # We extract them in NE, SE, SW, NW order
        r34_ne = int(np.nan_to_num(row['usa_r34'].iloc[0]))
        r34_se = int(np.nan_to_num(row['usa_r34'].iloc[1]))
        r34_sw = int(np.nan_to_num(row['usa_r34'].iloc[2]))
        r34_nw = int(np.nan_to_num(row['usa_r34'].iloc[3]))

        r50_ne = int(np.nan_to_num(row['usa_r50'].iloc[0]))
        r50_se = int(np.nan_to_num(row['usa_r50'].iloc[1]))
        r50_sw = int(np.nan_to_num(row['usa_r50'].iloc[2]))
        r50_nw = int(np.nan_to_num(row['usa_r50'].iloc[3]))

        r64_ne = int(np.nan_to_num(row['usa_r64'].iloc[0]))
        r64_se = int(np.nan_to_num(row['usa_r64'].iloc[1]))
        r64_sw = int(np.nan_to_num(row['usa_r64'].iloc[2]))
        r64_nw = int(np.nan_to_num(row['usa_r64'].iloc[3]))

        atcf_poci = str(int(np.nan_to_num(row['usa_poci']))).rjust(5)
        atcf_roci = str(int(np.nan_to_num(row['usa_roci']))).rjust(4)
        atcf_rmw = str(int(np.nan_to_num(row['usa_rmw']))).rjust(4)

        # --- Part 4: Build the full ATCF line ---
        # Note the extra fields for RADII, RMW, POCI, ROCI etc.
        line = (f"{atcf_basin}, {atcf_cyclone_num}, {atcf_datetime},{atcf_technique},"
                f"{atcf_lat},{atcf_lon},{atcf_wind},{atcf_pres}, {atcf_status}, "
                f"{'34'.rjust(3)}, {'NEQ'.rjust(4)}, {str(r34_ne).rjust(4)}, "
                f"{str(r34_se).rjust(4)}, {str(r34_sw).rjust(4)}, {str(r34_nw).rjust(4)},"
                f"{atcf_poci},{atcf_roci},{atcf_rmw}, "
                f"{'50'.rjust(3)}, {'NEQ'.rjust(4)}, {str(r50_ne).rjust(4)}, "
                f"{str(r50_se).rjust(4)}, {str(r50_sw).rjust(4)}, {str(r50_nw).rjust(4)}, "
                f"{'64'.rjust(3)}, {'NEQ'.rjust(4)}, {str(r64_ne).rjust(4)}, "
                f"{str(r64_se).rjust(4)}, {str(r64_sw).rjust(4)}, {str(r64_nw).rjust(4)}")
        atcf_lines.append(line)

    with open(output_atcf_path, 'w') as f:
        f.write('\n'.join(atcf_lines))
    print(f"✅ Successfully created ENRICHED ATCF file at '{output_atcf_path}'")


def generate_adcirc_inputs(storm: Storm, storm_ds: xr.Dataset, output_dir: str):
    """
    Generates a complete set of ADCIRC inputs for a single storm.
    """
    # 1. Load Mesh
    mesh = AdcircMesh.open(FORT14_PATH, crs="epsg:4326")

    # 2. Add Forcings -- no tides
    # tidal_forcing = Tides()
    # tidal_forcing.use_all()
    # mesh.add_forcing() #tidal_forcing)

    # --- Inside your generate_adcirc_inputs function ---

    # 1. Use the parent class to find the storm and create a track object
    # This step converts ('NICOLE', 2022) into a track object with the correct ID ('AL152022')
    # try:
    #     track = VortexTrack.from_storm_name(name=storm.name.lower(), year=storm.year)
    # except Exception as e:
    #     print(f"Could not find NHC data for {storm.name} {storm.year}: {e}")
    #     import traceback

    #     traceback.print_exc()
    #     # Decide how to handle this - skip the storm or raise the error
    #     return

    convert_ibtracs_netcdf_to_enriched_atcf(
        ds=na_landing_tcs(),
        output_atcf_path=os.path.join(output_dir, "atcf.txt"),
        storm_name=storm.name,
        storm_year=storm.year
    )
    # 2. Pass the track object to the BestTrackForcing constructor
    # Now you can correctly specify the 'nws' parameter
    wind_forcing = BestTrackForcing(Path(os.path.join(output_dir, "atcf.txt")), nws=20)  # NWS=20 for GAHM
    #wind_forcing = BestTrackForcing(storm=track, nws=20
    # wind_forcing = BestTrackForcing(storm.id, nws=20)  # NWS=20 for GAHM
    mesh.add_forcing(wind_forcing)

    # 3. Calculate Simulation Window
    sim_start, sim_end, spinup = calculate_simulation_window(storm)

    # 4. Configure AdcircRun Driver
    driver = AdcircRun(
        mesh=mesh, start_date=sim_start, end_date=sim_end, #spinup_duration=spinup
    )

    # --- Customize fort.15 parameters ---
    driver.timestep = 2.0
    driver.DRAMP = 2.0
    driver.set_elevation_surface_output(sampling_rate=timedelta(minutes=30))
    driver.set_velocity_surface_output(sampling_rate=timedelta(minutes=30))

    # 5. Write files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver.write(output_dir, overwrite=True)

    # 6. Copy static files
    # shutil.copy(FORT14_PATH, output_dir)
    shutil.copy(FORT13_PATH, os.path.join(output_dir, "fort.13"))

    print(
        f"Successfully generated inputs for {storm.name} {storm.year} in {output_dir}"
    )



# --- DEBUGGING CODE ---
# We will just check the first storm to see what's happening.


def generate_all_storm_inputs():
    # python -m adforce.generate_training_data

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

    outputs = {"storm_id": [], "storm_name": [], "year": [], "num_timepoints": []}

    for storm in target_storms:
        outputs["storm_id"].append(storm.id)
        outputs["storm_name"].append(storm.name)
        outputs["year"].append(storm.year)
        outputs["num_timepoints"].append(len(storm.time))

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv("debug_storms.csv", index=False)

    print(f"Found {len(target_storms)} U.S. landfalling storms in IBTrACS.")

    # 2. Loop and generate inputs for each storm
    for i, storm in enumerate(target_storms[0:1]):
        storm_name_safe = storm.name.upper().replace(" ", "_")
        run_directory = os.path.join(RUNS_PARENT_DIR, f"{storm_name_safe}_{storm.year}")

        try:
            print(
                f"Generating inputs for {storm.name} {storm.year}, {storm.id} {type(storm.id)}..."
            )
            generate_adcirc_inputs(storm, target_storms_ds.isel(storm=i),run_directory)
        except Exception as e:
            print(f"!!! FAILED to generate inputs for {storm.name} {storm.year}: {e}")
            print(" traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    # python -m adforce.generate_training_data
    # generate_all_storm_inputs()
    # track = VortexTrack('AL112017')
    # print(track)
    generate_all_storm_inputs()
