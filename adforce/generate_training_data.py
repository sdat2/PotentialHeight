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


def _decode_char_array(char_da: xr.DataArray) -> str:
    """
    Helper function to convert an xarray DataArray of characters into a
    clean, stripped Python string.
    """
    # .values gets the numpy array, .tobytes() joins the characters,
    # .decode() converts to string, and .strip() cleans whitespace.
    return char_da.values.tobytes().decode("utf-8").strip()


def convert_ibtracs_storm_to_atcf(ds: xr.Dataset, output_atcf_path: str):
    """
    Loads IBTrACS ds for one tropical cyclone and creates a complete ATCF file, including
    asymmetry (wind radii) and size (RMW, POCI, ROCI) information.

    Args:
        ds (xr.Dataset): xarray.Dataset for a single storm from IBTrACS.
        output_atcf_path (str): Path to save the new .txt ATCF file.
    """
    atcf_lines = []

    # Find all valid time steps by checking where the 'time' variable is not a fill value
    valid_time_indices = np.where(~np.isnat(ds["time"].values))[0]

    # Get static storm information once
    storm_number = int(ds["number"].values)
    storm_name = _decode_char_array(ds["name"]).upper().strip(" ").strip('\x00')
    print(f"Generating ATCF for storm {storm_name} ({storm_number})...")
    # storm_name = "GERT"

    print(f"Processing {len(valid_time_indices)} valid time steps...")

    # Loop through only the valid indices
    for time_idx in valid_time_indices:
        # Select a single time slice using .isel()
        row = ds.isel(date_time=time_idx)

        # --- Part 1: Basic Identifiers ---
        # Use the helper function to decode character arrays
        atcf_basin = _decode_char_array(row["basin"])
        atcf_cyclone_num = str(storm_number)
        atcf_datetime = pd.to_datetime(row["time"].values).strftime("%Y%m%d%H")

        # --- Part 2: Core Storm State ---
        # .item() cleanly extracts the scalar value from a 0-dimensional array
        lat = row["usa_lat"].item()
        lon = row["usa_lon"].item()

        lat_val = int(abs(lat) * 10)
        lat_hem = "N" if lat >= 0 else "S"
        atcf_lat = f"{lat_val}{lat_hem}"#.rjust(5)

        lon_val = int(abs(lon) * 10)
        lon_hem = "W" if lon < 0 else "E"
        atcf_lon = f"{lon_val}{lon_hem}"# .rjust(6)

        atcf_wind = str(int(np.nan_to_num(row["usa_wind"].item())))#.rjust(4)
        atcf_pres = str(int(np.nan_to_num(row["usa_pres"].item())))# .rjust(5)
        atcf_status = _decode_char_array(row["usa_status"])#.ljust(3)

        # --- Part 3: Enriched Data for Asymmetry and Size ---
        # Select quadrant data using .isel() and get scalar with .item()
        atcf_poci = str(int(np.nan_to_num(row["usa_poci"].item())))#.rjust(5)
        atcf_roci = str(int(np.nan_to_num(row["usa_roci"].item())))#.rjust(4)
        atcf_rmw = str(int(np.nan_to_num(row["usa_rmw"].item())))#.rjust(4)
        radii_data = {
            "34": [
                int(np.nan_to_num(row["usa_r34"].isel(quadrant=q).item()))
                for q in range(4)
            ],
            "50": [
                int(np.nan_to_num(row["usa_r50"].isel(quadrant=q).item()))
                for q in range(4)
            ],
            "64": [
                int(np.nan_to_num(row["usa_r64"].isel(quadrant=q).item()))
                for q in range(4)
            ],
        }

        for rad, rad_values in radii_data.items():
            # Only write a line if there is actual radii data (sum > 0)
            # if sum(rad_values) > 0:
            if True:
                # --- Part 3: Build the correctly formatted ATCF line ---
                # This line now includes all required placeholder commas
                line = (
                    f"{atcf_basin},{atcf_cyclone_num},{atcf_datetime},{'00'},{'BEST'},{'0'},",
                    f"{atcf_lat},{atcf_lon},{atcf_wind},{atcf_pres},{atcf_status},",
                    f"{rad},{'NEQ'},{str(rad_values[0])},",
                    f"{str(rad_values[1])},{str(rad_values[2])},{str(rad_values[3])},",
                    f"{atcf_poci},{atcf_roci},{atcf_rmw},",
                    f" , , , , , , ,{storm_name}"
                )
                atcf_lines.append("".join(line))

    with open(output_atcf_path, "wt") as f:
        f.write("\n".join(atcf_lines))
    print(f"✅ Successfully created ENRICHED ATCF file at '{output_atcf_path}'")


def generate_adcirc_inputs(storm: Storm,
                           storm_ds: xr.Dataset,
                           output_dir: str,
                           stormtracks = False) -> None:
    """
    Generates a complete set of ADCIRC inputs for a single storm.

    Args:
        storm (Storm): Storm object containing storm metadata.
        storm_ds (xr.Dataset): xarray.Dataset for the storm from IBTrACS.
        output_dir (str): Directory to save the generated ADCIRC input files.
        stormtracks (bool): Whether to use NHC storm tracks directly.

    Returns:
        None
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
    # option 1 - from NHC data
    os.makedirs(output_dir, exist_ok=True)

    if stormtracks: # rely on the NHC server
        try:
            track = VortexTrack.from_storm_name(name=storm.name.lower(), year=storm.year)
            print(track.data)
            wind_forcing = BestTrackForcing(storm=track, nws=20)

        except Exception as e:
            print(f"Could not find NHC data for {storm.name} {storm.year}: {e}")
            traceback.print_exc()
            assert False, "Stopping execution."
    else:
        # 2nd option: convert IBTrACS data to ATCF format
        convert_ibtracs_storm_to_atcf(
            ds=storm_ds,
            output_atcf_path=os.path.join(output_dir, "atcf.txt"),
        )
        wind_forcing = BestTrackForcing(
            Path(os.path.join(output_dir, "atcf.txt")), nws=20
        )  # NWS=20 for GAHM
        print(dir(wind_forcing))
        print(wind_forcing.tracks)
    # 2. Pass the track object to the BestTrackForcing constructor
    # Now you can correctly specify the 'nws' parameter

    # wind_forcing = BestTrackForcing(storm.id, nws=20)  # NWS=20 for GAHM
    mesh.add_forcing(wind_forcing)

    # 3. Calculate Simulation Window
    sim_start, sim_end, spinup = calculate_simulation_window(storm)

    # 4. Configure AdcircRun Driver
    driver = AdcircRun(
        mesh=mesh,
        start_date=sim_start,
        end_date=sim_end,  # spinup_duration=spinup
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

def drive_katrina():
    track = VortexTrack.from_storm_name(name='katrina', year=2005)
    print(track)
    # ds = na_landing_tcs().sel(name=b"KATRINA")
    # print(ds)
    # generate_adcirc_inputs(
    #     Storm(sid=ds.sid.values[0],
    #           name=ds.name.values[0],
    #           time=pd.to_datetime(ds.time.values).to_pydatetime().tolist()),
    #     ds.isel(storm=0),
    #     "./katrina_test/",
    # )


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

    # outputs_df = pd.DataFrame(outputs)
    ## outputs_df.to_csv("debug_storms.csv", index=False)

    print(f"Found {len(target_storms)} U.S. landfalling storms in IBTrACS.")

    # 2. Loop and generate inputs for each storm
    for i, storm in enumerate(target_storms[-1:]):
        storm_name_safe = storm.name.upper().replace(" ", "_")
        run_directory = os.path.join(RUNS_PARENT_DIR, f"{storm_name_safe}_{storm.year}")

        try:
            print(
                f"Generating inputs for {storm.name} {storm.year}, {storm.id} {type(storm.id)}..."
            )
            generate_adcirc_inputs(storm, target_storms_ds.isel(storm=i), run_directory)
        except Exception as e:
            print(f"!!! FAILED to generate inputs for {storm.name} {storm.year}: {e}")
            print(" traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    # python -m adforce.generate_training_data
    generate_all_storm_inputs()
    # track = VortexTrack('AL112017')
    # print(track)
    # generate_all_storm_inputs()
    # drive_katrina()
    # from stormevents.nhc import VortexTrack
    # track = VortexTrack('AL112017')
    # print(track.data)
    # drive_katrina()
    # vortex = VortexTrack.from_storm_name('irma', 2017)
    # print(vortex.data)
    # print(vortex)
    # vortex = VortexTrack.from_storm_name('katrina', 2005)
    # print(vortex.data, type(vortex.data))

