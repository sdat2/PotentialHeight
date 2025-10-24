"""Generate ADCIRC runs for historical U.S. landfalling storms.

Ideally we could do this with ADCIRCpy, but that package seemed to be somewhat incomplete.

Also there seem to be a lot of issues with using NOAA servers ATM.

Anyway, so the task in general is to extract the desired storms from IBTrACS,
get the data into a format that ASWIP can read,
get ASWIP to convert it into NWS=20 fort.22 format (Generalized Asymmetric Holland Model),
make AdCIRCpy create the correct fort.15 namelist file for this storm,
and then run the storm on ARCHER2 with padcirc etc.

After this is all done we will process the fort.*.nc files to get training data to train
our GNN model.
"""
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


def _decode_char_array(char_array_like) -> str:
    """
    PFunction to decode character arrays from xarray/numpy.

    Args:
        char_array_like: An object representing character data (e.g., numpy array of bytes/chars).

    Returns:
        The decoded string.

    >>> _decode_char_array(np.array([b'A', b'L'], dtype='|S1'))
    'AL'
    >>> _decode_char_array(np.array([b'G', b'E', b'R', b'T', b' ', b' '], dtype='|S1'))
    'GERT'
    >>> _decode_char_array(np.array([b'T', b'S'], dtype='|S1'))
    'TS'
    """
    try:
        # Attempt common decoding strategy for byte arrays
        if hasattr(char_array_like, 'tobytes'):
            decoded_bytes = char_array_like.tobytes()
            # Try decoding as UTF-8, ignore errors, remove null bytes and strip whitespace
            return decoded_bytes.decode('utf-8', errors='ignore').replace('\x00', '').strip()
        # Handle cases where it might already be strings or needs different handling
        elif isinstance(char_array_like, (str, bytes)):
             if isinstance(char_array_like, bytes):
                 return char_array_like.decode('utf-8', errors='ignore').replace('\x00', '').strip()
             return str(char_array_like).replace('\x00', '').strip()
        elif hasattr(char_array_like, 'item'): # Handle xarray DataArray extraction
             item_val = char_array_like.item()
             if isinstance(item_val, bytes):
                 return item_val.decode('utf-8', errors='ignore').replace('\x00', '').strip()
             return str(item_val).replace('\x00', '').strip()
        else:
             # Fallback for other array-like structures
             # This might need refinement based on actual data types
            joined_str = "".join([c.decode('utf-8', errors='ignore') if isinstance(c, bytes) else str(c) for c in char_array_like])
            return joined_str.replace('\x00', '').strip()

    except Exception:
        # Fallback if decoding fails
        return "UNKNOWN"


def convert_ibtracs_storm_to_aswip_input(ds: xr.Dataset, output_atcf_path: str):
    """
    Loads IBTrACS dataset for one tropical cyclone and creates a fixed-width
    ASWIP input file (fort.22 format) based on FORMAT(22) from ASWIP source.

    Includes asymmetry (wind radii) and size (RMW) information. Uses placeholders
    for fields not directly available in standard IBTrACS best tracks.

    Args:
        ds (xr.Dataset): xarray.Dataset for a single storm from IBTrACS,
                         containing variables like 'time', 'number', 'name',
                         'usa_lat', 'usa_lon', 'usa_wind', 'usa_pres',
                         'usa_status', 'usa_rmw', 'usa_r34', 'usa_r50', 'usa_r64'.
        output_atcf_path (str): Path to save the new fixed-width file for ASWIP input.

    Example:
        >>> # Create a mock xarray Dataset for testing
        >>> times = pd.to_datetime(['2005-08-28 12:00', '2005-08-28 18:00'])
        >>> mock_ds = xr.Dataset(
        ...     {
        ...         "time": (("date_time",), times),
        ...         "number": ((), 12), # Storm number 12
        ...         "name": ((), b'KATRINA'),
        ...         "basin": (("date_time", "max_basin_len"), np.array([[b'A', b'L'], [b'A', b'L']], dtype='|S1')),
        ...         "usa_lat": (("date_time",), np.array([25.7, 26.5])),
        ...         "usa_lon": (("date_time",), np.array([-84.7, -85.9])),
        ...         "usa_wind": (("date_time",), np.array([100, 110])), # knots
        ...         "usa_pres": (("date_time",), np.array([945, 940])), # mb
        ...         "usa_status": (("date_time", "max_status_len"), np.array([[b'H', b'U'], [b'H', b'U']], dtype='|S1')),
        ...         "usa_rmw": (("date_time",), np.array([20, 18])), # nm
        ...         "usa_r34": (("date_time", "quadrant"), np.array([[100, 100, 75, 75], [120, 120, 80, 80]], dtype=np.float32)), # nm
        ...         "usa_r50": (("date_time", "quadrant"), np.array([[50, 50, 35, 35], [60, 60, 40, 40]], dtype=np.float32)), # nm
        ...         "usa_r64": (("date_time", "quadrant"), np.array([[30, 30, 20, 20], [35, 35, 25, 25]], dtype=np.float32)), # nm
        ...         "usa_poci": (("date_time",), np.array([1010, 1008])), # Pressure of Outer Closed Isobar (placeholder)
        ...         "usa_roci": (("date_time",), np.array([200, 220])), # Radius of Outer Closed Isobar (placeholder)
        ...     },
        ...     coords={
        ...         "date_time": times,
        ...         "quadrant": np.arange(4),
        ...         "max_name_len": np.arange(10),
        ...         "max_basin_len": np.arange(2),
        ...         "max_status_len": np.arange(2)
        ...     }
        ... )
        >>> import tempfile
        >>> import os
        >>> with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmpfile:
        ...     output_path = tmpfile.name
        >>> convert_ibtracs_storm_to_aswip_input(mock_ds, output_path) # doctest: +ELLIPSIS
        Generating AWIP input for storm KATRINA (12)...
        Processing 2 valid time steps...
        ✅ Successfully created an ASWIP input file at ...
        >>> with open(output_path, 'r') as f:
        ...     lines = f.readlines()
        >>> print(lines[0].strip()) # First line, 34kt radii
           0  2005082812      BEST    0    257N   847W  100  945      34       100     100      75      75  1013        20                               0    0  KATRINA
        >>> print(lines[1].strip()) # First line, 50kt radii
           0  2005082812      BEST    0    257N   847W  100  945      50        50      50      35      35  1013        20                               0    0  KATRINA
        >>> print(lines[2].strip()) # First line, 64kt radii
           0  2005082812      BEST    0    257N   847W  100  945      64        30      30      20      20  1013        20                               0    0  KATRINA
        >>> print(lines[3].strip()) # Second line, 34kt radii
           0  2005082818      BEST    0    265N   859W  110  940      34       120     120      80      80  1013        18                               0    0  KATRINA
        >>> os.remove(output_path) # Clean up the temp file
    """
    atcf_lines = []
    fill_value_int = 0 # Default value for missing numeric fields

    # Find all valid time steps by checking where the 'time' variable is not NaT
    valid_time_indices = np.where(~np.isnat(ds["time"].values))[0]

    # Get static storm information once
    # Ensure storm number is treated safely, default if missing
    try:
        storm_number = int(ds["number"].item())
    except (ValueError, KeyError, AttributeError):
        storm_number = 0 # Default storm number if missing or invalid
        print("Warning: Could not read storm number, defaulting to 0.")

    try:
        storm_name = _decode_char_array(ds["name"]).upper().strip()
        if not storm_name or storm_name == "NOT_NAMED":
             storm_name = "NONAME"
    except (KeyError, AttributeError):
        storm_name = "NONAME" # Default storm name
        print("Warning: Could not read storm name, defaulting to NONAME.")

    print(f"Generating AWIP input for storm {storm_name} ({storm_number})...")
    print(f"Processing {len(valid_time_indices)} valid time steps...")

    start_time = pd.to_datetime(ds.isel(date_time=0)["time"].item())

    # Loop through only the valid indices
    for time_idx in valid_time_indices:
        # Select a single time slice using .isel()
        row = ds.isel(date_time=time_idx)

        # Safely extract and format data, providing defaults
        try:
            atcf_basin = _decode_char_array(row["basin"])[:2].ljust(2) # Ensure 2 chars
        except (KeyError, AttributeError):
            atcf_basin = "AL" # Default basin
            print(f"Warning: Missing basin at index {time_idx}, defaulting to AL.")

        try:
            atcf_cyclone_num = f"{storm_number:02d}" # Ensure 2 digits
        except ValueError:
             atcf_cyclone_num = "00"

        try:
            dt_val = pd.to_datetime(row["time"].item())
            atcf_datetime = dt_val.strftime("%Y%m%d%H")
            year = dt_val.year
            month = dt_val.month
            day = dt_val.day
            hour = dt_val.hour
        except (AttributeError, ValueError):
            print(f"Error: Invalid time data at index {time_idx}. Skipping this time step.")
            continue # Skip this time step if time is invalid

        technum = 0 # Placeholder for TECHNUM/MIN
        tech = "BEST" # Technique
        tau = int((start_time - dt_val).total_seconds() / 3600)

        try:
            lat = row["usa_lat"].item()
            lat_val = int(abs(lat) * 10)
            lat_hem = "N" if lat >= 0 else "S"
        except (ValueError, KeyError, AttributeError):
            lat_val, lat_hem = 0, "N"
            print(f"Warning: Missing latitude at index {time_idx}, defaulting to 0N.")

        try:
            lon = row["usa_lon"].item()
            lon_val = int(abs(lon) * 10)
            # Correct hemisphere logic: West is negative
            lon_hem = "W" if lon < 0 else "E"
        except (ValueError, KeyError, AttributeError):
            lon_val, lon_hem = 0, "E"
            print(f"Warning: Missing longitude at index {time_idx}, defaulting to 0E.")


        # Use np.nan_to_num to handle potential NaNs before converting to int
        atcf_wind = int(np.nan_to_num(row.get("usa_wind", fill_value_int).item(), nan=fill_value_int))
        atcf_pres = int(np.nan_to_num(row.get("usa_pres", 1013).item(), nan=1013)) # Default pressure 1013

        # Background pressure (Pn) - typically 1013 mb
        atcf_poci = 1013

        atcf_rmw = int(np.nan_to_num(row.get("usa_rmw", fill_value_int).item(), nan=fill_value_int))

        # Placeholder for Direction and Speed (ASWIP recalculates)
        storm_dir = 0
        storm_speed = 0

        # Define radii types and corresponding variable names
        radii_vars = {
            34: "usa_r34",
            50: "usa_r50",
            64: "usa_r64",
            # Add 100kt if available in your dataset
            # 100: "usa_r100",
        }

        for rad_kt, var_name in radii_vars.items():
            radii_nm = [fill_value_int] * 4 # Initialize with default
            has_data = False
            if var_name in row:
                try:
                    # Extract radii, handle NaNs, convert to int
                    radii_nm = [
                        int(np.nan_to_num(row[var_name].isel(quadrant=q).item(), nan=fill_value_int))
                        for q in range(4)
                    ]
                    # Check if any radius value is non-zero
                    if any(r > 0 for r in radii_nm):
                        has_data = True
                except (ValueError, IndexError, KeyError, AttributeError) as e:
                     print(f"Warning: Could not process radii for {rad_kt}kt at index {time_idx}. Error: {e}")
                     radii_nm = [fill_value_int] * 4 # Reset to default on error

            # ASWIP documentation suggests it can handle multiple lines per time step,
            # even if radii are zero, as it uses the presence of the isotach line.
            # Let's write a line for each isotach type found, regardless of whether radii are zero,
            # unless the variable itself was missing.
            if var_name in row:
                # FORMAT(22) breakdown and corresponding Python formatting
                # 3x, i3, 2x, i4, 3i2, 6x, a4, 1x, i4, 2x, i3, a1, 1x, i5, a1, 2x, i3, 2x, i4, 6x, i3, 7x, 4(i4,2x), i4, 8x, i3, 27x, 2(i3,2x), a10
                line = (
                    f"   "                          # 3x
                    f"{0: >3d}"                     # i3      (advr - placeholder 0)
                    f"  "                          # 2x
                    f"{year: >4d}"                  # i4      (iyear)
                    f"{month:02d}"                  # i2      (imth)
                    f"{day:02d}"                    # i2      (iday)
                    f"{hour:02d}"                   # i2      (ihr)
                    f"      "                      # 6x
                    f"{tech: <4}"                   # a4      (castType)
                    f" "                           # 1x
                    f"{tau: >4d}"                   # i4      (iFcstInc)
                    f"  "                          # 2x
                    f"{lat_val: >3d}"               # i3      (ilat)
                    f"{lat_hem: <1}"                # a1      (ns)
                    f" "                           # 1x
                    f"{lon_val: >5d}"               # i5      (ilon)
                    f"{lon_hem: <1}"                # a1      (ew)
                    f"  "                          # 2x
                    f"{atcf_wind: >3d}"             # i3      (ispd)
                    f"  "                          # 2x
                    f"{atcf_pres: >4d}"             # i4      (icpress)
                    f"      "                      # 6x
                    f"{rad_kt: >3d}"                # i3      (ivr - Wind intensity for radii)
                    f"       "                     # 7x
                    f"{radii_nm[0]: >4d}  "         # i4, 2x  (ir(1)) NE
                    f"{radii_nm[1]: >4d}  "         # i4, 2x  (ir(2)) SE
                    f"{radii_nm[2]: >4d}  "         # i4, 2x  (ir(3)) SW
                    f"{radii_nm[3]: >4d}  "         # i4, 2x  (ir(4)) NW
                    f"{atcf_poci: >4d}"             # i4      (ipn - Background pressure)
                    f"        "                    # 8x
                    f"{atcf_rmw: >3d}"              # i3      (atcfRMW)
                    f"                           " # 27x
                    f"{storm_dir: >3d}  "           # i3, 2x  (dir - placeholder 0)
                    f"{storm_speed: >3d}  "         # i3, 2x  (speed - placeholder 0)
                    f"{storm_name: <10}"            # a10     (stormname)
                )
                # Ensure the line length matches expected format length if needed, though Fortran often ignores trailing chars.
                # Total length from format: 3+3+2+4+2+2+2+6+4+1+4+2+3+1+1+5+1+2+3+2+4+6+3+7+4*6+4+8+3+27+2*5+10 = 143
                # Let's verify:
                # print(len(line)) # Check length if necessary
                atcf_lines.append(line)

    # Write the formatted lines to the output file
    try:
        with open(output_atcf_path, "wt") as f:
            f.write("\n".join(atcf_lines))
        # Use repr() to handle potential special characters in the path for printing
        print(f"✅ Successfully created an ASWIP input file at {output_atcf_path!r}")
    except IOError as e:
        print(f"Error: Could not write to output file {output_atcf_path!r}. Reason: {e}")


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

    # 1. Use the parent class to find the storm and create a track object
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
        convert_ibtracs_storm_to_aswip_input(
            ds=storm_ds,
            output_atcf_path=os.path.join(output_dir, "pre_aswip_fort.22"),
        )
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
    for i, storm in enumerate(target_storms[:]):
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

