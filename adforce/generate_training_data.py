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

The script relies on the IBtrACS netcdf being downloaded, and a couple of functions from tcpips.ibtracs to select storms.

It also relies on the aswip, padcirc and adcprep executables being compiled.

Usage:
    python -m adforce.generate_training_data [--test-single] [--test-nosubprocess] [--recommended-dt 5.0] [--runs-parent-name test_runs]

TODO: Add some yaml to each storm to track key characteristics for readability.
TODO: Add ability to split a task list between multiple slurm jobs.
TODO: Plot and summarise this training data.
"""

from typing import Tuple, Dict, Optional
import os
import math
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import shutil
from datetime import datetime, timedelta
from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.fort15 import Fort15
from adcircpy.forcing.winds import BestTrackForcing
import argparse

# from stormevents.nhc import VortexTrack

import hydra
from omegaconf import DictConfig, OmegaConf
from .subprocess import setoff_subprocess_job_and_wait
from tcpips.ibtracs import na_landing_tcs, select_tc_from_ds
from .constants import SETUP_PATH, PROJ_PATH

RUNS_PARENT_DIR = os.path.join(PROJ_PATH, "runs")
os.makedirs(RUNS_PARENT_DIR, exist_ok=True)
FORT14_PATH = os.path.join(SETUP_PATH, "fort.14.mid")
FORT13_PATH = os.path.join(SETUP_PATH, "fort.13.mid")


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
    storm: Storm,
    spinup_days: float = 0.0,
    extra_days: float = 0.0,
) -> Tuple[datetime, datetime]:
    """
    Calculate the simulation start and end dates based on storm data.

    Args:
       storm (Storm): Storm object with 'time' attribute (list of datetime objects).
       spinup_days (float, optional): Number of days for model spin-up before storm start. Defaults to 0.
       extra_days (float, optional): Number of days to continue simulation after storm end. Defaults to 0.

    Returns:
    - simulation_start_date: Start date for the simulation (datetime).
    - simulation_end_date: End date for the simulation (datetime).
    """
    storm_start_time = storm.time[0]
    storm_end_time = storm.time[-1]
    spinup_duration = timedelta(days=spinup_days)
    start_date = storm_start_time
    simulation_start_date = start_date - spinup_duration
    simulation_end_date = storm_end_time + timedelta(days=extra_days)
    return simulation_start_date, simulation_end_date


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
        if hasattr(char_array_like, "tobytes"):
            decoded_bytes = char_array_like.tobytes()
            # Try decoding as UTF-8, ignore errors, remove null bytes and strip whitespace
            return (
                decoded_bytes.decode("utf-8", errors="ignore")
                .replace("\x00", "")
                .strip()
            )
        # Handle cases where it might already be strings or needs different handling
        elif isinstance(char_array_like, (str, bytes)):
            if isinstance(char_array_like, bytes):
                return (
                    char_array_like.decode("utf-8", errors="ignore")
                    .replace("\x00", "")
                    .strip()
                )
            return str(char_array_like).replace("\x00", "").strip()
        elif hasattr(char_array_like, "item"):  # Handle xarray DataArray extraction
            item_val = char_array_like.item()
            if isinstance(item_val, bytes):
                return (
                    item_val.decode("utf-8", errors="ignore")
                    .replace("\x00", "")
                    .strip()
                )
            return str(item_val).replace("\x00", "").strip()
        else:
            # Fallback for other array-like structures
            # This might need refinement based on actual data types
            joined_str = "".join(
                [
                    (
                        c.decode("utf-8", errors="ignore")
                        if isinstance(c, bytes)
                        else str(c)
                    )
                    for c in char_array_like
                ]
            )
            return joined_str.replace("\x00", "").strip()

    except Exception:
        # Fallback if decoding fails
        return "UNKNOWN"


def clean_radii(radii_nm: list) -> list:
    """
    Cleans a list of 4-quadrant radii [NE, SE, SW, NW]
    to remove ambiguous zeros that crash ASWIP/ADCIRC.

    Args:
        radii_nm (list): A list of 4 radii [NE, SE, SW, NW].

    Returns:
        list: A corrected, unambiguous list of 4 radii.

    Doctests:
    >>> clean_radii([60.0, 60.0, 0.0, 0.0])  # Case 1: Simple mirroring
    [60.0, 60.0, 60.0, 60.0]
    >>> clean_radii([15.0, 0.0, 15.0, 0.0])  # Case 2: Un-mirrorable pair
    [15.0, 15.0, 15.0, 15.0]
    >>> clean_radii([15.0, 0.0, 0.0, 0.0])   # Case 3: Single value
    [15.0, 15.0, 15.0, 15.0]
    >>> clean_radii([0.0, 0.0, 0.0, 0.0])    # Case 4: All zeros (no storm)
    [0.0, 0.0, 0.0, 0.0]
    >>> clean_radii([60.0, 60.0, 15.0, 25.0]) # Case 5: Fully defined
    [60.0, 60.0, 15.0, 25.0]
    """
    # [NE, SE, SW, NW]
    #  0   1   2   3

    # Check if the line is "mixed" (has both zeros and non-zeros)
    has_zeros = any(r == 0 for r in radii_nm)
    has_non_zeros = any(r > 0 for r in radii_nm)

    if not (has_zeros and has_non_zeros):
        # This line is fine. It's either all zeros or all non-zeros.
        return radii_nm

    # print(f"Fixing ambiguous radii: {radii_nm}")

    # --- Pass 1: Try to mirror opposites ---

    # Check NE <-> SW pair
    if radii_nm[0] > 0 and radii_nm[2] == 0:
        radii_nm[2] = radii_nm[0]
    elif radii_nm[0] == 0 and radii_nm[2] > 0:
        radii_nm[0] = radii_nm[2]

    # Check SE <-> NW pair
    if radii_nm[1] > 0 and radii_nm[3] == 0:
        radii_nm[3] = radii_nm[1]
    elif radii_nm[1] == 0 and radii_nm[3] > 0:
        radii_nm[1] = radii_nm[3]

    # --- Pass 2: Check for remaining zeros ---
    # If we still have zeros, it means a pair was [0, 0]
    # (e.g., [15, 0, 15, 0] or [15, 0, 0, 0])

    if any(r == 0 for r in radii_nm):
        # Data is fundamentally incomplete.
        # Find the largest radius provided and make the storm symmetric.
        # This is the most conservative (strongest) and safest assumption.
        max_r = max(radii_nm)
        # print(f"  Mirroring failed. Symmetrizing to max radius: {max_r}")
        radii_nm = [max_r, max_r, max_r, max_r]

    return radii_nm


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
           0  2005082818      BEST    6  265N   859W  110   940       34        120   120    80    80  1013         18                             0    0  KATRINA
        >>> os.remove(output_path) # Clean up the temp file
    """
    atcf_lines = []
    fill_value_int = 0  # Default value for missing numeric fields

    # Find all valid time steps by checking where the 'time' variable is not NaT
    valid_time_indices = np.where(~np.isnat(ds["time"].values))[0]

    # Get static storm information once
    # Ensure storm number is treated safely, default if missing
    try:
        storm_number = int(ds["number"].item())
    except (ValueError, KeyError, AttributeError):
        storm_number = 0  # Default storm number if missing or invalid
        print("Warning: Could not read storm number, defaulting to 0.")

    try:
        storm_name = _decode_char_array(ds["name"]).upper().strip()
        if not storm_name or storm_name == "NOT_NAMED":
            storm_name = "NONAME"
    except (KeyError, AttributeError):
        storm_name = "NONAME"  # Default storm name
        print("Warning: Could not read storm name, defaulting to NONAME.")

    print(f"Generating AWIP input for storm {storm_name} ({storm_number})...")
    print(f"Processing {len(valid_time_indices)} valid time steps...")

    start_time = pd.to_datetime(ds.isel(date_time=0)["time"].item())

    # Loop through only the valid indices
    for time_idx in valid_time_indices:
        # Select a single time slice using .isel()
        row = ds.isel(date_time=time_idx)

        # Safely extract and format data, providing defaults where necessary
        try:
            dt_val = pd.to_datetime(row["time"].item())
            year = dt_val.year
            month = dt_val.month
            day = dt_val.day
            hour = dt_val.hour
        except (AttributeError, ValueError):
            print(
                f"Error: Invalid time data at index {time_idx}. Skipping this time step."
            )
            continue  # Skip this time step if time is invalid

        tech = "BEST"  # Technique
        tau = int((dt_val - start_time).total_seconds() / 3600)

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
        atcf_wind = int(
            np.nan_to_num(
                row.get("usa_wind", fill_value_int).item(), nan=fill_value_int
            )
        )
        atcf_pres = int(
            np.nan_to_num(row.get("usa_pres", 1013).item(), nan=1013)
        )  # Default pressure 1013

        # Background pressure (Pn) - typically 1013 mb
        atcf_poci = 1013

        atcf_rmw = int(
            np.nan_to_num(row.get("usa_rmw", fill_value_int).item(), nan=fill_value_int)
        )

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
            radii_nm = [fill_value_int] * 4  # Initialize with default
            if var_name in row:
                try:
                    # Extract radii, handle NaNs, convert to int
                    radii_nm = [
                        int(
                            np.nan_to_num(
                                row[var_name].isel(quadrant=q).item(),
                                nan=fill_value_int,
                            )
                        )
                        for q in range(4)
                    ]
                    # Check if any radius value is non-zero
                    radii_nm = clean_radii(radii_nm)
                except (ValueError, IndexError, KeyError, AttributeError) as e:
                    print(
                        f"Warning: Could not process radii for {rad_kt}kt at index {time_idx}. Error: {e}"
                    )
                    radii_nm = [fill_value_int] * 4  # Reset to default on error

            # ASWIP documentation suggests it can handle multiple lines per time step,
            # even if radii are zero, as it uses the presence of the isotach line.
            # Let's write a line for each isotach type found, regardless of whether radii are zero,
            # unless the variable itself was missing.
            if var_name in row:
                # FORMAT(22) breakdown and corresponding Python formatting
                # 3x, i3, 2x, i4, 3i2, 6x, a4, 1x, i4, 2x, i3, a1, 1x, i5, a1, 2x, i3, 2x, i4, 6x, i3, 7x, 4(i4,2x), i4, 8x, i3, 27x, 2(i3,2x), a10
                line = (
                    f"   "  # 3x
                    f"{0: >3d}"  # i3      (advr - placeholder 0)
                    f"  "  # 2x
                    f"{year: >4d}"  # i4      (iyear)
                    f"{month:02d}"  # i2      (imth)
                    f"{day:02d}"  # i2      (iday)
                    f"{hour:02d}"  # i2      (ihr)
                    f"      "  # 6x
                    f"{tech: <4}"  # a4      (castType)
                    f" "  # 1x
                    f"{tau: >4d}"  # i4      (iFcstInc)
                    f"  "  # 2x
                    f"{lat_val: >3d}"  # i3      (ilat)
                    f"{lat_hem: <1}"  # a1      (ns)
                    f" "  # 1x
                    f"{lon_val: >5d}"  # i5      (ilon)
                    f"{lon_hem: <1}"  # a1      (ew)
                    f"  "  # 2x
                    f"{atcf_wind: >3d}"  # i3      (ispd)
                    f"  "  # 2x
                    f"{atcf_pres: >4d}"  # i4      (icpress)
                    f"      "  # 6x
                    f"{rad_kt: >3d}"  # i3      (ivr - Wind intensity for radii)
                    f"       "  # 7x
                    f"{radii_nm[0]: >4d}  "  # i4, 2x  (ir(1)) NE
                    f"{radii_nm[1]: >4d}  "  # i4, 2x  (ir(2)) SE
                    f"{radii_nm[2]: >4d}  "  # i4, 2x  (ir(3)) SW
                    f"{radii_nm[3]: >4d}  "  # i4, 2x  (ir(4)) NW
                    f"{atcf_poci: >4d}"  # i4      (ipn - Background pressure)
                    f"        "  # 8x
                    f"{atcf_rmw: >3d}"  # i3      (atcfRMW)
                    f"                           "  # 27x
                    f"{storm_dir: >3d}  "  # i3, 2x  (dir - placeholder 0)
                    f"{storm_speed: >3d}  "  # i3, 2x  (speed - placeholder 0)
                    f"{storm_name: <10}"  # a10     (stormname)
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
        print(
            f"Error: Could not write to output file {output_atcf_path!r}. Reason: {e}"
        )


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
    storm_name = _decode_char_array(ds["name"]).upper().strip(" ").strip("\x00")
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
        atcf_lat = f"{lat_val}{lat_hem}"  # .rjust(5)

        lon_val = int(abs(lon) * 10)
        lon_hem = "W" if lon < 0 else "E"
        atcf_lon = f"{lon_val}{lon_hem}"  # .rjust(6)

        atcf_wind = str(int(np.nan_to_num(row["usa_wind"].item())))  # .rjust(4)
        atcf_pres = str(int(np.nan_to_num(row["usa_pres"].item())))  # .rjust(5)
        atcf_status = _decode_char_array(row["usa_status"])  # .ljust(3)

        # --- Part 3: Enriched Data for Asymmetry and Size ---
        # Select quadrant data using .isel() and get scalar with .item()
        atcf_poci = str(int(np.nan_to_num(row["usa_poci"].item())))  # .rjust(5)
        atcf_roci = str(int(np.nan_to_num(row["usa_roci"].item())))  # .rjust(4)
        atcf_rmw = str(int(np.nan_to_num(row["usa_rmw"].item())))  # .rjust(4)
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
                    f" , , , , , , ,{storm_name}",
                )
                atcf_lines.append("".join(line))

    with open(output_atcf_path, "wt") as f:
        f.write("\n".join(atcf_lines))
    print(f"✅ Successfully created ENRICHED ATCF file at '{output_atcf_path}'")


class CustomAdcircRun(AdcircRun):
    """
    An AdcircRun subclass that overrides the default namelists
    to match the "notide-example" (File 2) configuration.
    """

    def __init__(self, *args, **kwargs):
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


def calculate_cfl_timestep(
    mesh: AdcircMesh, cfl_target: float = 0.7, maxvel: float = 5.0, g: float = 9.81
) -> float:
    """
    Calculates a recommended timestep based on the full CFL condition
    for ADCIRC's shallow water equations.

    The condition is: CFL = (U + sqrt(gH)) * dt / dx <= cfl_target
    So, the maximum dt = cfl_target * dx / (U + sqrt(gH))

    Args:
        mesh: An AdcircMesh object containing node coordinates, depths,
              and element connectivity (implicitly used by properties).
        cfl_target: The desired Courant number (dimensionless). Typically
                    <= 1.0 for stability, often 0.5-0.7 is used for safety.
        maxvel: Estimated maximum expected flow speed (U) in m/s across the
                entire domain during the simulation. This is an estimate you
                provide based on the expected physics (e.g., 5-10 m/s for
                hurricanes). Default is 5.0 m/s.
        g: Acceleration due to gravity in m/s^2. Default is 9.81.

    Returns:
        Recommended maximum timestep (dt) in seconds.

    Raises:
        ValueError: If mesh properties (distances, depths) cannot be determined
                    or are invalid (e.g., non-positive).
        AttributeError: If the mesh object doesn't have the expected properties.
    """
    print(
        f"Calculating CFL timestep with cfl_target={cfl_target}, maxvel={maxvel} m/s..."
    )

    # 1. Find the minimum edge length (dx) in meters
    #    Uses the pre-calculated distances from the mesh object.
    min_dx = float("inf")
    try:
        # Accessing the property triggers calculation if needed
        node_distances = mesh.node_distances_in_meters

        if not node_distances:
            raise ValueError(
                "Node distances dictionary is empty. Cannot calculate min_dx."
            )

        for node_idx, neighbors in node_distances.items():
            if neighbors:  # Check if the neighbor dictionary is not empty
                min_neighbor_dist = min(neighbors.values())
                min_dx = min(min_dx, min_neighbor_dist)

    except AttributeError:
        raise AttributeError(
            "Mesh object requires 'node_distances_in_meters' property for CFL calculation."
        ) from None
    except Exception as e:
        raise ValueError(f"Error accessing node distances: {e}") from e

    if min_dx == float("inf") or min_dx <= 0:
        raise ValueError(
            f"Could not determine a valid minimum edge length (min_dx={min_dx}). Check mesh connectivity and units."
        )
    print(f"  - Minimum edge length (min_dx): {min_dx:.2f} m")

    # 2. Find the maximum depth (H) in meters
    #    Assumes mesh.values is a pandas DataFrame with depth data, positive downwards.
    max_h = 0.0
    try:
        # Get depth values, assuming it's a DataFrame (might have multiple columns)
        depth_df = mesh.values
        if not isinstance(depth_df, pd.DataFrame):
            raise TypeError("Expected mesh.values to be a pandas DataFrame")

        # Find the maximum value across all depth columns, ignoring NaNs
        numeric_depths = depth_df.select_dtypes(include=np.number)
        if numeric_depths.empty:
            raise ValueError("Mesh 'values' DataFrame contains no numeric depth data.")

        max_h = numeric_depths.max().max()  # Max of max of each column

        if pd.isna(max_h):
            raise ValueError(
                "Maximum depth calculation resulted in NaN. Check depth data."
            )

    except AttributeError:
        raise AttributeError(
            "Mesh object requires 'values' property (pandas DataFrame) for depths."
        ) from None
    except Exception as e:
        raise ValueError(f"Error accessing or processing mesh depths: {e}") from e

    if max_h < 0:
        print(
            f"  - Warning: Maximum mesh depth is negative ({max_h:.2f}m). Using absolute value."
        )
        max_h = abs(max_h)
    # Allow max_h == 0 (completely dry land), sqrt(0) is handled.

    print(f"  - Maximum depth (max_h): {max_h:.2f} m")

    # 3. Calculate gravity wave speed (C = sqrt(gH))
    wave_speed = math.sqrt(g * max_h)
    print(f"  - Max gravity wave speed (sqrt(gH)): {wave_speed:.2f} m/s")

    # 4. Calculate characteristic speed (S = U + C)
    #    Use absolute value of maxvel just in case.
    characteristic_speed = abs(maxvel) + wave_speed
    print(
        f"  - Characteristic speed (maxvel + sqrt(gH)): {characteristic_speed:.2f} m/s"
    )

    if characteristic_speed <= 1e-9:  # Check for effectively zero speed
        # This might happen if max_h=0 and maxvel=0
        raise ValueError(
            f"Characteristic speed ({characteristic_speed:.2f} m/s) is near zero. Cannot calculate timestep."
        )

    # 5. Calculate recommended timestep (dt = cfl_target * dx / S)
    recommended_dt = cfl_target * min_dx / characteristic_speed

    if recommended_dt <= 0:
        raise ValueError(
            f"Calculated timestep ({recommended_dt:.4f}s) is not positive. Check inputs."
        )

    print(f"✅ Recommended timestep (dt): {recommended_dt:.4f} seconds")
    return recommended_dt


def generate_adcirc_inputs(
    storm: Storm, storm_ds: xr.Dataset, output_dir: str, recalculate_timestep=False, recommended_dt = 1.0
) -> None:
    """
    Generates a complete set of ADCIRC inputs for a single storm.
    This creates:
    - pre_aswip_fort.22 (from IBTrACS data)
    - atcf.txt (from IBTrACS data)
    - fort.15 (via adcircpy)
    - fort.13 (copied)

    Args:
        storm (Storm): Storm object containing storm metadata.
        storm_ds (xr.Dataset): xarray.Dataset for the storm from IBTrACS.
        output_dir (str): Directory to save the generated ADCIRC input files.
        recalculate_timestep (bool): Whether to recalculate the timestep

    Returns:
        None
    """
    # 1. Load Mesh
    mesh = AdcircMesh.open(FORT14_PATH, crs="epsg:4326")
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

    # 2. Add Forcings -- no tides
    # tidal_forcing = Tides()
    # tidal_forcing.use_all()
    # mesh.add_forcing() #tidal_forcing)

    # 1. Use the parent class to find the storm and create a track object
    os.makedirs(output_dir, exist_ok=True)

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

    # 2. Pass the track object to the BestTrackForcing constructor
    mesh.add_forcing(wind_forcing)

    # 3. Calculate Simulation Window
    sim_start, sim_end = calculate_simulation_window(storm, extra_days=0, spinup_days=0)

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
    driver.DRAMP = 1.0
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
    shutil.copy(FORT13_PATH, os.path.join(output_dir, "fort.13"))

    print(
        f"Successfully generated inputs for {storm.name} {storm.year} in {output_dir}"
    )


def drive_all_adcirc(
    test_single=False,
    test_nosubprocess=False,
    runs_parent_name: Optional[str] = None,
    recommended_dt: float = None,
) -> None:
    """
    Generate all storm inputs using NWS=20.

    Args:
        test_single (bool, optional): If True just use Katrina 2005. Defaults to False.
        test_nosubprocess (bool, optional): If True do not set of subprocess to run ADCIRC. Defaults to False.

    Raises:
        IOError: Needs to find the adforce config.
    """
    # it could be good to add a flag just to run Katrina
    # python -m adforce.generate_training_data
    if runs_parent_name is None:
        runs_parent_dir = RUNS_PARENT_DIR
    else:
        runs_parent_dir = os.path.join(PROJ_PATH, runs_parent_name)

    # --- New: Load Hydra Config ---
    print("Loading ADCIRC run configuration...")

    # Determine config path relative to this file
    # __file__ is .../adforce/generate_training_data.py
    # os.path.dirname(__file__) is .../adforce
    # The config dir is .../adforce/config
    # So the relative path is just "config"
    relative_config_path = "config"

    if not os.path.isdir(relative_config_path):
        # Fallback if script is run from project root (e.g., python adforce/generate_training_data.py)
        alt_path = os.path.join(os.path.dirname(__file__), "config")
        if os.path.isdir(alt_path):
            relative_config_path = alt_path
        else:
            raise IOError(
                f"Could not find config directory at '{relative_config_path}' or '{alt_path}'. "
                "Ensure the 'config' directory is a sibling of this script."
            )

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="config", version_base=None)
    cfg = hydra.compose(config_name="wrap_config")
    print("✅ Default config loaded.")

    # --- End New Config ---
    target_storms_ds = na_landing_tcs()
    print("target storms", target_storms_ds)
    if test_single:
        # select just KATRINA 2005
        # i.e. name =  b"KATRINA"
        # and year = 2005
        i_ran = np.where([x == b"KATRINA" for x in target_storms_ds.name.values])[0]
        # i_ran = [i for i in i_ran if target_storms_ds.isel(storm=0).time.values[0]==2005]
        # print(i_ran)
        if len(i_ran) == 0:
            i_ran = [0]
        if len(i_ran) > 1:
            i_ran = [i_ran[-1]]
    else:
        i_ran = range(len(target_storms_ds.sid))

    target_storms = []
    for i in i_ran:
        # Isolate the raw numpy array of times for the storm
        raw_times = target_storms_ds.time[i].values
        # print("raw_times", raw_times)

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
        target_storms.append((i, storm))

    outputs = {"storm_id": [], "storm_name": [], "year": [], "num_timepoints": []}

    for i, storm in target_storms:
        outputs["storm_id"].append(storm.id)
        outputs["storm_name"].append(storm.name)
        outputs["year"].append(storm.year)
        outputs["num_timepoints"].append(len(storm.time))

    # outputs_df = pd.DataFrame(outputs)
    # outputs_df.to_csv("debug_storms.csv", index=False)

    print(f"Found {len(target_storms)} U.S. landfalling storms in IBTrACS.")

    # 2. Loop and generate inputs for each storm
    for i, storm in target_storms[:]:
        storm_name_safe = storm.name.upper().replace(" ", "_")
        run_directory = os.path.join(runs_parent_dir, f"{i}_{storm_name_safe}_{storm.year}")

        # --- Check for existing successful run (implements TODO #1) ---
        slurm_path = os.path.join(run_directory, "slurm.out")
        is_successful = False
        if os.path.exists(slurm_path):
            try:
                with open(slurm_path, "r") as slurm_out_file:
                    for line in slurm_out_file:
                        if "Job completed successfully.\n" in line:
                            is_successful = True
                            break
            except (IOError, FileNotFoundError) as e:
                print(f"Warning: Could not read {slurm_path}. Will attempt to rerun. Error: {e}")
                is_successful = False

        if is_successful:
            print(f"Run {run_directory} already completed successfully. Skipping.")
            continue  # Skip to the next storm
        elif os.path.exists(run_directory):
            print(f"Directory {run_directory} exists but is incomplete or failed. Rerunning...")
        # --- End check ---

        try:
            print(
                f"\n--- Processing Storm {i+1}/{len(target_storms)}: {storm.name} {storm.year} ---"
            )

            # 1. Generate ADCIRC input files (fort.15, pre_aswip_fort.22, fort.13)
            print(f"Generating inputs in: {run_directory}")
            generate_adcirc_inputs(storm, target_storms_ds.isel(storm=i), run_directory, recommended_dt=recommended_dt)

            # 2. Create a storm-specific config to pass to subprocess
            # Start with the default config
            storm_cfg = cfg.copy()

            # Set the specific run folder and name for logging
            OmegaConf.update(storm_cfg.files, "run_folder", run_directory)
            OmegaConf.update(storm_cfg, "name", f"{storm_name_safe}_{storm.year}")

            # Tell the subprocess runner to execute ASWIP
            # This converts 'pre_aswip_fort.22' to 'fort.22' with the NWS=20 format
            OmegaConf.update(storm_cfg, "use_aswip", True)

            # Ensure we're NOT using SLURM for this subprocess
            # (The main script is one SLURM job, but each storm is a local subprocess)
            OmegaConf.update(storm_cfg, "use_slurm", False)

            # 3. Run the simulation (ASWIP, adcprep, padcirc)
            print(f"Running ADCIRC simulation for {storm.name} via subprocess...")
            # This function will chdir into run_directory
            if not test_nosubprocess:
                setoff_subprocess_job_and_wait(run_directory, storm_cfg)
                print(f"✅ Successfully completed run for {storm.name} {storm.year}")
            else:
                print(f"✅ Successfully made inputs for {storm.name} {storm.year}")

        except Exception as e:
            print(f"!!! FAILED to process {storm.name} {storm.year}: {e}")
            print(" traceback:")
            traceback.print_exc()
    print(target_storms)


if __name__ == "__main__":
    # python -m adforce.generate_training_data
    parser = argparse.ArgumentParser(
        description="Generate ADCIRC inputs and run simulations for all U.S. landfalling storms."
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="If set, only process Katrina 2005 for testing.",
    )
    parser.add_argument(
        "--test-nosubprocess",
        action="store_true",
        help="If set, generate inputs but do not run subprocess simulations.",
    )
    parser.add_argument(
        "--recommended-dt",
        type=float,
        default=5.0,
        help="Recommended timestep (dt) in seconds for ADCIRC simulations. Default is 5s.",
    )
    parser.add_argument(
        "--runs-parent-name",
        type=str,
        default=None,
        help="Name of the parent directory for runs inside the project path. Defaults to the constant RUNS_PARENT_DIR.",
    )
    args = parser.parse_args()

    drive_all_adcirc(
        test_single=args.test_single,
        test_nosubprocess=args.test_nosubprocess,
        recommended_dt=args.recommended_dt,
        runs_parent_name=args.runs_parent_name,
    )
    # python -m adforce.generate_training_data --test-single --test-nosubprocess
    # python -m adforce.generate_training_data --recommended-dt 5.0 --test-nosubprocess --test-single --runs-parent-name test_runs

    # drive_all_adcirc(test_nosubprocess=True, test_single=True)
    # track = VortexTrack('AL112017')
    # print(track)
    # drive_all_adcirc()
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
