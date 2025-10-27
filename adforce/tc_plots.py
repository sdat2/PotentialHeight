"""
Module for plotting TC tracks and simulation summary data.

This script generates a summary plot of all U.S. landfalling TC tracks
on the ADCIRC mesh, colored by intensity.

Usage:
    python -m adforce.plotting [--test-single] [--output-name "track_summary.pdf"]
"""

import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
from adcircpy import AdcircMesh
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
from sithom.plot import plot_defaults, get_dim

# --- Assuming constants.py and ibtracs.py are accessible ---
# Use .constants and .ibtracs if running as a module,
# or adjust paths if running as a standalone script.
try:
    from .constants import FIGURE_PATH, SETUP_PATH, PROJ_PATH
    from tcpips.ibtracs import na_landing_tcs
except ImportError:
    print("Warning: Running as standalone. Assuming relative paths.")
    # Define fallback paths if run directly from project root
    PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SETUP_PATH = os.path.join(PROJ_PATH, 'setup')
    FIGURE_PATH = os.path.join(PROJ_PATH, 'figures')
    # This is complex, assumes tcpips is a sibling directory
    import sys
    sys.path.append(PROJ_PATH)
    from tcpips.ibtracs import na_landing_tcs

from .generate_training_data import _decode_char_array
from .constants import DATA_PATH

plot_defaults()

# --- Saffir-Simpson Scale Definitions (in m/s) ---
KNOTS_TO_MS = 0.514444

# Original wind speed bins in knots
SS_BINS_KNOTS = [0, 34, 64, 83, 96, 113, 137, 500]  # Bin edges

# Convert bins to m/s and round to one decimal
SS_BINS = [np.round(kt * KNOTS_TO_MS, 1) for kt in SS_BINS_KNOTS]
# Result: [0.0, 17.5, 32.9, 42.7, 49.4, 58.1, 70.5, 257.2]

SS_COLORS = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred', 'magenta']

# Labels updated to reflect m/s
SS_LABELS = [
    'TD (<17.5 m/s)', 'TS (17.5-32.8 m/s)', 'Cat 1 (32.9-42.6 m/s)',
    'Cat 2 (42.7-49.3 m/s)', 'Cat 3 (49.4-58.0 m/s)',
    'Cat 4 (58.1-70.4 m/s)', 'Cat 5 (70.5+ m/s)'
]
SS_CMAP = ListedColormap(SS_COLORS)
SS_NORM = BoundaryNorm(SS_BINS, SS_CMAP.N) # This now uses the m/s bins


# --- Plotting Function ---

def plot_all_tc_tracks_on_mesh(
    all_storms_ds: xr.Dataset,
    mesh_path: str,
    output_path: str,
    test_single: bool = False
) -> None:
    """
    Plots TC tracks from IBTrACS on the ADCIRC mesh.

    Tracks are colored by wind intensity using the Saffir-Simpson scale.

    Args:
        all_storms_ds (xr.Dataset): Dataset containing all storms from
                                    na_landing_tcs().
        mesh_path (str): Path to the fort.14 mesh file.
        output_path (str): Path to save the output .pdf plot.
        test_single (bool, optional): If True, plots only Katrina 2005.
                                     Defaults to False.
    """
    print(f"Loading mesh from {mesh_path}...")
    try:
        with nc.Dataset(mesh_path, 'r') as ds:
            x_nodes = ds.variables['x'][:]
            y_nodes = ds.variables['y'][:]
            triangles = ds.variables['element'][:]  -1  # adcircpy elements are 0-based
    except Exception as e:
        print(f"Error loading mesh: {e}. Cannot plot mesh background.")
        return

    print("Setting up plot...")
    fig, ax = plt.subplots() #figsize=(15, 12))

    # Plot the mesh (faintly)
    print("Plotting mesh background...")
    ax.triplot(
        x_nodes, y_nodes, triangles,
        color='grey', alpha=0.2, linewidth=0.1, label='ADCIRC Mesh'
    )

    # Determine storms to plot
    if test_single:
        # Find Katrina 2005, just as in drive_all_adcirc
        i_ran = np.where([x == b"KATRINA" for x in all_storms_ds.name.values])[0]
        if len(i_ran) == 0:
            indices = [0]  # Fallback to first storm
            print("Warning: Katrina 2005 not found. Plotting first storm.")
        else:
            indices = [i_ran[-1]]  # Use last Katrina entry
        print("Plotting in single mode (Katrina 2005)...")
    else:
        indices = range(len(all_storms_ds.storm))
        print(f"Plotting all {len(indices)} storms...")

    # Loop and plot each track
    tracks_plotted = 0
    for i in indices:
        storm_ds = all_storms_ds.isel(storm=i)
        storm_name = _decode_char_array(storm_ds['name'])

        # Extract and clean data
        lat = storm_ds['usa_lat'].values
        lon = storm_ds['usa_lon'].values
        wind = storm_ds['usa_wind'].values * KNOTS_TO_MS  # Convert to m/s

        valid_mask = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(wind)
        lat, lon, wind = lat[valid_mask], lon[valid_mask], wind[valid_mask]

        # --- CONVERT TO M/S ---
        # Data from IBTrACS (via na_landing_tcs) is in knots, convert to m/s
        wind = wind * KNOTS_TO_MS
        # ----------------------

        if len(lat) < 2:
            # print(f"Skipping {storm_name} (insufficient data).")
            continue

        # Create segments [[(x1, y1), (x2, y2)], [(x2, y2), (x3, y3)], ...]
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use wind at the start of each segment for coloring
        # This is now in m/s
        segment_winds = wind[:-1]

        # Create a LineCollection
        lc = LineCollection(
            segments, cmap=SS_CMAP, norm=SS_NORM,
            linewidths=0.5, alpha=0.7  # Thinner, semi-transparent lines
        )
        lc.set_array(segment_winds) # Set array with m/s values
        ax.add_collection(lc)
        tracks_plotted += 1

    print(f"Plotted {tracks_plotted} tracks.")

    # --- Final plot formatting ---
    ax.set_aspect('equal')
    ax.set_xlabel("Longitude [$^{\circ}$E]")
    ax.set_ylabel("Latitude [$^{\circ}$N]")
    # title = "Historical U.S. Landfalling TC Tracks on ADCIRC Mesh"
    #if test_single:
    #    title = "TC Track for Katrina (2005) on ADCIRC Mesh"
    #ax.set_title(title)

    # Set plot limits to the mesh extent
    ax.set_xlim(x_nodes.min(), x_nodes.max())
    ax.set_ylim(y_nodes.min(), y_nodes.max())

    # Add a custom colorbar
    # We create a dummy ScalarMappable to link the cmap and norm
    sm = plt.cm.ScalarMappable(cmap=SS_CMAP, norm=SS_NORM)
    sm.set_array([])  # Dummy array
    cbar = fig.colorbar(
        sm,
        ax=ax,
        boundaries=SS_BINS,
        # Center ticks in each color block
        ticks=[b + (SS_BINS[i+1]-b)/2 for i, b in enumerate(SS_BINS[:-1])],
        # spacing='proportional'
    )
    cbar.ax.set_yticklabels(SS_LABELS, fontsize='small')
    cbar.set_label("Wind Speed (m/s) - Saffir-Simpson Scale") # Updated label

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Successfully plotted tracks to {output_path}")


# --- Main execution block to make this script runnable ---

if __name__ == "__main__":
    # python -m adforce.plotting --test-single
    parser = argparse.ArgumentParser(
        description="Plot TC tracks on the ADCIRC mesh."
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="If set, only process Katrina 2005 for testing."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="all_tc_tracks.pdf",
        help="Name for the output plot file in the figures/tc_tracks directory."
    )
    args = parser.parse_args()

    print("Loading all storm data from IBTrACS...")
    try:
        # This data is still loaded with wind speed in knots
        all_storms_ds = na_landing_tcs()
    except Exception as e:
        print(f"Error loading IBTrACS data: {e}")
        print("Please ensure 'tcpips' package is installed and accessible.")
        sys.exit(1)

    output_file = args.output_name
    if args.test_single and "all_" in output_file:
        output_file = output_file.replace("all_", "test_single_")

    # Define a dedicated subdir for these plots
    output_folder = os.path.join(FIGURE_PATH, "tc_tracks")
    os.makedirs(output_folder, exist_ok=True)
    output_plot_path = os.path.join(output_folder, output_file)

    # The plotting function will now handle the conversion to m/s
    plot_all_tc_tracks_on_mesh(
        all_storms_ds=all_storms_ds,
        mesh_path=os.path.join(DATA_PATH, "exp_0049", "fort.63.nc"),
        output_path=output_plot_path,
        test_single=args.test_single
    )
