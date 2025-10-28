"""
Processes a single, completed ADCIRC run directory into SWE-GNN format.

This script is intended to be called by a Slurm job array, where each
task processes one directory.
"""
import os
import argparse
from .mesh import swegnn_netcdf_creation


def check_and_process_run(run_path: str, save_path: str, use_dask: bool) -> bool: # NEW: use_dask parameter
    """
    Checks a single run for success and processes it if valid.

    Args:
        run_path (str): Path to the single run directory (e.g., .../run_5sec/exp_0001).
        save_path (str): Full path for the output .nc file.
        use_dask (bool): Whether to use dask for loading.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    if not os.path.exists(run_path):
        print(f"Error: Run path not found: {run_path}")
        return False

    slurm_path = os.path.join(run_path, "slurm.out")
    run_name = os.path.basename(run_path)

    if not os.path.exists(slurm_path):
        print(f"Run {run_name}: FAILED (slurm.out not found)")
        return False

    try:
        with open(slurm_path, "r") as f:
            slurm_out_lines = f.readlines()

        if "Job completed successfully.\n" in slurm_out_lines:
            # NEW: Print the dask setting to the log
            print(f"Run {run_name}: SUCCESS. Processing with use_dask={use_dask}...")

            swegnn_netcdf_creation(
                path_in=run_path,
                path_out=save_path,
                use_dask=use_dask  # NEW: Use the parameter
            )
            print(f"Run {run_name}: Processing complete. Saved to {save_path}")
            return True
        else:
            print(f"Run {run_name}: FAILED (Did not find 'Job completed successfully.')")
            return False

    except Exception as e:
        print(f"Run {run_name}: ERROR during check or processing: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check and process a single ADCIRC training run."
    )
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Path to the single run directory (e.g., .../run_5sec/exp_0001)."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Full path for the output .nc file (e.g., .../swegnn_5sec/exp_0001.nc)."
    )
    # NEW: Add the --use-dask flag
    # action='store_true' means it defaults to False,
    # and becomes True if the flag '--use-dask' is present.
    parser.add_argument(
        "--use-dask",
        action='store_true',
        help="Enable Dask for loading (use_dask=True). If flag is absent, defaults to False."
    )
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # NEW: Pass the dask argument to the function
    check_and_process_run(args.run_path, args.save_path, args.use_dask)
