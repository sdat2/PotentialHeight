import json
import os
import tempfile
import doctest
from pathlib import Path
from typing import Optional, List
from adforce.mesh import swegnn_dg_from_mesh_ds_from_path


def find_best_run_from_json(json_path: str) -> Optional[str]:
    """
    Parses an 'experiments.json' file to find the run with the highest 'res'.

    This file format is defined in the `add_query_to_output` function
    in adbo/exp.py.

    Args:
        json_path: The full path to the 'experiments.json' file.

    Returns:
        The full path to the simulation directory (e.g., '.../exp_0025')
        that had the highest 'res' value, or None if not found.

    Doctests:
    >>> import tempfile
    >>> import json
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     p = Path(tmpdir)
    ...     run1_path = p / "exp_0001"
    ...     run2_path = p / "exp_0002"
    ...     run1_path.mkdir()
    ...     run2_path.mkdir()
    ...     json_path = p / "experiments.json"
    ...     data = {
    ...         "0": {"res": 5.2, "": str(run1_path), "displacement": 0.1},
    ...         "1": {"res": 8.5, "": str(run2_path), "displacement": 0.5},
    ...         "2": {"res": 7.1, "": str(run1_path), "displacement": 0.2}
    ...     }
    ...     with open(json_path, 'w') as f:
    ...         json.dump(data, f)
    ...     best_run = find_best_run_from_json(str(json_path))
    ...     # The best path should be the one from run "1"
    ...     str(best_run) == str(run2_path)
    True
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return None

    with open(json_path, "r") as f:
        try:
            exp_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            return None

    best_res = -float("inf")
    best_run_path = None

    if not exp_data:
        print(f"Warning: No data found in {json_path}")
        return None

    for run_key, run_details in exp_data.items():
        if not isinstance(run_details, dict):
            print(f"Warning: Skipping run {run_key}, not a valid entry.")
            continue

        if "res" not in run_details or "" not in run_details:
            print(f"Warning: Skipping run {run_key}, missing 'res' or path key.")
            continue

        try:
            current_res = float(run_details["res"])
        except (ValueError, TypeError):
            print(f"Warning: Skipping run {run_key}, invalid 'res' value.")
            continue

        if current_res > best_res:
            best_res = current_res
            best_run_path = run_details[""]

    if best_run_path and not os.path.exists(best_run_path):
        print(f"Warning: Best run path '{best_run_path}' not found on disk.")
        # We'll still return it, as the path might be relative or
        # fixed by the conversion function's logic.

    return best_run_path


def convert_best_bo_run_to_swegnn(
    bayesopt_run_folder: str, output_nc_path: str
) -> None:
    """
    Finds the best run from a BO folder, converts it to SWE-GNN format,
    and saves it.

    (This is the same function as before).
    """
    print(f"Processing BO folder: {bayesopt_run_folder}")

    json_path = os.path.join(bayesopt_run_folder, "experiments.json")

    best_run_path = find_best_run_from_json(json_path)

    if best_run_path is None:
        raise Exception(f"Could not determine best run for {bayesopt_run_folder}.")

    # The path in the JSON might be absolute or relative.
    # If it's not absolute, assume it's relative to the BO folder.
    if not os.path.isabs(best_run_path):
        best_run_path = os.path.join(bayesopt_run_folder, best_run_path)

    # Clean up path (e.g., remove '..')
    best_run_path = os.path.normpath(best_run_path)

    print(f"  Found best run path: {best_run_path}")

    if not os.path.exists(best_run_path):
        raise FileNotFoundError(f"Best run directory not found at {best_run_path}")

    swegnn_ds = None
    try:
        print(f"  Converting {best_run_path} to SWE-GNN format...")
        swegnn_ds = swegnn_dg_from_mesh_ds_from_path(
            path=best_run_path, time_downsampling=7
        )

        print(f"  Saving to {output_nc_path}...")
        swegnn_ds.to_netcdf(output_nc_path)

        print(f"✅ Successfully converted and saved.")

    finally:
        # Ensure we close the dataset file handle
        if swegnn_ds is not None:
            swegnn_ds.close()


# --- New Main Driver Function ---


def process_all_bo_runs(
    base_exp_dir: str,
    output_base_dir: str,
    folder_names: List[str],
    skip_existing: bool = True,
) -> None:
    """
    Iterates over a list of BO experiment folders and converts the best
    run from each into a SWE-GNN NetCDF file.

    Args:
        base_exp_dir: The root directory where all BO folders live
                      (e.g., '/work/n01/n01/sithom/adcirc-swan/tcpips/exp').
        output_base_dir: The target directory to save .nc files
                         (e.g., '/work/n01/n01/sithom/adcirc-swan/SurgenetTestPH').
        folder_names: A list of string names for the subfolders within
                      `base_exp_dir` to process.
        skip_existing: If True, skips processing if the output .nc file
                       already exists.
    """

    print(f"Starting batch conversion...")
    print(f"Input base directory: {base_exp_dir}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Found {len(folder_names)} folders to process.")

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    for i, folder_name in enumerate(folder_names):
        print(f"\n--- Processing folder {i+1}/{len(folder_names)}: {folder_name} ---")

        # 1. Construct full input path
        input_folder_path = os.path.join(base_exp_dir, folder_name)

        # 2. Construct output filename and path
        output_name = (
            folder_name.replace("-", "_").replace("galverston", "galveston")
            + ".nc"  # get rid of embarrassing typo
        )
        output_file_path = os.path.join(output_base_dir, output_name)

        # 3. Check if we should skip
        if skip_existing and os.path.exists(output_file_path):
            print(f"  Skipping: Output file already exists at {output_file_path}")
            continue

        # 4. Run the conversion
        try:
            convert_best_bo_run_to_swegnn(input_folder_path, output_file_path)
        except Exception as e:
            print(f"  ‼️ FAILED to process {folder_name}: {e}")
            # Log error and continue to the next folder
            pass

    print("\n--- Batch processing complete. ---")


# --- Main execution ---
if __name__ == "__main__":

    # 1. Run the doctest for the helper function
    print("Running doctests...")
    doctest.testmod()
    print("Doctests complete.\n")

    # 2. Define your batch job parameters
    BASE_EXP_DIR = "/work/n01/n01/sithom/adcirc-swan/tcpips/exp"
    OUTPUT_DATA_DIR = "/work/n01/n01/sithom/adcirc-swan/SurgeNetTestPH"

    # This is the raw string you provided
    RAW_FOLDER_LIST_STRING = """galverston-2015-1 galverson-2015-2 galverston-2015-3 galverston-2100-1 galverston-2100-2 galverston-2100-3 miami-2015-1 miami-2015-2 miami-2015-3 miami-2100-1 miami-2100-2 miami-2100-3 new-orleans-2015-1 new-orleans-2015-2 new-orleans-2015-3 new-orleans-2100-1 new-orleans-2100-2 new-orleans-2100-3"""

    """
    2d-ani-1             galverston-2015-3      galverston-2100-8  miami-2100-11           new-orleans-2015-6
    8729840-2015           galverston-2015-3-ei   galverston-2100-9  miami-2100-2            new-orleans-2015-7
    8729840-2100           galverston-2015-4      miami-2015         miami-2100-2-ei         new-orleans-2015-8
    8735180-2015           galverston-2015-4-ei   miami-2015-0-ei    miami-2100-3            new-orleans-2015-9
    8735180-2100           galverston-2015-5      miami-2015-1       miami-2100-3-ei         new-orleans-2100
    8760922-2015           galverston-2015-6      miami-2015-1-ei    miami-2100-4            new-orleans-2100-0-ei
    8760922-2100           galverston-2015-7      miami-2015-10      miami-2100-4-ei         new-orleans-2100-1
    8761724-2015           galverston-2015-8      miami-2015-11      miami-2100-5            new-orleans-2100-1-ei
    8761724-2100           galverston-2015-9      miami-2015-2       miami-2100-6            new-orleans-2100-10
    8762075-2015           galverston-2100        miami-2015-2-ei    miami-2100-7            new-orleans-2100-11
    8762075-2100           galverston-2100-0-ei   miami-2015-3       miami-2100-8            new-orleans-2100-2
    8762482-2015           galverston-2100-1      miami-2015-3-ei    miami-2100-9            new-orleans-2100-2-ei
    8762482-2100           galverston-2100-1-ei   miami-2015-4       new-orleans-2015        new-orleans-2100-3
    8764044-2015           galverston-2100-10     miami-2015-4-ei    new-orleans-2015-0-ei   new-orleans-2100-3-ei
    8764044-2100           galverston-2100-11     miami-2015-5       new-orleans-2015-1      new-orleans-2100-4
    bo-m32-ei              galverston-2100-2      miami-2015-6       new-orleans-2015-1-ei   new-orleans-2100-5
    galverston-2015        galverston-2100-2-ei   miami-2015-7       new-orleans-2015-10     new-orleans-2100-6
    galverston-2015-0-ei   galverston-2100-3      miami-2015-8       new-orleans-2015-11     new-orleans-2100-7
    galverston-2015-1      galverston-2100-3-ei   miami-2015-9       new-orleans-2015-2      new-orleans-2100-8
    galverston-2015-1-ei   galverston-2100-4      miami-2100         new-orleans-2015-2-ei   new-orleans-2100-9
    galverston-2015-10     galverston-2100-4-ei   miami-2100-0-ei    new-orleans-2015-3
    galverston-2015-11     galverston-2100-5      miami-2100-1       new-orleans-2015-3-ei
    galverston-2015-2      galverston-2100-6      miami-2100-1-ei    new-orleans-2015-4
    galverston-2015-2-ei   galverston-2100-7      miami-2100-10      new-orleans-2015-5
    """

    # 3. Clean the list
    # .split() automatically handles newlines and spaces
    FOLDER_NAMES = RAW_FOLDER_LIST_STRING.split()

    # 4. Run the main processing function
    # Set `run_main_job = True` to execute the batch job.
    run_main_job = True  # <-- SET THIS TO True TO RUN

    if run_main_job:
        process_all_bo_runs(
            base_exp_dir=BASE_EXP_DIR,
            output_base_dir=OUTPUT_DATA_DIR,
            folder_names=FOLDER_NAMES,
            skip_existing=True,  # Set to False to re-process all
        )
    else:
        print("Main job execution is skipped.")
        print("Set 'run_main_job = True' inside the")
        print('`if __name__ == "__main__":` block to run.')

    # python -m adbo.create_test_set
