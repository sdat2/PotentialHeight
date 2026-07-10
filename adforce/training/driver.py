"""Orchestration and CLI for generating the SurgeNet ADCIRC training runs.

Split out of ``adforce/generate_training_data.py``. ``drive_all_adcirc``
selects the U.S.-landfalling storms from IBTrACS (via ``tcpips.ibtracs``),
generates each storm's ADCIRC input deck (``adforce.training.inputs``) and
runs ASWIP/adcprep/padcirc per storm through
``adforce.subprocess.setoff_subprocess_job_and_wait``. ``main`` is the
argparse entry point behind ``python -m adforce.generate_training_data``.

Note: the run configuration is now composed via
``adforce.wrap.get_default_config()`` instead of a bare
``hydra.initialize``/``compose`` pair, so the ``ADCIRC_EXE_PATH`` /
``ADCIRC_NP`` / ``WORSTSURGE_MODULES`` environment overrides are honored and
no global hydra state is leaked.

Usage:
    python -m adforce.generate_training_data [--test-single] [--test-nosubprocess] [--recommended-dt 5.0] [--runs-parent-name test_runs]

TODO: Add some yaml to each storm to track key characteristics for readability.
TODO: Add ability to split a task list between multiple slurm jobs.
TODO: Plot and summarise this training data.
"""

from typing import Optional
import os
import traceback
import argparse
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..subprocess import setoff_subprocess_job_and_wait
from ..wrap import get_default_config
from ..constants import PROJ_PATH
from tcpips.ibtracs import na_landing_tcs
from .storms import Storm
from .inputs import generate_adcirc_inputs

RUNS_PARENT_DIR = os.path.join(PROJ_PATH, "runs")
os.makedirs(RUNS_PARENT_DIR, exist_ok=True)


def drive_all_adcirc(
    test_single=False,
    test_nosubprocess=False,
    runs_parent_name: Optional[str] = None,
    recommended_dt: float = None,
) -> None:
    """
    Generate all storm inputs using NWS=20.

    For every selected storm this creates ``<parent>/{i}_{NAME}_{year}/``
    under ``RUNS_PARENT_DIR`` (or ``<PROJ_PATH>/<runs_parent_name>``), writes
    the ADCIRC input deck there, and (unless ``test_nosubprocess``) runs the
    simulation via a local subprocess. Runs whose ``slurm.out`` already
    contains "Job completed successfully." are skipped, so the loop is
    resumable. Per-storm failures are printed (with traceback) and do not
    stop the loop.

    Args:
        test_single (bool, optional): If True just use Katrina 2005. Defaults to False.
        test_nosubprocess (bool, optional): If True do not set of subprocess to run ADCIRC. Defaults to False.
        runs_parent_name (Optional[str], optional): Name of the runs parent
            directory, joined onto PROJ_PATH. Defaults to None (use
            RUNS_PARENT_DIR, i.e. "<PROJ_PATH>/runs").
        recommended_dt (float, optional): ADCIRC timestep in seconds passed to
            generate_adcirc_inputs. Defaults to None (the CLI passes 5.0).
    """
    # it could be good to add a flag just to run Katrina
    # python -m adforce.generate_training_data
    if runs_parent_name is None:
        runs_parent_dir = RUNS_PARENT_DIR
    else:
        runs_parent_dir = os.path.join(PROJ_PATH, runs_parent_name)

    # --- Load Config (honors ADCIRC_EXE_PATH/ADCIRC_NP/WORSTSURGE_MODULES env overrides) ---
    print("Loading ADCIRC run configuration...")
    cfg = get_default_config()
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


def main() -> None:
    """Argparse CLI entry point (``python -m adforce.generate_training_data``).

    Parses the command-line flags and calls ``drive_all_adcirc``. ``prog`` is
    pinned to "generate_training_data.py" so the --help output is unchanged
    from before the module was split.
    """
    # python -m adforce.generate_training_data
    parser = argparse.ArgumentParser(
        prog="generate_training_data.py",
        description="Generate ADCIRC inputs and run simulations for all U.S. landfalling storms.",
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


if __name__ == "__main__":
    main()
