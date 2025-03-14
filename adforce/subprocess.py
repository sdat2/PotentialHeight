"""Run with subprocess rather than slurmpy."""

import os
import subprocess
from omegaconf import DictConfig  # Assuming omegaconf is available for DictConfig type


def setoff_subprocess_job_and_wait(direc: str, config: DictConfig) -> int:
    """
    Run the ADCIRC model in the given directory and wait for it to finish.
    This uses subprocess (no SLURM) to execute ADCIRC on the current node with parallelism.

    Args:
        direc (str): Path to ADCIRC run folder.
        config (DictConfig): Configuration object with required settings.

    Returns:
        int: An integer status code (0 if successful, non-zero if errors occurred).
    """
    # Prepare log file to capture output and errors (similar to slurm.out)
    log_path = os.path.join(direc, "slurm.out")
    # Open the main log file in write mode (overwrites if exists)
    with open(log_path, "w") as log_file:
        try:
            # 1. Load required modules (if any) for ADCIRC environment
            # If config specifies modules to load (e.g., for compilers/MPI), load them using subprocess.
            modules_to_load = getattr(config.slurm, "modules", None)
            if modules_to_load:
                log_file.write(f"Loading modules: {modules_to_load}\n")
                # Use a shell command to load modules. This assumes a module system is available.
                mod_proc = subprocess.run(
                    f"module load {modules_to_load}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if mod_proc.returncode != 0:
                    # Log the error and raise an exception if module loading failed.
                    error_msg = mod_proc.stderr.decode().strip() or "Unknown error"
                    log_file.write(
                        f"ERROR: Module load failed with message: {error_msg}\n"
                    )
                    raise RuntimeError(f"Module loading failed: {modules_to_load}")

            # 2. Change to the target directory for the ADCIRC run
            log_file.write(f"Changing working directory to: {direc}\n")
            try:
                os.chdir(direc)
            except Exception as e:
                log_file.write(f"ERROR: Could not change directory to {direc}: {e}\n")
                raise

            # 3. Set up environment variables for the run (single-threaded per MPI task)
            os.environ["OMP_NUM_THREADS"] = (
                "1"  # No OpenMP multi-threading within each MPI task
            )
            os.environ["SRUN_CPUS_PER_TASK"] = (
                "1"  # Ensure srun treats each task as single-core
            )

            # Determine the number of parallel tasks (np) based on config.
            # Use tasks_per_node * nodes (assuming one node unless specified otherwise in config).
            resolution_key = config.adcirc.resolution.value  # e.g., "high", "low", etc.
            tasks_per_node = (
                config.slurm.tasks_per_node - 20
            )  # 20 cores reserved for system
            num_nodes = config.slurm.options[resolution_key].nodes
            np = tasks_per_node * num_nodes  # total number of MPI processes for ADCIRC

            # Get the path to ADCIRC executables (adcprep and padcirc)
            exe_path = config.files.exe_path  # Directory containing ADCIRC executables

            # Derive a case name for logging (similar to SLURM job name if available)
            case_name = str(config.name) if hasattr(config, "name") else "ADCIRC_Case"

            # 4. Log the start of the case in the main log (mimicking SLURM job output)
            log_file.write("\n")  # blank line for clarity
            log_file.write("|---------------------------------------------|\n")
            log_file.write(f"    TEST CASE: {case_name}\n")
            log_file.write("\n")
            log_file.write("    Prepping case...")  # -n (no newline) equivalent

            # 5. Run ADCIRC preparation steps using adcprep
            # Step 5a: Partition the mesh (`adcprep --np <np> --partmesh`)
            adcprep_log = os.path.join(direc, "adcprep.log")
            try:
                with open(adcprep_log, "w") as prep_log:
                    proc_partmesh = subprocess.run(
                        [f"{exe_path}/adcprep", "--np", str(np), "--partmesh"],
                        stdout=prep_log,
                        stderr=subprocess.STDOUT,
                    )
            except FileNotFoundError:
                log_file.write(
                    "\nERROR: adcprep executable not found at path: "
                    f"{exe_path}/adcprep\n"
                )
                raise RuntimeError("adcprep not found. Check ADCIRC executable path.")

            # Step 5b: Prepare all input partitions (`adcprep --np <np> --prepall`)
            with open(adcprep_log, "a") as prep_log:
                proc_prepall = subprocess.run(
                    [f"{exe_path}/adcprep", "--np", str(np), "--prepall"],
                    stdout=prep_log,
                    stderr=subprocess.STDOUT,
                )

            # Check if adcprep steps were successful
            if proc_partmesh.returncode != 0 or proc_prepall.returncode != 0:
                # If either adcprep step failed, log and raise an error
                log_file.write(
                    "ERROR!\n"
                )  # completes the "Prepping case..." line with ERROR
                log_file.write("    ERROR: ADCIRC preprocessing (adcprep) failed.\n")
                raise RuntimeError(
                    "ADCIRC preprocessing failed (see adcprep.log for details)."
                )
            else:
                # Both adcprep steps succeeded
                log_file.write(
                    "done!\n"
                )  # completes the "Prepping case..." line with done!

            # 6. Run the main ADCIRC simulation using padcirc via srun
            log_file.write(
                "    Running case..."
            )  # similar to echo -n "Running case..."
            log_file.flush()  # flush so "Running case..." appears in log before the run

            padcirc_log = os.path.join(direc, "padcirc_log.txt")
            # Use srun to launch the ADCIRC MPI executable across CPUs
            with open(padcirc_log, "w") as run_log:
                proc_run = subprocess.run(
                    [
                        "srun",
                        "--distribution=block:block",
                        "--hint=nomultithread",  # set number of processes to np
                        "--ntasks=" + str(np),
                        f"{exe_path}/padcirc",
                    ],
                    stdout=run_log,
                    stderr=subprocess.STDOUT,
                )
            exit_code = proc_run.returncode  # Capture ADCIRC exit code
            log_file.write("Finished\n")  # completes the "Running case..." line
            log_file.write(f"    ADCIRC Exit Code: {exit_code}\n")

            # 7. Check for errors in the ADCIRC run
            if exit_code != 0:
                log_file.write("    ERROR: ADCIRC did not exit cleanly.\n")
                # Raise an exception to indicate failure (could also return non-zero code)
                raise RuntimeError(f"ADCIRC run failed with exit code {exit_code}.")

            # If we reach here, everything succeeded
            log_file.write("\nJob completed successfully.\n")
            return (
                0  # Return 0 (success code; no SLURM job ID available in no-slurm mode)
            )

        except Exception as e:
            # If any exception occurred, ensure it's logged
            log_file.write(f"\nException: {str(e)}\n")
            # (Optionally, we could return a non-zero code here instead of raising,
            #  but raising lets the caller handle it as an error condition.)
            raise
