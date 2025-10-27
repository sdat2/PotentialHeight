"""Let's go through the training ADICRC runs and check if they were sucessful.

# TODO: add additional checks for
- maxele values (find maxele, if too high, mark as failed)
- check if output files are generated (fort.63, fort.64, fort.73, fort.74)
"""
from typing import List
import os
import argparse
from .constants import PROJ_PATH

def status_list(parent_dir: str) -> List[bool]:
    """Check the status of ADCIRC training runs in the given parent directory.

    Args:
        parent_dir (str): Path to the parent directory containing run subdirectories.

    Returns:
        List[bool]: A list of booleans indicating success (True) or failure (False) for each run.
    """

    out_l = []
    run_dir_list = os.listdir(parent_dir)
    for run_dir in run_dir_list:
        run_path =  os.path.join(parent_dir, run_dir)
        slurm_path = os.path.join(run_path, "slurm.out")
        if os.path.exists(slurm_path) is False:
            print(f"Run {run_dir}: slurm.out not found, marking as FAILED")
            out_l.append(False)
            continue
        with open(slurm_path, "r") as slurm_out_file:
            slurm_out_lines = slurm_out_file.readlines() # produces list of strings
            # print(slurm_out_lines)
            if "Job completed successfully.\n" in slurm_out_lines:
                out_l.append(True)
                print(f"Run {run_dir}: SUCCESS")
            else:
                out_l.append(False)
                print(f"Run {run_dir}: FAILED")

    successful_runs_num = sum(out_l)
    print(f"Total successful runs: {successful_runs_num} out of {len(run_dir_list)}")
    print(f"Success rate {successful_runs_num/len(out_l)*100:.1f}%")
    return out_l


if __name__ == "__main__":
    # python -m adforce.check_training_runs --runs-parent-name run_5sec
    # python -m adforce.check_training_runs --runs-parent-name runs
    # python -m adforce.check_training_runs --runs-parent-name run_10sec

    parser = argparse.ArgumentParser(
        description="Check the status of ADCIRC training runs."
    )
    parser.add_argument("--runs-parent-name", type=str, default=None,
        help="Name of the parent directory for runs inside the project path. Defaults to the constant RUNS_PARENT_DIR.")
    args = parser.parse_args()

    parent_dir = os.path.join(PROJ_PATH, args.runs_parent_name) if args.runs_parent_name is not None else os.path.join(PROJ_PATH, "runs")

    status_list(parent_dir)
