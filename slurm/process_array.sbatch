#!/bin/bash --login

#SBATCH --job-name=process_swegnn
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

# --- JOB ARRAY ---
# Replace '499' with your (total_runs - 1)
# %128 means "run 128 tasks at a time"
# SBATCH --array=0-499%128 

# Output log file:
# %x = job name, %A = job ID, %a = array task ID
#SBATCH --output=logs/%x-%A_%a.out

##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=sdat2@cam.ac.uk

# --- Setup ---
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba activate t1

# Matplotlib cache
mkdir $(pwd)'/matplotlib'
export MPLCONFIGDIR=$(pwd)'/matplotlib'

echo "Which python:"
which python
echo "---"

# --- Define Paths ---
RUN_PARENT_DIR="/work/n02/n02/sdat2/adcirc-swan/worstsurge/run_5sec"
SAVE_PARENT_DIR="/work/n02/n02/sdat2/adcirc-swan/worstsurge/swegnn_5sec"

# Create the main output directory
mkdir -p $SAVE_PARENT_DIR

# --- Job Array Logic ---
# 1. Create a bash array of all run directories
DIRS=($(find $RUN_PARENT_DIR -mindepth 1 -maxdepth 1 -type d | sort))

# 2. Check if the array is empty
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Error: No run directories found in $RUN_PARENT_DIR"
    exit 1
fi

# 3. Get the specific directory for THIS Slurm array task
MY_RUN_DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME=$(basename $MY_RUN_DIR)

# 4. Check if the directory was found
if [ -z "$MY_RUN_DIR" ]; then
    echo "Error: Could not find directory for task ID $SLURM_ARRAY_TASK_ID."
    echo "Total directories found: ${#DIRS[@]}"
    exit 1
fi

# 5. Define the full path for the output file
MY_SAVE_PATH="$SAVE_PARENT_DIR/${RUN_NAME}.nc"

# --- Run the Job ---
echo "Slurm Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Processing Run Dir: $MY_RUN_DIR"
echo "Saving Output To:   $MY_SAVE_PATH"
echo "---"

# --- CHOOSE YOUR MODE ---

# OPTION 1: Run with use_dask=False (My recommendation for Lustre)
# The --use-dask flag is absent, so the script defaults to False.
echo "Running with use_dask=False"
python -m adforce.process_single \
    --run-path $MY_RUN_DIR \
    --save-path $MY_SAVE_PATH

# OPTION 2: Run with use_dask=True (To test your hypothesis)
# Comment out the block above and uncomment this block.
# The --use-dask flag is present, so the script sets it to True.
# echo "Running with use_dask=True"
# python -m adforce.process_single_run \
#     --run-path $MY_RUN_DIR \
#     --save-path $MY_SAVE_PATH \
#     --use-dask

echo "Task $SLURM_ARRAY_TASK_ID complete."