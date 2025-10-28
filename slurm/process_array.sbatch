#!/bin/bash --login

#SBATCH --job-name=pack_swegnn
#SBATCH --account=n02-bas
#SBATCH --partition=standard
#SBATCH --qos=standard

# --- STEP 1: THE "MANAGER" ARRAY ---
# We need to process ~500 runs.
# Each manager job will process 128 runs at a time.
# So, we need ceil(500 / 128) = 4 array tasks.
# Let's use 0-3 (which is 4 tasks).
#
#
#SBATCH --array=0-2

# --- STEP 2: RESOURCE REQUEST (PER MANAGER) ---
# Each *manager task* (0, 1, 2, or 3) gets ONE full node.
# We will then use all 128 cores on that node.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --time=01:00:00  # 1 hour to process 128 runs (adjust if needed)

# Log file for each manager task
#SBATCH --output=logs/%x-%A_%a.out

# --- Setup ---
source /work/n02/n02/sdat2/.bashrc
micromamba activate t1

# Matplotlib cache
mkdir -p $(pwd)'/matplotlib'
export MPLCONFIGDIR=$(pwd)'/matplotlib'

echo "---"
echo "Job Pack Manager Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_NTASKS"
echo "---"

# --- Define Paths ---
RUN_PARENT_DIR="/work/n02/n02/sdat2/adcirc-swan/worstsurge/run_5sec"
SAVE_PARENT_DIR="/work/n02/n02/sdat2/adcirc-swan/worstsurge/swegnn_5sec"
mkdir -p $SAVE_PARENT_DIR

# --- STEP 3: JOB PACKING LOGIC ---

# 1. Get the *full* list of all directories
DIRS=($(find $RUN_PARENT_DIR -mindepth 1 -maxdepth 1 -type d | sort))
NUM_DIRS=${#DIRS[@]}
echo "Found $NUM_DIRS total directories to process."

# 2. Define how many runs this manager task will process
TASKS_PER_NODE=128 # This MUST match --tasks-per-node above

# 3. Calculate the *chunk* of directories this manager task is responsible for
# Manager 0 (ID=0) processes dirs 0-127
# Manager 1 (ID=1) processes dirs 128-255
# ...and so on.
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $TASKS_PER_NODE ))

echo "This manager (Task ID $SLURM_ARRAY_TASK_ID) will process $TASKS_PER_NODE runs,"
echo "starting from directory index $START_INDEX."
echo "---"

# 4. Loop from 0 to 127
for i in $(seq 0 $(( $TASKS_PER_NODE - 1 ))); do
    
    # Get the *actual* directory index
    DIR_INDEX=$(( $START_INDEX + $i ))
    
    # Check if this directory exists (handles the last partial chunk)
    if [ $DIR_INDEX -ge $NUM_DIRS ]; then
        echo "Index $DIR_INDEX is out of bounds (>= $NUM_DIRS). No more dirs."
        continue
    fi
    
    # Get the directory name and output file name
    MY_RUN_DIR=${DIRS[$DIR_INDEX]}
    RUN_NAME=$(basename $MY_RUN_DIR)
    MY_SAVE_PATH="$SAVE_PARENT_DIR/${RUN_NAME}.nc"

    echo "Launching task for: $RUN_NAME"

    # --- STEP 4: LAUNCH THE PROCESS ---
    # This is the key:
    # 'srun'         : The Slurm command to run a task.
    # '--nodes=1'      : Use 1 node (from our allocation).
    # '--ntasks=1'     : This process is 1 task.
    # '--cpus-per-task=1': This process gets 1 CPU.
    # '--exclusive'  : Binds this process to its own core.
    # '--output=/dev/null': Discard stdout (logs go to the main file).
    # '&'              : Run this in the BACKGROUND.
    
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --output=/dev/null \
    python -m adforce.process_single \
        --run-path $MY_RUN_DIR \
        --save-path $MY_SAVE_PATH &

    # Add --use-dask here if you want to test it
    
done

# --- STEP 5: WAIT ---
# The script will get here in ~1 second after launching 128 jobs.
# 'wait' tells the script to pause here until all background (&)
# processes have finished.
echo "---"
echo "All $TASKS_PER_NODE tasks launched. Waiting for them to complete..."
wait
echo "All tasks complete. Manager job finished."