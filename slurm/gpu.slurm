#!/bin/bash --login

#SBATCH --job-name=single-gpu-max-walltime
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd 
#SBATCH --account=n02-bas

#SBATCH --time=12:00:00                # Maximal wall time for gpu-shd QoS

#SBATCH --output=logs/%x-%j.out


# Optional: Load any necessary modules for your GPU application environment
# ARCHER2 uses environment modules to manage software.
# The specific modules depend on your application and how it was compiled.
# Common modules for AMD GPU development include:
# module load PrgEnv-amd # Or PrgEnv-cray, PrgEnv-gnu depending on your compiler suite
module load rocm       # ROCm is AMD's Radeon Open Compute platform, crucial for their GPUs
# module load craype-accel-amd-gfx90a # Specifies the GPU target architecture (AMD MI210)
# module load craype-x86-milan       # Specifies the CPU target architecture
# (Refer to ARCHER2 documentation for specific modules your application needs [2])

# Echo some useful job information into the output log
echo "ARCHER2 GPU Job: $SLURM_JOB_NAME (Job ID: $SLURM_JOB_ID)"
echo "Running on host: $(hostname)"
echo "Job submitted to partition: $SLURM_JOB_PARTITION"
echo "Job requested $SLURM_GPUS GPU(s) on this node."
# ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES typically shows the specific GPU allocated
echo "Allocated GPU(s) (ROCR_VISIBLE_DEVICES): $ROCR_VISIBLE_DEVICES"
echo "Allocated GPU(s) (HIP_VISIBLE_DEVICES): $HIP_VISIBLE_DEVICES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo ""

# Command to check the assigned GPU (optional, but useful for verification and logging)
echo "Running rocm-smi to check GPU status:"
srun --ntasks=1 rocm-smi # rocm-smi is AMD's tool for monitoring GPU status [4]
echo ""

# Your GPU application command(s)
# Replace './my_gpu_program.x' with the actual command to run your program.
# Ensure your executable is in your PATH or provide the full path.
# Use srun to launch your parallel tasks.
# For a single GPU, you typically launch a single process that controls the GPU.
# The --cpus-per-task here is for the srun step, not an #SBATCH directive.
# It specifies how many CPUs the task launched by srun can use.
# Given that 1 GPU is associated with 8 CPU cores (see Section V.A),
# you might use --cpus-per-task=8 if your application can leverage them.

echo "Running rocm-smi to check GPU status:"
srun --ntasks=1 rocm-smi # rocm-smi is AMD's tool for monitoring GPU status [4]
echo ""


# try to activate micromamba
source /work/n02/n02/sdat2/.bashrc
which micromamba
micromamba --version
micromamba activate t1

mkdir $(pwd)'/matplotlib' # create matplotlib cache directory
export MPLCONFIGDIR=$(pwd)'/matplotlib'
echo "which python"
which python



echo "Launching GPU application..."
srun --ntasks=1 --cpus-per-task=8 python -c "import tensorflow as tf
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU'))); gpus = tf.config.experimental.list_physical_devices('GPU'); tf.config.experimental.set_memory_growth(gpus[0], True)"
                                                 # and resource allocation (see Section V.A)


echo "Launching Python/TensorFlow script..."
srun --ntasks=1 --cpus-per-task=8 python -m worst.vary_noise

echo ""
echo "Job finished at: $(date)"