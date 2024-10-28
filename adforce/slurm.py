import os
import numpy as np
import time
from omegaconf import DictConfig
from slurmpy import Slurm
from sithom.timeit import timeit
from sithom.constants import CONFIG_PATH


def is_slurm_job_finished(jid: int) -> bool:
    """
    Check if a SLURM job is finished using sacct.

    Args:
        jid (int): Job ID.

    Returns:
        bool: Whether the job is finished.
    """

    args = f"sacct -j {jid} -o state"
    job_states = [x.strip() for x in os.popen(args).read().strip().split("\n")]
    return (
        np.all([x == "COMPLETED" for x in job_states[2:]])
        if len(job_states) > 2
        else False
    )


@timeit
def setoff_slurm_job_and_wait(direc: str, config: DictConfig) -> int:
    """
    Run the ADCIRC model and wait for it to finish.

    Uses slurmpy to set off a slurm job and sacct to check if it has finished.

    Args:
        direc (str): Path to ADCIRC run folder.
        config (DictConfig): Configuration.

    Returns:
        int: Slurm Job ID.
    """
    # time_limit_str = "HH:mm:ss".format(time_limit)
    resolution = config.adcirc.resolution

    time_limit = config.slurm.options[config.adcirc.resolution].walltime
    s = Slurm(  # jobname,
        config.slurm.slurm_job_name,
        {
            "nodes": config.slurm.options[resolution].nodes,
            "account": config.options[resolution].account,
            "partition": config.partition,
            "qos": config[resolution].qos,
            "time": config.options[
                resolution
            ].walltime,  # str(timedelta(seconds=time_limit)),
            "tasks-per-node": config.slurm.tasks_per_node,  # number of cpus on archer2 node.
            "cpus-per-task": 1,
            "output": os.path.join(direc, "slurm.out"),
            "error": os.path.join(direc, "slurm.out"),
            "mail-type": "ALL",
            "mail-user": config.slurm.email,
        },
    )

    jid = s.run(
        f"""
module load {config.slurm.modules}

cd {direc}

EXE_PATH={config.files.exe_path}

# define variables
case_name=$SLURM_JOB_NAME # name for printing
np=$SLURM_NTASKS # how many parallel tasks to define

export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#...Run the case
echo ""
echo "|---------------------------------------------|"
echo "    TEST CASE: $case_name"
echo ""
echo -n "    Prepping case..."
$EXE_PATH/adcprep --np $np --partmesh >  adcprep.log
$EXE_PATH/adcprep --np $np --prepall  >> adcprep.log
if [ $? == 0 ] ; then
    echo "done!"
else
    echo "ERROR!"
    exit 1
fi

echo -n "    Runnning case..."
srun --distribution=block:block --hint=nomultithread $EXE_PATH/padcirc > padcirc_log.txt
exitstat=$?
echo "Finished"
echo "    ADCIRC Exit Code: $exitstat"
if [ "x$exitstat" != "x0" ] ; then
    echo "    ERROR: ADCIRC did not exit cleanly."
    exit 1
fi
echo ""

"""
    )

    time_total = 0
    is_finished = is_slurm_job_finished(jid)
    tinc = 1
    # wait twice as long as the time limit given to the slurm job
    # to account for queueing.
    while not is_finished and time_total < time_limit * 2:
        is_finished = is_slurm_job_finished(jid)
        time.sleep(tinc)
        time_total += tinc

    if not is_finished:
        print(f"Job {jid} did not finish in time")
    else:
        print(f"Job {jid} finished")
    return jid
