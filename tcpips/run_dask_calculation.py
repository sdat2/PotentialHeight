import time
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# Assuming this function is in your project and does the actual work
from tcpips.era5 import calculate_potential_sizes

# --- Dask Memory Configuration ---
# This is still good practice.
dask.config.set(
    {
        "distributed.worker.memory.target": 0.60,
        "distributed.worker.memory.spill": 0.70,
        "distributed.worker.memory.pause": 0.80,
        "distributed.worker.memory.terminate": 0.95,
    }
)

print("--- Setting up Dask Cluster ---")

# --- KEY CHANGES ---
# We define the cluster resources here instead of in the Slurm script.
cluster = SLURMCluster(
    # Slurm settings
    queue="standard",
    account="n02-bas",
    # Dask worker configuration
    cores=128,  # Total cores on one ARCHER2 node
    processes=16,  # THE FIX: Launch 16 workers per node
    memory="240GB",  # Total memory on one ARCHER2 node
    # Other settings
    walltime="24:00:00",
    interface="hsn0",  # High-speed network interface
    job_extra_directives=[
        "--qos=standard",  # Pass your QOS to worker jobs
        "--hint=nomultithread",  # Recommended for ARCHER2
    ],
)

print("Scaling cluster to 50 nodes...")
cluster.scale(jobs=10)

# Connect a client to the cluster
client = Client(cluster)

try:
    # Wait for at least one full node of workers to be ready
    print("Waiting for workers to start...")
    client.wait_for_workers(n_workers=16)

    print(f"Dask dashboard link: {client.dashboard_link}")
    print(f"Cluster is ready. Found {len(client.scheduler_info()['workers'])} workers.")

    print("\n--- Starting main computation ---")
    start_time = time.perf_counter()

    # The 5x5 chunking should be done inside this function
    # This is the calculation for the 2000s decade.
    calculate_potential_sizes(start_year=2000, end_year=2009, dry_run=False)

    end_time = time.perf_counter()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

finally:
    # Always shut down the cluster and client
    print("--- Computation finished, closing cluster. ---")
    client.close()
    cluster.close()
