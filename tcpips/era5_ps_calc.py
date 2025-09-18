"""
Calculating the potential size is very computationally intensive,
so we use dask to parallelize the work,
ideally across many nodes on the Archer2 cluster.
"""
import time
import xarray as xr
import numpy as np
import dask
import dask.array as da
from dask.distributed import Client
try:
    from dask_mpi import initialize
except ImportError:
    print(
        "dask-mpi is not installed. Please install it with `pip install dask-mpi`."
    )
from tcpips.era5 import calculate_potential_sizes


def process_large_dataset_ex(client: Client, size: int) -> float:
    """
    Process a large dataset using Dask.
    This function creates a large random dataset and performs a mean calculation on it.

    Args:
        client (Client): Dask client connected to the cluster.
        size (int): Size of the dataset to create (size x size).

    Returns:
        float: The mean of the dataset.
    """

    num_workers = len(client.scheduler_info()["workers"])
    print(f"Starting computation on a Dask cluster with {num_workers} workers.")

    random_data = da.random.random((size, size), chunks=(500, 500))

    ds = xr.Dataset({"temperature": (("x", "y"), random_data)})

    print(f"Dataset created with {ds.nbytes / 1e9:.2f} GB of data.")
    print("Performing computationally intense operation (mean calculation)...")
    result = ds.temperature.mean().compute()
    print("Computation finished successfully.")
    return result


def run_ps_calc(client: Client, start_year=1980, end_year=1989) -> None:
    """Run the potential size calculation using Dask.

    Args:
        client (Client): Dask client connected to the cluster.
    """
    num_workers = len(client.scheduler_info()["workers"])
    print(f"Starting computation on a Dask cluster with {num_workers} workers.")
    calculate_potential_sizes(start_year=start_year, end_year=end_year)
    print("Computation finished successfully.")
    return None


if __name__ == "__main__":
    # python -m tcpips.era5_ps_calc
    # --- Dask Memory Configuration ---
    # This is the key to preventing memory issues with Lustre.
    # We tell Dask to leave a 30-40% buffer for the OS file cache.
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.60,  # Start spilling to disk at 60% memory usage
            "distributed.worker.memory.spill": 0.70,  # Spill until usage is down to 70%
            "distributed.worker.memory.pause": 0.80,  # Pause worker threads at 80% usage
            "distributed.worker.memory.terminate": 0.95,  # Kill the worker at 95% usage
        }
    )

    # Initialize the dask-mpi cluster
    initialize(
        interface="hsn0"
    )  # 'ib0' is often the InfiniBand interface on HPCs, but letting it auto-detect is usually fine.

    # Connect to the cluster
    client = Client()

    print(f"Dask dashboard link: {client.dashboard_link}")
    print(f"Client info: {client}")
    print(f"Cluster info: {client.cluster}")
    print(f"Scheduler info: {client.scheduler_info()}")
    print(f"Workers info: {client.scheduler_info()['workers']}")

    print("\n--- Starting main computation ---")

    # Run your main computation
    start_time = time.perf_counter()
    # final_result = process_large_dataset(client, size=50000)
    # run_ps_calc(client, start_year=1980, end_year=1989)
    # run_ps_calc(client, start_year=1990, end_year=1999)
    # run_ps_calc(client, start_year=2000, end_year=2009)
    # run_ps_calc(client, start_year=2010, end_year=2019)
    run_ps_calc(client, start_year=2020, end_year=2024, dry_run=False)
    end_time = time.perf_counter()
    # print(f"\nFinal Result: {final_result:.4f}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
