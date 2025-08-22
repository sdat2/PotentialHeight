"""Dask utilities for parallel processing."""

from typing import Callable
import time
from dask.distributed import Client, LocalCluster
from sithom.time import hr_time


def dask_cluster_wrapper(
    func: Callable,
    *args,
    **kwargs,
) -> None:
    """
    A wrapper to run a function on a Dask cluster.
    This is useful for running functions that take a long time to compute
    and can be parallelized across multiple workers.

    I think this only works as long as you use it in the 'if __name__ == "__main__":' part of the script.
    Not really sure.
    """
    tick = time.perf_counter()
    # trying new option for Archer2
    if "/work/" in str(os.getcwd()):  # assume Archer2
        print("Using Archer2 Dask cluster configuration.")
        print(f"cluster options: n_workers=16, threads_per_worker=8, memory_limit='15GB'")
        cluster = LocalCluster(n_workers=16,
            threads_per_worker=8,
            interface='lo',
            memory_limit='15GB',
        )
    else:
        print("Using default Dask cluster configuration.")
        cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    func(*args, **kwargs)

    client.close()
    cluster.close()  # Also a good idea to close the cluster
    tock = time.perf_counter()
    print(
        f"Function {func.__name__} for {str(args)}, {str(kwargs)} completed in {hr_time(tock-tick)} seconds using a dask cluster."
    )
