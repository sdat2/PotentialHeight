"""File handling utilities for tc potential size and intensity calculations."""

import os
from sithom.time import time_stamp
from tcpips.constants import CMIP6_PATH, RAW_PATH


def locker(path: str) -> callable:
    """Decorator to lock a function to prevent multiple instances from running on the same file.

    Args:
        path (str): path to stage directory where locks are stored

    Returns:
        callable: decorator function
    """

    def decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs) -> callable:
            exp = kwargs["exp"]
            model = kwargs["model"]
            member = kwargs["member"]
            if "typ" in kwargs:
                typ = kwargs["typ"]
                key = f"{exp}.{typ}.{model}.{member}"
            else:
                key = f"{exp}.{model}.{member}"
            lock_file_path = os.path.join(path, f"{key}.lock")
            if os.path.exists(lock_file_path):
                print(f"Already running {func.__name__} on {key}")
                return  # already regridding this file.
            else:
                with open(lock_file_path, "w") as f:
                    f.write(time_stamp())  # create lock file containing time stamp
                result = func(*args, **kwargs)
                os.remove(lock_file_path)  # remove lock file if function completes
                return result

        return wrapper

    return decorator


def file_crawler(folder_to_search: str = RAW_PATH) -> dict:
    """Function to crawl through a directory and return a dictionary of experiments, models, members."""
    # print(RAW_PATH)
    # ${ROOT}/${processing-step}/${experiment}/${atm/oc}/${model}/${member_id}.nc
    out_d = {}

    experiments = os.listdir(folder_to_search)
    for experiment in experiments:
        print(experiment)
        atm_oc = os.listdir(os.path.join(folder_to_search, experiment))
        for aoc in atm_oc:
            models = os.listdir(os.path.join(folder_to_search, experiment, aoc))
            for model in models:
                members = os.listdir(
                    os.path.join(folder_to_search, experiment, aoc, model)
                )
                for member in members:
                    print(member)
                    out_d[".".join([experiment, aoc, model, member])] = os.path.join(
                        folder_to_search, experiment, aoc, model, member
                    )
    return out_d


if __name__ == "__main__":
    # python -m tcpips.files
    print(file_crawler())
