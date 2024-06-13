"""File handling utilities for tc potential size and intensity calculations."""

import os
from sithom.time import time_stamp


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
