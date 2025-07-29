"""File handling utilities for tc potential size and intensity calculations.

Keep the file pipeline clean and organized with these functions.
"""
from typing import Dict
import os
import collections
from sithom.io import write_json
from sithom.time import time_stamp, timeit
from sithom.misc import human_readable_size
from .constants import CMIP6_PATH, RAW_PATH, DATA_PATH, REGRIDDED_PATH, PI_PATH


def locker(path: str) -> callable:
    """Decorator to lock a function to prevent multiple instances from running on the same file.
    I plan to use this for regridding a reprocessing steps, but this could be used for downloads.

    Args:
        path (str): path to stage directory where locks are stored.

    Returns:
        callable: decorator function

    Example:
        >>> TEST_PATH = os.path.join(CMIP6_PATH, "test")
        >>> import shutil
        >>> shutil.rmtree(TEST_PATH, ignore_errors=True)
        >>> os.makedirs(TEST_PATH, exist_ok=True)
        >>> @locker(path=TEST_PATH)
        ... def example(*args, **kw) -> bool:
        ...     ret = os.path.exists(os.path.join(TEST_PATH, f"{kw['exp']}.{kw['model']}.{kw['member']}.lock"))
        ...     return ret
        >>> example(exp="historical", model="model", member="member")
        True
        >>> def write_lock_file():
        ...    with open(os.path.join(TEST_PATH, "historical.model.member.lock"), "w") as f:
        ...        f.write(time_stamp())
        >>> write_lock_file()
        >>> example(exp="historical", model="model", member="member")
        Already running example on historical.model.member
        >>> shutil.rmtree(TEST_PATH)

    """

    def decorator(func: callable) -> callable:
        """
        Decorator to lock a function to prevent multiple instances from running on the same file.

        Args:
            func (callable): function to lock.

        Returns:
            callable: wrapper function
        """

        def locker_for_cmip6_member(*args, **kwargs) -> callable:
            """Wrapper that runs before and after the function to create and remove lock files.

            If the lock file exists, the function will not run and return None.
            """
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

        return locker_for_cmip6_member

    return decorator


def file_crawler(folder_to_search: str = RAW_PATH) -> dict:
    """Function to crawl through a directory and return a dictionary of experiments, models, members.

    Args:
        folder_to_search (str, optional): folder to search for files. Defaults to RAW_PATH.

    Returns:
        dict: dictionary of file names and sizes.
    """
    # print(RAW_PATH)
    # ${ROOT}/${processing-step}/${experiment}/${atm/oc}/${model}/${member_id}.nc
    file_d = collections.defaultdict(dict)
    experiments = os.listdir(folder_to_search)
    for experiment in experiments:
        print(experiment)
        atm_oc = os.listdir(os.path.join(folder_to_search, experiment))
        for aoc in atm_oc:
            models = os.listdir(os.path.join(folder_to_search, experiment, aoc))
            for model in models:
                member_ncs = [
                    x
                    for x in os.listdir(
                        os.path.join(folder_to_search, experiment, aoc, model)
                    )
                    if x.endswith(".nc") or x.endswith(".zarr")
                ]
                for member_nc in member_ncs:
                    print(member_nc)
                    file_name = os.path.join(
                        folder_to_search, experiment, aoc, model, member_nc
                    )
                    file_size = os.stat(file_name).st_size
                    hr_size = human_readable_size(file_size)
                    file_d[".".join([model, member_nc[:-3]])][
                        ".".join([experiment, aoc])
                    ] = (file_name, file_size, hr_size)
    return file_d


def histogram_dict_from_file_dict(file_d: dict) -> dict:
    """Function to create a dictionary of file sizes for plotting histograms.

    Args:
        file_d (dict): dictionary of file sizes.

    Returns:
        dict: dictionary of file sizes for plotting histograms.
    """
    hist_d = collections.defaultdict(list)
    for model_id in file_d:
        for type in file_d[model_id]:
            hist_d[type] += [file_d[model_id][type][1]]
    return hist_d


def hist_dict_plot(hist_d: dict) -> None:
    """Plot histogram of file sizes.

    Args:
        hist_d (dict): dictionary of file sizes
    """
    import matplotlib.pyplot as plt
    from sithom.plot import plot_defaults, label_subplots
    from tcpips.constants import FIGURE_PATH

    plot_defaults()

    fig, axs = plt.subplots(2, 2, sharey=True)  # , sharex=True,

    for i, exp in enumerate(["historical", "ssp585"]):
        for j, aoc in enumerate(["atmos", "ocean"]):
            key = f"{exp}.{aoc}"
            axs[i, j].hist(hist_d[key], bins=200, alpha=0.5, label=key)
            axs[i, j].set_title(key)

    axs[0, 0].set_ylabel("Count")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_xlabel("File Size (bytes)")
    axs[1, 1].set_xlabel("File Size (bytes)")
    label_subplots(axs, override="outside")
    plt.savefig(os.path.join(FIGURE_PATH, "file_size_histogram.pdf"))
    plt.clf()
    plt.close()


def find_missing_raw_cmip6_nc(file_d: dict) -> list:
    """Find missing files in the file dictionary.

    Args:
        file_d (dict): dictionary of file sizes.

    Returns:
        list: list of missing files.
    """
    import intake
    from tcpips.constants import CONVERSION_NAMES, PANGEO_CMIP6_URL

    cat = intake.open_esm_datastore(PANGEO_CMIP6_URL)

    miss_list = []
    for model_id in file_d:
        model, member = model_id.split(".")
        for exp in ["historical", "ssp585"]:
            for aoc, table in [("atmos", "Amon"), ("ocean", "Omon")]:
                type = f"{exp}.{aoc}"
                print((model, member, exp, table))
                cat_subset = cat.search(
                    experiment_id=[exp],
                    table_id=[table],
                    source_id=[model],
                    member_id=[member],
                    variable_id=CONVERSION_NAMES.keys(),
                    # grid_label="gn",
                ).unique()

                if type not in file_d[model_id] and len(cat_subset["zstore"]) != 0:
                    miss_list.append(".".join([model_id, type]))
    return sorted(miss_list)


@timeit
def get_task_dict(
    original_root: str = RAW_PATH, new_root: str = REGRIDDED_PATH
) -> dict:
    """Find all tasks to be done. Return a dictionary of tasks.

    Args:
        original_root (str, optional): Original root path. Defaults to RAW_PATH.
        new_root (str, optional): New root path. Defaults to REGRIDDED_PATH.

    Returns:
        dict: dictionary of tasks.
    """
    tasks = {}
    for exp in os.listdir(original_root):
        for typ in [
            x
            for x in os.listdir(os.path.join(original_root, exp))
            if os.path.isdir(os.path.join(original_root, exp))
        ]:
            for model in os.listdir(os.path.join(original_root, exp, typ)):
                for member in os.listdir(os.path.join(original_root, exp, typ, model)):
                    if member.endswith(".nc"):

                        member = member.replace(".nc", "")
                        key = f"{exp}.{typ}.{model}.{member}"
                        tasks[key] = {
                            "exp": exp,
                            "typ": typ,
                            "model": model,
                            "member": member,
                            "processed_exists": os.path.exists(
                                os.path.join(new_root, exp, typ, model, member) + ".nc"
                            ),
                            "locked": os.path.exists(
                                os.path.join(new_root, key) + ".lock"
                            ),
                        }
    return tasks


def find_atmos_ocean_pairs(path: str = REGRIDDED_PATH,
                           new_path: str = PI_PATH,
                           ) -> Dict[str, Dict[str, any]]:
    """
    Find the atmospheric and oceanic data pairs that can be combined to calculate potential intensity.

    Args:
        path (str): Path to the regridded data directory. Defaults to REGRIDDED_PATH.
        new_path (str): Path to the directory where potential intensity data will be stored. Defaults to PI_PATH.

    Returns:
        Dict[str, Dict[str, any]]: Dictionary of pairs.
    """

    pairs = {}
    for exp in [
        x
        for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x))
    ]:
        for model in os.listdir(os.path.join(path, exp, "ocean")):
            for member in [
                x.strip(".nc")
                for x in os.listdir(os.path.join(path, exp, "ocean", model))
            ]:
                key = f"{exp}.{model}.{member}"
                pi_lock = os.path.join(new_path, key + ".lock")
                print(key)
                oc_path = os.path.exists(
                    os.path.join(path, exp, "ocean", model, member) + ".nc"
                )
                oc_lock = os.path.exists(
                    os.path.join(path) + f"{exp}.ocean.{model}.{member}.lock"
                )
                at_path = os.path.exists(
                    os.path.join(path, exp, "atmos", model, member) + ".nc"
                )
                at_lock = os.path.exists(
                    os.path.join(path) + f"{exp}.atmos.{model}.{member}.lock"
                )
                if oc_path and at_path and not oc_lock and not at_lock:
                    pairs[f"{exp}.{model}.{member}"] = {
                        "exp": exp,
                        "model": model,
                        "member": member,
                        "locked": os.path.exists(pi_lock),
                    }
                if oc_lock:
                    print(f"Ocean lock file exists for {key}")
                if at_lock:
                    print(f"Atmos lock file exists for {key}")
                if not oc_path:
                    print(f"File missing for {exp}.ocean.{model}.{member}")
                if not at_path:
                    print(f"File missing for {exp}.atmos.{model}.{member}")

    return pairs


def investigate_cmip6_pairs(path: str = REGRIDDED_PATH) -> None:
    """
    Investigate the CMIP6 pairs to see if they are the correct size.

    Args:
        path (str): Path to the regridded data directory. Defaults to REGRIDDED_PATH.
    """

    def hr_file_size(filename: str) -> str:
        st = os.stat(filename)
        return human_readable_size(st.st_size)

    pairs = find_atmos_ocean_pairs()
    for key in pairs:
        print(key)
        for i in ["ocean", "atmos"]:
            print(
                i,
                hr_file_size(
                    os.path.join(
                        path,
                        pairs[key]["exp"],
                        i,
                        pairs[key]["model"],
                        pairs[key]["member"] + ".nc",
                    )
                ),
            )


if __name__ == "__main__":
    # python -m tcpips.files
    file_d = file_crawler()
    write_json(file_d, os.path.join(DATA_PATH, "raw.json"))
    hist_d = histogram_dict_from_file_dict(file_d)
    write_json(hist_d, os.path.join(DATA_PATH, "hist.json"))
    # hist_dict_plot(hist_d)
    fm = find_missing_raw_cmip6_nc(file_d)
    print(fm)
    write_json({x: "" for x in fm}, os.path.join(DATA_PATH, "missing.json"))
