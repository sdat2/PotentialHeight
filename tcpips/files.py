"""File handling utilities for tc potential size and intensity calculations."""

import os
import collections
from sithom.io import write_json
from sithom.time import time_stamp
from sithom.misc import human_readable_size
from tcpips.constants import CMIP6_PATH, RAW_PATH, DATA_PATH


def locker(path: str) -> callable:
    """Decorator to lock a function to prevent multiple instances from running on the same file.
    I plan to use this for regridding a reprocessing steps, but this could be used for downloads.

    Args:
        path (str): path to stage directory where locks are stored.

    Returns:
        callable: decorator function
    """

    def decorator(func: callable) -> callable:
        """
        Decorator to lock a function to prevent multiple instances from running on the same file.

        Args:
            func (callable): function to lock.

        Returns:
            callable: wrapper function
        """

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
                    if x.endswith(".nc")
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


def hist_d_f(file_d: dict) -> dict:
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


def hist_plot(hist_d: dict) -> None:
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


def find_missing(file_d: dict) -> list:
    """Find missing files in the file dictionary.

    Args:
        file_d (dict): dictionary of file sizes.

    Returns:
        list: list of missing files.
    """
    import intake
    from tcpips.constants import CONVERSION_NAMES

    url: str = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    cat = intake.open_esm_datastore(url)

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


if __name__ == "__main__":
    # python -m tcpips.files
    file_d = file_crawler()
    write_json(file_d, os.path.join(DATA_PATH, "raw.json"))
    hist_d = hist_d_f(file_d)
    write_json(hist_d, os.path.join(DATA_PATH, "hist.json"))
    # hist_plot(hist_d)
    fm = find_missing(file_d)
    print(fm)
    write_json({x: "" for x in fm}, os.path.join(DATA_PATH, "missing.json"))
