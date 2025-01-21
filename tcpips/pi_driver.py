import os
from typing import Dict
from sithom.misc import human_readable_size
from tcpips.constants import REGRIDDED_PATH, PI_PATH


def find_atmos_ocean_pairs() -> Dict[str, Dict[str, any]]:
    """
    Find the atmospheric and oceanic data pairs that can be combined to calculate potential intensity.

    Returns:
        Dict[str, Dict[str, any]]: Dictionary of pairs.
    """

    pairs = {}
    for exp in [
        x
        for x in os.listdir(REGRIDDED_PATH)
        if os.path.isdir(os.path.join(REGRIDDED_PATH, x))
    ]:
        for model in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean")):
            for member in [
                x.strip(".nc")
                for x in os.listdir(os.path.join(REGRIDDED_PATH, exp, "ocean", model))
            ]:
                key = f"{exp}.{model}.{member}"
                pi_lock = os.path.join(PI_PATH, key + ".lock")
                print(key)
                oc_path = os.path.exists(
                    os.path.join(REGRIDDED_PATH, exp, "ocean", model, member) + ".nc"
                )
                oc_lock = os.path.exists(
                    os.path.join(REGRIDDED_PATH) + f"{exp}.ocean.{model}.{member}.lock"
                )
                at_path = os.path.exists(
                    os.path.join(REGRIDDED_PATH, exp, "atmos", model, member) + ".nc"
                )
                at_lock = os.path.exists(
                    os.path.join(REGRIDDED_PATH) + f"{exp}.atmos.{model}.{member}.lock"
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


def investigate_cmip6_pairs() -> None:
    """
    Investigate the CMIP6 pairs to see if they are the correct size.
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
                        REGRIDDED_PATH,
                        pairs[key]["exp"],
                        i,
                        pairs[key]["model"],
                        pairs[key]["member"] + ".nc",
                    )
                ),
            )


if __name__ == "__main__":
    investigate_cmip6_pairs()
