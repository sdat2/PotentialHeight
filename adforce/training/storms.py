"""Storm metadata container and simulation-window helper.

Pure-Python (stdlib-only) pieces of the SurgeNet training-data generator,
split out of ``adforce/generate_training_data.py``. A ``Storm`` holds the
identity and track times of one IBTrACS tropical cyclone, and
``calculate_simulation_window`` turns those track times into ADCIRC
simulation start/end dates.
"""

from typing import Tuple
from datetime import datetime, timedelta


class Storm:
    """Minimal container for one IBTrACS storm's identity and track times.

    Mirrors the shape of a ``tropycal`` storm object just enough for the
    training-data generator.

    Args:
        sid: Storm id as raw IBTrACS bytes (e.g. ``b"2005236N23285"``);
            decoded with ``bytes.decode`` (crashes if passed a ``str``).
        name: Storm name as raw IBTrACS bytes (e.g. ``b"KATRINA"``); decoded
            the same way.
        time: List of ``datetime`` track points (may be empty).

    Attributes:
        id (str): Decoded storm id.
        name (str): Decoded storm name.
        time (list): The track times, stored as passed (not copied).
        year (Optional[int]): Year of the first track point, or ``None`` when
            ``time`` is empty/falsy.
    """

    # I hate classes, but this was to mirror tropycal
    def __init__(self, sid, name, time):
        self.id = sid.decode()
        self.name = name.decode()
        self.time = time
        if self.time:
            self.year = self.time[0].year
        else:
            self.year = None


def calculate_simulation_window(
    storm: Storm,
    spinup_days: float = 0.0,
    extra_days: float = 0.0,
) -> Tuple[datetime, datetime]:
    """
    Calculate the simulation start and end dates based on storm data.

    Args:
       storm (Storm): Storm object with 'time' attribute (list of datetime objects).
       spinup_days (float, optional): Number of days for model spin-up before storm start. Defaults to 0.
       extra_days (float, optional): Number of days to continue simulation after storm end. Defaults to 0.

    Returns:
    - simulation_start_date: Start date for the simulation (datetime).
    - simulation_end_date: End date for the simulation (datetime).
    """
    storm_start_time = storm.time[0]
    storm_end_time = storm.time[-1]
    spinup_duration = timedelta(days=spinup_days)
    start_date = storm_start_time
    simulation_start_date = start_date - spinup_duration
    simulation_end_date = storm_end_time + timedelta(days=extra_days)
    return simulation_start_date, simulation_end_date
