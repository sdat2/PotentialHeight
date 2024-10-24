"""Some time conversion functions to deal with time for fort.22.nc."""

from typing import Union, Optional
import datetime


def time_to_datetime(time: int, units: str, calendar: str) -> any:
    """Convert time in minutes to datetime objects.

    We need to convert to the proleptic_gregorian calendar.

    Args:
        time (int): time in minutes.
        units (str): units of time.
        calendar (str): calendar type.

    Returns:
        datetime.datetime: datetime object

    Examples::
        >>> time_to_datetime(7680900, "minutes since 1990-01-01T01:00:00+00:00", "proleptic_gregorian") == datetime.datetime(2004, 8, 9, 0, 0)
        True
    """
    assert (
        calendar == "proleptic_gregorian"
        and units == "minutes since 1990-01-01T01:00:00+00:00"
    )
    return datetime.datetime(1990, 1, 1, 1, 0, 0) + datetime.timedelta(minutes=time)


def str_to_datetime(time_str: str) -> datetime.datetime:
    """
    Convert time string to datetime object.

    Args:
        time_str (str): time string in the format "%Y-%m-%dT%H:%M:%S"

    Returns:
        datetime.datetime: datetime object

    Examples::
        >>> str_to_datetime("2004-08-08T23:00:00") == datetime.datetime(2004, 8, 8, 23, 0)
        True
    """
    return datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")


def str_to_time(time_str: str, units: str, calendar: str) -> int:
    """
    Convert time string to time in minutes.

    Args:
        time_str (str): time string in the format "%Y-%m-%dT%H:%M:%S"

    Returns:
        int: time in minutes from 1990-01-01T01:00:00

    Examples::
        >>> str_to_time("2004-08-09T00:00:00") == 7680000
        True
    """
    return datetime_to_time(str_to_datetime(time_str), units, calendar)


def datetime_to_time(dt: int, units: str, calendar: str) -> any:
    """Convert time in minutes to datetime objects.

    We need to convert to the proleptic_gregorian calendar.

    Args:
        time (int): time in minutes.
        units (str): units of time.
        calendar (str): calendar type.

    Returns:
        time: datetime object

    Examples::
        >>> datetime_to_time(datetime.datetime(2004, 8, 9, 0, 0), "minutes since 1990-01-01T01:00:00+00:00", "proleptic_gregorian") == 7680900
        True
    """
    assert (
        calendar == "proleptic_gregorian"
        and units == "minutes since 1990-01-01T01:00:00+00:00"
    )
    return int((dt - datetime.datetime(1990, 1, 1, 1, 0, 0)).total_seconds() / 60)


def unknown_to_time(
    unknown_t: Union[str, datetime.datetime, int],
    units: Optional[str] = None,
    calendar: Optional[str] = None,
) -> int:
    """
    Convert unknown time to time in minutes.

    Args:
        unknown_t (Union[str, datetime.datetime, int]): unknown time.
        units (Optional[str]): units of time.
        calendar (Optional[str]): calendar type.

    Returns:
        int: time in minutes from 1990-01-01T01:00:00
    """

    if isinstance(unknown_t, str):
        time = str_to_time(unknown_t, units, calendar)
    elif isinstance(unknown_t, datetime.datetime):
        time = datetime_to_time(unknown_t, units, calendar)
    elif isinstance(unknown_t, int):
        time = unknown_t
    else:
        raise ValueError("start time must be a string, datetime object, or integer")
    return time
