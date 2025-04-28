"""
adforce.geo

"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Mean spherical Earth radius in metres (float32 for consistency)
EARTH_RADIUS: np.float32 = np.float32(6_371_008.8)


def _to_rad(*deg_arrays: ArrayLike) -> list[NDArray[np.float32]]:
    """Convert one or more degree arrays to radians (vectorised, float32)."""
    return [np.radians(np.asarray(a, dtype=np.float32)) for a in deg_arrays]


def haversine_dist_bearing(
    lon1: ArrayLike,
    lat1: ArrayLike,
    lon2: ArrayLike,
    lat2: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Great-circle distance and initial bearing on a spherical Earth (float32).

    Args:
        lon1: Longitudes of the start points (°E).
        lat1: Latitudes  of the start points (°N).
        lon2: Longitudes of the end   points (°E).
        lat2: Latitudes  of the end   points (°N).

    Returns:
        Tuple ``(distance_m, bearing_deg)``:

        * **distance_m** - float32 great-circle distance in metres.
        * **bearing_deg**  - float32 initial bearing, clockwise from north
          in degrees ``[0, 360)``.

    Examples:
        >>> d, b = haversine_dist_bearing(0.0, 0.0, 0.0, 1.0)
        >>> round(float(d), -3)        # ≈ 111. km
        111000.0
        >>> round(float(b), 1)
        0.0
    """
    lon1r, lat1r, lon2r, lat2r = _to_rad(lon1, lat1, lon2, lat2)
    dlon: NDArray[np.float32] = lon2r - lon1r
    dlat: NDArray[np.float32] = lat2r - lat1r

    a: NDArray[np.float32] = np.sin(dlat / 2, dtype=np.float32) ** np.float32(
        2.0
    ) + np.cos(lat1r, dtype=np.float32) * np.cos(lat2r, dtype=np.float32) * (
        np.sin(dlon / 2, dtype=np.float32) ** np.float32(2.0)
    )
    c: NDArray[np.float32] = np.float32(2.0) * np.arcsin(np.sqrt(a, dtype=np.float32))
    dist: NDArray[np.float32] = EARTH_RADIUS * c

    y: NDArray[np.float32] = np.sin(dlon, dtype=np.float32) * np.cos(
        lat2r, dtype=np.float32
    )
    x: NDArray[np.float32] = np.cos(lat1r, dtype=np.float32) * np.sin(
        lat2r, dtype=np.float32
    ) - np.sin(lat1r, dtype=np.float32) * np.cos(lat2r, dtype=np.float32) * np.cos(
        dlon, dtype=np.float32
    )
    bearing: NDArray[np.float32] = (
        np.degrees(np.arctan2(y, x), dtype=np.float32) + 360.0
    ) % 360.0
    return dist.astype(np.float32), bearing.astype(np.float32)


def forward_point_sphere(
    lon0: ArrayLike,
    lat0: ArrayLike,
    bearing: ArrayLike,
    distance: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Destination point along a great-circle path on a sphere (float32).

    Args:
        lon0: Start longitude(s) (° E).
        lat0: Start latitude(s)  (° N).
        bearing: Initial bearing(s) (° clockwise from north).
        distance: Metres travelled along the bearing (positive = forward).

    Returns:
        ``(lon_dest, lat_dest)`` in degrees, float32.

    Examples:
        >>> lon2, lat2 = forward_point_sphere(0.0, 0.0, 90.0, 1000.0)
        >>> np.isclose(float(lat2), 0.0)
        True
        >>> round(float(lon2), 4)
        0.009
    """
    lon0r, lat0r, brgr = _to_rad(lon0, lat0, bearing)
    dist_rad: NDArray[np.float32] = (
        np.asarray(distance, dtype=np.float32) / EARTH_RADIUS
    )

    sin_lat0, cos_lat0 = np.sin(lat0r, dtype=np.float32), np.cos(
        lat0r, dtype=np.float32
    )
    sin_d, cos_d = np.sin(dist_rad, dtype=np.float32), np.cos(
        dist_rad, dtype=np.float32
    )
    sin_b, cos_b = np.sin(brgr, dtype=np.float32), np.cos(brgr, dtype=np.float32)

    lat2 = np.arcsin(sin_lat0 * cos_d + cos_lat0 * sin_d * cos_b, dtype=np.float32)
    lon2 = lon0r + np.arctan2(
        sin_b * sin_d * cos_lat0,
        cos_d - sin_lat0 * np.sin(lat2, dtype=np.float32),
        dtype=np.float32,
    )

    lon_deg = ((np.degrees(lon2, dtype=np.float32) + 540.0) % 360.0) - 180.0
    lat_deg = np.degrees(lat2, dtype=np.float32)
    return lon_deg.astype(np.float32), lat_deg.astype(np.float32)


def dist_bearing_to_centres_sphere(
    lon_mat: ArrayLike,
    lat_mat: ArrayLike,
    lon_c: ArrayLike,
    lat_c: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Distance and bearing from each grid point to its centre(s) on a sphere.

    Args:
        lon_mat: 2-D or 3-D array of longitudes (°E).
        lat_mat: 2-D or 3-D array of latitudes  (°N).
        lon_c:   Scalar or 1-D array of centre longitudes.
        lat_c:   Scalar or 1-D array of centre latitudes.

    Returns:
        ``(distance_m, bearing_deg)`` matching ``lon_mat`` shape (float32).

    Examples:
        >>> import numpy as np
        >>> lon = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
        >>> lat = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
        >>> d, b = dist_bearing_to_centres_sphere(lon, lat, 0.0, 0.0)
        >>> d.dtype, b.dtype
        (dtype('float32'), dtype('float32'))
        >>> round(float(b[0, 1]), 1)
        225.0
    """
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(lon_mat, lat_mat, lon_c, lat_c)
    return haversine_dist_bearing(lon1, lat1, lon2, lat2)


def line_with_impact_sphere(
    impact_time: float,
    impact_lon: float,
    impact_lat: float,
    translation_speed: float,
    bearing: float,
    times: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Great-circle trajectory that passes a known impact point at a given time.

    Args:
        impact_time: Time when the object is at (impact_lon, impact_lat).
        impact_lon:  Impact longitude (°E).
        impact_lat:  Impact latitude  (°N).
        translation_speed: Speed along the bearing (m s⁻¹).
        bearing: Constant bearing (° clockwise from north).
        times: Array of times at which to evaluate the position.

    Returns:
        ``(lon_arr, lat_arr)`` arrays (float32) matching *times*.

    Examples:
        >>> times = np.array([9., 10., 11.], dtype=np.float32)
        >>> lon, lat = line_with_impact_sphere(
        ...     impact_time=10.0,
        ...     impact_lon=2.0,
        ...     impact_lat=50.0,
        ...     translation_speed=100.0,
        ...     bearing=0.0,
        ...     times=times,
        ... )
        >>> round(float(lat[1]), 5)
        50.0
        >>> float(lat[2]) > float(lat[1])
        True
    """
    dt: NDArray[np.float32] = np.asarray(times, dtype=np.float32) - np.float32(
        impact_time
    )
    dist_m: NDArray[np.float32] = dt * np.float32(translation_speed)
    return forward_point_sphere(impact_lon, impact_lat, bearing, dist_m)
