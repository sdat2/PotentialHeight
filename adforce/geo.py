"""
adforce.geo

Do some geographic calculations, including either using a sphere or a GEOID.

"""

from typing import Tuple, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .constants import GEOD  # type: ignore[import]

# Mean spherical Earth radius in metres (float32 for consistency)
EARTH_RADIUS: np.float32 = np.float32(6_371_008.8)


def line_with_impact_pyproj(
    impact_time: float,
    impact_lon: float,
    impact_lat: float,
    translation_speed: float,
    bearing: float,
    times: Union[np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a line of constant bearing that passes through (impact_lon, impact_lat)
    at `impact_time`. For each t in `times`, the object travels along this bearing at `translation_speed` (meters/second).

    The bearing is assumed to be in degrees (clockwise from north), and longitudes
    and latitudes are in degrees.

    Args:
        impact_time (float): The time (e.g. in seconds) at which the path intersects
                (impact_lon, impact_lat).
        impact_lon (float): Impact longitude (in degrees).
        impact_lat (float): Impact latitude (in degrees).
        translation_speed (float): Constant speed (in m/s) along the bearing.
        bearing (float): Constant bearing (in degrees, clockwise from north).
        times (array-like): Array of time values (same units as impact_time).

    Returns:
        (np.ndarray, np.ndarray):
            A tuple of arrays (lon_arr, lat_arr) in degrees for each time in `times`.
            The shape matches the shape of the input `times`.

    Examples:
        >>> # Suppose the object passes through (2°E, 50°N) at t=10s,
        >>> # traveling due north at 100 m/s.
        >>> import numpy as np
        >>> times = np.array([9.0, 10.0, 11.0])
        >>> lon_arr, lat_arr = line_with_impact_pyproj(
        ...     impact_time=10.0,
        ...     impact_lon=2.0,
        ...     impact_lat=50.0,
        ...     translation_speed=100.0,  # m/s
        ...     bearing=0.0,             # due north
        ...     times=times
        ... )
        >>> # At t=10.0, we should be exactly at (2, 50).
        >>> round(lon_arr[1], 5), round(lat_arr[1], 5)
        (2.0, 50.0)
        >>> # 1 second earlier, we are about 100m south => ~0.0009 degrees of latitude
        >>> round(lat_arr[0], 5)
        49.9991
        >>> # 1 second later, we are about 100m north => ~50.00090 degrees of latitude
        >>> round(lat_arr[2], 5)
        50.0009
    """

    # Convert times to a NumPy array so we can do vector math
    times = np.asarray(times, dtype=float)

    # Time difference from impact_time (in seconds)
    dt = times - impact_time

    # Distance traveled from impact point, can be negative if t < impact_time
    # (i.e. behind the point along the same line).
    distances = dt * translation_speed  # in meters

    # Build arrays for initial conditions
    lon_init = np.full_like(distances, impact_lon, dtype=float)
    lat_init = np.full_like(distances, impact_lat, dtype=float)
    bearing_arr = np.full_like(distances, bearing, dtype=float)

    # Use forward geodesic to find location at each distances along bearing
    lon_arr, lat_arr, _ = GEOD.fwd(lon_init, lat_init, bearing_arr, distances)

    return lon_arr, lat_arr


def distances_bearings_to_center_pyproj(
    lon_mat: np.ndarray, lat_mat: np.ndarray, lon_c: np.ndarray, lat_c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the geodesic distances (meters) and bearing (degrees) from each location
    in a 2D or 3D array of longitudes/latitudes to the corresponding central point(s).

    The central longitude(s) and latitude(s) can be scalars (0D) or 1D arrays,
    while the location arrays can be 2D or 3D. Broadcasting rules apply:
      - If lon_c and lat_c are scalars (0D), they apply to all points in lon_mat/lat_mat.
      - If lon_c and lat_c are 1D, their shape must match the first dimension(s) of lon_mat/lat_mat.

    This is the main bottleneck of the code. It use the pyproj Geod class to calculate
    the geodesic distances and bearing.

    Bearing is measured clockwise from north (0° = north, 90° = east, etc.),
    and is the direction from each (lon_mat, lat_mat) location toward (lon_c, lat_c).

    Args:
        lon_mat (array-like): 2D or 3D array of longitudes (in degrees).
        lat_mat (array-like): 2D or 3D array of latitudes (in degrees).
        lon_c (float or array-like): 0D or 1D array of central longitude(s) (in degrees).
        lat_c (float or array-like): 0D or 1D array of central latitude(s) (in degrees).

    Returns:
        dist_arr (np.ndarray): 2D or 3D array of geodesic distancess (in meters),
            broadcasted to match the shape of lon_mat/lat_mat.
        bearing_arr (np.ndarray): 2D or 3D array of bearings (in degrees, [0, 360)),
            broadcasted to match the shape of lon_mat/lat_mat.

    Examples:
        >>> # Example 1: Single (scalar) center point, 2D location arrays
        >>> import numpy as np
        >>> lon_mat = np.array([[0.0,  1.0], [2.0,  3.0]])
        >>> lat_mat = np.array([[50.0, 51.0], [52.0, 53.0]])
        >>> center_lon, center_lat = 1.5, 51.5
        >>> dist, bearing = distances_bearings_to_center_pyproj(lon_mat, lat_mat, center_lon, center_lat)
        >>> dist.shape, bearing.shape
        ((2, 2), (2, 2))
        >>> # Example 2: 1D center arrays, 2D location arrays: each row uses a different center
        >>> lon_mat2 = np.array([[0.0,  1.0], [10.0,  11.0]])
        >>> lat_mat2 = np.array([[ 0.0,  1.0], [ 5.0,   6.0]])
        >>> center_lons = np.array([0.0, 10.0])  # shape (2,)
        >>> center_lats = np.array([0.0,  5.0])  # shape (2,)
        >>> dist2, bearing2 = distances_bearings_to_center_pyproj(lon_mat2, lat_mat2, center_lons, center_lats)
        >>> dist2.shape, bearing2.shape
        ((2, 2), (2, 2))
        >>> # The first row is measured to center (0,0), second row to center (10,5).
    """
    # Convert all inputs to NumPy arrays of float for consistent operations
    # lon_mat = np.asarray(lon_mat, dtype=float)
    # lat_mat = np.asarray(lat_mat, dtype=float)
    # lon_c = np.asarray(lon_c, dtype=float)
    # lat_c = np.asarray(lat_c, dtype=float)

    # We want the distances & bearing from each (lon_mat, lat_mat) -> (lon_c, lat_c).
    # geod.inv(lon1, lat1, lon2, lat2):
    #   forward_azimuth, back_azimuth, distances
    # Here, (lon1, lat1) are the "locations", (lon2, lat2) are the "center points."
    # Use np.broadcast_arrays to handle matching shapes or dimension expansions:
    lon1_b, lat1_b, lon2_b, lat2_b = np.broadcast_arrays(lon_mat, lat_mat, lon_c, lat_c)

    assert lon1_b.shape == lon2_b.shape
    assert lat1_b.shape == lat2_b.shape
    assert lon1_b.shape == lat1_b.shape

    fwd_az, _, dist = GEOD.inv(lon1_b, lat1_b, lon2_b, lat2_b)

    assert dist.shape == lon1_b.shape
    assert fwd_az.shape == lon1_b.shape

    # Bearing (forward azimuth) is from location -> center
    bearing = (fwd_az + 360) % 360  # normalize to [0, 360)

    return dist, bearing


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
    Great-circle distances and initial bearing on a spherical Earth (float32).

    Args:
        lon1: Longitudes of the start points (°E).
        lat1: Latitudes of the start points (°N).
        lon2: Longitudes of the end points (°E).
        lat2: Latitudes of the end points (°N).

    Returns:
        Tuple ``(distances_m, bearing_deg)``:

        * **distances_m** - float32 great-circle distances in metres.
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
    distances: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Destination point along a great-circle path on a sphere (float32).

    Args:
        lon0: Start longitude(s) (° E).
        lat0: Start latitude(s)  (° N).
        bearing: Initial bearing(s) (° clockwise from north).
        distances: Metres travelled along the bearing (positive = forward).

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
        np.asarray(distances, dtype=np.float32) / EARTH_RADIUS
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


def distances_bearings_to_center_sphere(
    lon_mat: ArrayLike,
    lat_mat: ArrayLike,
    lon_c: ArrayLike,
    lat_c: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Distance and bearing from each grid point to its center(s) on a sphere.

    Args:
        lon_mat: 2-D or 3-D array of longitudes (°E).
        lat_mat: 2-D or 3-D array of latitudes  (°N).
        lon_c:   Scalar or 1-D array of center longitudes.
        lat_c:   Scalar or 1-D array of center latitudes.

    Returns:
        ``(distances_m, bearing_deg)`` matching ``lon_mat`` shape (float32).

    Examples:
        >>> import numpy as np
        >>> lon = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
        >>> lat = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
        >>> d, b = distances_bearings_to_center_sphere(lon, lat, 0.0, 0.0)
        >>> d.dtype, b.dtype
        (dtype('float32'), dtype('float32'))
        >>> round(float(b[0, 1]), 1)
        225.0
    """
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(lon_mat, lat_mat, lon_c, lat_c)
    assert lon1.shape == lon2.shape
    assert lat1.shape == lat2.shape
    assert lon1.shape == lat1.shape
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
        impact_time: Time when the object is at (impact_lon, impact_lat). In seconds from beginning of calendar.
        impact_lon:  Impact longitude (°E).
        impact_lat:  Impact latitude  (°N).
        translation_speed: Speed along the bearing (meters/second).
        bearing: Constant bearing (° clockwise from north).
        times: Array of times at which to evaluate the position. In seconds from beginning of calendar.

    Returns:
        Tuple[np.ndarray, np.ndarray] ``(lon_arr, lat_arr)`` arrays (float32) matching *times*.

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


DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def parabolic_track_with_impact_sphere(
    impact_time: float,
    impact_lon: float,
    impact_lat: float,
    translation_speed: float,
    bearing: float,
    curvature: float,
    times: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Return the longitude/latitude of a storm centre that follows Ide-style
    *parabolic* track and crosses a chosen impact point at a specified time.

    Args:
        impact_time (float): Epoch seconds when the eye is exactly over
            ``(impact_lon, impact_lat)``.
        impact_lon (float): Impact longitude in degrees East.
        impact_lat (float): Impact latitude in degrees North.
        translation_speed (float): Forward speed along the initial bearing
            (metres s⁻¹).
        bearing (float): Initial bearing, degrees clockwise from north
            (0 ° = due north).
        curvature (float): Parabolic curvature *r*.
            * Positive → track bends **right** of the bearing.
            * Negative → bends **left**.
            * |r| ≈ 0.5 ≙ turning-radius ≈ 110 km.
        times (ArrayLike): 1-D epoch seconds at which to evaluate the position.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(lon_arr, lat_arr)`` — float32 arrays matching *times*.

    Notes:
        * Setting ``curvature = 0`` collapses to great-circle motion identical
          to the original ``line_with_impact_sphere`` helper.
        * Uses small-angle ENU projection (good for ≤ 300 km cross-track).
          Swap Section 4 with a geodesic solver for long crossings.

    Examples:
        Straight track (no curvature) heading north::

            >>> import numpy as np
            >>> t = np.array([9., 10., 11.], dtype=np.float32)
            >>> lon, lat = parabolic_track_with_impact_sphere(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=0.0,
            ...     times=t)
            >>> float(lat[1])
            50.0
            >>> lat[2] > lat[1] > lat[0]
            True

        Right-hand bend (positive curvature) increases longitude::

            >>> lon2, _ = parabolic_track_with_impact_sphere(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=0.5,
            ...     times=t)
            >>> round(float(lon2[1]), 6)
            2.0
            >>> lon2[2] > lon2[1]
            True

        Left-hand bend (negative curvature) decreases longitude::

            >>> lon3, _ = parabolic_track_with_impact_sphere(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=-0.5,
            ...     times=t)
            >>> lon3[0] < 2.0 and lon3[2] < 2.0   # middle point is exactly 2.0
            True

    """
    # 1 ─ along-track distance from impact vertex (metres)
    dt: NDArray[np.float32] = np.asarray(times, dtype=np.float32) - np.float32(
        impact_time
    )
    s: NDArray[np.float32] = dt * np.float32(translation_speed)

    # 2 ─ cross-track offset: n(s) = r · s² (metres)
    n: NDArray[np.float32] = np.float32(curvature) * s**2

    # 3 ─ rotate from storm-local axes to ENU (metres)
    θ = bearing * DEG2RAD
    cosθ, sinθ = np.cos(θ), np.sin(θ)
    east = n * cosθ + s * sinθ
    north = -n * sinθ + s * cosθ

    # 4 ─ convert ENU metres to lon/lat degrees (small-angle)
    lat0_rad = impact_lat * DEG2RAD
    dlat = north / EARTH_RADIUS
    dlon = east / (EARTH_RADIUS * np.cos(lat0_rad))
    lat = impact_lat + dlat * RAD2DEG
    lon = impact_lon + dlon * RAD2DEG
    return lon.astype(np.float32), lat.astype(np.float32)


def _arc_len(s: NDArray[np.float64], r: float) -> NDArray[np.float64]:
    """Arc-length L(s) for n = r·s² (see Ide et al.)."""
    if r == 0.0:
        return s
    k = 2.0 * r
    return 0.5 * s * np.sqrt(1.0 + (k * s) ** 2) + np.arcsinh(k * s) / (2.0 * k)


def _inv_arc_len(L: NDArray[np.float64], r: float) -> NDArray[np.float64]:
    """Inverse of L(s) by ≤3 Newton steps (good to <10-6 m)."""
    if r == 0.0:
        return L
    s = L.copy()
    k = 2.0 * r
    for _ in range(3):  # convergence in ≤2, keep a 3rd for safety
        f = 0.5 * s * np.sqrt(1.0 + (k * s) ** 2) + np.arcsinh(k * s) / (2.0 * k) - L
        fp = np.sqrt(1.0 + (k * s) ** 2)  # dL/ds
        s -= f / fp
    return s


def parabolic_track_with_impact_pyproj(
    impact_time: float,
    impact_lon: float,
    impact_lat: float,
    translation_speed: float,
    bearing: float,
    curvature: float,
    times: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Accurate Ide-style parabolic track **with constant ground speed** on WGS-84.

    Args:
        impact_time (float): Epoch seconds when the eye is over the point
            (``impact_lon``, ``impact_lat``).
        impact_lon (float): Impact longitude (°E).
        impact_lat (float): Impact latitude  (°N).
        translation_speed (float): True ground speed along the curved centre-line
            (m s⁻¹).
        bearing (float): Initial bearing (° clockwise from north; 0 ° = due N).
        curvature (float): Parabolic curvature *r*.  Positive bends right,
            negative bends left.  |r| ≃ 5 × 10⁻⁶ m⁻¹ reproduces Ide’s r = ±0.5
            when s ≈ 100 km.
        times (ArrayLike): 1-D epoch seconds at which to sample the trajectory.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(lon_arr, lat_arr)`` – float32 arrays
            matching *times*.

    Examples:
        Straight great-circle (curvature = 0) heading north::

            >>> import numpy as np, pyproj, math
            >>> t = np.array([9., 10., 11.], dtype=np.float64)
            >>> lon, lat = parabolic_track_with_impact_pyproj(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=0.0,
            ...     times=t)
            >>> float(lat[1])                                   # at impact
            50.0
            >>> lat[2] > lat[1] > lat[0]                        # moves north
            True
            >>> # -- distance between successive eye positions ≈ 100 m
            >>> az12, az21, d12 = GEOD.inv(
            ...     lon[1], lat[1], lon[2], lat[2])
            >>> abs(d12 - 100.0) < 2.0
            True

        Right-hand bend (positive curvature) moves both flanks eastward::

            >>> lon2, _ = parabolic_track_with_impact_pyproj(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=5e-6,
            ...     times=t)
            >>> lon2[0] > lon2[1] and lon2[2] > lon2[1]       # middle = impact
            True

        Left-hand bend (negative curvature) decreases longitude::

            >>> lon3, _ = parabolic_track_with_impact_pyproj(
            ...     impact_time=10.0, impact_lon=2.0, impact_lat=50.0,
            ...     translation_speed=100.0, bearing=0.0, curvature=-5e-6,
            ...     times=t)
            >>> lon3[0] < 2.0 and lon3[2] < 2.0                # middle = 2.0
            True
    """
    # 1 ─ arc-length travelled since impact (can be ±)
    L = (np.asarray(times, dtype=np.float64) - impact_time) * translation_speed

    # 2 ─ invert to storm-local parameter s
    s = _inv_arc_len(L, curvature)

    # 3 ─ cross-track offset n(s) = r·s² and rotate to ENU
    n = curvature * s**2
    θ = bearing * DEG2RAD
    cosθ, sinθ = np.cos(θ), np.sin(θ)
    east = n * cosθ + s * sinθ
    north = -n * sinθ + s * cosθ

    # 4 ─ forward geodesic from the impact point by (azimuth, distance)
    dist = np.hypot(east, north)
    azimuth = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    lon, lat, _ = GEOD.fwd(
        np.full_like(dist, impact_lon),
        np.full_like(dist, impact_lat),
        azimuth,
        dist,
    )
    return lon.astype(np.float32), lat.astype(np.float32)
