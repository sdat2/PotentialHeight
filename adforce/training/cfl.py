"""CFL timestep recommendation for ADCIRC meshes.

Split out of ``adforce/generate_training_data.py``. The mesh argument is only
duck-typed (it needs ``node_distances_in_meters`` and ``values``), so this
module does not import adcircpy at runtime and stays cheap to import; the
adcircpy ``AdcircMesh`` type is referenced for type checking only.
"""

from typing import TYPE_CHECKING
import math
import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid a runtime adcircpy dependency for a type hint
    from adcircpy import AdcircMesh


def calculate_cfl_timestep(
    mesh: "AdcircMesh", cfl_target: float = 0.7, maxvel: float = 5.0, g: float = 9.81
) -> float:
    """
    Calculates a recommended timestep based on the full CFL condition
    for ADCIRC's shallow water equations.

    The condition is: CFL = (U + sqrt(gH)) * dt / dx <= cfl_target
    So, the maximum dt = cfl_target * dx / (U + sqrt(gH))

    Args:
        mesh: An AdcircMesh object containing node coordinates, depths,
              and element connectivity (implicitly used by properties).
        cfl_target: The desired Courant number (dimensionless). Typically
                    <= 1.0 for stability, often 0.5-0.7 is used for safety.
        maxvel: Estimated maximum expected flow speed (U) in m/s across the
                entire domain during the simulation. This is an estimate you
                provide based on the expected physics (e.g., 5-10 m/s for
                hurricanes). Default is 5.0 m/s.
        g: Acceleration due to gravity in m/s^2. Default is 9.81.

    Returns:
        Recommended maximum timestep (dt) in seconds.

    Raises:
        ValueError: If mesh properties (distances, depths) cannot be determined
                    or are invalid (e.g., non-positive).
        AttributeError: If the mesh object doesn't have the expected properties.
    """
    print(
        f"Calculating CFL timestep with cfl_target={cfl_target}, maxvel={maxvel} m/s..."
    )

    # 1. Find the minimum edge length (dx) in meters
    #    Uses the pre-calculated distances from the mesh object.
    min_dx = float("inf")
    try:
        # Accessing the property triggers calculation if needed
        node_distances = mesh.node_distances_in_meters

        if not node_distances:
            raise ValueError(
                "Node distances dictionary is empty. Cannot calculate min_dx."
            )

        for node_idx, neighbors in node_distances.items():
            if neighbors:  # Check if the neighbor dictionary is not empty
                min_neighbor_dist = min(neighbors.values())
                min_dx = min(min_dx, min_neighbor_dist)

    except AttributeError:
        raise AttributeError(
            "Mesh object requires 'node_distances_in_meters' property for CFL calculation."
        ) from None
    except Exception as e:
        raise ValueError(f"Error accessing node distances: {e}") from e

    if min_dx == float("inf") or min_dx <= 0:
        raise ValueError(
            f"Could not determine a valid minimum edge length (min_dx={min_dx}). Check mesh connectivity and units."
        )
    print(f"  - Minimum edge length (min_dx): {min_dx:.2f} m")

    # 2. Find the maximum depth (H) in meters
    #    Assumes mesh.values is a pandas DataFrame with depth data, positive downwards.
    max_h = 0.0
    try:
        # Get depth values, assuming it's a DataFrame (might have multiple columns)
        depth_df = mesh.values
        if not isinstance(depth_df, pd.DataFrame):
            raise TypeError("Expected mesh.values to be a pandas DataFrame")

        # Find the maximum value across all depth columns, ignoring NaNs
        numeric_depths = depth_df.select_dtypes(include=np.number)
        if numeric_depths.empty:
            raise ValueError("Mesh 'values' DataFrame contains no numeric depth data.")

        max_h = numeric_depths.max().max()  # Max of max of each column

        if pd.isna(max_h):
            raise ValueError(
                "Maximum depth calculation resulted in NaN. Check depth data."
            )

    except AttributeError:
        raise AttributeError(
            "Mesh object requires 'values' property (pandas DataFrame) for depths."
        ) from None
    except Exception as e:
        raise ValueError(f"Error accessing or processing mesh depths: {e}") from e

    if max_h < 0:
        print(
            f"  - Warning: Maximum mesh depth is negative ({max_h:.2f}m). Using absolute value."
        )
        max_h = abs(max_h)
    # Allow max_h == 0 (completely dry land), sqrt(0) is handled.

    print(f"  - Maximum depth (max_h): {max_h:.2f} m")

    # 3. Calculate gravity wave speed (C = sqrt(gH))
    wave_speed = math.sqrt(g * max_h)
    print(f"  - Max gravity wave speed (sqrt(gH)): {wave_speed:.2f} m/s")

    # 4. Calculate characteristic speed (S = U + C)
    #    Use absolute value of maxvel just in case.
    characteristic_speed = abs(maxvel) + wave_speed
    print(
        f"  - Characteristic speed (maxvel + sqrt(gH)): {characteristic_speed:.2f} m/s"
    )

    if characteristic_speed <= 1e-9:  # Check for effectively zero speed
        # This might happen if max_h=0 and maxvel=0
        raise ValueError(
            f"Characteristic speed ({characteristic_speed:.2f} m/s) is near zero. Cannot calculate timestep."
        )

    # 5. Calculate recommended timestep (dt = cfl_target * dx / S)
    recommended_dt = cfl_target * min_dx / characteristic_speed

    if recommended_dt <= 0:
        raise ValueError(
            f"Calculated timestep ({recommended_dt:.4f}s) is not positive. Check inputs."
        )

    print(f"✅ Recommended timestep (dt): {recommended_dt:.4f} seconds")
    return recommended_dt
