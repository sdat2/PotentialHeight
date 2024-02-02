"""
Process ADCIRC meshes efficiently.
"""
import numpy as np
from adpy.fort63 import xr_loader


f63 = xr_loader("data/fort.63.nc")


def trim_tri(
    x: np.ndarray,
    y: np.ndarray,
    tri: np.ndarray,
    bbox: BoundingBox,
    z: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Trim triangular mesh to x and y points within an area.

    Args:
        x (np.ndarray): longitude [degrees East].
        y (np.ndarray): latitude [degrees North].
        tri (np.ndarray): triangular mesh.
        bbox (BoundingBox): bounding box to trim by.
        z (Optional[np.ndarray], optional): z parameter. Defaults to None.

    Returns:
        Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
             ]: trimmed x, y, tri, and z.
    """

    @np.vectorize
    def in_bbox(xi: float, yi: float) -> bool:
        return (
            xi > bbox.lon[0]
            and xi < bbox.lon[1]
            and yi > bbox.lat[0]
            and yi < bbox.lat[1]
        )

    tindices = in_bbox(x, y)
    indices = np.where(tindices)[0]
    new_indices = np.where(indices)[0]
    neg_indices = np.where(~tindices)[0]
    tri_list = tri.tolist()
    new_tri_list = []
    for el in tri_list:
        if np.any([x in neg_indices for x in el]):
            continue
        else:
            new_tri_list.append(el)

    tri_new = np.array(new_tri_list)
    # should there be an off by one error here?
    tri_new = np.select(
        [tri_new == x for x in indices.tolist()], new_indices.tolist(), tri_new
    )
    if z is None:
        return x[indices], y[indices], tri_new
    elif len(z.shape) == 1:
        return x[indices], y[indices], tri_new, z[indices]
    elif len(z.shape) == 2:
        return x[indices], y[indices], tri_new, z[:, indices]
    else:
        return None
