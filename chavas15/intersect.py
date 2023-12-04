from typing import List, Tuple
import numpy as np
from sithom.time import timeit


@timeit
def curveintersect(
    x1: List[float], y1: List[float], x2: List[float], y2: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Find intersection points of two curves (x1, y1) and (x2, y2).

    Parameters:
    x1, y1 : array-like, coordinates of the first curve
    x2, y2 : array-like, coordinates of the second curve

    Returns:
    Tuple[List[float], List[float]]: intersections x and y coordinates.
    """

    def line_intersect(p1, p2, q1, q2):
        """Check if line segments p1p2 and q1q2 intersect."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    def segment_intersection(p1, p2, q1, q2):
        """Find intersection point of line segments p1p2 and q1q2 if they intersect."""

        def line(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0] * p2[1] - p2[0] * p1[1]
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            return None

        L1 = line(p1, p2)
        L2 = line(q1, q2)
        return intersection(L1, L2)

    intersections_x = []
    intersections_y = []
    for i in range(len(x1) - 1):
        for j in range(len(x2) - 1):
            p1, p2 = (x1[i], y1[i]), (x1[i + 1], y1[i + 1])
            q1, q2 = (x2[j], y2[j]), (x2[j + 1], y2[j + 1])
            if line_intersect(p1, p2, q1, q2):
                intersect_point = segment_intersection(p1, p2, q1, q2)
                if intersect_point:
                    intersections_x.append(intersect_point[0])
                    intersections_y.append(intersect_point[1])

    return intersections_x, intersections_y


if __name__ == "__main__":
    # Example usage
    x1 = np.random.rand(50)
    y1 = np.random.rand(50)
    x2 = np.random.rand(50)
    y2 = np.random.rand(50)

    import matplotlib.pyplot as plt

    plt.plot(x1, y1, color="orange")
    plt.plot(x2, y2, color="blue")

    x, y = curveintersect(x1, y1, x2, y2)
    print(x, len(x))
    plt.plot(x, y, "o", color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("intersect.pdf")
