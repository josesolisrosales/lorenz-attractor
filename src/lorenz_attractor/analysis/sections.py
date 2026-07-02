"""Poincaré section computation for trajectories."""

from typing import List, Optional

import numpy as np


def poincare_section(
    trajectory: np.ndarray,
    plane_normal: Optional[np.ndarray] = None,
    plane_offset: float = 27.0,
) -> np.ndarray:
    """Return points where ``trajectory`` crosses the plane n·x = offset.

    Crossings are found by linear interpolation between consecutive samples
    that lie on opposite sides of the plane.
    """
    if plane_normal is None:
        plane_normal = np.array([0, 0, 1])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    intersections: List[np.ndarray] = []
    for i in range(len(trajectory) - 1):
        p1, p2 = trajectory[i], trajectory[i + 1]
        d1 = np.dot(p1, plane_normal) - plane_offset
        d2 = np.dot(p2, plane_normal) - plane_offset
        if d1 * d2 < 0:
            t = -d1 / (d2 - d1)
            intersections.append(p1 + t * (p2 - p1))

    return np.array(intersections) if intersections else np.array([]).reshape(0, 3)
