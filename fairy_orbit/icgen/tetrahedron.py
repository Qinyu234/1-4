"""Regular tetrahedron vertices and Rodrigues rotations A→X."""

from __future__ import annotations

import numpy as np

# Unit tetrahedron vertices (normalized).
RA = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
RB = np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)
RC = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0)
RD = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)

VERTICES = {"A": RA, "B": RB, "C": RC, "D": RD}
FAIRY_ORDER = ("A", "B", "C", "D")


def tetra_rotation(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """Rodrigues rotation mapping unit vector r_from onto r_to.

    Axis is r_from × r_to (plane of the two radii). The angle-bisector form
    does not map r_from → r_to; cross-product axis is the correct SO(3) map
    and preserves ∠(direction, radius) under the same R.
    """
    r_from = np.asarray(r_from, dtype=float)
    r_to = np.asarray(r_to, dtype=float)
    r_from = r_from / np.linalg.norm(r_from)
    r_to = r_to / np.linalg.norm(r_to)

    cos_theta = float(np.clip(np.dot(r_from, r_to), -1.0, 1.0))
    if cos_theta > 1.0 - 1e-14:
        return np.eye(3)
    if cos_theta < -1.0 + 1e-14:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, r_from)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(r_from, tmp)
        axis = axis / np.linalg.norm(axis)
        theta = np.pi
    else:
        axis = np.cross(r_from, r_to)
        axis = axis / np.linalg.norm(axis)
        theta = float(np.arccos(cos_theta))

    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    I = np.eye(3)
    R = I * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K
    return R


def rotations_from_A() -> dict[str, np.ndarray]:
    """Identity for A; Rodrigues maps for B, C, D."""
    return {
        "A": np.eye(3),
        "B": tetra_rotation(RA, RB),
        "C": tetra_rotation(RA, RC),
        "D": tetra_rotation(RA, RD),
    }


def local_frame(r_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (radial, tangential, binormal) frame at r_hat."""
    r_hat = np.asarray(r_hat, dtype=float)
    r_hat = r_hat / np.linalg.norm(r_hat)
    # Stable tangent: project a reference vector off the radial direction.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, r_hat)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t_hat = ref - np.dot(ref, r_hat) * r_hat
    t_hat = t_hat / np.linalg.norm(t_hat)
    n_hat = np.cross(r_hat, t_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)
    return r_hat, t_hat, n_hat


def escape_speed(G: float, M: float, R: float) -> float:
    return float(np.sqrt(2.0 * G * M / R))


def circular_speed(G: float, M: float, R: float) -> float:
    return float(np.sqrt(G * M / R))
