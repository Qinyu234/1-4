"""Collision and energy-based escape criteria (PROMPT §6)."""

from __future__ import annotations

from enum import Enum

import numpy as np

from fairy_orbit.core.body import System


class SimulationStatus(Enum):
    SUCCESS = "success"
    COLLISION = "collision"
    ESCAPE = "escape"
    TIMEOUT = "timeout"


def specific_orbital_energy(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float,
) -> float:
    """Specific two-body energy relative to the central mass: E = v²/2 − μ/r."""
    r = float(np.linalg.norm(position))
    v2 = float(np.dot(velocity, velocity))
    if r <= 0.0:
        return float("inf")
    return 0.5 * v2 - mu / r


def check_collision(system: System, soft_floor: float = 0.0) -> bool:
    """True if any pair satisfies r_ij < R_i + R_j (and ≥ soft_floor)."""
    pos = system.positions()
    radii = system.radii()
    n = system.n
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            thresh = max(radii[i] + radii[j], soft_floor)
            if thresh > 0.0 and dist < thresh:
                return True
    return False


def check_escape(system: System, central_index: int = 0) -> bool:
    """Escape if any non-central body has E = v²/2 − GM/r > 0 (equiv. a ≤ 0)."""
    bodies = system.bodies
    if not bodies:
        return False
    central = bodies[central_index]
    mu = system.G * central.mass
    # Use relative state w.r.t. central body
    for i, body in enumerate(bodies):
        if i == central_index:
            continue
        r = body.position - central.position
        v = body.velocity - central.velocity
        if specific_orbital_energy(r, v, mu) > 0.0:
            return True
    return False


def evaluate_status(
    system: System,
    *,
    soft_floor: float = 0.0,
    central_index: int = 0,
) -> SimulationStatus:
    if check_collision(system, soft_floor=soft_floor):
        return SimulationStatus.COLLISION
    if check_escape(system, central_index=central_index):
        return SimulationStatus.ESCAPE
    return SimulationStatus.SUCCESS
