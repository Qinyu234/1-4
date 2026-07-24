"""Initial-condition generator: 2D near-escape + tetrahedron symmetry."""

from __future__ import annotations

import numpy as np

from fairy_orbit.icgen.tetrahedron import (
    FAIRY_ORDER,
    RA,
    VERTICES,
    escape_speed,
    local_frame,
    rotations_from_A,
)
from fairy_orbit.physics.body import Body, System
from fairy_orbit.physics.gravity import total_momentum


def velocity_at_A(v_rad: float, v_tan: float) -> np.ndarray:
    r_hat, t_hat, _ = local_frame(RA)
    return v_rad * r_hat + v_tan * t_hat


def generate_system(
    v_rad: float,
    v_tan: float,
    *,
    planet_mass: float = 1.0,
    fairy_mass: float = 0.01,
    radius: float = 20.0,
    G: float = 1.0,
) -> System:
    """Build Planet + 4 fairies with Rodrigues-propagated velocities."""
    v_A = velocity_at_A(v_rad, v_tan)
    rotations = rotations_from_A()

    fairies: list[Body] = []
    for label in FAIRY_ORDER:
        r_hat = VERTICES[label]
        pos = radius * r_hat
        vel = rotations[label] @ v_A
        fairies.append(Body(fairy_mass, pos, vel, name=label))

    # Planet at origin; cancel total linear momentum.
    planet = Body(planet_mass, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], name="Planet")
    system = System([planet, *fairies], G=G)
    P = total_momentum(system)
    planet.velocity = -P / planet_mass
    return system


def orbital_period(G: float, M: float, R: float) -> float:
    return float(2.0 * np.pi * np.sqrt(R**3 / (G * M)))


def default_grid_bounds(
    G: float = 1.0,
    M: float = 1.0,
    R: float = 20.0,
    lo: float = 0.7,
    hi: float = 1.3,
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Near-escape grid for (v_rad, v_tan)."""
    vesc = escape_speed(G, M, R)
    values = np.linspace(lo * vesc, hi * vesc, n)
    return values, values
