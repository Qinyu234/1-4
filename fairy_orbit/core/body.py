"""
Inertial ↔ barycentric (COM) frame for an isolated N-body system.

Translation only — overall rotation is left for SO(3) matching in the PEO filter:

    R_cm = (1/M) Σ m_i r_i
    V_cm = (1/M) Σ m_i v_i     (constant for isolated Newton systems)

    r'_i = r_i − R_cm
    v'_i = v_i − V_cm

    ⇒ Σ m_i r'_i = 0,  Σ m_i v'_i = 0

Inverse:

    r_i = r'_i + R_cm
    v_i = v'_i + V_cm

Newtonian forces depend only on r_i − r_j = r'_i − r'_j, so the equations
of motion are identical in the COM frame; one may integrate there directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def cross3(a, b) -> np.ndarray:
    """Fast 3-vector cross product (np.cross has heavy axis machinery)."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=float,
    )


def norm3(v) -> float:
    """Fast Euclidean norm of a 3-vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@dataclass
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    name: str = ""
    radius: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)


@dataclass
class System:
    bodies: list[Body]
    G: float = 1.0
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.labels:
            self.labels = [b.name or f"body_{i}" for i, b in enumerate(self.bodies)]

    @property
    def n(self) -> int:
        return len(self.bodies)

    def positions(self) -> np.ndarray:
        return np.stack([b.position for b in self.bodies], axis=0)

    def velocities(self) -> np.ndarray:
        return np.stack([b.velocity for b in self.bodies], axis=0)

    def masses(self) -> np.ndarray:
        return np.array([b.mass for b in self.bodies], dtype=float)

    def radii(self) -> np.ndarray:
        return np.array([b.radius for b in self.bodies], dtype=float)

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        for i, body in enumerate(self.bodies):
            body.position = np.asarray(positions[i], dtype=float).copy()
            body.velocity = np.asarray(velocities[i], dtype=float).copy()

    def copy(self) -> System:
        return System(
            bodies=[
                Body(
                    mass=b.mass,
                    position=b.position.copy(),
                    velocity=b.velocity.copy(),
                    name=b.name,
                    radius=b.radius,
                )
                for b in self.bodies
            ],
            G=self.G,
            labels=list(self.labels),
        )


def kinetic_energy(system: System) -> float:
    masses = system.masses()
    vel = system.velocities()
    return 0.5 * float(np.sum(masses * np.sum(vel**2, axis=1)))


def potential_energy(system: System) -> float:
    bodies = system.bodies
    G = system.G
    n = len(bodies)
    pe = 0.0
    for i in range(n):
        pi = bodies[i].position
        mi = bodies[i].mass
        for j in range(i + 1, n):
            pj = bodies[j].position
            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            dz = pj[2] - pi[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            pe -= G * mi * bodies[j].mass / dist
    return pe


def total_energy(system: System) -> float:
    return kinetic_energy(system) + potential_energy(system)


def angular_momentum(system: System) -> np.ndarray:
    L = np.zeros(3, dtype=float)
    for b in system.bodies:
        p = b.position
        v = b.velocity
        m = b.mass
        L[0] += m * (p[1] * v[2] - p[2] * v[1])
        L[1] += m * (p[2] * v[0] - p[0] * v[2])
        L[2] += m * (p[0] * v[1] - p[1] * v[0])
    return L


def total_mass(system: System) -> float:
    return float(np.sum(system.masses()))


def com_position(system: System) -> np.ndarray:
    """R_cm = (1/M) Σ m_i r_i  (inertial)."""
    m = system.masses()
    r = system.positions()
    return (m[:, None] * r).sum(axis=0) / float(np.sum(m))


def com_velocity(system: System) -> np.ndarray:
    """V_cm = (1/M) Σ m_i v_i  (inertial; constant if isolated)."""
    m = system.masses()
    v = system.velocities()
    return (m[:, None] * v).sum(axis=0) / float(np.sum(m))


@dataclass(frozen=True)
class ComShift:
    """Inertial COM state used for a frame change (needed for the inverse)."""

    R_cm: np.ndarray
    V_cm: np.ndarray


def to_com_inertial_frame(system: System) -> ComShift:
    """
    Translate into the inertial COM frame in place:

        r'_i = r_i − R_cm
        v'_i = v_i − V_cm

    Does NOT remove rotations (those are matched later by R ∈ SO(3)).
    Returns the (R_cm, V_cm) that were subtracted so `from_com_inertial_frame`
    can restore the original inertial embedding.
    """
    R = com_position(system)
    V = com_velocity(system)
    for b in system.bodies:
        b.position = b.position - R
        b.velocity = b.velocity - V
    return ComShift(R_cm=R.copy(), V_cm=V.copy())


def from_com_inertial_frame(system: System, shift: ComShift) -> System:
    """
    Inverse of `to_com_inertial_frame`:

        r_i = r'_i + R_cm
        v_i = v'_i + V_cm
    """
    for b in system.bodies:
        b.position = b.position + shift.R_cm
        b.velocity = b.velocity + shift.V_cm
    return system
