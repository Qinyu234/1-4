"""Body and System data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    pos = system.positions()
    masses = system.masses()
    G = system.G
    n = system.n
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[j] - pos[i])
            pe -= G * masses[i] * masses[j] / dist
    return pe


def total_energy(system: System) -> float:
    return kinetic_energy(system) + potential_energy(system)


def angular_momentum(system: System) -> np.ndarray:
    pos = system.positions()
    vel = system.velocities()
    masses = system.masses()
    L = np.zeros(3, dtype=float)
    for i in range(system.n):
        L += masses[i] * np.cross(pos[i], vel[i])
    return L
