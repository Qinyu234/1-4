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
                )
                for b in self.bodies
            ],
            G=self.G,
            labels=list(self.labels),
        )
