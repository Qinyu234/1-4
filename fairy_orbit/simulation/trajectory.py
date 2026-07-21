"""Trajectory abstraction class."""

from __future__ import annotations
import numpy as np

class Trajectory:
    def __init__(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        energies: np.ndarray,
        angular_momenta: np.ndarray,
        labels: list[str],
        G: float = 1.0,
        masses: np.ndarray | None = None,
    ):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)  # (T, N, 3)
        self.velocities = np.asarray(velocities, dtype=float)  # (T, N, 3)
        self.energies = np.asarray(energies, dtype=float)
        self.angular_momenta = np.asarray(angular_momenta, dtype=float)
        self.labels = list(labels)
        self.G = G
        self.masses = np.asarray(masses, dtype=float) if masses is not None else None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.positions.shape

    def __getitem__(self, index) -> np.ndarray:
        return self.positions[index]
