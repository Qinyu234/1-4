"""Newtonian gravity: accelerations, energy, angular momentum."""

from __future__ import annotations

import numpy as np

from fairy_orbit.physics.body import System


def accelerations(system: System) -> np.ndarray:
    """Return shape (N, 3) accelerations for all bodies."""
    pos = system.positions()
    masses = system.masses()
    G = system.G
    n = system.n
    acc = np.zeros((n, 3), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r)
            if dist == 0.0:
                raise ValueError("Bodies coincide; singular gravity")
            acc[i] += G * masses[j] * r / (dist**3)
    return acc


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
    """Total angular momentum vector about the origin."""
    pos = system.positions()
    vel = system.velocities()
    masses = system.masses()
    L = np.zeros(3, dtype=float)
    for i in range(system.n):
        L += masses[i] * np.cross(pos[i], vel[i])
    return L


def total_momentum(system: System) -> np.ndarray:
    masses = system.masses()
    vel = system.velocities()
    return np.sum(masses[:, None] * vel, axis=0)


def center_of_mass(system: System) -> np.ndarray:
    masses = system.masses()
    pos = system.positions()
    mtot = float(np.sum(masses))
    return np.sum(masses[:, None] * pos, axis=0) / mtot
