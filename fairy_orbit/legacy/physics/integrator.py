"""Symplectic Leapfrog (kick-drift-kick) integrator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fairy_orbit.physics.body import System
from fairy_orbit.physics.gravity import accelerations


class Integrator(ABC):
    @abstractmethod
    def step(self, system: System, dt: float) -> None:
        raise NotImplementedError


class Leapfrog(Integrator):
    """Velocity-Verlet / leapfrog: half-kick, drift, half-kick."""

    def step(self, system: System, dt: float) -> None:
        pos = system.positions()
        vel = system.velocities()
        acc = accelerations(system)

        vel = vel + 0.5 * dt * acc
        pos = pos + dt * vel
        system.set_state(pos, vel)

        acc = accelerations(system)
        vel = vel + 0.5 * dt * acc
        system.set_state(pos, vel)


class Euler(Integrator):
    """Forward Euler — only for comparison tests; do not use in search."""

    def step(self, system: System, dt: float) -> None:
        pos = system.positions()
        vel = system.velocities()
        acc = accelerations(system)
        vel_new = vel + dt * acc
        pos_new = pos + dt * vel
        system.set_state(pos_new, vel_new)
