"""Numerical integration for active subsystems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.physics.body import System


@dataclass
class IntegratorConfig:
    """Configuration for numerical integration."""
    dt: float = 0.01  # Integration timestep
    max_steps: int = 10000  # Maximum integration steps


class SubsystemIntegrator:
    """Numerical integration for active subsystems."""
    
    def __init__(self, config: IntegratorConfig | None = None):
        self.config = config or IntegratorConfig()
    
    def compute_accelerations(
        self,
        system: System,
        active_indices: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        """
        Compute accelerations for bodies in active subsystem.
        
        All bodies in subsystem interact with each other.
        Inactive bodies are propagated analytically (not handled here).
        
        Args:
            system: Current system state
            active_indices: Indices of bodies in active subsystem
        
        Returns:
            Dictionary mapping body index to acceleration vector
        """
        G = system.G
        acc = {i: np.zeros(3) for i in active_indices}
        
        # Compute mutual interactions within active subsystem
        for i in active_indices:
            for j in active_indices:
                if i != j:
                    body_i = system.bodies[i]
                    body_j = system.bodies[j]
                    r = body_j.position - body_i.position
                    dist = np.linalg.norm(r)
                    if dist > 1e-10:
                        acc[i] += G * body_j.mass * r / dist**3
        
        return acc
    
    def velocity_verlet_step(
        self,
        system: System,
        active_indices: tuple[int, ...],
        dt: float,
    ) -> None:
        """
        Single Velocity Verlet integration step for active subsystem.
        
        Args:
            system: System to integrate
            active_indices: Indices of bodies in active subsystem
            dt: Timestep
        """
        # Half-step velocity update
        acc = self.compute_accelerations(system, active_indices)
        for i in active_indices:
            system.bodies[i].velocity += 0.5 * acc[i] * dt
        
        # Full-step position update
        for i in active_indices:
            system.bodies[i].position += system.bodies[i].velocity * dt
        
        # Recompute accelerations at new positions
        new_acc = self.compute_accelerations(system, active_indices)
        
        # Half-step velocity update
        for i in active_indices:
            system.bodies[i].velocity += 0.5 * new_acc[i] * dt
    
    def integrate(
        self,
        system: System,
        active_indices: tuple[int, ...],
        t_end: float,
        monitor_callback=None,
    ) -> tuple[float, bool]:
        """
        Integrate active subsystem until t_end or monitor callback returns True.
        
        Args:
            system: System to integrate
            active_indices: Indices of bodies in active subsystem
            t_end: End time for integration
            monitor_callback: Optional callback to check for exit condition
                              Returns True if integration should stop early
        
        Returns:
            (final_time, early_exit) tuple
        """
        t = 0.0
        dt = self.config.dt
        steps = 0
        
        while t < t_end and steps < self.config.max_steps:
            # Check monitor callback
            if monitor_callback and monitor_callback(system, t):
                return t, True
            
            # Integration step
            self.velocity_verlet_step(system, active_indices, dt)
            t += dt
            steps += 1
        
        return t, steps >= self.config.max_steps
