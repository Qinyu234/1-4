"""REBOUND simulator interface for high-accuracy reference and verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import rebound
    REBOUND_AVAILABLE = True
except ImportError:
    REBOUND_AVAILABLE = False

from fairy_orbit.physics.body import System


@dataclass
class ReboundConfig:
    """Configuration for REBOUND simulator."""
    integrator: str = "ias15"  # REBOUND integrator (ias15, whfast, etc.)
    dt: float = 0.01  # Timestep for fixed-step integrators


class ReboundSimulator:
    """
    REBOUND simulator for high-accuracy reference.
    
    Always integrates the complete system.
    Used for verification and benchmarking.
    """
    
    def __init__(self, config: ReboundConfig | None = None):
        if not REBOUND_AVAILABLE:
            raise ImportError("REBOUND is not installed. Install with: pip install rebound")
        
        self.config = config or ReboundConfig()
    
    def system_to_rebound(self, system: System) -> rebound.Simulation:
        """
        Convert fairy_orbit System to REBOUND Simulation.
        
        Args:
            system: fairy_orbit System
        
        Returns:
            REBOUND Simulation object
        """
        sim = rebound.Simulation()
        sim.G = system.G
        
        # Add particles
        for body in system.bodies:
            sim.add(
                m=body.mass,
                x=body.position[0],
                y=body.position[1],
                z=body.position[2],
                vx=body.velocity[0],
                vy=body.velocity[1],
                vz=body.velocity[2],
            )
        
        # Set integrator
        sim.integrator = self.config.integrator
        
        return sim
    
    def rebound_to_system(self, sim: rebound.Simulation, original_system: System) -> System:
        """
        Convert REBOUND Simulation back to fairy_orbit System.
        
        Args:
            sim: REBOUND Simulation
            original_system: Original fairy_orbit System (for metadata)
        
        Returns:
            fairy_orbit System
        """
        new_system = original_system.copy()
        
        for i, p in enumerate(sim.particles):
            new_system.bodies[i].position = np.array([p.x, p.y, p.z])
            new_system.bodies[i].velocity = np.array([p.vx, p.vy, p.vz])
        
        return new_system
    
    def simulate(
        self,
        system: System,
        t_end: float,
        dt: float | None = None,
    ) -> tuple[System, np.ndarray, np.ndarray]:
        """
        Run REBOUND simulation.
        
        Args:
            system: Initial system state
            t_end: End time
            dt: Output interval (not integration step)
        
        Returns:
            (final_system, times, positions) tuple
        """
        sim = self.system_to_rebound(system)
        
        if dt is None:
            dt = t_end / 1000  # Default to 1000 output points
        
        # Integrate and record
        times = []
        positions = []
        
        t = 0.0
        while t < t_end:
            times.append(t)
            pos = np.array([[p.x, p.y, p.z] for p in sim.particles])
            positions.append(pos)
            
            sim.integrate(t + dt)
            t += dt
        
        # Final state
        final_system = self.rebound_to_system(sim, system)
        
        return final_system, np.array(times), np.array(positions)
    
    def compare_with_hierarchical(
        self,
        system: System,
        t_end: float,
        hierarchical_result: tuple[System, np.ndarray, np.ndarray],
    ) -> dict[str, float]:
        """
        Compare hierarchical simulator result with REBOUND reference.
        
        Args:
            system: Initial system state
            t_end: End time
            hierarchical_result: Result from hierarchical simulator
        
        Returns:
            Dictionary with comparison metrics
        """
        # Run REBOUND
        final_rebound, times_rebound, positions_rebound = self.simulate(system, t_end)
        
        final_hierarchical, _, _ = hierarchical_result
        
        # Compare final positions
        pos_rebound = np.array([[p.x, p.y, p.z] for p in final_rebound.bodies])
        pos_hierarchical = np.array([b.position for b in final_hierarchical.bodies])
        
        position_error = np.linalg.norm(pos_rebound - pos_hierarchical)
        
        # Compare energy
        from fairy_orbit.physics.gravity import total_energy
        energy_rebound = total_energy(final_rebound)
        energy_hierarchical = total_energy(final_hierarchical)
        
        energy_error = abs(energy_rebound - energy_hierarchical)
        
        return {
            "position_error": float(position_error),
            "energy_error": float(energy_error),
            "relative_position_error": float(position_error / (np.linalg.norm(pos_rebound) + 1e-10)),
            "relative_energy_error": float(energy_error / (abs(energy_rebound) + 1e-10)),
        }
