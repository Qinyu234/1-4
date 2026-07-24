"""Constraint filter for collision and escape detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.physics.body import System
from fairy_orbit.simulation.status import SimulationStatus


@dataclass
class ConstraintConfig:
    """Configuration for constraint filtering."""
    collision_radius: float = 1.0  # Minimum allowed distance between bodies
    outer_radius: float = 200.0  # Maximum allowed distance from origin


class ConstraintFilter:
    """Filter trajectories based on collision and escape constraints."""
    
    def __init__(self, config: ConstraintConfig | None = None):
        self.config = config or ConstraintConfig()
    
    def check_collision(self, positions: np.ndarray) -> bool:
        """
        Check if any pair of bodies has collided.
        
        Args:
            positions: Array of shape (n_bodies, 3) with body positions
        
        Returns:
            True if collision detected, False otherwise
        """
        n = positions.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < self.config.collision_radius:
                    return True
        return False
    
    def check_escape(self, positions: np.ndarray) -> bool:
        """
        Check if any body has escaped.
        
        Args:
            positions: Array of shape (n_bodies, 3) with body positions
        
        Returns:
            True if escape detected, False otherwise
        """
        for pos in positions:
            if np.linalg.norm(pos) > self.config.outer_radius:
                return True
        return False
    
    def check_constraints(
        self,
        system: System,
    ) -> SimulationStatus:
        """
        Check all constraints on current system state.
        
        Args:
            system: Current system state
        
        Returns:
            SimulationStatus indicating constraint violation or success
        """
        positions = system.positions()
        
        if self.check_collision(positions):
            return SimulationStatus.COLLISION
        
        if self.check_escape(positions):
            return SimulationStatus.ESCAPE
        
        return SimulationStatus.SUCCESS
