"""State model for N-body system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """
    System state at a given time.
    
    Contains the fundamental physical state of the system.
    This is the ground truth - all other representations derive from this.
    """
    time: float
    positions: np.ndarray  # Shape: (n_bodies, 3)
    velocities: np.ndarray  # Shape: (n_bodies, 3)
    
    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        
        # Validate shapes
        assert self.positions.shape[1] == 3, "Positions must be (n_bodies, 3)"
        assert self.velocities.shape[1] == 3, "Velocities must be (n_bodies, 3)"
        assert self.positions.shape[0] == self.velocities.shape[0], "Position and velocity arrays must have same length"
    
    @property
    def n_bodies(self) -> int:
        """Number of bodies in the system."""
        return self.positions.shape[0]
    
    def copy(self) -> State:
        """Create a copy of the state."""
        return State(
            time=self.time,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
        )
    
    def get_body_state(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get position and velocity of a specific body.
        
        Args:
            index: Body index
        
        Returns:
            (position, velocity) tuple
        """
        return self.positions[index].copy(), self.velocities[index].copy()
