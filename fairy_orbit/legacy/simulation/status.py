"""Simulation status enumeration for constraint filtering."""

from __future__ import annotations

from enum import Enum


class SimulationStatus(Enum):
    """Status of a simulation run."""
    SUCCESS = "success"  # Simulation completed without constraint violations
    COLLISION = "collision"  # Bodies collided (distance < collision_radius)
    ESCAPE = "escape"  # Body escaped (norm(position) > outer_radius)
    TIMEOUT = "timeout"  # Simulation reached time limit without completion
