"""Candidate dataclass for storing promising orbit solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fairy_orbit.simulation.status import SimulationStatus


@dataclass
class Candidate:
    """A candidate orbit solution with full metadata."""
    
    # Initial conditions
    v_rad: float
    v_tan: float
    k: float  # Mass ratio
    planet_mass: float
    fairy_mass: float
    radius: float
    G: float
    
    # Simulation results
    status: SimulationStatus
    score: float
    integration_time: float
    
    # Score components (modular for future evaluators)
    score_components: dict[str, float] = field(default_factory=dict)
    
    # Trajectory data (optional, for verification)
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    times: np.ndarray | None = None
    
    # Verification status
    verified: bool = False
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: Candidate) -> bool:
        """Sort candidates by score (lower is better)."""
        return self.score < other.score
    
    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to dictionary for serialization."""
        return {
            "v_rad": self.v_rad,
            "v_tan": self.v_tan,
            "k": self.k,
            "planet_mass": self.planet_mass,
            "fairy_mass": self.fairy_mass,
            "radius": self.radius,
            "G": self.G,
            "status": self.status.value,
            "score": self.score,
            "integration_time": self.integration_time,
            "score_components": self.score_components,
            "verified": self.verified,
            "metadata": self.metadata,
            # Convert numpy arrays to lists if present
            "positions": self.positions.tolist() if self.positions is not None else None,
            "velocities": self.velocities.tolist() if self.velocities is not None else None,
            "times": self.times.tolist() if self.times is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        """Create candidate from dictionary."""
        # Convert lists back to numpy arrays
        positions = np.array(data["positions"]) if data.get("positions") else None
        velocities = np.array(data["velocities"]) if data.get("velocities") else None
        times = np.array(data["times"]) if data.get("times") else None
        
        return cls(
            v_rad=data["v_rad"],
            v_tan=data["v_tan"],
            k=data["k"],
            planet_mass=data["planet_mass"],
            fairy_mass=data["fairy_mass"],
            radius=data["radius"],
            G=data["G"],
            status=SimulationStatus(data["status"]),
            score=data["score"],
            integration_time=data["integration_time"],
            score_components=data.get("score_components", {}),
            positions=positions,
            velocities=velocities,
            times=times,
            verified=data.get("verified", False),
            metadata=data.get("metadata", {}),
        )
