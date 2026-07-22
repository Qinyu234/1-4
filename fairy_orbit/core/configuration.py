"""Configuration model for physical parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Configuration:
    """
    Physical configuration for the simulation.
    
    Explicitly saves all physical assumptions.
    No hidden parameters.
    """
    G: float  # Gravitational constant
    central_mass: float  # Mass of central body
    mass_ratio: float  # Ratio of fairy mass to central mass
    central_radius: float  # Radius of central body
    canonical_units: str = "dimensionless"  # Unit system description
    
    @property
    def fairy_mass(self) -> float:
        """Mass of fairy bodies."""
        return self.mass_ratio * self.central_mass
    
    def copy(self) -> Configuration:
        """Create a copy of the configuration."""
        return Configuration(
            G=self.G,
            central_mass=self.central_mass,
            mass_ratio=self.mass_ratio,
            central_radius=self.central_radius,
            canonical_units=self.canonical_units,
        )
