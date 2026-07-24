"""Canonical units and system configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalUnits:
    """Dimensionless units fixed by a reference semi-major axis and GM."""

    a_ref: float = 1.0
    GM: float = 1.0

    @property
    def time_unit(self) -> float:
        """Orbital time unit ~ period / (2π) for a circular orbit at a_ref."""
        return float((self.a_ref**3 / self.GM) ** 0.5)

    @property
    def velocity_unit(self) -> float:
        return float((self.GM / self.a_ref) ** 0.5)

    def period(self, a: float) -> float:
        return float(2.0 * 3.141592653589793 * (a**3 / self.GM) ** 0.5)


@dataclass
class SystemConfig:
    """Physical configuration for 1 central + N fairy bodies."""

    G: float = 1.0
    central_mass: float = 1.0
    mass_ratio: float = 1e-4
    central_radius: float = 0.01
    fairy_radius: float = 0.001
    units: CanonicalUnits | None = None

    def __post_init__(self) -> None:
        if self.units is None:
            self.units = CanonicalUnits(a_ref=1.0, GM=self.G * self.central_mass)

    @property
    def fairy_mass(self) -> float:
        return self.mass_ratio * self.central_mass

    @property
    def mu(self) -> float:
        return self.G * self.central_mass
