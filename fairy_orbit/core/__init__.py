"""Core units, config, state carriers, and physical criteria."""

from fairy_orbit.core.body import Body, System
from fairy_orbit.core.config import CanonicalUnits, SystemConfig
from fairy_orbit.core.criteria import (
    SimulationStatus,
    check_collision,
    check_escape,
    specific_orbital_energy,
)

__all__ = [
    "Body",
    "System",
    "CanonicalUnits",
    "SystemConfig",
    "SimulationStatus",
    "check_collision",
    "check_escape",
    "specific_orbital_energy",
]
