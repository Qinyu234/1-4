"""REBOUND adapter to convert System to REBOUND particles."""

from __future__ import annotations
import rebound
from fairy_orbit.physics.body import System

def system_to_rebound(system: System) -> rebound.Simulation:
    """Convert System state to a rebound.Simulation object."""
    sim = rebound.Simulation()
    sim.G = system.G
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
    return sim
