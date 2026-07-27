"""REBOUND (IAS15) integrator producing Trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.core.body import System, angular_momentum, total_energy
from fairy_orbit.core.criteria import SimulationStatus, evaluate_status
from fairy_orbit.engine.trajectory import Trajectory

try:
    import rebound

    REBOUND_AVAILABLE = True
except ImportError:  # pragma: no cover
    rebound = None  # type: ignore
    REBOUND_AVAILABLE = False


@dataclass
class ReboundConfig:
    integrator: str = "ias15"
    dt: float = 0.01
    soft_floor: float = 0.0
    stop_on_collision: bool = True
    stop_on_escape: bool = True
    # IAS15 accuracy: smaller epsilon -> smaller adaptive timesteps -> higher precision.
    # None keeps REBOUND's default (1e-9). Set epsilon=0 with dt for fixed timestep.
    epsilon: float | None = None
    # Floor on the adaptive timestep. Default guards exact-symmetry configs where
    # IAS15's error estimator can collapse dt to ~1e-16 and hang.
    min_dt: float = 1e-8


def _apply_ias15_precision(sim: "rebound.Simulation", config: ReboundConfig) -> None:
    """Apply IAS15 precision controls (no-op for other integrators)."""
    if config.integrator.lower() != "ias15":
        return
    if config.epsilon is not None:
        sim.ri_ias15.epsilon = float(config.epsilon)
        # epsilon == 0 means fixed timestep; honour the requested dt.
        if config.epsilon == 0.0:
            sim.dt = config.dt
    if config.min_dt > 0.0:
        sim.ri_ias15.min_dt = float(config.min_dt)


def system_to_rebound(system: System) -> "rebound.Simulation":
    if not REBOUND_AVAILABLE:
        raise ImportError("REBOUND is not installed. Install with: pip install rebound")
    sim = rebound.Simulation()
    sim.G = system.G
    for body in system.bodies:
        sim.add(
            m=body.mass,
            x=float(body.position[0]),
            y=float(body.position[1]),
            z=float(body.position[2]),
            vx=float(body.velocity[0]),
            vy=float(body.velocity[1]),
            vz=float(body.velocity[2]),
            r=float(body.radius),
        )
    return sim


def rebound_to_system(sim: "rebound.Simulation", template: System) -> System:
    out = template.copy()
    for i, body in enumerate(out.bodies):
        p = sim.particles[i]
        body.position = np.array([p.x, p.y, p.z], dtype=float)
        body.velocity = np.array([p.vx, p.vy, p.vz], dtype=float)
    return out


def integrate(
    system: System,
    t_end: float,
    n_outputs: int = 1000,
    config: ReboundConfig | None = None,
) -> Trajectory:
    """
    Integrate with REBOUND IAS15 and record Trajectory samples.
    """
    config = config or ReboundConfig()
    if not REBOUND_AVAILABLE:
        raise ImportError("REBOUND is not installed. Install with: pip install rebound")

    sim = system_to_rebound(system)
    sim.integrator = config.integrator
    if config.integrator.lower() != "ias15":
        sim.dt = config.dt
    _apply_ias15_precision(sim, config)

    times = np.linspace(0.0, float(t_end), int(n_outputs))
    n = system.n
    positions = np.zeros((len(times), n, 3), dtype=float)
    velocities = np.zeros((len(times), n, 3), dtype=float)
    energies = np.zeros(len(times), dtype=float)
    angular_momenta = np.zeros((len(times), 3), dtype=float)

    status = SimulationStatus.SUCCESS
    last_i = 0
    working = system.copy()

    for i, t in enumerate(times):
        sim.integrate(float(t))
        working = rebound_to_system(sim, working)
        positions[i] = working.positions()
        velocities[i] = working.velocities()
        energies[i] = total_energy(working)
        angular_momenta[i] = angular_momentum(working)
        last_i = i

        st = evaluate_status(working, soft_floor=config.soft_floor)
        if st == SimulationStatus.COLLISION and config.stop_on_collision:
            status = st
            break
        if st == SimulationStatus.ESCAPE and config.stop_on_escape:
            status = st
            break

    # Trim if stopped early
    if last_i < len(times) - 1:
        sl = slice(0, last_i + 1)
        times = times[sl]
        positions = positions[sl]
        velocities = velocities[sl]
        energies = energies[sl]
        angular_momenta = angular_momenta[sl]

    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        energies=energies,
        angular_momenta=angular_momenta,
        labels=list(system.labels),
        G=system.G,
        masses=system.masses(),
        status=status.value,
    )


def compute_megno(
    system: System,
    t_end: float,
    config: ReboundConfig | None = None,
) -> float:
    """Run a MEGNO integration and return the final MEGNO value."""
    config = config or ReboundConfig()
    if not REBOUND_AVAILABLE:
        raise ImportError("REBOUND is not installed. Install with: pip install rebound")

    sim = system_to_rebound(system)
    sim.integrator = config.integrator
    _apply_ias15_precision(sim, config)
    sim.init_megno()
    sim.integrate(float(t_end))
    return float(sim.megno())
