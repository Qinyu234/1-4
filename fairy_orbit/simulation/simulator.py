"""Simulator entry point implementing unified simulation pipeline."""

from __future__ import annotations
import numpy as np
from fairy_orbit.physics.body import System
from fairy_orbit.simulation.cache import build_cache_key, ensure_cache_db, save_cached_trajectory
from fairy_orbit.simulation.trajectory import Trajectory

def simulate(
    system: System,
    dt: float,
    t_end: float,
    solver_type: str = "own",  # "own", "rebound"
    record_every: int = 1,
    cache_path: str | None = None,
) -> Trajectory:
    """Run simulation on the given system and return a Trajectory."""
    if solver_type == "rebound":
        from fairy_orbit.physics.rebound_adapter import system_to_rebound
        sim = system_to_rebound(system)
        sim.dt = dt
        n_steps = int(np.ceil(t_end / dt))
        times = []
        positions = []
        velocities = []
        energies = []
        angular_momenta = []
        
        def record(t: float) -> None:
            times.append(t)
            pos = np.zeros((sim.N, 3))
            vel = np.zeros((sim.N, 3))
            for idx in range(sim.N):
                p = sim.particles[idx]
                pos[idx] = [p.x, p.y, p.z]
                vel[idx] = [p.vx, p.vy, p.vz]
            positions.append(pos)
            velocities.append(vel)
            energies.append(sim.energy())
            angular_momenta.append(np.array(sim.angular_momentum()))

        t = 0.0
        record(t)
        for step in range(1, n_steps + 1):
            sim.integrate(step * dt)
            t = step * dt
            if step % record_every == 0 or step == n_steps:
                record(t)
                
        traj = Trajectory(
            times=np.asarray(times, dtype=float),
            positions=np.stack(positions, axis=0),
            velocities=np.stack(velocities, axis=0),
            energies=np.asarray(energies, dtype=float),
            angular_momenta=np.stack(angular_momenta, axis=0),
            labels=list(system.labels),
            G=system.G,
            masses=system.masses(),
        )
        if cache_path is not None:
            conn = ensure_cache_db(cache_path)
            cache_key = build_cache_key(system, dt, t_end, record_every, solver_type)
            save_cached_trajectory(conn, cache_key, traj)
        return traj
    else:
        # Default own solver (Adaptive / Encounter integrator)
        from fairy_orbit.simulation.adaptive_simulator import AdaptiveSimulator, AdaptiveConfig
        config = AdaptiveConfig(influence_threshold=1.5)
        sim = AdaptiveSimulator(config)
        # run adaptive simulation
        sim.run(system, dt, t_end, record_every=record_every)
        return Trajectory(
            times=np.array(sim.times),
            positions=np.stack(sim.positions_history, axis=0),
            velocities=np.stack(sim.velocities_history, axis=0),
            energies=np.array(sim.energies),
            angular_momenta=np.stack(sim.angular_momenta, axis=0),
            labels=list(system.labels),
            G=system.G,
            masses=system.masses(),
        )
